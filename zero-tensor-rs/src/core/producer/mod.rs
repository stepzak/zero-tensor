pub mod builder;
pub mod error;
mod helpers;
mod msg;
#[cfg(test)]
mod tests;
pub use error::*;
use parking_lot::Mutex;

use crate::core::{
    dataset::item::StrideType,
    writer::{TensorWriter, TensorWriterCache},
};

use super::{
    buffer::{ZeroTensorBuffer, control_block::ZeroTensorControlBlock, tensor_meta::TensorHeader},
    dataset::{
        ZeroTensorDataset,
        item::{ShapeType, TensorDT},
    },
};

use std::{
    fs,
    io::{self, Read, Write},
    mem::offset_of,
    os::unix::net::{UnixListener, UnixStream},
    sync::atomic::AtomicU8,
    thread,
    time::Duration,
};
use std::{
    io::{BufRead, BufReader},
    path::PathBuf,
};
use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

pub use builder::*;

pub const DEFAULT_SLOTS: u64 = 2;
pub const CONSUMER_START: &str = "START";
pub const CONSUMER_STOP: &str = "STOP";
pub const PRODUCER_EPOCH_DONE: &str = "EPOCH_DONE\n";
pub const CONSUMER_RESP_BUFFER: usize = get_max_resp_len() * 2;
pub const DEFAULT_TIMEOUT_CHECK_CTRLC: u64 = 500;
pub const VERSION: &str = "0.5.0";

const fn const_max_usize(a: usize, b: usize) -> usize {
    if a > b { a } else { b }
}

const fn get_max_resp_len() -> usize {
    const_max_usize(CONSUMER_START.len(), CONSUMER_STOP.len())
}

#[derive(Debug, Clone)]
enum ZTConsumerCmd {
    Start,
    Stop,
}

#[derive(Debug, Clone)]
enum ZTProducerCmd {
    EpochEnd,
}

pub struct ZeroTensorProducer {
    buffer: ZeroTensorBuffer,
    listener: UnixListener,
    sock_path: PathBuf,
    running: Arc<AtomicBool>,
    read_timeout: Option<u64>,
    shuffle: bool,
    seed: Option<u64>,
    max_steps: Option<usize>,
}

impl ZeroTensorProducer {
    fn next_command(
        &self,
        reader: &mut BufReader<UnixStream>,
        buf: &mut String,
    ) -> Result<ZTConsumerCmd, io::Error> {
        buf.clear();

        if !self.running.load(Ordering::SeqCst) {
            return Err(io::Error::from(io::ErrorKind::Interrupted));
        }

        match reader.read_line(buf) {
            Ok(0) => Ok(ZTConsumerCmd::Stop),
            Ok(_) => match buf.trim() {
                CONSUMER_STOP => Ok(ZTConsumerCmd::Stop),
                CONSUMER_START => Ok(ZTConsumerCmd::Start),
                _ => Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "Unexpected protocol violation from consumer: '{}'",
                        buf.trim()
                    ),
                )),
            },
            Err(e)
                if e.kind() == io::ErrorKind::WouldBlock || e.kind() == io::ErrorKind::TimedOut =>
            {
                Err(io::ErrorKind::WouldBlock.into())
            }
            Err(e) => Err(e),
        }
    }

    fn send_cmd(stream: &mut UnixStream, cmd: ZTProducerCmd) -> Result<(), io::Error> {
        match cmd {
            ZTProducerCmd::EpochEnd => {
                stream.write_all(PRODUCER_EPOCH_DONE.as_bytes())?;
            }
        }
        stream.flush()
    }

    fn stop(&mut self) {
        self.buffer.control_block_mut().stop();
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    fn send_handshake(&self, stream: &mut UnixStream, keys: &[&str]) -> Result<(), io::Error> {
        let cb = self.buffer.control_block();
        let joined_keys = keys.join(",");
        let msg = msg::HandshakeBuilder::new()
            .add_key("cb_size", ZeroTensorControlBlock::SIZE)
            .add_key("head_offset", offset_of!(ZeroTensorControlBlock, head))
            .add_key("head_size", size_of_val(&cb.head))
            .add_key("tail_offset", offset_of!(ZeroTensorControlBlock, tail))
            .add_key("tail_size", size_of_val(&cb.tail))
            .add_key("nslots_offset", ZeroTensorControlBlock::nslots_offset())
            .add_key("nslots_size", size_of_val(&cb.nslots()))
            .add_key(
                "slot_size_offset",
                ZeroTensorControlBlock::slot_size_offset(),
            )
            .add_key("slot_size_size", size_of_val(&cb.slot_size()))
            .add_key(
                "is_running_offset",
                ZeroTensorControlBlock::is_running_offset(),
            )
            .add_key("is_running_size", cb.is_running_size())
            .add_key("header_size", size_of::<TensorHeader>())
            .add_key("dt_offset", TensorHeader::dt_offset())
            .add_key("dt_size", size_of::<TensorDT>())
            .add_key("ndims_offset", TensorHeader::ndims_offset())
            .add_key("ndims_size", size_of::<u8>())
            .add_key("is_ready_offset", TensorHeader::is_ready_offset())
            .add_key("is_ready_size", size_of::<AtomicU8>())
            .add_key("shape_type_size", size_of::<ShapeType>())
            .add_key("header_size", size_of::<TensorHeader>())
            .add_key("slot_alignment", TensorWriter::ALIGNMENT)
            .add_key("stride_type_size", size_of::<StrideType>())
            .add_key("keys", joined_keys)
            .build();

        stream.write_all(msg.as_bytes())
    }

    fn reshuffle_indices(&self, indices: &mut [usize], epoch: usize) {
        for (i, val) in indices.iter_mut().enumerate() {
            *val = i;
        }

        if self.shuffle {
            let effective_seed = match self.seed {
                Some(base_seed) => base_seed.wrapping_add(epoch as u64),
                None => fastrand::u64(..),
            };
            let mut rng = fastrand::Rng::with_seed(effective_seed);
            rng.shuffle(indices);
        }
    }

    fn is_peer_alive(stream: &mut UnixStream) -> bool {
        let mut buf = [0u8; 1];
        match stream.read(&mut buf) {
            Ok(0) => false,
            Ok(_) => true,
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => true,
            Err(_) => false,
        }
    }

    fn start_streaming_loop<'a, D: ZeroTensorDataset<'a>>(
        &mut self,
        dataset: &'a D,
        batch_size: usize,
        stream: &mut UnixStream,
    ) -> Result<(), ZTProducerErr<D::Error>> {
        if dataset.len() == 0 || batch_size == 0 {
            return Ok(());
        }

        let mut reader = BufReader::new(stream.try_clone().map_err(ZTProducerErr::IoError)?);
        let mut buf = String::with_capacity(CONSUMER_RESP_BUFFER);
        let layout = if let Some(s) = dataset.static_layouts() {
            s
        } else {
            &dataset
                .dynamic_layouts(&[0])
                .map_err(|e| ZTProducerErr::DatasetError {
                    idx: 0.into(),
                    source: e,
                })?
        };
        let keys: Vec<&str> = layout.keys().copied().collect();
        let tensors_per_sample = keys.len();
        let mut caches: Vec<Mutex<TensorWriterCache<'_>>> = (0..batch_size)
            .map(|_| Mutex::new(TensorWriterCache::with_capacity(tensors_per_sample)))
            .collect();

        match self.next_command(&mut reader, &mut buf)? {
            ZTConsumerCmd::Start => {
                self.send_handshake(stream, &keys)?;
            }
            ZTConsumerCmd::Stop => return Ok(()),
        }

        let timeout = std::cmp::min(
            DEFAULT_TIMEOUT_CHECK_CTRLC,
            self.read_timeout.unwrap_or(DEFAULT_TIMEOUT_CHECK_CTRLC),
        );
        stream
            .set_read_timeout(Some(std::time::Duration::from_millis(timeout)))
            .map_err(ZTProducerErr::IoError)?;

        let steps_per_epoch = dataset.len().div_ceil(batch_size);
        let mut current_epoch = 0;
        let mut epoch_step = steps_per_epoch;
        let mut indices: Vec<usize> = (0..dataset.len()).collect();

        loop {
            if !self.is_running() {
                self.stop();
                return Err(ZTProducerErr::IoError(std::io::Error::from(
                    std::io::ErrorKind::Interrupted,
                )));
            }

            let cb = self.buffer.control_block();
            if !cb.is_running() {
                return Ok(());
            }

            let total_steps = epoch_step + current_epoch * steps_per_epoch;
            if let Some(max) = self.max_steps
                && total_steps >= max
            {
                return Ok(());
            }

            if epoch_step >= steps_per_epoch {
                current_epoch += 1;
                epoch_step = 0;
                if self.shuffle {
                    self.reshuffle_indices(&mut indices, current_epoch);
                }
                if current_epoch != 1 {
                    Self::send_cmd(stream, ZTProducerCmd::EpochEnd)?;
                }
            }

            let start_idx = batch_size * epoch_step;
            let end_idx = std::cmp::min(start_idx + batch_size, dataset.len());
            if start_idx >= end_idx {
                epoch_step += 1;
                continue;
            }

            let batch_indices = &indices[start_idx..end_idx];

            let cur_head = cb.head.fetch_add(1, Ordering::AcqRel);
            let cur_tail = cb.tail.load(Ordering::Acquire);

            if cur_head - cur_tail >= cb.nslots() {
                cb.head.fetch_sub(1, Ordering::Release);
                while cb.head.load(Ordering::Acquire) - cb.tail.load(Ordering::Acquire)
                    >= cb.nslots()
                {
                    if !self.is_running() {
                        self.stop();
                        return Err(ZTProducerErr::IoError(std::io::Error::from(
                            std::io::ErrorKind::Interrupted,
                        )));
                    }
                    if !cb.is_running() {
                        return Ok(());
                    }
                    if !Self::is_peer_alive(stream) {
                        return Err(
                            std::io::Error::from(std::io::ErrorKind::ConnectionAborted).into()
                        );
                    }
                    std::hint::spin_loop();
                }
                continue;
            }

            let slot_idx = (cur_head % cb.nslots()) as usize;
            let offset = ZeroTensorControlBlock::slot_offset(slot_idx, cb.slot_size() as usize);

            let (single_layouts, batch_layouts, element_size_bytes, meta, total_data_bytes) =
                helpers::prepare_batch_metadata(dataset, batch_indices)?;

            let caches_ref: &mut [Mutex<TensorWriterCache<'_>>] =
                unsafe { std::mem::transmute(&mut *caches) };
            let batch_meta = (
                &single_layouts,
                &batch_layouts,
                element_size_bytes,
                meta,
                total_data_bytes,
            );
            helpers::copy_batch_to_shm(
                &mut self.buffer,
                &self.running,
                dataset,
                batch_indices,
                offset,
                batch_meta,
                caches_ref,
            )?;

            for cache_mutex in caches.iter_mut() {
                cache_mutex.get_mut().clear();
            }

            self.buffer.set_slot_ready(offset);

            epoch_step += 1;
        }
    }

    pub fn start_streaming<'a, D: ZeroTensorDataset<'a>>(
        &'a mut self,
        dataset: &'a D,
        batch_size: usize,
    ) -> Result<(), ZTProducerErr<D::Error>> {
        self.listener
            .set_nonblocking(true)
            .map_err(ZTProducerErr::IoError)?;

        let poll_interval = Duration::from_millis(DEFAULT_TIMEOUT_CHECK_CTRLC);

        loop {
            if !self.running.load(Ordering::SeqCst) {
                return Err(ZTProducerErr::IoError(io::Error::from(
                    io::ErrorKind::Interrupted,
                )));
            }

            let mut stream = match self.listener.accept() {
                Ok((stream, _addr)) => {
                    stream
                        .set_nonblocking(false)
                        .map_err(ZTProducerErr::IoError)?;
                    stream
                }
                Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                    thread::sleep(poll_interval);
                    continue;
                }
                Err(e) => {
                    return Err(ZTProducerErr::IoError(e));
                }
            };

            return self.start_streaming_loop(dataset, batch_size, &mut stream);
        }
    }
}

impl Drop for ZeroTensorProducer {
    fn drop(&mut self) {
        let cb = self.buffer.control_block();
        let head = cb.head.load(Ordering::Acquire);
        let timeout = std::time::Instant::now() + std::time::Duration::from_secs(5);

        while cb.tail.load(Ordering::Acquire) < head {
            if std::time::Instant::now() > timeout {
                eprintln!("[ZeroTensor] Warning: Consumer did not drain all data");
                break;
            }
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
        if self.sock_path.exists() {
            _ = fs::remove_file(&self.sock_path);
        }
    }
}
