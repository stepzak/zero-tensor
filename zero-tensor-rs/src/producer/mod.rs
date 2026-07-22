use crate::{
    buffer::{ZTBufErr, ZeroTensorBuffer, get_dt_size, tensor_meta::TensorHeader},
    dataset::{
        ZeroTensorDataset,
        item::{ShapeType, StrideType},
    },
};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::{
    fs,
    io::{self, Write},
    os::unix::net::{UnixListener, UnixStream},
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
use thiserror::Error;

pub const DEFAULT_SLOTS: usize = 2;
pub const CONSUMER_RESP_BUFFER: usize = b"RELEASE".len() * 2;
pub const DEFAULT_TIMEOUT_CHECK_CTRLC: u64 = 500;

pub struct ZeroTensorProducer {
    buffer: ZeroTensorBuffer,
    slot_size: usize,
    current_step: usize,
    steps: usize,
    nslots: usize,
    listener: UnixListener,
    sock_path: PathBuf,
    running: Arc<AtomicBool>,
    read_timeout: Option<u64>,
    shuffle: bool,
    seed: Option<u64>,
}

#[derive(Debug, Error)]
pub enum ZTProducerErr {
    #[error("ZT Buffer Error: {0}")]
    ZTBufferError(ZTBufErr),

    #[error("Io error: {0}")]
    IoError(io::Error),
}

impl ZeroTensorProducer {
    pub fn new<P: AsRef<Path>, N: Into<Option<usize>>, R: Into<Option<u64>>>(
        steps: usize,
        step_size: usize,
        shm_filename: &str,
        socket_addr: P,
        num_slots: N,
        read_timeout: R,
        overwrite_socket: bool,
        shuffle: bool,
        seed: R,
    ) -> Result<Self, ZTProducerErr> {
        let running = Arc::new(AtomicBool::new(true));
        let rclone = running.clone();
        let _ = ctrlc::set_handler(move || {
            println!("Handler worked");
            rclone.store(false, Ordering::SeqCst);
        });

        let nslots = num_slots.into().unwrap_or(DEFAULT_SLOTS);
        let total_size = nslots * step_size;
        let buffer = ZeroTensorBuffer::new(shm_filename, total_size)
            .map_err(ZTProducerErr::ZTBufferError)?;

        let path = socket_addr.as_ref();
        if path.exists() {
            if overwrite_socket {
                fs::remove_file(path).map_err(ZTProducerErr::IoError)?;
            } else {
                return Err(ZTProducerErr::IoError(io::Error::from(
                    io::ErrorKind::AddrInUse,
                )));
            }
        }
        let listener = UnixListener::bind(path).map_err(ZTProducerErr::IoError)?;

        Ok(ZeroTensorProducer {
            buffer,
            slot_size: step_size,
            steps,
            current_step: 0,
            listener,
            nslots,
            sock_path: path.into(),
            read_timeout: read_timeout.into(),
            running,
            shuffle,
            seed: seed.into(),
        })
    }

    fn start_streaming_loop<D: ZeroTensorDataset>(
        &mut self,
        dataset: &D,
        batch_size: usize,
        stream: &mut UnixStream,
    ) -> Result<(), ZTProducerErr> {
        if dataset.len() == 0 || batch_size == 0 {
            return Ok(());
        }

        let mut buf = String::with_capacity(CONSUMER_RESP_BUFFER);
        let timeout = std::cmp::min(
            DEFAULT_TIMEOUT_CHECK_CTRLC,
            self.read_timeout.unwrap_or(DEFAULT_TIMEOUT_CHECK_CTRLC),
        );

        let steps_per_epoch = (dataset.len() + batch_size - 1) / batch_size;
        let mut current_epoch = usize::MAX;
        let mut indices: Vec<usize> = (0..dataset.len()).collect();

        let mut reader = BufReader::new(stream.try_clone().map_err(ZTProducerErr::IoError)?);
        stream
            .set_read_timeout(Some(std::time::Duration::from_millis(timeout)))
            .map_err(ZTProducerErr::IoError)?;

        loop {
            if !self.running.load(Ordering::SeqCst) {
                return Err(ZTProducerErr::IoError(io::Error::from(
                    io::ErrorKind::Interrupted,
                )));
            }
            if self.current_step == self.steps {
                return Ok(());
            }

            // 1. Управление эпохами и shuffle
            let step_epoch = self.current_step / steps_per_epoch;
            if step_epoch != current_epoch {
                current_epoch = step_epoch;
                self.reshuffle_indices(&mut indices, step_epoch);
            }

            let epoch_step = self.current_step % steps_per_epoch;
            let start_idx = epoch_step * batch_size;
            let end_idx = std::cmp::min(start_idx + batch_size, dataset.len());

            if start_idx >= end_idx {
                self.current_step += 1;
                continue;
            }

            let batch_indices = &indices[start_idx..end_idx];
            let offset = (self.current_step % self.nslots) * self.slot_size;

            let (data_start_offset, total_data_bytes, element_size_bytes) =
                self.prepare_batch_metadata(dataset, batch_indices, offset)?;

            self.copy_batch_to_shm(
                dataset,
                batch_indices,
                offset,
                data_start_offset,
                total_data_bytes,
                element_size_bytes,
            )?;

            let msg = format!("READY {}\n", offset);
            stream
                .write_all(msg.as_bytes())
                .map_err(ZTProducerErr::IoError)?;
            stream.flush().map_err(ZTProducerErr::IoError)?;

            self.wait_for_release(&mut reader, &mut buf)?;

            self.current_step += 1;
        }
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

    fn prepare_batch_metadata<D: ZeroTensorDataset>(
        &mut self,
        dataset: &D,
        batch_indices: &[usize],
        offset: usize,
    ) -> Result<(usize, usize, usize), ZTProducerErr> {
        let current_batch_size = batch_indices.len();
        let first_idx = batch_indices[0];

        let (_, first_meta) = dataset.get_item(first_idx).unwrap_or_else(|| {
            panic!("Failed to get first item of batch {first_idx} to extract metadata");
        });

        let dt = first_meta.dt();
        let ndims = (first_meta.shape().len() + 1) as u8;

        let mut batch_shape = vec![current_batch_size as ShapeType];
        batch_shape.extend_from_slice(first_meta.shape());

        let element_strides = first_meta.strides();
        let element_size_elements = first_meta
            .shape()
            .iter()
            .zip(element_strides.iter())
            .map(|(dim, stride)| (dim - 1) * stride)
            .sum::<StrideType>()
            + 1;

        let element_size_bytes = element_size_elements as usize * get_dt_size(dt);
        let mut batch_strides = vec![element_size_bytes as StrideType];

        let mut converted_element_strides = element_strides.to_vec();
        for stride in &mut converted_element_strides {
            *stride *= get_dt_size(dt) as StrideType;
        }
        batch_strides.extend_from_slice(&converted_element_strides);

        let header_meta = TensorHeader::new(dt, ndims);
        let offs = header_meta.get_offsets();

        self.buffer
            .write_tensor(offset, &batch_shape, &batch_strides, dt, &[]);

        let total_data_bytes = current_batch_size * element_size_bytes;

        Ok((offs.data(), total_data_bytes, element_size_bytes))
    }

    fn copy_batch_to_shm<D: ZeroTensorDataset>(
        &mut self,
        dataset: &D,
        batch_indices: &[usize],
        offset: usize,
        data_start_offset: usize,
        total_data_bytes: usize,
        element_size_bytes: usize,
    ) -> Result<(), ZTProducerErr> {
        let raw_shm_slice = unsafe {
            self.buffer
                .get_item_slice_mut(offset, data_start_offset, total_data_bytes)
        };

        let shm_chunks: Vec<&mut [u8]> = raw_shm_slice.chunks_mut(element_size_bytes).collect();

        let interrupted = batch_indices
            .par_iter()
            .zip(shm_chunks)
            .any(|(&i, shm_chunk)| {
                if !self.running.load(Ordering::SeqCst) {
                    return true;
                }
                let (raw_data, _) = dataset.get_item(i).unwrap_or_else(|| {
                    panic!("Failed to get item {i} from dataset");
                });

                if !self.running.load(Ordering::SeqCst) {
                    return true;
                }

                shm_chunk[..raw_data.len()].copy_from_slice(&raw_data);
                false
            });

        if interrupted {
            return Err(ZTProducerErr::IoError(io::Error::from(
                io::ErrorKind::Interrupted,
            )));
        }

        Ok(())
    }

    fn wait_for_release(
        &self,
        reader: &mut BufReader<UnixStream>,
        buf: &mut String,
    ) -> Result<(), ZTProducerErr> {
        let start_time = std::time::Instant::now();

        loop {
            if !self.running.load(Ordering::SeqCst) {
                return Err(ZTProducerErr::IoError(io::Error::from(
                    io::ErrorKind::Interrupted,
                )));
            }

            match reader.read_line(buf) {
                Ok(0) => return Ok(()),
                Ok(_) => {
                    let trimmed = buf.trim();
                    if trimmed != "RELEASE" {
                        panic!("Unexpected protocol violation from consumer: '{}'", trimmed);
                    }
                    buf.clear();
                    return Ok(());
                }
                Err(e)
                    if e.kind() == io::ErrorKind::WouldBlock
                        || e.kind() == io::ErrorKind::TimedOut =>
                {
                    let el = start_time.elapsed();
                    if let Some(rt) = self.read_timeout {
                        if el.as_millis() >= rt as u128 {
                            return Err(ZTProducerErr::IoError(e));
                        }
                    }
                    continue;
                }
                Err(e) => return Err(ZTProducerErr::IoError(e)),
            }
        }
    }

    pub fn start_streaming<D: ZeroTensorDataset>(
        &mut self,
        dataset: &D,
        batch_size: usize,
    ) -> Result<(), ZTProducerErr> {
        self.current_step = 0;
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
        if self.sock_path.exists() {
            _ = fs::remove_file(&self.sock_path);
        }
    }
}
