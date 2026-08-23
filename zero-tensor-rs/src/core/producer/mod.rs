#[cfg(test)]
mod tests;

use crate::{
    core::dataset::item::{TensorBatchLayout, TensorViewError},
    pipeline::{Pipeline, PipelineError},
};

use super::{
    buffer::{
        ZTBufErr, ZeroTensorBuffer, control_block::ZeroTensorControlBlock, get_dt_size,
        tensor_meta::TensorHeader,
    },
    dataset::{
        ZTDatasetError, ZeroTensorDataset,
        item::{ShapeType, ShapeVec, StrideVec, TensorDT},
    },
};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
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
use thiserror::Error;

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

struct CopyCtx {
    pub data_start_offset: usize,
    pub total_data_bytes: usize,
    pub element_size_bytes: usize,
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
    pipeline: Option<Pipeline>,
}

#[derive(Debug, Error)]
pub enum ZTProducerNewErr {
    #[error("ZT Buffer Error: {0}")]
    ZTBufferError(#[from] ZTBufErr),

    #[error("IO error: {0}")]
    IoError(#[from] io::Error),
}

#[derive(Debug, Error)]
pub enum ZTProducerErr<E: ZTDatasetError + 'static> {
    #[error("ZT Buffer Error: {0}")]
    ZTBufferError(#[from] ZTBufErr),

    #[error("IO error at: {0}")]
    IoError(#[from] io::Error),

    #[error("Dataset error {source}")]
    DatasetError {
        idx: Option<usize>,
        #[source]
        source: E,
    },

    #[error("{0}")]
    ProtocolError(String),

    #[error("Pipeline error {0}")]
    PipelineError(#[from] PipelineError),

    #[error("Tensor View conv error {0}")]
    TensorViewError(#[from] TensorViewError),
}

#[derive(Clone)]
pub struct ZeroTensorProducerBuilder {
    // Required
    slot_size: u64,
    shm_filename: String,
    socket_addr: PathBuf,

    // Optional
    num_slots: u64,
    read_timeout: Option<u64>,
    overwrite_socket: bool,
    shuffle: bool,
    seed: Option<u64>,
    max_steps: Option<usize>,
    pipeline: Option<Pipeline>,
}

impl ZeroTensorProducerBuilder {
    pub fn new<P: AsRef<Path>>(slot_size: u64, shm_filename: &str, socket_addr: P) -> Self {
        let mut s_size = slot_size;
        let rec_slot_size_align = ZeroTensorControlBlock::recommended_slot_alignment() as u64;
        let min_slot_size_align = ZeroTensorControlBlock::min_slot_alignment() as u64;

        if !slot_size.is_multiple_of(min_slot_size_align) {
            //TODO: warning
            s_size = Self::round_slot_size(slot_size, rec_slot_size_align);
        }

        Self {
            slot_size: s_size,
            shm_filename: shm_filename.to_string(),
            socket_addr: socket_addr.as_ref().to_path_buf(),
            num_slots: DEFAULT_SLOTS,
            read_timeout: None,
            overwrite_socket: false,
            shuffle: false,
            seed: None,
            max_steps: None,
            pipeline: None,
        }
    }

    fn round_slot_size(slot_size: u64, rec: u64) -> u64 {
        slot_size.div_ceil(rec) * rec
    }

    pub fn num_slots(mut self, slots: u64) -> Self {
        self.num_slots = slots;
        self
    }

    pub fn read_timeout(mut self, timeout_ms: u64) -> Self {
        self.read_timeout = Some(timeout_ms);
        self
    }

    pub fn overwrite_socket(mut self, overwrite: bool) -> Self {
        self.overwrite_socket = overwrite;
        self
    }

    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn max_steps<M: Into<Option<usize>>>(mut self, max_steps: M) -> Self {
        self.max_steps = max_steps.into();
        self
    }

    pub fn pipeline(mut self, pipeline: Pipeline) -> Self {
        self.pipeline = pipeline.into();
        self
    }

    pub fn build(self) -> Result<ZeroTensorProducer, ZTProducerNewErr> {
        let running = Arc::new(AtomicBool::new(true));
        let rclone = running.clone();

        let _ = ctrlc::set_handler(move || {
            rclone.store(false, Ordering::SeqCst);
        });

        let buffer = ZeroTensorBuffer::new(&self.shm_filename, self.slot_size, self.num_slots)?;

        if self.socket_addr.exists() {
            if self.overwrite_socket {
                fs::remove_file(&self.socket_addr)?;
            } else {
                return Err(ZTProducerNewErr::IoError(io::Error::from(
                    io::ErrorKind::AddrInUse,
                )));
            }
        }

        let listener = UnixListener::bind(&self.socket_addr)?;

        Ok(ZeroTensorProducer {
            buffer,
            listener,
            sock_path: self.socket_addr,
            read_timeout: self.read_timeout,
            running,
            shuffle: self.shuffle,
            seed: self.seed,
            max_steps: self.max_steps,
            pipeline: self.pipeline,
        })
    }
}

impl TryFrom<ZeroTensorProducerBuilder> for ZeroTensorProducer {
    type Error = ZTProducerNewErr;

    fn try_from(value: ZeroTensorProducerBuilder) -> Result<Self, Self::Error> {
        value.build()
    }
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

    fn send_handshake(&self, stream: &mut UnixStream) -> Result<(), io::Error> {
        let cb = self.buffer.control_block();

        let msg = format!(
            "ZT {} \
            cb_size={} \
            head_offset={} head_size={} \
            tail_offset={} tail_size={} \
            nslots_offset={} nslots_size={} \
            slot_size_offset={} slot_size_size={} \
            is_running_offset={} is_running_size={} \
            header_size={} \
            dt_offset={} dt_size={} \
            ndims_offset={} ndims_size={} \
            is_ready_offset={} is_ready_size={} \
            shape_type_size={}\n",
            VERSION,
            ZeroTensorControlBlock::SIZE,
            offset_of!(ZeroTensorControlBlock, head),
            size_of_val(&cb.head),
            offset_of!(ZeroTensorControlBlock, tail),
            size_of_val(&cb.tail),
            ZeroTensorControlBlock::nslots_offset(),
            size_of_val(&cb.nslots()),
            ZeroTensorControlBlock::slot_size_offset(),
            size_of_val(&cb.slot_size()),
            ZeroTensorControlBlock::is_running_offset(),
            cb.is_running_size(),
            size_of::<TensorHeader>(),
            TensorHeader::dt_offset(),
            size_of::<TensorDT>(),
            TensorHeader::ndims_offset(),
            size_of::<u8>(),
            TensorHeader::is_ready_offset(),
            size_of::<AtomicU8>(),
            size_of::<ShapeType>()
        );
        stream.write_all(msg.as_bytes())
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

    fn start_streaming_loop<D: ZeroTensorDataset>(
        &mut self,
        dataset: &D,
        batch_size: usize,
        stream: &mut UnixStream,
    ) -> Result<(), ZTProducerErr<D::Error>> {
        if dataset.len() == 0 || batch_size == 0 {
            return Ok(());
        }

        let mut reader = BufReader::new(stream.try_clone().map_err(ZTProducerErr::IoError)?);
        let mut buf = String::with_capacity(CONSUMER_RESP_BUFFER);

        match self.next_command(&mut reader, &mut buf)? {
            ZTConsumerCmd::Start => {
                self.send_handshake(stream)?;
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
            let cb = self.buffer.control_block();
            if !self.is_running() {
                self.stop();
                return Err(ZTProducerErr::IoError(io::Error::from(
                    io::ErrorKind::Interrupted,
                )));
            }

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
                        return Err(ZTProducerErr::IoError(io::Error::from(
                            io::ErrorKind::Interrupted,
                        )));
                    }
                    if !cb.is_running() {
                        return Ok(());
                    }
                    if !Self::is_peer_alive(stream) {
                        return Err(io::Error::from(io::ErrorKind::ConnectionAborted).into());
                    }
                    std::hint::spin_loop();
                }
                continue;
            }

            let slot_idx = (cur_head % cb.nslots()) as usize;
            let offset = ZeroTensorControlBlock::slot_offset(slot_idx, cb.slot_size() as usize);
            let (data_start_offset, total_data_bytes, element_size_bytes, layout) =
                self.prepare_batch_metadata(dataset, batch_indices, offset)?;
            let ctx = CopyCtx {
                data_start_offset,
                total_data_bytes,
                element_size_bytes,
            };
            self.copy_batch_to_shm(dataset, batch_indices, offset, ctx, &layout)?;
            self.buffer.set_slot_ready(offset);
            epoch_step += 1;
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

    /// Returns: (data_start_offset, total_data_bytes, element_size_bytes, layout)
    fn prepare_batch_metadata<D: ZeroTensorDataset>(
        &mut self,
        dataset: &D,
        batch_indices: &[usize],
        offset: usize,
    ) -> Result<(usize, usize, usize, TensorBatchLayout), ZTProducerErr<D::Error>> {
        let current_batch_size = batch_indices.len();

        let layout =
            dataset
                .get_batch_layout(batch_indices)
                .map_err(|e| ZTProducerErr::DatasetError {
                    idx: e.index(),
                    source: e,
                })?;
        let dt = layout.dt();
        let shape = layout.shape();
        let ndims = shape.len() + 1;
        let strides = layout.strides();

        if strides.len() + 1 != ndims {
            return Err(ZTBufErr::InvalidShape(strides.len() as u8 + 1, ndims as u8).into());
        }

        let mut batch_shape = ShapeVec::with_capacity(ndims);
        batch_shape.push(current_batch_size as ShapeType);
        batch_shape.extend_from_slice(shape);

        let element_size_bytes = layout.total_elements() * get_dt_size(dt);

        let mut batch_strides = StrideVec::with_capacity(ndims);
        batch_strides.push(layout.total_elements());
        for &s in layout.strides() {
            batch_strides.push(s);
        }

        let header_meta = TensorHeader::new(dt, ndims as u8);
        let offs = header_meta.get_offsets();

        self.buffer
            .write_tensor(offset, &batch_shape, &batch_strides, dt, &[])?;

        let total_data_bytes = current_batch_size * element_size_bytes;
        Ok((offs.data(), total_data_bytes, element_size_bytes, layout))
    }

    fn check_running(running: &Arc<AtomicBool>) -> Result<(), io::Error> {
        if !running.load(Ordering::SeqCst) {
            return Err(io::Error::from(io::ErrorKind::Interrupted));
        }
        Ok(())
    }

    fn process_chunk<D: ZeroTensorDataset>(
        running: &Arc<AtomicBool>,
        pipeline: &Option<Pipeline>,
        shm_chunk: &mut [u8],
        dataset: &D,
        i: usize,
        layout: &TensorBatchLayout,
        atomic: bool,
    ) -> Result<(), ZTProducerErr<D::Error>> {
        if atomic {
            Self::check_running(running)?;
        }
        let l = shm_chunk.len();
        let written =
            dataset
                .write_item_into(i, shm_chunk)
                .map_err(|e| ZTProducerErr::DatasetError {
                    idx: Some(i),
                    source: e,
                })?;
        if atomic {
            Self::check_running(running)?;
        }
        if written > l {
            return Err(ZTProducerErr::ProtocolError(format!(
                "Dataset has written {written} but chunk size was only {} as index {i}",
                shm_chunk.len()
            )));
        }
        if written < shm_chunk.len() {
            shm_chunk[written..].fill(0);
        }
        if let Some(p) = &pipeline {
            let mut view = layout.try_view_mut(shm_chunk)?;
            p.exec(&mut view)?;
        }
        Ok(())
    }

    fn copy_batch_to_shm<D: ZeroTensorDataset>(
        &mut self,
        dataset: &D,
        batch_indices: &[usize],
        offset: usize,
        ctx: CopyCtx,
        layout: &TensorBatchLayout,
    ) -> Result<(), ZTProducerErr<D::Error>> {
        let CopyCtx {
            element_size_bytes,
            total_data_bytes,
            data_start_offset,
        } = ctx;
        let raw_shm_slice = unsafe {
            self.buffer
                .get_item_slice_mut(offset, data_start_offset, total_data_bytes)
        }?;

        const RAYON_THRESHOLD: usize = 256 * 1024;

        if total_data_bytes < RAYON_THRESHOLD {
            for (shm_chunk, &i) in raw_shm_slice
                .chunks_mut(element_size_bytes)
                .zip(batch_indices)
            {
                Self::process_chunk(
                    &self.running,
                    &self.pipeline,
                    shm_chunk,
                    dataset,
                    i,
                    layout,
                    false,
                )?;
            }
        } else {
            raw_shm_slice
                .par_chunks_mut(element_size_bytes)
                .zip(batch_indices)
                .try_for_each(|(shm_chunk, &i)| -> Result<(), ZTProducerErr<D::Error>> {
                    Self::process_chunk(
                        &self.running,
                        &self.pipeline,
                        shm_chunk,
                        dataset,
                        i,
                        layout,
                        true,
                    )
                })?;
        }

        Ok(())
    }

    pub fn start_streaming<D: ZeroTensorDataset>(
        &mut self,
        dataset: &D,
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
