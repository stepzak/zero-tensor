use crate::{
    buffer::{ZTBufErr, ZeroTensorBuffer, get_dt_size, tensor_meta::TensorHeader},
    dataset::{
        ZTDatasetError, ZeroTensorDataset,
        item::{ShapeType, ShapeVec, StrideType, StrideVec},
    },
};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use std::{
    fs,
    io::{self, Write},
    os::unix::net::{UnixListener, UnixStream},
    thread,
    time::{Duration, Instant},
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
pub const CONSUMER_RELEASE: &str = "RELEASE";
pub const CONSUMER_STOP: &str = "STOP";
pub const PRODUCER_EPOCH_DONE: &str = "EPOCH_DONE\n";
pub const CONSUMER_RESP_BUFFER: usize = get_max_resp_len() * 2;
pub const DEFAULT_TIMEOUT_CHECK_CTRLC: u64 = 500;

const fn const_max_usize(a: usize, b: usize) -> usize {
    if a > b { a } else { b }
}

const fn get_max_resp_len() -> usize {
    const_max_usize(CONSUMER_RELEASE.len(), CONSUMER_STOP.len())
}

#[derive(Debug, Clone)]
enum ZTConsumerCmd {
    Release,
    Stop,
}

#[derive(Debug, Clone)]
enum ZTProducerCmd {
    Ready { offset: usize },
    EpochEnd,
}

pub struct ZeroTensorProducer {
    buffer: ZeroTensorBuffer,
    slot_size: usize,
    nslots: usize,
    listener: UnixListener,
    sock_path: PathBuf,
    running: Arc<AtomicBool>,
    read_timeout: Option<u64>,
    shuffle: bool,
    seed: Option<u64>,
    max_steps: Option<usize>,
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
}

#[derive(Clone, Debug)]
pub struct ZeroTensorProducerBuilder {
    // Required
    step_size: usize,
    shm_filename: String,
    socket_addr: PathBuf,

    // Optional
    num_slots: usize,
    read_timeout: Option<u64>,
    overwrite_socket: bool,
    shuffle: bool,
    seed: Option<u64>,
    max_steps: Option<usize>,
}

impl ZeroTensorProducerBuilder {
    pub fn new<P: AsRef<Path>>(step_size: usize, shm_filename: &str, socket_addr: P) -> Self {
        Self {
            step_size,
            shm_filename: shm_filename.to_string(),
            socket_addr: socket_addr.as_ref().to_path_buf(),
            num_slots: DEFAULT_SLOTS,
            read_timeout: None,
            overwrite_socket: false,
            shuffle: false,
            seed: None,
            max_steps: None,
        }
    }

    pub fn num_slots(mut self, slots: usize) -> Self {
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

    pub fn build(self) -> Result<ZeroTensorProducer, ZTProducerNewErr> {
        let running = Arc::new(AtomicBool::new(true));
        let rclone = running.clone();

        let _ = ctrlc::set_handler(move || {
            rclone.store(false, Ordering::SeqCst);
        });

        let total_size = self.num_slots * self.step_size;
        let buffer = ZeroTensorBuffer::new(&self.shm_filename, total_size)?;

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
            slot_size: self.step_size,
            listener,
            nslots: self.num_slots,
            sock_path: self.socket_addr,
            read_timeout: self.read_timeout,
            running,
            shuffle: self.shuffle,
            seed: self.seed,
            max_steps: self.max_steps,
        })
    }
}

impl ZeroTensorProducer {
    pub fn from_builder(builder: ZeroTensorProducerBuilder) -> Result<Self, ZTProducerNewErr> {
        builder.build()
    }

    fn next_command(
        &self,
        reader: &mut BufReader<UnixStream>,
        buf: &mut String,
    ) -> Result<ZTConsumerCmd, io::Error> {
        let start_time = Instant::now();
        buf.clear();
        loop {
            if !self.running.load(Ordering::SeqCst) {
                return Err(io::Error::from(io::ErrorKind::Interrupted));
            }

            match reader.read_line(buf) {
                Ok(0) => {
                    return Ok(ZTConsumerCmd::Stop);
                }
                Ok(_) => match buf.trim() {
                    CONSUMER_RELEASE => {
                        return Ok(ZTConsumerCmd::Release);
                    }
                    CONSUMER_STOP => return Ok(ZTConsumerCmd::Stop),
                    _ => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "Unexpected protocol violation from consumer: '{}'",
                                buf.trim()
                            ),
                        ));
                    }
                },
                Err(e)
                    if e.kind() == io::ErrorKind::WouldBlock
                        || e.kind() == io::ErrorKind::TimedOut =>
                {
                    let el = start_time.elapsed();
                    if let Some(rt) = self.read_timeout
                        && el.as_millis() >= rt as u128
                    {
                        return Err(e);
                    }
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
    }

    fn wait_for_next_command(
        &self,
        reader: &mut BufReader<UnixStream>,
        buf: &mut String,
    ) -> Result<ZTConsumerCmd, io::Error> {
        self.next_command(reader, buf)
    }

    fn send_cmd(stream: &mut UnixStream, cmd: ZTProducerCmd) -> Result<(), io::Error> {
        match cmd {
            ZTProducerCmd::EpochEnd => {
                stream.write_all(PRODUCER_EPOCH_DONE.as_bytes())?;
            }
            ZTProducerCmd::Ready { offset } => {
                writeln!(stream, "READY {}", offset)?;
            }
        }
        stream.flush()
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

        let mut buf = String::with_capacity(CONSUMER_RESP_BUFFER);
        let timeout = std::cmp::min(
            DEFAULT_TIMEOUT_CHECK_CTRLC,
            self.read_timeout.unwrap_or(DEFAULT_TIMEOUT_CHECK_CTRLC),
        );
        let steps_per_epoch = dataset.len().div_ceil(batch_size);
        let mut current_epoch = 0;
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

            if self.shuffle {
                self.reshuffle_indices(&mut indices, current_epoch);
            }

            let mut epoch_step = 0;
            while epoch_step < steps_per_epoch {
                if let Some(max) = self.max_steps
                    && current_epoch * steps_per_epoch + epoch_step >= max
                {
                    return Ok(());
                }

                let start_idx = epoch_step * batch_size;
                let end_idx = std::cmp::min(start_idx + batch_size, dataset.len());

                if start_idx >= end_idx {
                    epoch_step += 1;
                    continue;
                }

                let batch_indices = &indices[start_idx..end_idx];
                let offset = (epoch_step % self.nslots) * self.slot_size;

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

                let cmd_to_send = ZTProducerCmd::Ready { offset };
                Self::send_cmd(stream, cmd_to_send).map_err(ZTProducerErr::IoError)?;
                let cmd = match self.wait_for_next_command(&mut reader, &mut buf) {
                    Ok(cmd) => cmd,
                    Err(ref e) if e.kind() == io::ErrorKind::InvalidData => {
                        return Err(ZTProducerErr::ProtocolError(e.to_string()));
                    }
                    Err(e) => {
                        return Err(e.into());
                    }
                };

                match cmd {
                    ZTConsumerCmd::Stop => return Ok(()),
                    ZTConsumerCmd::Release => {}
                }
                epoch_step += 1;
            }

            current_epoch += 1;
            Self::send_cmd(stream, ZTProducerCmd::EpochEnd).map_err(ZTProducerErr::IoError)?;
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

    /// Returns: (data_start_offset, total_data_bytes, element_size_bytes)
    fn prepare_batch_metadata<D: ZeroTensorDataset>(
        &mut self,
        dataset: &D,
        batch_indices: &[usize],
        offset: usize,
    ) -> Result<(usize, usize, usize), ZTProducerErr<D::Error>> {
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

        let dt_size = get_dt_size(dt) as StrideType;
        let mut batch_strides = StrideVec::with_capacity(ndims);

        batch_strides.push(element_size_bytes as StrideType);

        for &s in layout.strides() {
            batch_strides.push(s * dt_size);
        }

        let header_meta = TensorHeader::new(dt, ndims as u8);
        let offs = header_meta.get_offsets();

        self.buffer
            .write_tensor(offset, &batch_shape, &batch_strides, dt, &[])?;

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
    ) -> Result<(), ZTProducerErr<D::Error>> {
        let raw_shm_slice = unsafe {
            self.buffer
                .get_item_slice_mut(offset, data_start_offset, total_data_bytes)
        }?;

        raw_shm_slice.fill(0);
        const RAYON_THRESHOLD: usize = 256 * 1024;

        if element_size_bytes < RAYON_THRESHOLD {
            raw_shm_slice
                .chunks_mut(element_size_bytes)
                .zip(batch_indices)
                .try_for_each(|(shm_chunk, &i)| -> Result<(), ZTProducerErr<D::Error>> {
                    dataset.write_item_into(i, shm_chunk).map_err(|e| {
                        ZTProducerErr::DatasetError {
                            idx: Some(i),
                            source: e,
                        }
                    })?;

                    Ok(())
                })?;
        } else {
            raw_shm_slice
                .par_chunks_mut(element_size_bytes)
                .zip(batch_indices)
                .try_for_each(|(shm_chunk, &i)| -> Result<(), ZTProducerErr<D::Error>> {
                    if !self.running.load(Ordering::SeqCst) {
                        return Err(io::Error::from(io::ErrorKind::Interrupted).into());
                    }
                    dataset.write_item_into(i, shm_chunk).map_err(|e| {
                        ZTProducerErr::DatasetError {
                            idx: Some(i),
                            source: e,
                        }
                    })?;

                    if !self.running.load(Ordering::SeqCst) {
                        return Err(io::Error::from(io::ErrorKind::Interrupted).into());
                    }

                    Ok(())
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
        if self.sock_path.exists() {
            _ = fs::remove_file(&self.sock_path);
        }
    }
}
