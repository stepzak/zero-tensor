use crate::{
    buffer::{ZTBufErr, ZeroTensorBuffer, get_dt_size, tensor_meta::TensorHeader},
    dataset::{
        ZeroTensorDataset,
        item::{ShapeType, StrideType},
    },
};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
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
        })
    }

    fn start_streaming_loop<D: ZeroTensorDataset>(
        &mut self,
        dataset: &D,
        batch_size: usize,
        stream: &mut UnixStream,
    ) -> Result<(), ZTProducerErr> {
        let mut buf = String::with_capacity(CONSUMER_RESP_BUFFER);
        let timeout = std::cmp::min(
            DEFAULT_TIMEOUT_CHECK_CTRLC,
            self.read_timeout.unwrap_or(DEFAULT_TIMEOUT_CHECK_CTRLC),
        );
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
            let offset = (self.current_step % self.nslots) * self.slot_size;
            let start_idx = self.current_step * batch_size;
            let end_idx = std::cmp::min(start_idx + batch_size, dataset.len());
            if start_idx >= end_idx {
                return Ok(());
            }
            let idxs = start_idx..end_idx;
            let current_batch_size = end_idx - start_idx;

            let (_, first_meta) = dataset.get_item(start_idx).unwrap_or_else(|| {
                panic!("Failed to get first item of batch to extract metadata");
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
            let data_start_offset = offs.data();

            self.buffer
                .write_tensor(offset, &batch_shape, &batch_strides, dt, &[]);

            let total_data_bytes = current_batch_size * element_size_bytes;

            let raw_shm_slice = unsafe {
                self.buffer
                    .get_item_slice_mut(offset, data_start_offset, total_data_bytes)
            };

            let shm_chunks: Vec<&mut [u8]> = raw_shm_slice.chunks_mut(element_size_bytes).collect();

            let interrupted = idxs.into_par_iter().zip(shm_chunks).any(|(i, shm_chunk)| {
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

            let msg = format!("READY {}\n", offset);
            stream
                .write_all(msg.as_bytes())
                .map_err(ZTProducerErr::IoError)?;
            stream.flush().map_err(ZTProducerErr::IoError)?;
            let start_time = std::time::Instant::now();
            loop {
                if !self.running.load(Ordering::SeqCst) {
                    return Err(ZTProducerErr::IoError(io::Error::from(
                        io::ErrorKind::Interrupted,
                    )));
                }

                match reader.read_line(&mut buf) {
                    Ok(0) => return Ok(()),
                    Ok(_) => {
                        let trimmed = buf.trim();
                        if trimmed != "RELEASE" {
                            panic!("Unexpected protocol violation from consumer: '{}'", trimmed);
                        }
                        buf.clear();
                        break;
                    }
                    Err(e)
                        if e.kind() == io::ErrorKind::WouldBlock
                            || e.kind() == io::ErrorKind::TimedOut =>
                    {
                        let el = start_time.elapsed();
                        if let Some(rt) = self.read_timeout
                            && el.as_millis() >= rt as u128
                        {
                            return Err(ZTProducerErr::IoError(e));
                        }

                        continue;
                    }
                    Err(e) => return Err(ZTProducerErr::IoError(e)),
                }
            }

            self.current_step += 1;
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
