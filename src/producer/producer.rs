use std::{fs, io, os::unix::net::UnixListener};
use thiserror::Error;
use std::path::Path;
use crate::buffer::buffer::{ZTBufErr, ZeroTensorBuffer};

const DEFAULT_SLOTS: usize = 2;

pub struct ZeroTensorProducer {
    buffer: ZeroTensorBuffer,
    slot_size: usize,
    current_step: usize,
    steps: usize,
    nslots: usize,
    listener: UnixListener
}

#[derive(Debug, Error)]
pub enum ZTProducerErr {
    #[error("ZT Buffer Error: {0}")]
    ZTBufferError(ZTBufErr),

    #[error("Io error: {0}")]
    IoError(io::Error)
}

impl ZeroTensorProducer {
    pub fn new<P: AsRef<Path>, N: Into<Option<usize>>>(steps: usize, step_size: usize, shm_filename: &str, socket_addr: P, num_slots: N) -> Result<Self, ZTProducerErr> {
        let nslots = num_slots.into().unwrap_or(DEFAULT_SLOTS);
        let total_size = nslots * step_size;
        let buffer = ZeroTensorBuffer::new(shm_filename, total_size).map_err(|e| {
            ZTProducerErr::ZTBufferError(e)
        })?;

        let path = socket_addr.as_ref();
        if path.exists() {
            fs::remove_file(path).map_err(|e| {
                ZTProducerErr::IoError(e)
            })?;
        }
        let listener = UnixListener::bind(path).map_err(|e| {
            ZTProducerErr::IoError(e)
        })?;

        Ok(
            ZeroTensorProducer { buffer, slot_size: step_size, steps, current_step: 0, listener, nslots }
        )
    }
}