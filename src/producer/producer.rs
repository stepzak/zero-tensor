use crate::{
    buffer::buffer::{ZTBufErr, ZeroTensorBuffer, get_dt_size},
    dataset::{
        dataset::ZeroTensorDataset,
        item::{ShapeType, StrideType, TensorItemMeta},
    },
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::path::Path;
use std::{
    fs,
    io::{self, Read, Write},
    os::unix::net::{UnixListener, UnixStream},
};
use thiserror::Error;

const DEFAULT_SLOTS: usize = 2;
const PYTHON_RESP_BUFFER: usize = b"RELEASE".len() * 2;

pub struct ZeroTensorProducer {
    buffer: ZeroTensorBuffer,
    slot_size: usize,
    current_step: usize,
    steps: usize,
    nslots: usize,
    listener: UnixListener,
}

#[derive(Debug, Error)]
pub enum ZTProducerErr {
    #[error("ZT Buffer Error: {0}")]
    ZTBufferError(ZTBufErr),

    #[error("Io error: {0}")]
    IoError(io::Error),
}

impl ZeroTensorProducer {
    pub fn new<P: AsRef<Path>, N: Into<Option<usize>>>(
        steps: usize,
        step_size: usize,
        shm_filename: &str,
        socket_addr: P,
        num_slots: N,
    ) -> Result<Self, ZTProducerErr> {
        let nslots = num_slots.into().unwrap_or(DEFAULT_SLOTS);
        let total_size = nslots * step_size;
        let buffer = ZeroTensorBuffer::new(shm_filename, total_size)
            .map_err(|e| ZTProducerErr::ZTBufferError(e))?;

        let path = socket_addr.as_ref();
        if path.exists() {
            fs::remove_file(path).map_err(|e| ZTProducerErr::IoError(e))?;
        }
        let listener = UnixListener::bind(path).map_err(|e| ZTProducerErr::IoError(e))?;

        Ok(ZeroTensorProducer {
            buffer,
            slot_size: step_size,
            steps,
            current_step: 0,
            listener,
            nslots,
        })
    }

    fn start_streaming_loop<D: ZeroTensorDataset>(
        &mut self,
        dataset: &D,
        batch_size: usize,
        stream: &mut UnixStream,
    ) -> Result<(), ZTProducerErr> {
        let mut buf = vec![0; PYTHON_RESP_BUFFER];
        loop {
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

            let items: Vec<(Vec<u8>, TensorItemMeta)> = idxs
                .into_par_iter()
                .map(|i| {
                    dataset.get_item(i).unwrap_or_else(|| {
                        panic!("Failed to get item {i} from dataset");
                    })
                })
                .collect();

            let meta = &items[0].1;
            let dt = meta.dt();
            let mut batch_shape = vec![items.len() as ShapeType];
            batch_shape.extend_from_slice(meta.shape());

            let element_strides = meta.strides();

            let element_size_elements = meta
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

            let mut raw_batch_data = Vec::with_capacity(items.iter().map(|x| x.0.len()).sum());
            for (raw_data, _) in items {
                raw_batch_data.extend_from_slice(&raw_data);
            }

            self.buffer
                .write_tensor(offset, &batch_shape, &batch_strides, dt, &raw_batch_data);

            let msg = format!("READY {}\n", offset);
            stream
                .write_all(msg.as_bytes())
                .map_err(ZTProducerErr::IoError)?;
            stream.flush().map_err(ZTProducerErr::IoError)?;

            let n = stream.read(&mut buf).map_err(ZTProducerErr::IoError)?;
            let response = std::str::from_utf8(&buf[..n]).unwrap_or("");
            if !response.starts_with("RELEASE") {
                panic!("Unexpected protocol violation from consumer: {}", response);
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

        let (mut stream, addr) = self
            .listener
            .accept()
            .map_err(|e| ZTProducerErr::IoError(e))?;
        dbg!(format!("Accepted {addr:?}"));
        self.start_streaming_loop(dataset, batch_size, &mut stream)
    }
}
