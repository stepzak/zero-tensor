use std::io;

use thiserror::Error;

use crate::{
    core::{
        buffer::ZTBufErr,
        dataset::{ZTDatasetError, item::TensorViewError},
    },
    pipeline::PipelineError,
};

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
