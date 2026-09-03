use thiserror::Error;

use crate::{
    augmentation::AugmentationError, core::writer::TensorWriteError, decoder::DecodeError,
};

#[derive(Error, Debug)]
pub enum TarJpegProcessorError {
    #[error("Invalid DT")]
    InvalidDT,

    #[error("DecodeError at file {0}: {1}")]
    DecodeError(String, DecodeError<turbojpeg::Error>),

    #[error("Buffer too small at {filename}. Needed: {needed}, got: {got}")]
    BufferTooSmall {
        filename: String,
        needed: usize,
        got: usize,
    },

    #[error("Augmentation error at {filename}: {source}")]
    Augmentation {
        filename: String,
        #[source]
        source: AugmentationError,
    },

    #[error("Writer error at {filename}: {source}")]
    TensorWriter {
        filename: String,
        #[source]
        source: Box<TensorWriteError<Self>>,
    },
}
