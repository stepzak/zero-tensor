use std::path::PathBuf;

use thiserror::Error;

use crate::{
    augmentation::AugmentationError,
    core::{dataset::ZTDatasetError, writer::TensorWriteError},
    decoder::DecodeError,
};

#[derive(Error, Debug)]
pub enum JpegFolderDatasetNewError<D: std::error::Error = turbojpeg::Error> {
    #[error("Not a dir: {0}")]
    NotDirectory(PathBuf),

    #[error("No JPEG files found in {0}")]
    Empty(PathBuf),

    #[error("Io error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Decode Error at {0}: {1}")]
    DecodeError(PathBuf, DecodeError<D>),

    #[error("WalkDirError: {0}")]
    WalkDirError(#[from] walkdir::Error),

    #[error("Unsupported dtype: {0}")]
    UnsupportedType(String),
}

#[derive(Error, Debug)]
pub enum JpegFolderDatasetError<D: std::error::Error = turbojpeg::Error> {
    #[error("Empty batch")]
    EmptyBatch,

    #[error("Decode error at {0}: {1}")]
    DecodeError(PathBuf, DecodeError<D>),

    #[error("TensorWrite error at {0}: {1}")]
    WriteError(PathBuf, Box<TensorWriteError<Self>>),

    #[error("Buffer too small. Needeed : {needed}, got {got}")]
    BufferTooSmall { needed: usize, got: usize },

    #[error("Augmentation error: {0}")]
    Augmentation(AugmentationError),
}

impl ZTDatasetError for JpegFolderDatasetError {
    fn index(&self) -> Option<usize> {
        None
    }
}
