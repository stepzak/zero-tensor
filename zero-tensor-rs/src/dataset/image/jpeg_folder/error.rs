use std::path::PathBuf;

use thiserror::Error;

use crate::decoder::DecodeError;

#[derive(Error, Debug)]
pub enum JpegFolderDatasetNewError<D: std::error::Error> {
    #[error("Not a dir: {0}")]
    NotDirectory(PathBuf),

    #[error("No JPEG files found in {0}")]
    Empty(PathBuf),

    #[error("Io error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Decode Error at {0}: {1}")]
    DecodeError(PathBuf, DecodeError<D>),
}
