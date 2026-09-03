use std::{error::Error, fmt::Debug};

use thiserror::Error;

use crate::{
    core::dataset::{ZTDatasetError, item::MergeError},
    dataset::tar::tar_reader::TarReaderError,
};

#[derive(Error, Debug)]
pub enum TarDatasetError<T: Error + Sync + Send + Debug> {
    #[error("Tar reader error at file {filename}: {source}")]
    TarReader {
        filename: String,
        #[source]
        source: TarReaderError,
    },

    #[error("Item error at file {filename}: {source} ")]
    ItemError {
        filename: String,
        #[source]
        source: T,
    },

    #[error("Io error at file {filename}: {source}")]
    IoError {
        filename: String,
        #[source]
        source: std::io::Error,
    },

    #[error("No shards")]
    Empty,

    #[error("Exhausted")]
    Exhausted,

    #[error("Layout merge error: {0}")]
    Merge(#[from] MergeError),
}

impl<T: Error + Sync + Send + Debug> TarDatasetError<T> {
    pub fn from_tar_reader_err(filename: String, source: TarReaderError) -> Self {
        Self::TarReader { filename, source }
    }
}

impl<E: Send + Sync + Debug + Error + 'static> ZTDatasetError for TarDatasetError<E> {
    fn index(&self) -> Option<usize> {
        None
    }
}
