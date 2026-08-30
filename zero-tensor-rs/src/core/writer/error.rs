use thiserror::Error;

#[derive(Error, Debug)]
pub enum TensorWriteError<E> {
    #[error("Unknown key: {0}")]
    UnknownKey(String),

    #[error(
        "For {key}. Buffer out of bounds. Offset is {offset} while total length is {total_size}"
    )]
    BufferOutOfBounds {
        key: String,
        offset: usize,
        total_size: usize,
    },

    #[error("Dataset error: {source}")]
    DatasetError {
        #[source]
        source: E,
    },

    #[error("Key already exists: {0}")]
    KeyExists(String),
}

#[derive(Error, Debug)]
pub enum TensorWriterError {
    #[error(
        "Failed to create writer: invalid layout. Required size: {required}, available: {available}"
    )]
    BufferTooSmall { required: usize, available: usize },

    #[error("Finalization failed. Missing keys: {0:?}")]
    MissingKeys(Vec<String>),
}
