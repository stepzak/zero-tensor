use thiserror::Error;

#[derive(Error, Debug)]
pub enum TensorWriteError<'a, E> {
    #[error("Unknown key: {0}")]
    UnknownKey(&'a str),

    #[error(
        "For {key}. Buffer out of bounds. Offset is {offset} while total length is {total_size}"
    )]
    BufferOutOfBounds {
        key: &'a str,
        offset: usize,
        total_size: usize,
    },

    #[error("Dataset error: {source}")]
    DatasetError {
        #[source]
        source: E,
    },

    #[error("Key already exists: {0}")]
    KeyExists(&'a str),
}

#[derive(Error, Debug)]
pub enum TensorWriterError<'a> {
    #[error(
        "Failed to create writer: invalid layout. Required size: {required}, available: {available}"
    )]
    BufferTooSmall { required: usize, available: usize },

    #[error("Finalization failed. Missing keys: {0:?}")]
    MissingKeys(Vec<&'a str>),
}
