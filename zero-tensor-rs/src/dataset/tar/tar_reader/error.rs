use thiserror::Error;

#[derive(Error, Debug)]
pub enum TarReaderError {
    #[error("Unable to read header")]
    HeaderError,

    #[error("Overflow, needed: {needed}, got: {got}")]
    Overflow { needed: usize, got: usize },

    #[error("EOF")]
    Eof,
}
