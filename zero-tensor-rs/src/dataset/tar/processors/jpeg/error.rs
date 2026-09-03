use thiserror::Error;

#[derive(Error, Debug)]
pub enum TarJpegProcessorError {
    #[error("Invalid DT")]
    InvalidDT,
}