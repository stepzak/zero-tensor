use std::error::Error;

#[derive(thiserror::Error, Debug)]
pub enum DecodeError<D: Error> {
    #[error("Buffer overflow during decoding. available: {available}, requested: {requested}")]
    BufferOverflow { available: usize, requested: usize },

    #[error("Decoder error: {0}")]
    DecoderError(#[from] D),

    #[error("Invalid stride: {0} (minimum is: {1}")]
    InvalidStride(usize, usize)
}
