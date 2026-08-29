pub mod pixel;
pub mod error;
pub mod default;
pub use pixel::Pixel;
pub use error::DecoderError;

#[derive(Debug, Clone, Copy)]
pub enum ImageFormat {
    JPEG
}

pub struct ImageInfo {
    width: usize,
    height: usize,
    channels: usize,
    format: ImageFormat
}

pub trait ImageDecoder: Send + Sync {
    fn decode<P: Pixel>(&self, compressed: &[u8], output: &mut [u8]) -> Result<ImageInfo, DecoderError>;

    fn info(&self, compressed: &[u8]) -> Result<ImageInfo, DecoderError>;
}