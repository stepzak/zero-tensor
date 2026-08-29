pub mod default;
pub mod error;
pub mod pixel;
pub use error::DecodeError;
pub use pixel::Pixel;

#[derive(Debug, Clone, Copy)]
pub enum ImageFormat {
    JPEG,
}

#[derive(Debug, Clone, Copy)]
pub struct ImageInfo {
    width: usize,
    height: usize,
    channels: usize,
    format: ImageFormat,
}

pub trait ImageDecoder: Send + Sync {
    type Error: std::error::Error;

    fn decode<P: Pixel>(
        &self,
        compressed: &[u8],
        output: &mut [P],
    ) -> Result<ImageInfo, DecodeError<Self::Error>>;

    fn info(&self, compressed: &[u8]) -> Result<ImageInfo, DecodeError<Self::Error>>;
}
