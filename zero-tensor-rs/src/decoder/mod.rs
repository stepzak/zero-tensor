pub mod default;
pub mod error;
pub mod pixel;
pub use error::DecodeError;
pub use pixel::Pixel;

#[derive(Debug, Clone, Copy)]
pub enum ImageFormat {
    Jpeg,
}

#[derive(Debug, Clone, Copy)]
pub struct ImageInfo {
    width: usize,
    height: usize,
    channels: usize,
    _format: ImageFormat,
}

impl ImageInfo {
    pub fn new(width: usize, height: usize, channels: usize, image_format: ImageFormat) -> Self {
        Self {
            width,
            height,
            channels,
            _format: image_format,
        }
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    #[allow(dead_code)]
    pub fn format(&self) -> ImageFormat {
        self._format
    }
}

pub trait ImageDecoder: Send + Sync {
    type Error: std::error::Error;

    fn decode<P: Pixel, T: Into<Option<usize>>>(
        &self,
        compressed: &[u8],
        output: &mut [P],
        stride: T,
    ) -> Result<ImageInfo, DecodeError<Self::Error>>;

    fn info(&self, compressed: &[u8]) -> Result<ImageInfo, DecodeError<Self::Error>>;
}
