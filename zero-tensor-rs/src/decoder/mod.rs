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

#[derive(Debug, Clone, Copy)]
pub struct PaddingConfig {
    pub stride: usize,
    pub max_height: usize,
}

impl PaddingConfig {
    pub fn new(stride: usize, max_height: usize) -> Self {
        Self { stride, max_height }
    }
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

    fn decode<P: Pixel, T: Into<Option<PaddingConfig>>>(
        &self,
        compressed: &[u8],
        output: &mut [P],
        stride: T,
    ) -> Result<ImageInfo, DecodeError<Self::Error>>;

    fn info(&self, compressed: &[u8]) -> Result<ImageInfo, DecodeError<Self::Error>>;
}
