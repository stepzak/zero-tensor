use std::cell::RefCell;

use turbojpeg::{Decompressor, Image, PixelFormat};

use crate::decoder::{DecodeError, ImageDecoder, ImageFormat, ImageInfo, PaddingConfig};

pub struct JpegDecoder;

impl JpegDecoder {
    pub fn new() -> Self {
        Self {}
    }
}

const DEFAULT_CAP: usize = 3 * 1024 * 1024;

thread_local! {
    static RAW_PIXELS: RefCell<Vec<u8>> = RefCell::new(Vec::with_capacity(DEFAULT_CAP))
}

impl ImageDecoder for JpegDecoder {
    type Error = turbojpeg::Error;

    fn info(&self, compressed: &[u8]) -> Result<ImageInfo, DecodeError<Self::Error>> {
        let mut decompressor = Decompressor::new()?;
        let header = decompressor.read_header(compressed)?;

        Ok(ImageInfo::new(
            header.width,
            header.height,
            3,
            ImageFormat::Jpeg,
        ))
    }

    fn decode<P: crate::decoder::Pixel, T: Into<Option<PaddingConfig>>>(
        &self,
        compressed: &[u8],
        output: &mut [P],
        padding_config: T,
    ) -> Result<crate::decoder::ImageInfo, DecodeError<Self::Error>> {
        let header = self.info(compressed)?;
        let width = header.width;
        let height = header.height;
        let channels = header.channels;

        let PaddingConfig { stride, max_height } = padding_config.into().unwrap_or(PaddingConfig {
            stride: width,
            max_height: height,
        });

        if stride < width {
            return Err(DecodeError::InvalidStride(stride, width));
        }
        if max_height < height {
            return Err(DecodeError::InvalidStride(max_height, width));
        }

        let total = stride * max_height * channels;
        if total > output.len() {
            return Err(DecodeError::BufferOverflow {
                available: output.len(),
                requested: total,
            });
        }

        let mut decompressor = Decompressor::new()?;

        // TurboJPEG emits interleaved RGB. Dataset layouts and image
        // augmentations use planar CHW, including the padded destination.
        RAW_PIXELS.with_borrow_mut(|raw_pixels| -> Result<(), DecodeError<Self::Error>> {
            let dense_len = width * height * channels;
            if raw_pixels.len() < dense_len {
                raw_pixels.resize(dense_len, 0);
            }
            let pixels = &mut raw_pixels[..dense_len];
            decompressor.decompress(
                compressed,
                Image {
                    pixels: &mut *pixels,
                    width,
                    pitch: width * channels,
                    height,
                    format: PixelFormat::RGB,
                },
            )?;

            let plane_len = stride * max_height;
            for channel in 0..channels {
                let plane = &mut output[channel * plane_len..(channel + 1) * plane_len];
                for y in 0..height {
                    let row = &mut plane[y * stride..(y + 1) * stride];
                    for x in 0..width {
                        row[x] = P::from_u8(pixels[(y * width + x) * channels + channel]);
                    }
                    row[width..].fill(P::from_u8(0));
                }
                plane[height * stride..].fill(P::from_u8(0));
            }
            Ok(())
        })?;

        Ok(header)
    }
}

#[cfg(test)]
mod tests;
