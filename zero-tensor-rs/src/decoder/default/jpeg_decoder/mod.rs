use std::cell::RefCell;

use turbojpeg::{Decompressor, Image, PixelFormat};

use crate::decoder::{DecodeError, ImageDecoder, ImageFormat, ImageInfo};

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

    fn decode<P: crate::decoder::Pixel, T: Into<Option<usize>>>(
        &self,
        compressed: &[u8],
        output: &mut [P],
        stride: T,
    ) -> Result<crate::decoder::ImageInfo, DecodeError<Self::Error>> {
        let header = self.info(compressed)?;

        let stride = stride.into().unwrap_or(header.width);
        if stride < header.width {
            return Err(DecodeError::InvalidStride(stride, header.width));
        }
        let total = stride * header.height * header.channels;
        if total > output.len() {
            return Err(DecodeError::BufferOverflow {
                available: output.len(),
                requested: total,
            });
        }
        let mut decompressor = Decompressor::new()?;
        if std::any::TypeId::of::<P>() == std::any::TypeId::of::<u8>() {
            let u8_output: &mut [u8] = bytemuck::cast_slice_mut(&mut output[..total]);

            let target_image = Image {
                pixels: u8_output,
                width: header.width,
                pitch: stride * 3,
                height: header.height,
                format: PixelFormat::RGB,
            };
            decompressor.decompress(compressed, target_image)?;
            return Ok(header);
        }

        RAW_PIXELS.with_borrow_mut(|raw_pixels| -> Result<(), DecodeError<Self::Error>> {
            if raw_pixels.len() < total {
                raw_pixels.resize(total, 0);
            }
            let pixels = &mut raw_pixels[..total];

            let target_image = Image {
                pixels,
                width: header.width,
                pitch: header.width * 3,
                height: header.height,
                format: PixelFormat::RGB,
            };
            decompressor.decompress(compressed, target_image)?;

            let width = header.width;
            let height = header.height;
            let channels = header.channels;
            for y in 0..height {
                for x in 0..width {
                    for c in 0..channels {
                        let src_idx = (y * width + x) * channels + c;
                        let dst_idx = (y * stride + x) * channels + c;
                        output[dst_idx] = P::from_u8(raw_pixels[src_idx]);
                    }
                }
            }

            if stride > width {
                for y in 0..height {
                    for x in width..stride {
                        for c in 0..channels {
                            let dst_idx = (y * stride + x) * channels + c;
                            output[dst_idx] = P::from_u8(0);
                        }
                    }
                }
            }
            Ok(())
        })?;

        Ok(header)
    }
}

#[cfg(test)]
mod tests;
