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

        if std::any::TypeId::of::<P>() == std::any::TypeId::of::<u8>() {
            let u8_output: &mut [u8] = bytemuck::cast_slice_mut(&mut output[..total]);

            let target_image = Image {
                pixels: u8_output,
                width,
                pitch: stride * channels,
                height,
                format: PixelFormat::RGB,
            };
            decompressor.decompress(compressed, target_image)?;

            if stride > width {
                let u8_output: &mut [u8] = bytemuck::cast_slice_mut(&mut output[..total]);

                for y in 0..height {
                    let pad_start = (y * stride + width) * channels;
                    let pad_end = (y * stride + stride) * channels;
                    u8_output[pad_start..pad_end].fill(0);
                }
            }

            if max_height > height {
                let u8_output: &mut [u8] = bytemuck::cast_slice_mut(&mut output[..total]);

                let bottom_start = height * stride * channels;
                let bottom_end = max_height * stride * channels;
                u8_output[bottom_start..bottom_end].fill(0);
            }

            return Ok(header);
        }

        RAW_PIXELS.with_borrow_mut(|raw_pixels| -> Result<(), DecodeError<Self::Error>> {
            if raw_pixels.len() < total {
                raw_pixels.resize(total, 0);
            }
            let pixels = &mut raw_pixels[..total];

            let target_image = Image {
                pixels,
                width,
                pitch: stride * channels,
                height,
                format: PixelFormat::RGB,
            };
            decompressor.decompress(compressed, target_image)?;
            let pixels = &mut raw_pixels[..total];

            for y in 0..height {
                let row_start = y * stride * channels;
                let row_end = row_start + width * channels;

                for i in row_start..row_end {
                    output[i] = P::from_u8(pixels[i]);
                }

                if stride > width {
                    let row_max_end = row_start + stride * channels;
                    output[row_end..row_max_end].fill(P::from_u8(0));
                }
            }

            if max_height > height {
                let bottom_start = height * stride * channels;
                let bottom_end = max_height * stride * channels;
                output[bottom_start..bottom_end].fill(P::from_u8(0));
            }

            Ok(())
        })?;

        Ok(header)
    }
}

#[cfg(test)]
mod tests;
