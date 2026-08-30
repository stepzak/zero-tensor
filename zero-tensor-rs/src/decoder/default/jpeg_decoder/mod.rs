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
            ImageFormat::JPEG,
        ))
    }

    fn decode<P: crate::decoder::Pixel>(
        &self,
        compressed: &[u8],
        output: &mut [P],
    ) -> Result<crate::decoder::ImageInfo, DecodeError<Self::Error>> {
        let header = self.info(compressed)?;
        let total = header.width * header.height * header.channels;
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
                pitch: header.width * 3,
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
                pixels: pixels,
                width: header.width,
                pitch: header.width * 3,
                height: header.height,
                format: PixelFormat::RGB,
            };
            decompressor.decompress(compressed, target_image)?;
            let pixels = &mut raw_pixels[..total];

            for (pixel, byte) in output[..total].iter_mut().zip(pixels.iter()) {
                *pixel = P::from_u8(*byte);
            }
            Ok(())
        })?;

        Ok(header)
    }
}

#[cfg(test)]
mod tests;
