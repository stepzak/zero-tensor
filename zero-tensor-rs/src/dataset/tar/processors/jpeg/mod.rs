use std::cell::RefCell;

use crate::{
    augmentation::{AugmentationItem, AugmentationPipeline, ImageShape},
    core::{
        dataset::item::{ShapeVec, StrideVec, TensorBatchLayout, TensorDT},
        writer::TensorWriter,
    },
    dataset::tar::{TarRecordProcessor, tar_reader::TarHeader},
    decoder::{ImageDecoder, PaddingConfig, Pixel, default::JpegDecoder},
};
use indexmap::IndexMap;
use rand::rngs::ThreadRng;

pub mod error;
pub use error::*;

struct AugBuffers {
    a: Vec<u8>,
    b: Vec<u8>,
}

thread_local! {
    static AUG_BUFS: RefCell<AugBuffers> = RefCell::new(AugBuffers {
        a: Vec::with_capacity(3 * 224 * 224 * 4),
        b: Vec::with_capacity(3 * 224 * 224 * 4),
    });
    static AUG_RNG: RefCell<ThreadRng> = RefCell::new(rand::rng());
}

pub struct TarJpegProcessor<T: AugmentationItem, F: Fn(&str) -> i64 + Send + Sync> {
    decoder: JpegDecoder,
    dt: TensorDT,
    augmentation: Option<AugmentationPipeline<T>>,
    label_fn: F,
    target_h_w: Option<(usize, usize)>,
}

impl<T: AugmentationItem, F: Fn(&str) -> i64 + Send + Sync> TarJpegProcessor<T, F> {
    pub fn new(
        augmentation: Option<AugmentationPipeline<T>>,
        label_fn: F,
    ) -> Result<Self, TarJpegProcessorError> {
        let dt = TensorDT::from_type::<T>().ok_or(TarJpegProcessorError::InvalidDT)?;
        let decoder = JpegDecoder::new();
        let target_h_w = augmentation.as_ref().and_then(|aug| aug.output_size());

        Ok(Self {
            decoder,
            dt,
            augmentation,
            label_fn,
            target_h_w,
        })
    }

    fn copy_with_padding(
        augmented: &[T],
        shape: ImageShape,
        output: &mut [T],
        max_h: usize,
        max_w: usize,
    ) {
        if augmented.len() == output.len() {
            output.copy_from_slice(augmented);
            return;
        }
        let c = shape.channels;
        let h = shape.height;
        let w = shape.width;

        let zero = T::zeroed();

        output.fill(zero);

        for channel in 0..c {
            let src_offset = channel * h * w;
            let dst_offset = channel * max_h * max_w;

            for y in 0..h {
                let src_row = src_offset + y * w;
                let dst_row = dst_offset + y * max_w;
                output[dst_row..dst_row + w].copy_from_slice(&augmented[src_row..src_row + w]);
            }
        }
    }
}

impl<'data, T: AugmentationItem + Pixel, F: Fn(&str) -> i64 + Send + Sync> TarRecordProcessor<'data>
    for TarJpegProcessor<T, F>
{
    type Error = TarJpegProcessorError;

    fn get_layout(
        &self,
        filename: &str,
        _header: &TarHeader,
        compressed_data: &[u8],
    ) -> Result<IndexMap<&'data str, TensorBatchLayout>, Self::Error> {
        let mut layout = IndexMap::new();

        let (height, width) = if let Some((h, w)) = self.target_h_w {
            (h, w)
        } else {
            let info = self
                .decoder
                .info(compressed_data)
                .map_err(|e| TarJpegProcessorError::DecodeError(filename.into(), e))?;
            (info.height(), info.width())
        };

        let mut img_shape = ShapeVec::with_capacity(3);
        img_shape.extend_from_slice(&[3, height, width]);
        let mut img_strides = StrideVec::with_capacity(3);
        img_strides.extend_from_slice(&[height * width, width, 1]);
        layout.insert(
            "image",
            TensorBatchLayout::new(img_shape, img_strides, self.dt),
        );

        layout.insert(
            "label",
            TensorBatchLayout::new(ShapeVec::new(), StrideVec::new(), TensorDT::I64),
        );

        Ok(layout)
    }

    fn write_into<'layout, 'b, 'c>(
        &self,
        filename: &str,
        data: &[u8],
        writer: &mut TensorWriter<'layout, 'b, 'c>,
    ) -> Result<(), Self::Error> {
        let info = self
            .decoder
            .info(data)
            .map_err(|e| TarJpegProcessorError::DecodeError(filename.into(), e))?;

        let c = info.channels();
        let h = info.height();
        let w = info.width();
        let elem_size = std::mem::size_of::<T>();

        writer
            .write("image", |buf| {
                AUG_BUFS.with_borrow_mut(|bufs| {
                    let AugBuffers { a, b } = &mut *bufs;

                    let (target_h, target_w) = self.target_h_w.unwrap_or((h, w));
                    let target_pixels = target_h * target_w * c;
                    let target_bytes = target_pixels * elem_size;

                    if buf.len() < target_bytes {
                        return Err(TarJpegProcessorError::BufferTooSmall {
                            filename: filename.into(),
                            needed: target_bytes,
                            got: buf.len(),
                        });
                    }

                    if let Some(aug) = &self.augmentation {
                        let inter_max = aug
                            .max_intermediate_size()
                            .map(|(ih, iw)| ih * iw * c)
                            .unwrap_or(0);

                        let max_dense_size =
                            (target_h * target_w * c).max(inter_max).max(h * w * c);
                        let max_bytes_needed = max_dense_size * elem_size;

                        if a.len() < max_bytes_needed {
                            a.resize(max_bytes_needed, 0);
                        }
                        if b.len() < max_bytes_needed {
                            b.resize(max_bytes_needed, 0);
                        }

                        let dense_size = c * h * w;
                        let bytes_needed = dense_size * elem_size;
                        let decoded: &mut [T] = bytemuck::cast_slice_mut(&mut a[..bytes_needed]);
                        let augmented: &mut [T] =
                            bytemuck::cast_slice_mut(&mut b[..max_bytes_needed]);

                        self.decoder
                            .decode::<T, PaddingConfig>(data, decoded, PaddingConfig::new(w, h))
                            .map_err(|e| TarJpegProcessorError::DecodeError(filename.into(), e))?;

                        let input_shape = ImageShape::new(c, h, w);
                        let output_shape = AUG_RNG
                            .with_borrow_mut(|rng| {
                                aug.apply(decoded, input_shape, augmented, Some(&mut *rng))
                            })
                            .map_err(|e| TarJpegProcessorError::Augmentation {
                                filename: filename.into(),
                                source: e,
                            })?;

                        let output: &mut [T] = bytemuck::cast_slice_mut(&mut buf[..target_bytes]);
                        Self::copy_with_padding(
                            augmented,
                            output_shape,
                            output,
                            target_h,
                            target_w,
                        );
                    } else {
                        let output: &mut [T] = bytemuck::cast_slice_mut(&mut buf[..target_bytes]);

                        self.decoder
                            .decode::<T, PaddingConfig>(data, output, PaddingConfig::new(w, h))
                            .map_err(|e| TarJpegProcessorError::DecodeError(filename.into(), e))?;
                    }

                    Ok(target_bytes)
                })
            })
            .map_err(|e| TarJpegProcessorError::TensorWriter {
                filename: filename.into(),
                source: e.into(),
            })?;

        let label = (self.label_fn)(filename);
        writer
            .write("label", |buf| {
                const LABEL_SIZE: usize = std::mem::size_of::<i64>();
                if buf.len() < LABEL_SIZE {
                    return Err(TarJpegProcessorError::BufferTooSmall {
                        filename: filename.into(),
                        needed: LABEL_SIZE,
                        got: buf.len(),
                    });
                }
                buf[..LABEL_SIZE].copy_from_slice(&label.to_le_bytes());
                Ok(LABEL_SIZE)
            })
            .map_err(|e| TarJpegProcessorError::TensorWriter {
                filename: filename.into(),
                source: e.into(),
            })?;

        Ok(())
    }
}

#[cfg(test)]
mod tests;
