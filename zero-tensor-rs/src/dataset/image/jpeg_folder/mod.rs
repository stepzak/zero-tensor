pub mod error;
pub use error::*;
use indexmap::IndexMap;
use memmap2::Mmap;
use parking_lot::RwLock;
use rand::rngs::ThreadRng;

use std::{
    cell::RefCell,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

use crate::{
    augmentation::{AugmentationItem, AugmentationPipeline, ImageShape},
    core::dataset::{
        ZeroTensorDataset,
        item::{ShapeVec, StrideVec, TensorBatchLayout, TensorDT},
    },
    decoder::{ImageDecoder, ImageInfo, PaddingConfig, Pixel, default::JpegDecoder},
};

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

pub struct JpegFolderDataset<T: AugmentationItem = f32> {
    samples: Vec<(PathBuf, i64)>,
    infos: Vec<ImageInfo>,
    decoder: JpegDecoder,
    dt: TensorDT,
    current_batch_max: RwLock<PaddingConfig>,
    augmentation: Option<AugmentationPipeline<T>>,
}

impl<T: AugmentationItem + Pixel> JpegFolderDataset<T> {
    pub fn new<F>(
        root: &Path,
        label_fn: F,
    ) -> Result<Self, JpegFolderDatasetNewError<turbojpeg::Error>>
    where
        F: Fn(&Path) -> Option<i64>,
    {
        if !root.is_dir() {
            return Err(JpegFolderDatasetNewError::NotDirectory(root.into()));
        }
        let mut samples: Vec<(PathBuf, i64)> = Vec::new();
        for entry in walkdir::WalkDir::new(root).into_iter() {
            let entry = entry.map_err(JpegFolderDatasetNewError::WalkDirError)?;
            let path = entry.path();
            if path.is_file()
                && let Some(extos) = path.extension()
            {
                let ext = extos.to_string_lossy();
                if (ext == "jpg" || ext == "jpeg")
                    && let Some(label) = label_fn(path)
                {
                    samples.push((path.into(), label));
                }
            }
        }

        if samples.is_empty() {
            return Err(JpegFolderDatasetNewError::Empty(root.into()));
        }

        samples.sort_by(|a, b| a.0.cmp(&b.0));

        let mut infos = Vec::with_capacity(samples.len());
        let decoder = JpegDecoder::new();

        for (path, _) in samples.iter() {
            let file = File::open(path)?;
            const HEADER_SIZE: usize = 1024;
            let mut header_buf = [0u8; HEADER_SIZE];
            let mut reader = std::io::BufReader::new(&file);

            reader.read_exact(&mut header_buf)?;

            let info = decoder
                .info(&header_buf)
                .map_err(|e| JpegFolderDatasetNewError::DecodeError(path.into(), e))?;
            infos.push(info);
        }

        let dt = TensorDT::from_type::<T>().ok_or_else(|| {
            JpegFolderDatasetNewError::UnsupportedType(std::any::type_name::<T>().to_string())
        })?;

        Ok(Self {
            samples,
            infos,
            decoder,
            dt,
            current_batch_max: RwLock::new(PaddingConfig {
                stride: 0,
                max_height: 0,
            }),
            augmentation: None,
        })
    }

    pub fn with_augmentation(mut self, augmentation: AugmentationPipeline<T>) -> Self {
        self.augmentation = Some(augmentation);
        self
    }

    fn inner_write(
        &self,
        idx: usize,
        output: &mut [T],
    ) -> Result<usize, JpegFolderDatasetError<turbojpeg::Error>> {
        let file = File::open(&self.samples[idx].0)?;
        let compressed = unsafe { Mmap::map(&file)? };
        compressed.advise(memmap2::Advice::Sequential)?;

        let padding = *self.current_batch_max.read();
        let info = &self.infos[idx];
        let c = info.channels();

        if let Some(aug) = &self.augmentation {
            return self.inner_write_with_augmentation(
                idx,
                &compressed,
                output,
                padding.max_height,
                padding.stride,
                aug,
            );
        }

        self.decoder
            .decode::<T, PaddingConfig>(&compressed, output, padding)
            .map_err(|e| JpegFolderDatasetError::DecodeError(self.samples[idx].0.clone(), e))?;

        Ok(padding.stride * padding.max_height * c * size_of::<T>())
    }

    fn inner_write_with_augmentation(
        &self,
        idx: usize,
        compressed: &[u8],
        output: &mut [T],
        max_h: usize,
        max_w: usize,
        aug: &AugmentationPipeline<T>,
    ) -> Result<usize, JpegFolderDatasetError<turbojpeg::Error>> {
        let info = &self.infos[idx];
        let c = info.channels();
        let h = info.height();
        let w = info.width();
        let elem_size = std::mem::size_of::<T>();

        AUG_BUFS.with_borrow_mut(|bufs| {
            let AugBuffers { a, b } = &mut *bufs;

            let inter_max = aug
                .max_intermediate_size()
                .map(|(ih, iw)| ih * iw * c)
                .unwrap_or(0);

            let max_dense_size = (max_h * max_w * c).max(inter_max).max(h * w * c);
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

            let augmented: &mut [T] = bytemuck::cast_slice_mut(&mut b[..max_bytes_needed]);

            self.decoder
                .decode::<T, PaddingConfig>(compressed, decoded, PaddingConfig::new(w, h))
                .map_err(|e| JpegFolderDatasetError::DecodeError(self.samples[idx].0.clone(), e))?;

            let input_shape = ImageShape::new(c, h, w);

            let output_shape = AUG_RNG
                .with_borrow_mut(|rng| aug.apply(decoded, input_shape, augmented, Some(&mut *rng)))
                .map_err(JpegFolderDatasetError::Augmentation)?;

            Self::copy_with_padding(augmented, output_shape, output, max_h, max_w);

            Ok(max_h * max_w * c * elem_size)
        })
    }

    fn copy_with_padding(
        augmented: &[T],
        shape: ImageShape,
        output: &mut [T],
        max_h: usize,
        max_w: usize,
    ) {
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

    fn compute_max_size(&self, idxs: &[usize]) -> (usize, usize) {
        let mut max_h = 0;
        let mut max_w = 0;

        for &idx in idxs {
            let info = &self.infos[idx];
            if info.height() > max_h {
                max_h = info.height();
            }
            if info.width() > max_w {
                max_w = info.width();
            }
        }

        (max_h, max_w)
    }
}

impl<'a, T: AugmentationItem + Pixel> ZeroTensorDataset<'a> for JpegFolderDataset<T> {
    type Error = JpegFolderDatasetError;

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn dynamic_layouts(
        &self,
        idxs: &[usize],
    ) -> Result<IndexMap<&'a str, TensorBatchLayout>, Self::Error> {
        if idxs.is_empty() {
            return Err(JpegFolderDatasetError::EmptyBatch);
        }

        let mut im = IndexMap::new();
        let (max_h, max_w) = if let Some(aug) = &self.augmentation {
            aug.max_intermediate_size()
                .or_else(|| aug.output_size())
                .unwrap_or_else(|| self.compute_max_size(idxs))
        } else {
            self.compute_max_size(idxs)
        };

        *self.current_batch_max.write() = PaddingConfig {
            stride: max_w,
            max_height: max_h,
        };

        let mut img_shape = ShapeVec::new();
        img_shape.extend_from_slice(&[3, max_h, max_w]);
        let mut img_strides = StrideVec::new();
        img_strides.extend_from_slice(&[max_h * max_w, max_w, 1]);
        let img_layout = TensorBatchLayout::new(img_shape, img_strides, self.dt);
        im.insert("image", img_layout);

        let label_shape = ShapeVec::new();
        let label_stride = StrideVec::new();
        let lbl_layout = TensorBatchLayout::new(label_shape, label_stride, TensorDT::I64);
        im.insert("label", lbl_layout);
        Ok(im)
    }

    fn write_item_into<'layout, 'b, 'c>(
        &self,
        idx: usize,
        writer: &mut crate::core::writer::TensorWriter<'layout, 'b, 'c>,
    ) -> Result<(), Self::Error> {
        let (path, lbl) = &self.samples[idx];
        let i_res = writer.write("image", |buf| -> Result<usize, JpegFolderDatasetError> {
            let buf_cast: &mut [T] = bytemuck::cast_slice_mut(buf);
            let res = self.inner_write(idx, buf_cast)?;
            Ok(res)
        });
        i_res.map_err(|e| JpegFolderDatasetError::WriteError(path.clone(), Box::new(e)))?;
        writer
            .write("label", |buf| -> Result<usize, JpegFolderDatasetError> {
                if buf.len() < size_of::<i64>() {
                    return Err(JpegFolderDatasetError::BufferTooSmall {
                        needed: size_of::<i64>(),
                        got: buf.len(),
                    });
                }
                let sl = bytemuck::cast_slice_mut::<u8, i64>(buf);
                sl[0] = *lbl;
                Ok(size_of::<i64>())
            })
            .map_err(|e| JpegFolderDatasetError::WriteError(path.clone(), Box::new(e)))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
