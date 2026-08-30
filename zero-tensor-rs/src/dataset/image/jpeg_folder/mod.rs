pub mod error;
use bytemuck::{AnyBitPattern, NoUninit};
pub use error::*;
use indexmap::IndexMap;
use parking_lot::RwLock;

use std::{
    fs::File,
    path::{Path, PathBuf},
};

use memmap2::Mmap;

use crate::{
    core::dataset::{
        ZeroTensorDataset,
        item::{ShapeVec, StrideVec, TensorBatchLayout, TensorDT},
    },
    decoder::{DecodeError, ImageDecoder, ImageInfo, Pixel, default::JpegDecoder},
};

pub struct JpegFolderDataset {
    samples: Vec<(PathBuf, i64)>,
    mmaps: Vec<Mmap>,
    infos: Vec<ImageInfo>,
    decoder: JpegDecoder,
    dt: TensorDT,
    current_batch_max: RwLock<usize>,
}

impl JpegFolderDataset {
    pub fn new<F, D>(
        root: &Path,
        label_fn: F,
        dt: D,
    ) -> Result<Self, JpegFolderDatasetNewError<turbojpeg::Error>>
    where
        F: Fn(&Path) -> Option<i64>,
        D: Into<Option<TensorDT>>,
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

        let mut mmaps = Vec::with_capacity(samples.len());
        let mut infos = Vec::with_capacity(samples.len());
        let decoder = JpegDecoder::new();

        for (path, _) in samples.iter() {
            let file = File::open(path)?;
            let mmap = unsafe { Mmap::map(&file) }?;
            mmap.advise(memmap2::Advice::Sequential)?;

            let info = decoder
                .info(&mmap)
                .map_err(|e| JpegFolderDatasetNewError::DecodeError(path.into(), e))?;
            mmaps.push(mmap);
            infos.push(info);
        }

        let dt = dt.into().unwrap_or(TensorDT::F32);

        Ok(Self {
            samples,
            mmaps,
            infos,
            decoder,
            dt,
            current_batch_max: RwLock::new(0),
        })
    }

    fn inner_write<T: NoUninit + AnyBitPattern + Pixel>(
        &self,
        idx: usize,
        buf: &mut [u8],
    ) -> Result<usize, DecodeError<turbojpeg::Error>> {
        let elem_size = std::mem::size_of::<T>();
        let aligned_len = (buf.len() / elem_size) * elem_size;
        let aligned_buf = &mut buf[..aligned_len];

        let output: &mut [T] = bytemuck::cast_slice_mut(aligned_buf);
        let compressed = &self.mmaps[idx];
        let stride = *self.current_batch_max.read();
        let info = self
            .decoder
            .decode::<T, usize>(compressed, output, stride)?;
        let actual_size = stride * info.height() * info.channels() * elem_size;
        Ok(actual_size)
    }
}

impl<'a> ZeroTensorDataset<'a> for JpegFolderDataset {
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

        *self.current_batch_max.write() = max_w;

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
            let res = match self.dt {
                TensorDT::F16 => self.inner_write::<half::f16>(idx, buf),
                TensorDT::BF16 => self.inner_write::<half::bf16>(idx, buf),
                TensorDT::F32 => self.inner_write::<f32>(idx, buf),
                TensorDT::F64 => self.inner_write::<f64>(idx, buf),
                TensorDT::I8 => self.inner_write::<i8>(idx, buf),
                TensorDT::I32 => self.inner_write::<i32>(idx, buf),
                TensorDT::I64 => self.inner_write::<i64>(idx, buf),
                TensorDT::U8 => self.inner_write::<u8>(idx, buf),
            }
            .map_err(|e| JpegFolderDatasetError::DecodeError(path.clone(), e))?;
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