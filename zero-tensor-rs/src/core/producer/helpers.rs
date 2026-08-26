use indexmap::IndexMap;
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::core::buffer::{ZTBufErr, ZeroTensorBuffer};
use crate::core::dataset::item::{LayoutError, TensorBatchLayout};
use crate::core::dataset::{ZTDatasetError, ZeroTensorDataset};
use crate::core::producer::ZTProducerErr;
use crate::core::writer::TensorWriter;

pub fn prepare_batch_metadata<'a, D: ZeroTensorDataset>(
    dataset: &'a D,
    batch_indices: &[usize],
) -> Result<
    (
        IndexMap<&'a str, TensorBatchLayout>, // single-item layouts
        IndexMap<&'a str, TensorBatchLayout>, // batch layouts
        usize,                                // element_size_bytes
        usize,                                // total_data_bytes
    ),
    ZTProducerErr<D::Error>,
> {
    let current_batch_size = batch_indices.len();

    let mut single_layouts =
        dataset
            .get_batch_layouts(batch_indices)
            .map_err(|e| ZTProducerErr::DatasetError {
                idx: e.index(),
                source: e,
            })?;

    let mut batch_layouts = single_layouts.clone();
    let mut element_size_bytes = 0usize;

    for (_, s_layout) in single_layouts.iter_mut() {
        element_size_bytes += s_layout.total_bytes().next_multiple_of(TensorWriter::ALIGNMENT);
    }

    for (_, b_layout) in batch_layouts.iter_mut() {
        b_layout.add_batch_dimension(current_batch_size).map_err(
            |e| -> ZTProducerErr<D::Error> {
                match e {
                    LayoutError::ShapeStrideMismatch { strides, shape } => {
                        ZTBufErr::InvalidShape(strides, shape).into()
                    }
                }
            },
        )?;
    }

    let total_data_bytes = element_size_bytes * current_batch_size;
    Ok((
        single_layouts,
        batch_layouts,
        element_size_bytes,
        total_data_bytes,
    ))
}

pub fn process_chunk<'a, D: ZeroTensorDataset>(
    running: &Arc<AtomicBool>,
    shm_chunk: &'a mut [u8],
    dataset: &D,
    layouts: &'a IndexMap<&str, TensorBatchLayout>,
    i: usize,
) -> Result<(), ZTProducerErr<D::Error>> {
    if !running.load(Ordering::SeqCst) {
        return Err(ZTProducerErr::IoError(std::io::Error::from(
            std::io::ErrorKind::Interrupted,
        )));
    }

    let mut writer =
        TensorWriter::new(&layouts.clone(), shm_chunk).map_err(ZTProducerErr::TensorWriterError)?;

    dataset
        .write_item_into(i, &mut writer)
        .map_err(|e| ZTProducerErr::DatasetError {
            idx: Some(i),
            source: e,
        })?;

    writer
        .finalize()
        .map_err(ZTProducerErr::TensorWriterError)?;

    if !running.load(Ordering::SeqCst) {
        return Err(ZTProducerErr::IoError(std::io::Error::from(
            std::io::ErrorKind::Interrupted,
        )));
    }

    Ok(())
}


pub fn copy_batch_to_shm<D: ZeroTensorDataset>(
    buffer: &mut ZeroTensorBuffer,
    running: &Arc<AtomicBool>,
    dataset: &D,
    batch_indices: &[usize],
    slot_offset: usize,
    single_layouts: &IndexMap<&str, TensorBatchLayout>,
    batch_layouts: &IndexMap<&str, TensorBatchLayout>,
    element_size_bytes: usize,
    total_data_bytes: usize,
) -> Result<(), ZTProducerErr<D::Error>> {

    let raw_shm_slice = unsafe { buffer.get_item_slice_mut(slot_offset, 0, total_data_bytes) }?;

    const RAYON_THRESHOLD: usize = 256 * 1024;

    if total_data_bytes < RAYON_THRESHOLD {
        for (shm_chunk, &i) in raw_shm_slice
            .chunks_mut(element_size_bytes)
            .zip(batch_indices)
        {
            process_chunk(running, shm_chunk, dataset, single_layouts, i)?;
        }
    } else {
        raw_shm_slice
            .par_chunks_mut(element_size_bytes)
            .zip(batch_indices)
            .try_for_each(
                |(shm_chunk, &i)| -> Result<(), ZTProducerErr<D::Error>> {
                    process_chunk(running, shm_chunk, dataset, single_layouts, i)
                },
            )?;
    }


    let offsets: Vec<(&str, usize)> = {
        let meta_writer = TensorWriter::new(&batch_layouts.clone(), raw_shm_slice)
            .map_err(ZTProducerErr::TensorWriterError)?;

        batch_layouts
            .keys()
            .map(|&k| {
                let (off, _size) = meta_writer
                    .get_offset_size(k)
                    .ok_or(ZTBufErr::InvalidShape(0, 0))?;
                Ok((k, off))
            })
            .collect::<Result<Vec<_>, ZTBufErr>>()
            .map_err(ZTProducerErr::ZTBufferError)?
    };

    for (k, off) in offsets {
        let v = &batch_layouts[k];
        buffer
            .write_tensor_metadata(slot_offset + off, v.shape(), v.strides(), v.dt())
            .map_err(ZTProducerErr::ZTBufferError)?;
    }

    Ok(())
}
