use indexmap::IndexMap;
use parking_lot::Mutex;
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::core::buffer::tensor_meta::TensorHeader;
use crate::core::buffer::{ZTBufErr, ZeroTensorBuffer};
use crate::core::dataset::ZeroTensorDataset;
use crate::core::dataset::item::{LayoutError, ShapeType, StrideType, TensorBatchLayout};
use crate::core::helpers::align_to;
use crate::core::producer::ZTProducerErr;
use crate::core::writer::{TensorWriter, TensorWriterCache};

pub fn prepare_batch_metadata<'a, D: ZeroTensorDataset<'a>>(
    dataset: &'a D,
    batch_indices: &[usize],
) -> Result<
    (
        IndexMap<&'a str, TensorBatchLayout>,
        IndexMap<&'a str, TensorBatchLayout>,
        usize, // data_size_per_item (no meta)
        usize, // total_batch_metadata_size
        usize, // total_slot_size
    ),
    ZTProducerErr<D::Error>,
> {
    let current_batch_size = batch_indices.len();

    let single_layouts = if let Some(s) = dataset.static_layouts() {
        s.clone()
    } else {
        dataset
            .dynamic_layouts(&[0])
            .map_err(|e| ZTProducerErr::DatasetError {
                idx: 0.into(),
                source: e,
            })?
    };

    let mut batch_layouts = single_layouts.clone();

    for (_, b_layout) in batch_layouts.iter_mut() {
        b_layout
            .add_batch_dimension(current_batch_size)
            .map_err(|e| match e {
                LayoutError::ShapeStrideMismatch { strides, shape } => {
                    ZTBufErr::InvalidShape(strides, shape)
                }
            })?;
    }

    let data_size_per_item: usize = single_layouts
        .iter()
        .map(|(_, layout)| align_to(layout.total_bytes(), TensorWriter::ALIGNMENT))
        .sum();

    let total_batch_metadata_size: usize = batch_layouts
        .iter()
        .map(|(_, layout)| {
            let ndims = layout.shape().len();
            let raw = size_of::<TensorHeader>()
                + ndims * (size_of::<StrideType>() + size_of::<ShapeType>());
            align_to(raw, TensorWriter::ALIGNMENT)
        })
        .sum();

    let total_slot_size = total_batch_metadata_size + (data_size_per_item * current_batch_size);

    Ok((
        single_layouts,
        batch_layouts,
        data_size_per_item,
        total_batch_metadata_size,
        total_slot_size,
    ))
}

fn process_chunk<'a, 'layout, 'chunk, D: ZeroTensorDataset<'a>>(
    running: &Arc<AtomicBool>,
    shm_chunk: &'chunk mut [u8],
    dataset: &D,
    layouts: &'layout IndexMap<&'layout str, TensorBatchLayout>,
    i: usize,
    cache: &mut TensorWriterCache<'layout>,
) -> Result<(), ZTProducerErr<D::Error>> {
    if !running.load(Ordering::SeqCst) {
        return Err(ZTProducerErr::IoError(std::io::Error::from(
            std::io::ErrorKind::Interrupted,
        )));
    }

    let mut writer =
        TensorWriter::new(&layouts, shm_chunk, cache).map_err(ZTProducerErr::TensorWriterError)?;

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

pub fn copy_batch_to_shm<'a, 'layout, 'c, D: ZeroTensorDataset<'a>>(
    buffer: &mut ZeroTensorBuffer,
    running: &Arc<AtomicBool>,
    dataset: &D,
    batch_indices: &[usize],
    slot_offset: usize,
    single_layouts: &'layout IndexMap<&'layout str, TensorBatchLayout>,
    batch_layouts: &IndexMap<&'layout str, TensorBatchLayout>,
    data_size_per_item: usize,
    total_batch_metadata_size: usize,
    total_slot_size: usize,
    caches: &'c mut [Mutex<TensorWriterCache<'layout>>],
) -> Result<(), ZTProducerErr<D::Error>> {
    let mut current_meta_offset = slot_offset;
    for (_, layout) in batch_layouts.iter() {
        buffer
            .write_tensor_metadata(
                current_meta_offset,
                layout.shape(),
                layout.strides(),
                layout.dt(),
            )
            .map_err(ZTProducerErr::ZTBufferError)?;

        let ndims = layout.shape().len();
        current_meta_offset += align_to(
            size_of::<TensorHeader>() + ndims * (size_of::<StrideType>() + size_of::<ShapeType>()),
            TensorWriter::ALIGNMENT,
        );
    }

    let data_offset_in_slot = total_batch_metadata_size;
    let data_total_size = total_slot_size - data_offset_in_slot;

    if data_total_size == 0 || total_slot_size < data_offset_in_slot {
        return Err(ZTProducerErr::ZTBufferError(ZTBufErr::InvalidShape(0, 0)));
    }

    let raw_shm_slice =
        unsafe { buffer.get_item_slice_mut(slot_offset, data_offset_in_slot, data_total_size)? };

    const RAYON_THRESHOLD: usize = 256 * 1024;
    let n_tensors = single_layouts.len();
    if data_total_size < RAYON_THRESHOLD {
        for (idx, (shm_chunk, &i)) in raw_shm_slice
            .chunks_mut(data_size_per_item)
            .zip(batch_indices)
            .enumerate()
        {
            let mut cache_guard = caches[idx % caches.len()].lock();
            process_chunk(
                running,
                shm_chunk,
                dataset,
                single_layouts,
                i,
                &mut cache_guard,
            )?;
        }
    } else {
        raw_shm_slice
            .par_chunks_mut(data_size_per_item)
            .zip(batch_indices)
            .try_for_each(|(shm_chunk, &i)| -> Result<(), ZTProducerErr<D::Error>> {
                let mut local_cache = TensorWriterCache::with_capacity(n_tensors);
                process_chunk(
                    running,
                    shm_chunk,
                    dataset,
                    single_layouts,
                    i,
                    &mut local_cache,
                )
            })?;
    }

    Ok(())
}
