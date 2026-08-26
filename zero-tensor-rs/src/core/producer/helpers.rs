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
        IndexMap<&'a str, TensorBatchLayout>, // single-item layouts
        IndexMap<&'a str, TensorBatchLayout>, // batch layouts
        usize,                                // element_size_bytes
        usize,                                // total_data_bytes
    ),
    ZTProducerErr<D::Error>,
> {
    let current_batch_size = batch_indices.len();

    let mut single_layouts = if let Some(s) = dataset.static_layouts() {
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
    let mut element_size_bytes = 0usize;

    for (_, s_layout) in single_layouts.iter_mut() {
        element_size_bytes += s_layout
            .total_bytes()
            .next_multiple_of(TensorWriter::ALIGNMENT);
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

    let mut writer = TensorWriter::new(&layouts.clone(), shm_chunk, cache)
        .map_err(ZTProducerErr::TensorWriterError)?;

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
    element_size_bytes: usize,
    total_data_bytes: usize,
    caches: &'c mut [Mutex<TensorWriterCache<'layout>>]
) -> Result<(), ZTProducerErr<D::Error>> {
    let mut metadata_offset = slot_offset;
    for (_, layout) in batch_layouts.iter() {
        buffer
            .write_tensor_metadata(
                metadata_offset,
                layout.shape(),
                layout.strides(),
                layout.dt(),
            )
            .map_err(ZTProducerErr::ZTBufferError)?;

        let ndims = layout.shape().len();
        metadata_offset += align_to(
            size_of::<TensorHeader>() + ndims * (size_of::<StrideType>() + size_of::<ShapeType>()),
            TensorWriter::ALIGNMENT,
        );
    }

    let missing_keys_res = {
        let raw_shm_slice: &mut [u8] =
            unsafe { buffer.get_item_slice_mut(slot_offset, 0, total_data_bytes)? };

        const RAYON_THRESHOLD: usize = 256 * 1024;

        if total_data_bytes < RAYON_THRESHOLD {
            let mut res = Ok(());
            for (shm_chunk, &i) in raw_shm_slice
                .chunks_mut(element_size_bytes)
                .zip(batch_indices)
            {
                let mut cache_guard = caches[i].lock();
                if let Err(e) = process_chunk(
                    running,
                    shm_chunk,
                    dataset,
                    single_layouts,
                    i,
                    &mut cache_guard,
                ) {
                    res = Err(e);
                    break;
                }
            }
            res
        } else {
            raw_shm_slice
                .par_chunks_mut(element_size_bytes)
                .zip(batch_indices)
                .enumerate()
                .try_for_each(
                    |(idx, (shm_chunk, &i))| -> Result<(), ZTProducerErr<D::Error>> {
                        let mut cache_guard = caches[idx % caches.len()].lock();
                        process_chunk(
                            running,
                            shm_chunk,
                            dataset,
                            single_layouts,
                            i,
                            &mut cache_guard,
                        )
                    },
                )
        }
    };

    missing_keys_res?;

    Ok(())
}
