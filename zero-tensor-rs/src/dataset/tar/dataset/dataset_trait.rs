use std::sync::atomic::Ordering;

use indexmap::{IndexMap, map::Entry};
use rand::{Rng, seq::SliceRandom};

use crate::{
    core::{
        dataset::{ZeroTensorDataset, item::TensorBatchLayout},
        producer::epoch_context::EpochContext,
        writer::TensorWriter,
    },
    dataset::tar::{
        TarDataset, TarDatasetError, TarRecordProcessor,
        tar_reader::{MAX_PATH_LEN, TarReaderError},
    },
};

impl<'data, P: TarRecordProcessor<'data>, R: Rng + Send> ZeroTensorDataset<'data>
    for TarDataset<'data, P, R>
{
    type Error = TarDatasetError<P::Error>;

    fn len(&self) -> usize {
        self.buffer_cap
    }

    fn total_epoch_len(&self) -> usize {
        self.total_samples
    }

    fn next_epoch(&self, ctx: &EpochContext) -> Result<(), Self::Error> {
        if ctx.shuffle {
            let mut shards = self.shards.write();
            let rng = &mut self.rng.lock();
            shards.shuffle(rng);
        }
        self.current_shard_idx.store(0, Ordering::Release);
        let shards = self.shards.read();
        if !shards.is_empty() {
            self.exhausted.store(false, Ordering::Release);

            let mut reader = self.tar_reader.lock();
            reader
                .open_file(&shards[0])
                .map_err(|e| TarDatasetError::IoError {
                    filename: shards[0].to_string_lossy().into_owned(),
                    source: e,
                })?;
        }
        self.prime_initial_buffer()
    }

    fn dynamic_layouts(
        &self,
        idxs: &[usize],
    ) -> Result<
        indexmap::IndexMap<&'data str, crate::core::dataset::item::TensorBatchLayout>,
        Self::Error,
    > {
        let mut merged: IndexMap<&str, TensorBatchLayout> = IndexMap::new();

        for &idx in idxs {
            let cell = self.shuffle_buffer[idx].lock();
            if let Some(layout) = &cell.layout {
                for (key, batch_layout) in layout {
                    match merged.entry(*key) {
                        Entry::Occupied(mut entry) => {
                            entry.get_mut().merge_with(batch_layout)?;
                        }
                        Entry::Vacant(entry) => {
                            entry.insert(batch_layout.clone());
                        }
                    }
                }
            }
        }
        Ok(merged)
    }

    fn write_item_into<'layout, 'b, 'c>(
        &self,
        idx: usize,
        writer: &mut TensorWriter<'layout, 'b, 'c>,
    ) -> Result<(), Self::Error> {
        let mut cell = self.shuffle_buffer[idx].lock();

        if cell.data.is_empty() {
            return Err(TarDatasetError::Exhausted);
        }

        self.processor
            .write_into(&cell.filename, &cell.data, writer)
            .map_err(|e| TarDatasetError::ItemError {
                filename: cell.filename.clone(),
                source: e,
            })?;

        if self.exhausted.load(Ordering::Acquire) {
            cell.data.clear();
            cell.layout = None;
            return Ok(());
        }

        let mut name_buf = [0u8; MAX_PATH_LEN];

        loop {
            let mut reader = self.tar_reader.lock();

            match reader.next_record(&mut name_buf) {
                Ok(record) => {
                    self.update_cell(&mut cell, record.name, record.data, record.header)
                        .map_err(|e| TarDatasetError::ItemError {
                            filename: cell.filename.clone(),
                            source: e,
                        })?;
                    return Ok(());
                }
                Err(TarReaderError::Eof) => {
                    drop(reader);
                    let shards = self.shards.read();
                    self.move_to_next_shard(&shards)?;

                    if self.exhausted.load(Ordering::Acquire) {
                        cell.data.clear();
                        cell.layout = None;
                        return Ok(());
                    }
                }
                Err(e) => {
                    return Err(TarDatasetError::from_tar_reader_err(
                        cell.filename.clone(),
                        e,
                    ));
                }
            }
        }
    }
}
