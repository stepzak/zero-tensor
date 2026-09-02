pub mod dataset_trait;
pub mod error;
pub mod item;
use std::{
    marker::PhantomData,
    path::PathBuf,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

use indexmap::IndexMap;
use parking_lot::{Mutex, RwLock};
use rand::Rng;

use crate::{
    core::dataset::item::TensorBatchLayout,
    dataset::tar::tar_reader::{MAX_PATH_LEN, TarHeader, TarReader, TarReaderError},
};

pub use error::*;
pub use item::*;

const DEFAULT_ENTRY_CAP: usize = 512 * 1024;

pub struct TarBufferEntry<'data, P: TarRecordProcessor<'data>> {
    data: Vec<u8>,
    layout: Option<IndexMap<&'data str, TensorBatchLayout>>,
    filename: String,
    _marker: PhantomData<P>,
}

pub struct TarDataset<'data, P: TarRecordProcessor<'data>, R: Rng + Send> {
    shards: RwLock<Vec<PathBuf>>,
    current_shard_idx: AtomicUsize,
    tar_reader: Mutex<TarReader>,
    exhausted: AtomicBool,

    shuffle_buffer: Vec<Mutex<TarBufferEntry<'data, P>>>,
    buffer_cap: usize,
    total_samples: usize,
    processor: P,
    rng: Mutex<R>,
}

impl<'data, P: TarRecordProcessor<'data>, R: Rng + Send> TarDataset<'data, P, R> {
    pub fn new<F>(
        shards: Vec<PathBuf>,
        buffer_cap: usize,
        shard_size_fn: Option<F>,
        processor: P,
        rng: R,
    ) -> Result<Self, TarDatasetError<P::Error>>
    where
        F: Fn(&PathBuf) -> Result<usize, P::Error> + Send + Sync,
    {
        if shards.is_empty() {
            return Err(TarDatasetError::Empty);
        }
        let total_samples = if let Some(fn_) = shard_size_fn {
            shards
                .iter()
                .map(|x| {
                    fn_(x).map_err(|e| TarDatasetError::ItemError {
                        filename: x.to_string_lossy().into_owned(),
                        source: e,
                    })
                })
                .try_fold(0, |acc, val| -> Result<usize, TarDatasetError<P::Error>> {
                    let v = val?;
                    Ok(acc + v)
                })?
        } else {
            Self::count_items(&shards)?
        };

        let mut shuffle_buffer = Vec::with_capacity(buffer_cap);
        for _ in 0..buffer_cap {
            let mut entry = TarBufferEntry {
                data: Vec::new(),
                layout: None,
                filename: String::new(),
                _marker: std::marker::PhantomData,
            };

            entry.data.reserve(DEFAULT_ENTRY_CAP);
            shuffle_buffer.push(Mutex::new(entry));
        }

        let reader = TarReader::open(&shards[0]).map_err(|e| TarDatasetError::IoError {
            filename: shards[0].to_string_lossy().into(),
            source: e,
        })?;

        let dataset = Self {
            shards: RwLock::new(shards),
            current_shard_idx: AtomicUsize::new(0),
            tar_reader: Mutex::new(reader),
            exhausted: false.into(),

            shuffle_buffer,
            buffer_cap,
            total_samples,
            processor,
            rng: Mutex::new(rng),
        };

        Ok(dataset)
    }

    fn count_items(shard_paths: &[PathBuf]) -> Result<usize, TarDatasetError<P::Error>> {
        let mut total = 0;
        let mut name_buf = [0u8; MAX_PATH_LEN];

        for path in shard_paths {
            let mut reader = TarReader::open(path).map_err(|e| TarDatasetError::IoError {
                filename: path.to_string_lossy().into_owned(),
                source: e,
            })?;

            while let Ok(record) = reader.next_record(&mut name_buf) {
                if record.header.is_regular_file() {
                    total += 1;
                }
            }
        }
        Ok(total)
    }

    fn prime_initial_buffer(&self) -> Result<(), TarDatasetError<P::Error>> {
        let shards = self.shards.read();
        if shards.is_empty() {
            return Err(TarDatasetError::Empty);
        }

        let mut name_buf = [0u8; MAX_PATH_LEN];
        let mut cell_idx = 0;

        while cell_idx < self.buffer_cap {
            if self.exhausted.load(Ordering::Acquire) {
                break;
            }
            let mut reader = self.tar_reader.lock();

            match reader.next_record(&mut name_buf) {
                Ok(record) => {
                    let mut cell = self.shuffle_buffer[cell_idx].lock();
                    self.update_cell(&mut cell, record.name, record.data, record.header)
                        .map_err(|e| TarDatasetError::ItemError {
                            filename: cell.filename.clone(),
                            source: e,
                        })?;
                    cell_idx += 1;
                }
                Err(TarReaderError::EOF) => {
                    drop(reader);
                    self.move_to_next_shard(&shards)?;
                }
                Err(e) => {
                    let filename = self.shuffle_buffer[cell_idx].lock().filename.clone();
                    return Err(TarDatasetError::from_tar_reader_err(filename, e));
                }
            }
        }

        Ok(())
    }

    fn open_shard(&self, shards: &[PathBuf], idx: usize) -> Result<(), TarDatasetError<P::Error>> {
        if idx >= shards.len() {
            self.exhausted.store(true, Ordering::Release);
            return Ok(());
        }

        let mut reader = self.tar_reader.lock();
        reader
            .open_file(&shards[idx])
            .map_err(|e| TarDatasetError::IoError {
                filename: shards[idx].to_string_lossy().into_owned(),
                source: e,
            })?;
        Ok(())
    }

    fn move_to_next_shard(&self, shards: &[PathBuf]) -> Result<(), TarDatasetError<P::Error>> {
        let next_idx = self.current_shard_idx.fetch_add(1, Ordering::Relaxed) + 1;
        self.open_shard(shards, next_idx)
    }

    fn update_cell(
        &self,
        cell: &mut TarBufferEntry<'data, P>,
        filename: &str,
        data: &[u8],
        tar_header: &TarHeader,
    ) -> Result<(), P::Error> {
        cell.layout = Some(self.processor.get_layout(filename, tar_header, data)?);
        cell.data.clear();
        cell.data.extend_from_slice(data);
        cell.filename = filename.to_string();
        Ok(())
    }
}
