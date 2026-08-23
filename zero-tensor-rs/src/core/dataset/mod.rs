use std::{error::Error, fmt::Debug};

use item::TensorBatchLayout;

pub mod item;

pub type TensorBytes = Vec<u8>;

pub trait ZTDatasetError: Debug + Error + Send + Sync {
    fn index(&self) -> Option<usize>;
}

impl ZTDatasetError for std::io::Error {
    fn index(&self) -> Option<usize> {
        None
    }
}

pub trait ZeroTensorDataset: Send + Sync {
    type Error: ZTDatasetError;

    fn len(&self) -> usize;

    /// # Safety contract
    /// Must return `Ok(bytes_written)` if the write operation was a success
    fn write_item_into(&self, idx: usize, buf: &mut [u8]) -> Result<usize, Self::Error>;

    fn get_batch_layout(&self, idxs: &[usize]) -> Result<TensorBatchLayout, Self::Error>;

    fn is_empty(&self) -> bool;
}
