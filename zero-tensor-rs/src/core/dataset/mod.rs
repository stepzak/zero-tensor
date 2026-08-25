use std::{error::Error, fmt::Debug};

use indexmap::IndexMap;
use item::TensorBatchLayout;

use crate::core::writer::TensorWriter;

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
    fn write_item_into(&self, idx: usize, writer: &mut TensorWriter) -> Result<(), Self::Error>;

    fn get_batch_layouts(
        &self,
        idxs: &[usize],
    ) -> Result<IndexMap<&str, TensorBatchLayout>, Self::Error>;

    fn is_empty(&self) -> bool;
}
