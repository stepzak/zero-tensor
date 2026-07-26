use std::{error::Error, fmt::Debug};

use crate::dataset::item::TensorBatchLayout;

pub mod item;


pub type TensorBytes = Vec<u8>;

pub trait ZeroTensorDataset: Send + Sync {
    type Error: Debug + Error;
    type Meta: Clone + Send + Sync;

    fn len(&self) -> usize;

    fn write_item_into(&self, idx: usize, buf: &mut [u8]) -> Result<(), Self::Error>;

    fn get_batch_layout(&self, idxs: &[usize]) -> Result<TensorBatchLayout, Self::Error>;

    fn is_empty(&self) -> bool;
}
