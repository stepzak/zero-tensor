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

pub trait ZeroTensorDataset<'data>: Send + Sync {
    type Error: ZTDatasetError;

    fn len(&self) -> usize;

    fn write_item_into<'layout, 'b, 'c>(
        &self,
        idx: usize,
        writer: &mut TensorWriter<'layout, 'b, 'c>,
    ) -> Result<(), Self::Error>;

    fn static_layouts(&self) -> Option<&IndexMap<&'static str, TensorBatchLayout>> {
        None
    }

    fn dynamic_layouts(
        &self,
        idxs: &[usize],
    ) -> Result<IndexMap<&'data str, TensorBatchLayout>, Self::Error> {
        let _ = idxs;
        unimplemented!("Either static_layouts() or dynamic_layouts() must be implemented")
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
