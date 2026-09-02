use std::error::Error;

use indexmap::IndexMap;

use crate::{
    core::{dataset::item::TensorBatchLayout, writer::TensorWriter},
    dataset::tar::tar_reader::TarHeader,
};

pub trait TarRecordProcessor<'data>: Send + Sync {
    type Error: std::fmt::Debug + Send + Sync + Error;

    fn get_layout(
        &self,
        filename: &str,
        header: &TarHeader,
        compressed_data: &[u8],
    ) -> Result<IndexMap<&'data str, TensorBatchLayout>, Self::Error>;

    fn write_into<'layout, 'b, 'c>(
        &self,
        filename: &str,
        data: &[u8],
        writer: &mut TensorWriter<'layout, 'b, 'c>,
    ) -> Result<(), Self::Error>;
}
