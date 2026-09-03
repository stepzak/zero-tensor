use std::cell::RefCell;

use crate::{augmentation::{AugmentationItem, AugmentationPipeline}, core::dataset::item::TensorDT, dataset::tar::TarRecordProcessor, decoder::{ImageInfo, PaddingConfig, default::JpegDecoder}};
use parking_lot::RwLock;
use rand::rngs::ThreadRng;
pub mod error;
pub use error::*;

struct AugBuffers {
    a: Vec<u8>,
    b: Vec<u8>,
}

thread_local! {
    static AUG_BUFS: RefCell<AugBuffers> = RefCell::new(AugBuffers {
        a: Vec::with_capacity(3 * 224 * 224 * 4),
        b: Vec::with_capacity(3 * 224 * 224 * 4),
    });
    static AUG_RNG: RefCell<ThreadRng> = RefCell::new(rand::rng());

}

pub struct TarJpegProcessor<T: AugmentationItem, F: Fn(&str) -> i64> {
    decoder: JpegDecoder,
    dt: TensorDT,
    augmentation: Option<AugmentationPipeline<T>>,
    label_fn: F
}

impl<T: AugmentationItem, F: Fn(&str) -> i64> TarJpegProcessor<T, F> {
    pub fn new(
        augmentation: Option<AugmentationPipeline<T>>,
        label_fn: F
    ) -> Result<Self, TarJpegProcessorError> {
        let dt = TensorDT::from_type::<T>().ok_or(TarJpegProcessorError::InvalidDT)?;
        let decoder = JpegDecoder::new();
        Ok(Self {
            decoder,
            dt,
            augmentation,
            label_fn
        })
    }
}

impl<'data, T: AugmentationItem, F: Fn(&str) -> i64> TarRecordProcessor<'data> for TarJpegProcessor<T, F> {
    type Error = TarJpegProcessorError;

    
}