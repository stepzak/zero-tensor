pub mod error;
pub mod scale;

use crate::core::dataset::item::TensorViewMut;

pub use scale::Scale;
pub use error::TransformError;

pub trait Transform {
    type Error;

    fn apply(&self, tensor: &mut TensorViewMut) -> Result<(), Self::Error>;
}
