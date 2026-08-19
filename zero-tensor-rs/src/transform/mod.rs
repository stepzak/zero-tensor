pub mod error;
pub mod scale;

use crate::core::dataset::item::TensorViewMut;

pub trait Transform {
    type Error;

    fn apply(&self, tensor: &mut TensorViewMut) -> Result<(), Self::Error>;
}
