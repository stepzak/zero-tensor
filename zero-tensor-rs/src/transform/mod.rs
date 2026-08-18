pub mod scale;
pub mod error;

use crate::core::dataset::item::TensorViewMut;

pub trait Transform {
    type Error;

    fn apply(
        &self,
        tensor: &mut TensorViewMut
    ) -> Result<(), Self::Error>;
}