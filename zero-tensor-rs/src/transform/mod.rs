pub mod add;
pub mod clamp;
pub mod error;
mod helpers;
pub mod scale;

use crate::core::dataset::item::TensorViewMut;

pub use add::Add;
pub use clamp::Clamp;
pub use error::TransformError;
pub use scale::Scale;

pub trait Transform {
    type Error;

    fn apply(&self, tensor: &mut TensorViewMut) -> Result<(), Self::Error>;
}
