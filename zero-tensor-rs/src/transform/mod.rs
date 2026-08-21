pub mod add;
pub mod clamp;
pub mod error;
mod helpers;
pub mod scalar;
pub mod scale;

use crate::core::dataset::item::TensorViewMut;

pub use add::Add;
pub use clamp::Clamp;
pub use error::{ScalarConversionError, TransformError};
pub use scalar::{IntoScalarOption, Scalar};
pub use scale::Scale;

pub trait Transform {
    type Error;

    fn apply(&self, tensor: &mut TensorViewMut) -> Result<(), Self::Error>;
}
