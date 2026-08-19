pub mod add;
pub mod error;
pub mod scale;

use crate::core::dataset::item::TensorViewMut;

pub use add::Add;
pub use error::TransformError;
pub use scale::Scale;

pub trait Transform {
    type Error;

    fn apply(&self, tensor: &mut TensorViewMut) -> Result<(), Self::Error>;
}
