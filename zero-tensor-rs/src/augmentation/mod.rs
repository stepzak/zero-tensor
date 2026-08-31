pub mod default;
pub mod error;
pub mod item;
pub mod pipeline;
pub use error::*;
pub use item::*;
pub use pipeline::*;
use rand::Rng;

#[derive(Clone, Debug, Copy, PartialEq)]
pub struct ImageShape {
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

impl ImageShape {
    pub fn new(channels: usize, height: usize, width: usize) -> Self {
        Self {
            channels,
            height,
            width,
        }
    }
}

pub trait Augmentation: Send + Sync + std::fmt::Debug {
    type InputItem: AugmentationItem;
    type OutputItem: AugmentationItem;

    fn name(&self) -> &'static str;

    fn fixed_output_size(&self) -> Option<(usize, usize)> {
        None
    }

    fn apply(
        &self,
        input: &[Self::InputItem],
        input_shape: ImageShape,
        output: &mut [Self::OutputItem],
        rng: Option<&mut dyn Rng>,
    ) -> Result<ImageShape, AugmentationError>;

    fn changes_size(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests;
