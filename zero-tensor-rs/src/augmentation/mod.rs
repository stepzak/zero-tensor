pub mod error;
pub mod item;
pub mod pipeline;
pub use error::*;
pub use item::*;
pub use pipeline::*;
use rand::Rng;

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
        input_shape: (usize, usize, usize), // (C, H, W)
        output: &mut [Self::OutputItem],
        rng: Option<&mut dyn Rng>,
    ) -> Result<(usize, usize, usize), AugmentationError>;

    fn changes_size(&self) -> bool {
        false
    }
}
