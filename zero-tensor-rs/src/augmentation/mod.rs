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

    fn apply(&self, input: &[Self::InputItem], output: &mut [Self::OutputItem], rng: Option<&mut dyn Rng>) -> Result<(), AugmentationError>;

    fn changes_size(&self) -> bool {
        false
    }
}