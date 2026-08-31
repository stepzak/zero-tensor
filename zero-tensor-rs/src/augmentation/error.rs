use thiserror::Error;

#[derive(Error, Debug)]
pub enum AugmentationError {
    #[error(
        "Cannot add size-changing {new_step} at index {idx} to a pipeline as it already has size-preserving. All size-changing augmentations must be added first."
    )]
    InvalidOrder { new_step: &'static str, idx: usize },
}
