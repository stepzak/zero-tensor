use thiserror::Error;

#[derive(Error, Debug)]
pub enum AugmentationError {
    #[error(
        "Cannot add size-changing {new_step} at index {idx} to a pipeline as it already has size-preserving. All size-changing augmentations must be added first."
    )]
    InvalidOrder { new_step: &'static str, idx: usize },

    #[error("Rng is required")]
    MissingRng,

    #[error("Invalid parameter at {name}: {message}")]
    InvalidParameter { name: &'static str, message: String },

    #[error("UnsupportedDtype for {name}: {dtype}")]
    UnsupportedDtype { name: &'static str, dtype: String },

    #[error("Custom: {0}")]
    Custom(String),
}
