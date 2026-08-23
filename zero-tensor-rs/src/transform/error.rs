use std::{error::Error, sync::Arc};

#[derive(Debug, thiserror::Error, Clone)]
pub enum TransformError {
    #[error("UnsupportedDtype")]
    UnsupportedDtype,

    #[error("Overflow")]
    Overflow,

    #[error("Invalid value")]
    InvalidValue,

    #[error("Scalar conversion error: {0}")]
    ScalarConversion(#[from] ScalarConversionError),

    #[error("Custom error: {0}")]
    Custom(Arc<dyn Error + Send + Sync>),
}

#[derive(Debug, thiserror::Error, Clone, Copy)]
pub enum ScalarConversionError {
    #[error("Overlow")]
    Overflow,

    #[error("Unsupported dtype")]
    UnsupportedDtype,

    #[error("Invalid value")]
    InvalidValue,

    #[error("FractionalValue")]
    FractionalValue,
}
