use crate::transform::TransformError;
use std::{error::Error, fmt};

#[derive(Debug)]
pub struct PipelineError {
    pub step: usize,
    pub error: TransformError,
}

impl PipelineError {
    pub fn new(step: usize, error: TransformError) -> Self {
        Self { step, error }
    }
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Pipeline error during step {}: {}",
            self.step, self.error
        )
    }
}

impl Error for PipelineError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.error)
    }
}
