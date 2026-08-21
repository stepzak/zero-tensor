pub mod error;
use std::sync::Arc;

pub use error::PipelineError;

use crate::{core::dataset::item::TensorViewMut, transform::Transform};

#[derive(Clone)]
pub struct Pipeline {
    steps: Vec<Arc<dyn Transform>>,
}

impl Pipeline {
    pub fn new() -> Self {
        let steps = Vec::new();
        Self { steps }
    }

    pub fn then<T: Transform + 'static>(mut self, step: T) -> Self {
        self.steps.push(Arc::new(step));
        self
    }

    pub fn exec(&self, tensor: &mut TensorViewMut) -> Result<(), PipelineError> {
        self.steps
            .iter()
            .enumerate()
            .try_for_each(|(i, step)| step.apply(tensor).map_err(|e| PipelineError::new(i + 1, e)))
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
