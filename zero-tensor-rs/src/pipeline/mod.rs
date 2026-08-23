pub mod error;
use crate::{core::dataset::item::TensorViewMut, transform::Transform};
pub use error::PipelineError;
use ndarray::Axis;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::sync::Arc;

const PARALLEL_THRESHOLD_BYTES: usize = 256 * 1024;

/// In parallel execution mode (for large tensors), the "all-or-nothing"
/// atomicity guarantee is relaxed. If an error occurs, some elements
/// may already be modified. Use sequential mode if strict atomicity is required.
#[derive(Clone)]
pub struct Pipeline {
    steps: Vec<Arc<dyn Transform>>,
    force_atomic: bool,
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            force_atomic: false,
        }
    }

    pub fn force_atomic(mut self, force: bool) -> Self {
        self.force_atomic = force;
        self
    }

    pub fn then<T: Transform + 'static>(mut self, step: T) -> Self {
        self.steps.push(Arc::new(step));
        self
    }

    pub fn exec(&self, tensor: &mut TensorViewMut) -> Result<(), PipelineError> {
        if self.steps.is_empty() {
            return Ok(());
        }

        let total_bytes = tensor.total_bytes();

        if self.force_atomic || total_bytes < PARALLEL_THRESHOLD_BYTES {
            self.exec_seq(tensor)
        } else {
            self.exec_parallel(tensor)
        }
    }

    fn exec_seq(&self, tensor: &mut TensorViewMut) -> Result<(), PipelineError> {
        self.steps
            .iter()
            .enumerate()
            .try_for_each(|(i, step)| step.apply(tensor).map_err(|e| PipelineError::new(i + 1, e)))
    }

    fn exec_parallel(&self, tensor: &mut TensorViewMut) -> Result<(), PipelineError> {
        macro_rules! par_exec {
            ($t:expr, $variant:ident) => {{
                $t.axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .try_for_each(|item| {
                        let mut view = TensorViewMut::$variant(item);
                        self.exec_seq(&mut view)
                    })
            }};
        }

        match tensor {
            TensorViewMut::F16(t) => par_exec!(t, F16),
            TensorViewMut::F32(t) => par_exec!(t, F32),
            TensorViewMut::F64(t) => par_exec!(t, F64),
            TensorViewMut::BF16(t) => par_exec!(t, BF16),
            TensorViewMut::I8(t) => par_exec!(t, I8),
            TensorViewMut::I32(t) => par_exec!(t, I32),
            TensorViewMut::I64(t) => par_exec!(t, I64),
            TensorViewMut::U8(t) => par_exec!(t, U8),
        }
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
