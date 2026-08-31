use rand::Rng;
use std::cell::RefCell;

use crate::augmentation::{Augmentation, AugmentationError, AugmentationItem, ImageShape};

pub type AugVec<T> = Vec<Box<dyn Augmentation<InputItem = T, OutputItem = T>>>;

const SCRATCH_INIT_CAP: usize = 3 * 224 * 224;

thread_local! {
    static DOUBLE_SCRATCH: RefCell<Vec<u8>> =
        RefCell::new(Vec::with_capacity(2 * SCRATCH_INIT_CAP * std::mem::size_of::<f32>()));
}

#[derive(Debug, Default)]
pub struct AugmentationPipeline<T: AugmentationItem> {
    augmentations: AugVec<T>,
    size_preserving_idx: Option<usize>,
}

impl<T: AugmentationItem> AugmentationPipeline<T> {
    pub fn new() -> Self {
        Self {
            augmentations: Vec::new(),
            size_preserving_idx: None,
        }
    }

    pub fn then<A: Augmentation<InputItem = T, OutputItem = T> + 'static>(
        mut self,
        augmentation: A,
    ) -> Result<Self, AugmentationError> {
        if !augmentation.changes_size() && self.size_preserving_idx.is_none() {
            self.size_preserving_idx = Some(self.augmentations.len());
        }

        if augmentation.changes_size() && let Some(idx) = self.size_preserving_idx {
            return Err(AugmentationError::InvalidOrder {
                new_step: augmentation.name(),
                idx,
            });
        }

        self.augmentations.push(Box::new(augmentation));
        Ok(self)
    }

    pub fn output_size(&self) -> Option<(usize, usize)> {
        let split_idx = self.size_preserving_idx.unwrap_or(self.augmentations.len());
        let size_changing = &self.augmentations[..split_idx];

        if size_changing.is_empty() {
            return None;
        }

        size_changing.last().and_then(|aug| aug.fixed_output_size())
    }

    pub fn apply(
        &self,
        input: &[T],
        input_shape: ImageShape,
        output: &mut [T],
        rng: Option<&mut dyn Rng>,
    ) -> Result<ImageShape, AugmentationError> {
        let mut local_rng;
        let rng_ref: &mut dyn Rng = match rng {
            Some(r) => r,
            None => {
                local_rng = rand::rng();
                &mut local_rng
            }
        };

        if self.augmentations.is_empty() {
            output[..input.len()].copy_from_slice(input);
            return Ok(input_shape);
        }

        let split_idx = self.size_preserving_idx.unwrap_or(self.augmentations.len());
        let size_changing = &self.augmentations[..split_idx];
        let size_preserving = &self.augmentations[split_idx..];

        let mut current_shape = input_shape;
        let mut current_len = input.len();

        DOUBLE_SCRATCH.with_borrow_mut(|scratch| -> Result<(), AugmentationError> {
            let initial_bytes = std::cmp::max(input.len(), output.len()) * std::mem::size_of::<T>();
            let bytes_per_buf = initial_bytes;

            let total_bytes = bytes_per_buf * 2;
            if scratch.len() < total_bytes {
                scratch.resize(total_bytes, 0);
            }

            let (buf_a_bytes, buf_b_bytes) = scratch.split_at_mut(bytes_per_buf);
            let buf_a: &mut [T] = bytemuck::cast_slice_mut(buf_a_bytes);
            let buf_b: &mut [T] = bytemuck::cast_slice_mut(buf_b_bytes);

            let mut active = 0;

            if !size_changing.is_empty() {
                for (i, aug) in size_changing.iter().enumerate() {
                    if i == 0 {
                        current_shape = aug.apply(input, current_shape, buf_a, Some(rng_ref))?;
                        active = 0;
                    } else {
                        if active == 0 {
                            current_shape =
                                aug.apply(&*buf_a, current_shape, buf_b, Some(rng_ref))?;
                            active = 1;
                        } else {
                            current_shape =
                                aug.apply(&*buf_b, current_shape, buf_a, Some(rng_ref))?;
                            active = 0;
                        }
                    }
                    current_len =
                        current_shape.channels * current_shape.width * current_shape.height;
                }
            }

            if !size_preserving.is_empty() {
                if active == 1 {
                    buf_a[..current_len].copy_from_slice(&buf_b[..current_len]);
                    active = 0;
                } else if size_changing.is_empty() {
                    buf_a[..current_len].copy_from_slice(input);
                }

                for aug in size_preserving {
                    if active == 0 {
                        aug.apply(
                            &buf_a[..current_len],
                            current_shape,
                            &mut buf_b[..current_len],
                            Some(rng_ref),
                        )?;
                        active = 1;
                    } else {
                        aug.apply(
                            &buf_b[..current_len],
                            current_shape,
                            &mut buf_a[..current_len],
                            Some(rng_ref),
                        )?;
                        active = 0;
                    }
                }
            }

            if active == 0 {
                output[..current_len].copy_from_slice(&buf_a[..current_len]);
            } else {
                output[..current_len].copy_from_slice(&buf_b[..current_len]);
            }

            Ok(())
        })?;

        Ok(current_shape)
    }
}
