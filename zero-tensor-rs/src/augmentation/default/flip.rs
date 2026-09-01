use std::marker::PhantomData;

use rand::{Rng, RngExt};

use crate::augmentation::{Augmentation, AugmentationError, AugmentationItem, ImageShape};

#[derive(Debug, Clone)]
pub struct RandomHorizontalFlip<T> {
    prob: f32,
    _marker: PhantomData<T>,
}

impl<T> RandomHorizontalFlip<T> {
    pub fn new(prob: f32) -> Result<Self, AugmentationError> {
        if !(0.0..=1.0).contains(&prob) {
            return Err(AugmentationError::InvalidParameter {
                name: "RandomHorizontalFlip",
                message: format!("prob must be in [0, 1], got {}", prob),
            });
        }
        Ok(Self {
            prob,
            _marker: PhantomData,
        })
    }
}

impl<T: AugmentationItem + std::fmt::Debug> Augmentation for RandomHorizontalFlip<T> {
    type InputItem = T;
    type OutputItem = T;

    fn name(&self) -> &'static str {
        "RandomHorizontalFlip"
    }

    fn changes_size(&self) -> bool {
        false
    }

    fn apply(
        &self,
        input: &[T],
        input_shape: ImageShape,
        output: &mut [T],
        rng: Option<&mut dyn Rng>,
    ) -> Result<ImageShape, AugmentationError> {
        let ImageShape {
            channels: c,
            height: h,
            width: w,
        } = input_shape;

        let expected_len = c * h * w;
        if input.len() != expected_len || output.len() != expected_len {
            return Err(AugmentationError::InvalidParameter {
                name: self.name(),
                message: format!(
                    "Expected input/output len {}, got input={}, output={}",
                    expected_len,
                    input.len(),
                    output.len()
                ),
            });
        }

        let rng = rng.ok_or(AugmentationError::MissingRng)?;

        if rng.random_range(0.0..1.0) >= self.prob {
            output.copy_from_slice(input);
            return Ok(input_shape);
        }

        for channel in 0..c {
            let c_offset = channel * h * w;
            for y in 0..h {
                let row_start = c_offset + y * w;
                for x in 0..w {
                    output[row_start + x] = input[row_start + (w - 1 - x)];
                }
            }
        }

        Ok(input_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn create_indexed_image(c: usize, h: usize, w: usize) -> Vec<f32> {
        let mut img = vec![0.0f32; c * h * w];
        for channel in 0..c {
            let c_offset = channel * h * w;
            for y in 0..h {
                for x in 0..w {
                    img[c_offset + y * w + x] = (channel * 1000 + y * w + x) as f32;
                }
            }
        }
        img
    }

    #[test]
    fn test_flip_prob_1_always_flips() {
        let flip = RandomHorizontalFlip::<f32>::new(1.0).unwrap();
        let input = create_indexed_image(1, 2, 3);
        let mut output = vec![0.0f32; 6];

        let mut rng = StdRng::seed_from_u64(42);
        flip.apply(
            &input,
            ImageShape::new(1, 2, 3),
            &mut output,
            Some(&mut rng),
        )
        .unwrap();

        assert_eq!(output, vec![2.0, 1.0, 0.0, 5.0, 4.0, 3.0]);
    }

    #[test]
    fn test_flip_prob_0_never_flips() {
        let flip = RandomHorizontalFlip::<f32>::new(0.0).unwrap();
        let input = create_indexed_image(1, 2, 3);
        let mut output = vec![0.0f32; 6];

        let mut rng = StdRng::seed_from_u64(42);
        flip.apply(
            &input,
            ImageShape::new(1, 2, 3),
            &mut output,
            Some(&mut rng),
        )
        .unwrap();

        assert_eq!(input, output);
    }

    #[test]
    fn test_flip_multichannel() {
        let flip = RandomHorizontalFlip::<f32>::new(1.0).unwrap();
        let input = create_indexed_image(2, 2, 2);
        let mut output = vec![0.0f32; 8];

        let mut rng = StdRng::seed_from_u64(42);
        flip.apply(
            &input,
            ImageShape::new(2, 2, 2),
            &mut output,
            Some(&mut rng),
        )
        .unwrap();

        assert_eq!(
            output,
            vec![1.0, 0.0, 3.0, 2.0, 1001.0, 1000.0, 1003.0, 1002.0]
        );
    }

    #[test]
    fn test_flip_missing_rng() {
        let flip = RandomHorizontalFlip::<f32>::new(0.5).unwrap();
        let input = vec![0.0f32; 4];
        let mut output = vec![0.0f32; 4];

        let result = flip.apply(&input, ImageShape::new(1, 2, 2), &mut output, None);
        assert!(matches!(result, Err(AugmentationError::MissingRng)));
    }

    #[test]
    fn test_flip_invalid_prob() {
        assert!(RandomHorizontalFlip::<f32>::new(1.5).is_err());
        assert!(RandomHorizontalFlip::<f32>::new(-0.1).is_err());
        assert!(RandomHorizontalFlip::<f32>::new(0.5).is_ok());
    }

    #[test]
    fn test_flip_does_not_change_size() {
        let flip = RandomHorizontalFlip::<f32>::new(0.5).unwrap();
        assert!(!flip.changes_size());
        assert_eq!(flip.fixed_output_size(), None);
    }
}
