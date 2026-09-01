use std::marker::PhantomData;

use rand::Rng;

use crate::augmentation::{Augmentation, AugmentationError, AugmentationItem, ImageShape};

#[derive(Debug, Clone)]
pub struct Normalize<T> {
    mean: Vec<f64>,
    std: Vec<f64>,
    _marker: PhantomData<T>,
}

impl<T> Normalize<T> {
    pub fn new(mean: Vec<f64>, std: Vec<f64>) -> Result<Self, AugmentationError> {
        if mean.len() != std.len() {
            return Err(AugmentationError::InvalidParameter {
                name: "Normalize",
                message: format!(
                    "mean and std must have same length, got {} and {}",
                    mean.len(),
                    std.len()
                ),
            });
        }
        for (i, &s) in std.iter().enumerate() {
            if s.abs() < 1e-10 {
                return Err(AugmentationError::InvalidParameter {
                    name: "Normalize",
                    message: format!("std[{}] must be non-zero, got {}", i, s),
                });
            }
        }
        Ok(Self {
            mean,
            std,
            _marker: PhantomData,
        })
    }

    pub fn imagenet() -> Self {
        Self {
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
            _marker: PhantomData,
        }
    }
}

impl<T: AugmentationItem + std::fmt::Debug> Augmentation for Normalize<T> {
    type InputItem = T;
    type OutputItem = T;

    fn name(&self) -> &'static str {
        "Normalize"
    }

    fn changes_size(&self) -> bool {
        false
    }

    fn apply(
        &self,
        input: &[T],
        input_shape: ImageShape,
        output: &mut [T],
        _rng: Option<&mut dyn Rng>,
    ) -> Result<ImageShape, AugmentationError> {
        let ImageShape {
            channels: c,
            height: h,
            width: w,
        } = input_shape;

        if std::any::TypeId::of::<T>() != std::any::TypeId::of::<f32>() {
            return Err(AugmentationError::UnsupportedDtype {
                name: self.name(),
                dtype: std::any::type_name::<T>().to_string(),
            });
        }

        if self.mean.len() != c {
            return Err(AugmentationError::InvalidParameter {
                name: self.name(),
                message: format!(
                    "Normalize has {} channels, but image has {}",
                    self.mean.len(),
                    c
                ),
            });
        }

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

        let input_f32: &[f32] = bytemuck::cast_slice(input);
        let output_f32: &mut [f32] = bytemuck::cast_slice_mut(output);

        let hw = h * w;
        for channel in 0..c {
            let offset = channel * hw;
            let m = self.mean[channel] as f32;
            let s = self.std[channel] as f32;
            let inv_s = 1.0 / s;

            let src_chunk = &input_f32[offset..offset + hw];
            let dst_chunk = &mut output_f32[offset..offset + hw];

            for (out, &inp) in dst_chunk.iter_mut().zip(src_chunk.iter()) {
                *out = (inp - m) * inv_s;
            }
        }

        Ok(input_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_basic() {
        let norm = Normalize::<f32>::new(vec![0.5], vec![0.5]).unwrap();
        let input = vec![0.0f32, 0.5, 1.0];
        let mut output = vec![0.0f32; 3];

        norm.apply(&input, ImageShape::new(1, 1, 3), &mut output, None)
            .unwrap();

        assert!((output[0] - (-1.0)).abs() < 1e-5);
        assert!((output[1] - 0.0).abs() < 1e-5);
        assert!((output[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_imagenet() {
        let norm = Normalize::<f32>::imagenet();
        let input = vec![0.485f32, 0.456, 0.406];
        let mut output = vec![0.0f32; 3];

        norm.apply(&input, ImageShape::new(3, 1, 1), &mut output, None)
            .unwrap();

        for &v in &output {
            assert!(v.abs() < 1e-5, "Expected ~0, got {}", v);
        }
    }

    #[test]
    fn test_normalize_multichannel() {
        let norm = Normalize::<f32>::new(vec![0.0, 1.0], vec![1.0, 2.0]).unwrap();
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];

        norm.apply(&input, ImageShape::new(2, 1, 2), &mut output, None)
            .unwrap();

        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] - 2.0).abs() < 1e-5);
        assert!((output[2] - 1.0).abs() < 1e-5);
        assert!((output[3] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_unsupported_dtype() {
        let norm = Normalize::<u8>::new(vec![0.0], vec![1.0]).unwrap();
        let input = vec![0u8; 4];
        let mut output = vec![0u8; 4];

        let result = norm.apply(&input, ImageShape::new(1, 2, 2), &mut output, None);
        assert!(matches!(
            result,
            Err(AugmentationError::UnsupportedDtype { name, .. }) if name == "Normalize"
        ));
    }

    #[test]
    fn test_normalize_invalid_params() {
        assert!(Normalize::<f32>::new(vec![0.0], vec![1.0, 1.0]).is_err());
        assert!(Normalize::<f32>::new(vec![0.0], vec![0.0]).is_err());
    }

    #[test]
    fn test_normalize_does_not_change_size() {
        let norm = Normalize::<f32>::imagenet();
        assert!(!norm.changes_size());
        assert_eq!(norm.fixed_output_size(), None);
    }
}
