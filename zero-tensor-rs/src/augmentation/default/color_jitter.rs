use std::marker::PhantomData;

use rand::{Rng, RngExt};

use crate::augmentation::{Augmentation, AugmentationError, AugmentationItem, ImageShape};

#[derive(Debug, Clone)]
pub struct ColorJitter<T> {
    brightness: f32,
    contrast: f32,
    saturation: f32,
    _marker: PhantomData<T>,
}

impl<T> ColorJitter<T> {
    pub fn new(brightness: f32, contrast: f32, saturation: f32) -> Result<Self, AugmentationError> {
        for (name, val) in [
            ("brightness", brightness),
            ("contrast", contrast),
            ("saturation", saturation),
        ] {
            if val < 0.0 {
                return Err(AugmentationError::InvalidParameter {
                    name: "ColorJitter",
                    message: format!("{} must be >= 0, got {}", name, val),
                });
            }
        }
        Ok(Self {
            brightness,
            contrast,
            saturation,
            _marker: PhantomData,
        })
    }
}

impl<T: AugmentationItem + std::fmt::Debug> Augmentation for ColorJitter<T> {
    type InputItem = T;
    type OutputItem = T;

    fn name(&self) -> &'static str {
        "ColorJitter"
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

        if std::any::TypeId::of::<T>() != std::any::TypeId::of::<f32>() {
            return Err(AugmentationError::UnsupportedDtype {
                name: self.name(),
                dtype: std::any::type_name::<T>().to_string(),
            });
        }

        if c != 3 {
            return Err(AugmentationError::InvalidParameter {
                name: self.name(),
                message: format!("ColorJitter requires 3 channels (RGB), got {}", c),
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

        let rng = rng.ok_or(AugmentationError::MissingRng)?;

        let input_f32: &[f32] = bytemuck::cast_slice(input);
        let output_f32: &mut [f32] = bytemuck::cast_slice_mut(output);

        let brightness_factor = if self.brightness > 0.0 {
            let lo = (1.0 - self.brightness).max(0.0);
            let hi = 1.0 + self.brightness;
            rng.random_range(lo..=hi)
        } else {
            1.0
        };

        let contrast_factor = if self.contrast > 0.0 {
            let lo = (1.0 - self.contrast).max(0.0);
            let hi = 1.0 + self.contrast;
            rng.random_range(lo..=hi)
        } else {
            1.0
        };

        let saturation_factor = if self.saturation > 0.0 {
            let lo = (1.0 - self.saturation).max(0.0);
            let hi = 1.0 + self.saturation;
            rng.random_range(lo..=hi)
        } else {
            1.0
        };

        let hw = h * w;

        output_f32.copy_from_slice(input_f32);
        for v in output_f32.iter_mut() {
            *v *= brightness_factor;
        }

        if contrast_factor != 1.0 {
            let mean: f32 = output_f32.iter().copied().sum::<f32>() / output_f32.len() as f32;
            for v in output_f32.iter_mut() {
                *v = (*v - mean) * contrast_factor + mean;
            }
        }

        if saturation_factor != 1.0 {
            const LUMA_R: f32 = 0.299;
            const LUMA_G: f32 = 0.587;
            const LUMA_B: f32 = 0.114;

            for i in 0..hw {
                let r = output_f32[i];
                let g = output_f32[hw + i];
                let b = output_f32[2 * hw + i];

                let luma = LUMA_R * r + LUMA_G * g + LUMA_B * b;

                output_f32[i] = luma + saturation_factor * (r - luma);
                output_f32[hw + i] = luma + saturation_factor * (g - luma);
                output_f32[2 * hw + i] = luma + saturation_factor * (b - luma);
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

    #[test]
    fn test_color_jitter_zero_is_identity() {
        let jitter = ColorJitter::<f32>::new(0.0, 0.0, 0.0).unwrap();
        let input = vec![0.5f32; 12];
        let mut output = vec![0.0f32; 12];

        let mut rng = StdRng::seed_from_u64(42);
        jitter
            .apply(
                &input,
                ImageShape::new(3, 2, 2),
                &mut output,
                Some(&mut rng),
            )
            .unwrap();

        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-5, "Expected {}, got {}", a, b);
        }
    }

    #[test]
    fn test_color_jitter_brightness_only() {
        let jitter = ColorJitter::<f32>::new(0.5, 0.0, 0.0).unwrap();
        let input = vec![1.0f32; 12];
        let mut output = vec![0.0f32; 12];

        let mut rng = StdRng::seed_from_u64(42);
        jitter
            .apply(
                &input,
                ImageShape::new(3, 2, 2),
                &mut output,
                Some(&mut rng),
            )
            .unwrap();

        let first = output[0];
        for &v in &output {
            assert!((v - first).abs() < 1e-5);
        }
        assert!(first >= 0.5 - 1e-5 && first <= 1.5 + 1e-5);
    }

    #[test]
    fn test_color_jitter_requires_rgb() {
        let jitter = ColorJitter::<f32>::new(0.5, 0.5, 0.5).unwrap();
        let input = vec![0.0f32; 4];
        let mut output = vec![0.0f32; 4];

        let mut rng = StdRng::seed_from_u64(42);
        let result = jitter.apply(
            &input,
            ImageShape::new(1, 2, 2),
            &mut output,
            Some(&mut rng),
        );

        assert!(matches!(
            result,
            Err(AugmentationError::InvalidParameter { name, .. }) if name == "ColorJitter"
        ));
    }

    #[test]
    fn test_color_jitter_invalid_params() {
        assert!(ColorJitter::<f32>::new(-0.1, 0.0, 0.0).is_err());
        assert!(ColorJitter::<f32>::new(0.0, -0.1, 0.0).is_err());
        assert!(ColorJitter::<f32>::new(0.0, 0.0, -0.1).is_err());
        assert!(ColorJitter::<f32>::new(0.5, 0.5, 0.5).is_ok());
    }

    #[test]
    fn test_color_jitter_missing_rng() {
        let jitter = ColorJitter::<f32>::new(0.5, 0.5, 0.5).unwrap();
        let input = vec![0.5f32; 12];
        let mut output = vec![0.0f32; 12];

        let result = jitter.apply(&input, ImageShape::new(3, 2, 2), &mut output, None);
        assert!(matches!(result, Err(AugmentationError::MissingRng)));
    }

    #[test]
    fn test_color_jitter_zero_saturation_is_identity() {
        let jitter = ColorJitter::<f32>::new(0.0, 0.0, 0.0).unwrap();
        let input = vec![
            1.0f32, 0.5, 0.0, 0.8, 0.3, 0.1, 0.9, 0.4, 0.2, 0.7, 0.6, 0.5,
        ];
        let mut output = vec![0.0f32; 12];

        let mut rng = StdRng::seed_from_u64(123);
        jitter
            .apply(
                &input,
                ImageShape::new(3, 2, 2),
                &mut output,
                Some(&mut rng),
            )
            .unwrap();

        for (a, b) in input.iter().zip(output.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "Expected identity, but got {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_color_jitter_saturation_actually_changes_colors() {
        let jitter = ColorJitter::<f32>::new(0.0, 0.0, 1.0).unwrap();

        let mut input = vec![0.0f32; 12];
        for i in 0..4 {
            input[i] = 1.0;
            input[4 + i] = 0.5;
            input[8 + i] = 0.0;
        }
        let mut output = vec![0.0f32; 12];

        let mut rng = StdRng::seed_from_u64(123);
        jitter
            .apply(
                &input,
                ImageShape::new(3, 2, 2),
                &mut output,
                Some(&mut rng),
            )
            .unwrap();

        let any_different = input
            .iter()
            .zip(output.iter())
            .any(|(a, b)| (a - b).abs() > 1e-5);
        assert!(
            any_different,
            "Saturation should change colors, but output equals input"
        );
    }

    #[test]
    fn test_color_jitter_saturation_preserves_luma() {
        let jitter = ColorJitter::<f32>::new(0.0, 0.0, 1.0).unwrap();

        let mut input = vec![0.0f32; 12];
        for i in 0..4 {
            input[i] = 0.8;
            input[4 + i] = 0.4;
            input[8 + i] = 0.2;
        }
        let mut output = vec![0.0f32; 12];

        let mut rng = StdRng::seed_from_u64(42);
        jitter
            .apply(
                &input,
                ImageShape::new(3, 2, 2),
                &mut output,
                Some(&mut rng),
            )
            .unwrap();

        const LUMA_R: f32 = 0.299;
        const LUMA_G: f32 = 0.587;
        const LUMA_B: f32 = 0.114;

        for i in 0..4 {
            let input_luma = LUMA_R * input[i] + LUMA_G * input[4 + i] + LUMA_B * input[8 + i];
            let output_luma = LUMA_R * output[i] + LUMA_G * output[4 + i] + LUMA_B * output[8 + i];
            assert!(
                (input_luma - output_luma).abs() < 1e-4,
                "Luma should be preserved: input={}, output={}",
                input_luma,
                output_luma
            );
        }
    }

    #[test]
    fn test_color_jitter_does_not_change_size() {
        let jitter = ColorJitter::<f32>::new(0.5, 0.5, 0.5).unwrap();
        assert!(!jitter.changes_size());
        assert_eq!(jitter.fixed_output_size(), None);
    }
}
