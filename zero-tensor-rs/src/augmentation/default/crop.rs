use std::{fmt::Debug, marker::PhantomData};

use crate::augmentation::{Augmentation, AugmentationError, AugmentationItem, ImageShape};
use rand::{Rng, RngExt};

#[derive(Debug, Clone)]
pub struct RandomCrop<T> {
    target_h: usize,
    target_w: usize,
    _marker: PhantomData<T>,
}

impl<T> RandomCrop<T> {
    pub fn new(h: usize, w: usize) -> Self {
        Self {
            target_h: h,
            target_w: w,
            _marker: PhantomData,
        }
    }
}

impl<T: AugmentationItem + Debug> Augmentation for RandomCrop<T> {
    type InputItem = T;
    type OutputItem = T;

    fn name(&self) -> &'static str {
        "RandomCrop"
    }

    fn changes_size(&self) -> bool {
        true
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
        let rng = rng.ok_or(AugmentationError::MissingRng)?;

        if h < self.target_h || w < self.target_w {
            return Err(AugmentationError::InvalidParameter {
                name: self.name(),
                message: format!(
                    "Input {}x{} smaller than crop {}x{}",
                    h, w, self.target_h, self.target_w
                ),
            });
        }

        let max_y = h - self.target_h;
        let max_x = w - self.target_w;
        let y = rng.random_range(0..=max_y);
        let x = rng.random_range(0..=max_x);

        for channel in 0..c {
            let src_offset = channel * h * w;
            let dst_offset = channel * self.target_h * self.target_w;

            for dy in 0..self.target_h {
                let src_row = src_offset + (y + dy) * w + x;
                let dst_row = dst_offset + dy * self.target_w;
                output[dst_row..dst_row + self.target_w]
                    .copy_from_slice(&input[src_row..src_row + self.target_w]);
            }
        }

        Ok(ImageShape {
            channels: c,
            height: self.target_h,
            width: self.target_w,
        })
    }

    fn fixed_output_size(&self) -> Option<(usize, usize)> {
        Some((self.target_h, self.target_w))
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
    fn test_random_crop_basic_and_deterministic() {
        let crop = RandomCrop::<f32>::new(2, 2);
        let input = create_indexed_image(1, 4, 4);
        let mut output = vec![0.0f32; 4];

        let mut rng = StdRng::seed_from_u64(42);
        let shape = ImageShape {
            channels: 1,
            height: 4,
            width: 4,
        };
        let shape = crop
            .apply(&input, shape, &mut output, Some(&mut rng))
            .unwrap();

        assert_eq!(shape.channels, 1);
        assert_eq!(shape.height, 2);
        assert_eq!(shape.width, 2);

        let top_left = output[0] as usize;
        let x = top_left % 4;
        let y = top_left / 4;

        assert_eq!(output[0], (y * 4 + x) as f32);
        assert_eq!(output[1], (y * 4 + x + 1) as f32);
        assert_eq!(output[2], ((y + 1) * 4 + x) as f32);
        assert_eq!(output[3], ((y + 1) * 4 + x + 1) as f32);

        assert!(x <= 2);
        assert!(y <= 2);
    }

    #[test]
    fn test_random_crop_multichannel() {
        let crop = RandomCrop::<f32>::new(1, 1);
        let input = create_indexed_image(2, 2, 2);
        let mut output = vec![0.0f32; 2];
        let shape = ImageShape {
            channels: 2,
            height: 2,
            width: 2,
        };
        let mut rng = StdRng::seed_from_u64(123);
        let shape = crop
            .apply(&input, shape, &mut output, Some(&mut rng))
            .unwrap();

        assert_eq!(shape.channels, 2);
        assert_eq!(shape.height, 1);
        assert_eq!(shape.width, 1);
        let val_c0 = output[0] as usize;
        let val_c1 = output[1] as usize;
        assert!(
            val_c0 < 1000,
            "Channel 0 value should be small, got {}",
            val_c0
        );
        assert!(
            val_c1 >= 1000,
            "Channel 1 value should be large, got {}",
            val_c1
        );

        let pos_c0 = val_c0 % 1000;
        let pos_c1 = val_c1 % 1000;
        assert_eq!(pos_c0, pos_c1, "Spatial position mismatch between channels");
    }

    #[test]
    fn test_random_crop_exact_size_match() {
        let crop = RandomCrop::<f32>::new(3, 3);
        let input = create_indexed_image(1, 3, 3);
        let mut output = vec![0.0f32; 9];
        let shape = ImageShape {
            channels: 1,
            height: 3,
            width: 3,
        };

        let mut rng = StdRng::seed_from_u64(0);
        let shape = crop
            .apply(&input, shape, &mut output, Some(&mut rng))
            .unwrap();

        assert_eq!(shape.channels, 1);
        assert_eq!(shape.width, 3);
        assert_eq!(shape.height, 3);
        assert_eq!(input, output);
    }

    #[test]
    fn test_random_crop_input_too_small() {
        let crop = RandomCrop::<f32>::new(5, 5);
        let input = create_indexed_image(1, 4, 4);
        let mut output = vec![0.0f32; 25];

        let mut rng = StdRng::seed_from_u64(0);
        let shape = ImageShape {
            channels: 1,
            height: 4,
            width: 4,
        };

        let result = crop.apply(&input, shape, &mut output, Some(&mut rng));

        assert!(matches!(
            result,
            Err(AugmentationError::InvalidParameter { name, .. }) if name == "RandomCrop"
        ));
    }

    #[test]
    fn test_random_crop_missing_rng() {
        let crop = RandomCrop::<f32>::new(2, 2);
        let input = create_indexed_image(1, 4, 4);
        let mut output = vec![0.0f32; 4];

        let shape = ImageShape {
            channels: 1,
            height: 4,
            width: 4,
        };
        let result = crop.apply(&input, shape, &mut output, None);

        assert!(matches!(result, Err(AugmentationError::MissingRng)));
    }

    #[test]
    fn test_random_crop_fixed_output_size() {
        let crop = RandomCrop::<f32>::new(224, 224);
        assert_eq!(crop.fixed_output_size(), Some((224, 224)));

        let crop_non_square = RandomCrop::<f32>::new(100, 200);
        assert_eq!(crop_non_square.fixed_output_size(), Some((100, 200)));
    }

    #[test]
    fn test_random_crop_changes_size() {
        let crop = RandomCrop::<f32>::new(10, 10);
        assert!(crop.changes_size());
    }
}
