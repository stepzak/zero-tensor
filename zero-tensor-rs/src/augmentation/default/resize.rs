use std::{cell::RefCell, marker::PhantomData};

use crate::augmentation::{Augmentation, AugmentationError, AugmentationItem, ImageShape};
use fast_image_resize::{
    FilterType::Bilinear,
    PixelType, ResizeAlg, ResizeOptions, Resizer,
    images::{Image, ImageRef},
};
use rand::Rng;

thread_local! {
    static RESIZER: RefCell<Resizer> = RefCell::new(Resizer::new());
}

#[derive(Debug, Clone)]
pub struct Resize<T> {
    target_h: usize,
    target_w: usize,
    _marker: PhantomData<T>,
}

impl<T> Resize<T> {
    pub fn new(h: usize, w: usize) -> Self {
        Self {
            target_h: h,
            target_w: w,
            _marker: PhantomData,
        }
    }
}

impl<T: AugmentationItem + std::fmt::Debug> Augmentation for Resize<T> {
    type InputItem = T;
    type OutputItem = T;

    fn name(&self) -> &'static str {
        "Resize"
    }

    fn changes_size(&self) -> bool {
        true
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

        if h == self.target_h && w == self.target_w {
            output[..input.len()].copy_from_slice(input);
            return Ok(input_shape);
        }

        let pixel_type = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            PixelType::F32
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u8>() {
            PixelType::U8
        } else {
            return Err(AugmentationError::UnsupportedDtype {
                name: self.name(),
                dtype: std::any::type_name::<T>().to_string(),
            });
        };

        let src_len = h * w;
        let dst_len = self.target_h * self.target_w;
        let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(Bilinear));

        RESIZER.with_borrow_mut(|resizer| {
            for channel in 0..c {
                let src_offset = channel * src_len;
                let dst_offset = channel * dst_len;

                let src_bytes =
                    bytemuck::cast_slice::<T, u8>(&input[src_offset..src_offset + src_len]);
                let dst_bytes = bytemuck::cast_slice_mut::<T, u8>(
                    &mut output[dst_offset..dst_offset + dst_len],
                );

                let src_image =
                    ImageRef::new(w as u32, h as u32, src_bytes, pixel_type).map_err(|e| {
                        AugmentationError::Custom(format!("Failed to create src image: {}", e))
                    })?;

                let mut dst_image = Image::from_slice_u8(
                    self.target_w as u32,
                    self.target_h as u32,
                    dst_bytes,
                    pixel_type,
                )
                .map_err(|e| {
                    AugmentationError::Custom(format!("Failed to create dst image: {}", e))
                })?;

                resizer
                    .resize(&src_image, &mut dst_image, &options)
                    .map_err(|e| AugmentationError::Custom(format!("Resize failed: {}", e)))?;
            }
            Ok(())
        })?;

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

    fn create_gradient_image(c: usize, h: usize, w: usize) -> Vec<f32> {
        let mut img = vec![0.0f32; c * h * w];
        for channel in 0..c {
            let c_offset = channel * h * w;
            for y in 0..h {
                for x in 0..w {
                    img[c_offset + y * w + x] = (x as f32 / w as f32) + (channel as f32 * 0.1);
                }
            }
        }
        img
    }

    #[test]
    fn test_resize_downscale() {
        let resize = Resize::<f32>::new(2, 2);
        let input = create_gradient_image(1, 4, 4);
        let mut output = vec![0.0f32; 4];

        let shape = resize
            .apply(
                &input,
                ImageShape {
                    channels: 1,
                    height: 4,
                    width: 4,
                },
                &mut output,
                None,
            )
            .unwrap();

        assert_eq!(
            shape,
            ImageShape {
                channels: 1,
                height: 2,
                width: 2
            }
        );

        assert!(output.iter().any(|&x| x > 0.0));
        assert!(output.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }

    #[test]
    fn test_resize_upscale() {
        let resize = Resize::<f32>::new(4, 4);
        let input = create_gradient_image(1, 2, 2);
        let mut output = vec![0.0f32; 16];

        let shape = resize
            .apply(
                &input,
                ImageShape {
                    channels: 1,
                    height: 2,
                    width: 2,
                },
                &mut output,
                None,
            )
            .unwrap();

        assert_eq!(
            shape,
            ImageShape {
                channels: 1,
                height: 4,
                width: 4
            }
        );
        assert!(output.iter().any(|&x| x > 0.0));
    }

    #[test]
    fn test_resize_multichannel() {
        let resize = Resize::<f32>::new(2, 2);
        let input = create_gradient_image(3, 4, 4);
        let mut output = vec![0.0f32; 12];

        let shape = resize
            .apply(
                &input,
                ImageShape {
                    channels: 3,
                    height: 4,
                    width: 4,
                },
                &mut output,
                None,
            )
            .unwrap();

        assert_eq!(
            shape,
            ImageShape {
                channels: 3,
                height: 2,
                width: 2
            }
        );

        let val_c0 = output[0];
        let val_c1 = output[4];
        assert!((val_c1 - val_c0 - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_resize_fast_path() {
        let resize = Resize::<f32>::new(3, 3);
        let input = vec![1.0f32; 9];
        let mut output = vec![0.0f32; 9];

        let shape = resize
            .apply(
                &input,
                ImageShape {
                    channels: 1,
                    height: 3,
                    width: 3,
                },
                &mut output,
                None,
            )
            .unwrap();

        assert_eq!(
            shape,
            ImageShape {
                channels: 1,
                height: 3,
                width: 3
            }
        );
        assert_eq!(input, output);
    }

    #[test]
    fn test_resize_unsupported_dtype() {
        let resize = Resize::<i32>::new(2, 2);
        let input = vec![0i32; 16];
        let mut output = vec![0i32; 4];

        let result = resize.apply(
            &input,
            ImageShape {
                channels: 1,
                height: 4,
                width: 4,
            },
            &mut output,
            None,
        );

        assert!(matches!(
            result,
            Err(AugmentationError::UnsupportedDtype { name, .. }) if name == "Resize"
        ));
    }

    #[test]
    fn test_resize_fixed_output_size() {
        let resize = Resize::<f32>::new(224, 224);
        assert_eq!(resize.fixed_output_size(), Some((224, 224)));
    }
}
