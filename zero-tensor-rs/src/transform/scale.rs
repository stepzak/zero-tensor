use super::{Transform, error::TransformError};
use crate::{core::dataset::item::TensorViewMut, transform::Scalar};

pub struct Scale {
    factor: Scalar,
}

impl Scale {
    pub fn new<T: Into<Scalar>>(factor: T) -> Self {
        Scale {
            factor: factor.into(),
        }
    }
}

impl Transform for Scale {
    fn apply(&self, tensor: &mut TensorViewMut) -> Result<(), TransformError> {
        match tensor {
            TensorViewMut::BF16(t) => {
                let factor: half::bf16 = self.factor.try_into()?;
                t.map_inplace(|x| *x *= factor);
            }
            TensorViewMut::F32(t) => {
                let factor: f32 = self.factor.try_into()?;

                if let Some(slice) = t.as_slice_mut() {
                    for x in slice.iter_mut() {
                        *x *= factor;
                    }
                } else {
                    *t *= factor;
                }
            }
            TensorViewMut::F64(t) => {
                let factor: f64 = self.factor.try_into()?;
                if let Some(slice) = t.as_slice_mut() {
                    for x in slice.iter_mut() {
                        *x *= factor;
                    }
                } else {
                    *t *= factor;
                }
            }
            TensorViewMut::U8(t) => {
                let factor = self.factor.try_into()?;

                for x in t.iter() {
                    x.checked_mul(factor).ok_or(TransformError::Overflow)?;
                }

                t.map_inplace(|x| *x = unsafe { x.unchecked_mul(factor) });
            }
            TensorViewMut::F16(t) => {
                let factor: half::f16 = self.factor.try_into()?;

                t.map_inplace(|x| *x *= factor);
            }
            TensorViewMut::I8(t) => {
                let factor = self.factor.try_into()?;

                for x in t.iter() {
                    x.checked_mul(factor).ok_or(TransformError::Overflow)?;
                }

                t.map_inplace(|x| *x = unsafe { x.unchecked_mul(factor) });
            }
            TensorViewMut::I32(t) => {
                let factor = self.factor.try_into()?;

                for x in t.iter() {
                    x.checked_mul(factor).ok_or(TransformError::Overflow)?;
                }

                t.map_inplace(|x| *x = unsafe { x.unchecked_mul(factor) });
            }
            TensorViewMut::I64(t) => {
                let factor = self.factor.try_into()?;

                for x in t.iter() {
                    x.checked_mul(factor).ok_or(TransformError::Overflow)?;
                }

                t.map_inplace(|x| *x = unsafe { x.unchecked_mul(factor) });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::dataset::item::{TensorBatchLayout, TensorDT},
        transform::ScalarConversionError,
    };
    use rstest::rstest;

    macro_rules! bytes {
        ($data:expr) => {{
            let data = $data;
            let len = data.len();
            let raw_bytes: Vec<u8> = bytemuck::pod_collect_to_vec(&data);
            (raw_bytes, len)
        }};
    }

    macro_rules! assert_result {
        ($raw_bytes:expr, $expected:expr, $ty:ty) => {{
            let result: Vec<$ty> = bytemuck::pod_collect_to_vec(&$raw_bytes);

            assert_eq!(result, $expected);
        }};
    }

    // ============================================================
    // FLOATS
    // ============================================================

    #[rstest]
    #[case(TensorDT::BF16)]
    #[case(TensorDT::F16)]
    #[case(TensorDT::F32)]
    #[case(TensorDT::F64)]
    fn scale_float_positive_factor(#[case] dt: TensorDT) {
        let factor = 2.0;

        let (mut raw_bytes, len) = match dt {
            TensorDT::BF16 => {
                let data = vec![
                    half::bf16::from_f64(-2.0),
                    half::bf16::from_f64(0.0),
                    half::bf16::from_f64(3.0),
                ];

                bytes!(data)
            }

            TensorDT::F16 => {
                let data = vec![
                    half::f16::from_f64(-2.0),
                    half::f16::from_f64(0.0),
                    half::f16::from_f64(3.0),
                ];

                bytes!(data)
            }

            TensorDT::F32 => {
                let data = vec![-2.0f32, 0.0, 3.0];

                bytes!(data)
            }

            TensorDT::F64 => {
                let data = vec![-2.0f64, 0.0, 3.0];

                bytes!(data)
            }

            _ => unreachable!(),
        };

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), dt);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        Scale::new(factor).apply(&mut tensor).unwrap();

        match dt {
            TensorDT::BF16 => {
                assert_result!(
                    raw_bytes,
                    vec![
                        half::bf16::from_f64(-4.0),
                        half::bf16::from_f64(0.0),
                        half::bf16::from_f64(6.0),
                    ],
                    half::bf16
                );
            }

            TensorDT::F16 => {
                assert_result!(
                    raw_bytes,
                    vec![
                        half::f16::from_f64(-4.0),
                        half::f16::from_f64(0.0),
                        half::f16::from_f64(6.0),
                    ],
                    half::f16
                );
            }

            TensorDT::F32 => {
                assert_result!(raw_bytes, vec![-4.0f32, 0.0, 6.0], f32);
            }

            TensorDT::F64 => {
                assert_result!(raw_bytes, vec![-4.0f64, 0.0, 6.0], f64);
            }

            _ => unreachable!(),
        }
    }

    #[rstest]
    #[case(TensorDT::BF16)]
    #[case(TensorDT::F16)]
    #[case(TensorDT::F32)]
    #[case(TensorDT::F64)]
    fn scale_float_negative_factor(#[case] dt: TensorDT) {
        let factor = -2.0;

        let (mut raw_bytes, len) = match dt {
            TensorDT::BF16 => {
                let data = vec![
                    half::bf16::from_f64(-2.0),
                    half::bf16::from_f64(0.0),
                    half::bf16::from_f64(3.0),
                ];

                bytes!(data)
            }

            TensorDT::F16 => {
                let data = vec![
                    half::f16::from_f64(-2.0),
                    half::f16::from_f64(0.0),
                    half::f16::from_f64(3.0),
                ];

                bytes!(data)
            }

            TensorDT::F32 => {
                let data = vec![-2.0f32, 0.0, 3.0];

                bytes!(data)
            }

            TensorDT::F64 => {
                let data = vec![-2.0f64, 0.0, 3.0];

                bytes!(data)
            }

            _ => unreachable!(),
        };

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), dt);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        Scale::new(factor).apply(&mut tensor).unwrap();

        match dt {
            TensorDT::BF16 => {
                assert_result!(
                    raw_bytes,
                    vec![
                        half::bf16::from_f64(4.0),
                        half::bf16::from_f64(0.0),
                        half::bf16::from_f64(-6.0),
                    ],
                    half::bf16
                );
            }

            TensorDT::F16 => {
                assert_result!(
                    raw_bytes,
                    vec![
                        half::f16::from_f64(4.0),
                        half::f16::from_f64(0.0),
                        half::f16::from_f64(-6.0),
                    ],
                    half::f16
                );
            }

            TensorDT::F32 => {
                assert_result!(raw_bytes, vec![4.0f32, 0.0, -6.0], f32);
            }

            TensorDT::F64 => {
                assert_result!(raw_bytes, vec![4.0f64, 0.0, -6.0], f64);
            }

            _ => unreachable!(),
        }
    }

    #[rstest]
    #[case(TensorDT::BF16)]
    #[case(TensorDT::F16)]
    #[case(TensorDT::F32)]
    #[case(TensorDT::F64)]
    fn scale_float_zero_factor(#[case] dt: TensorDT) {
        let (mut raw_bytes, len) = match dt {
            TensorDT::BF16 => {
                let data = vec![half::bf16::from_f64(-10.0), half::bf16::from_f64(5.0)];

                bytes!(data)
            }

            TensorDT::F16 => {
                let data = vec![half::f16::from_f64(-10.0), half::f16::from_f64(5.0)];

                bytes!(data)
            }

            TensorDT::F32 => {
                let data = vec![-10.0f32, 5.0];

                bytes!(data)
            }

            TensorDT::F64 => {
                let data = vec![-10.0f64, 5.0];

                bytes!(data)
            }

            _ => unreachable!(),
        };

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), dt);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        Scale::new(0.0).apply(&mut tensor).unwrap();

        match dt {
            TensorDT::BF16 => {
                assert_result!(
                    raw_bytes,
                    vec![half::bf16::from_f64(0.0), half::bf16::from_f64(0.0),],
                    half::bf16
                );
            }

            TensorDT::F16 => {
                assert_result!(
                    raw_bytes,
                    vec![half::f16::from_f64(0.0), half::f16::from_f64(0.0),],
                    half::f16
                );
            }

            TensorDT::F32 => {
                assert_result!(raw_bytes, vec![0.0f32, 0.0], f32);
            }

            TensorDT::F64 => {
                assert_result!(raw_bytes, vec![0.0f64, 0.0], f64);
            }

            _ => unreachable!(),
        }
    }

    // ============================================================
    // INTEGERS
    // ============================================================

    #[rstest]
    #[case(TensorDT::U8)]
    #[case(TensorDT::I8)]
    #[case(TensorDT::I32)]
    #[case(TensorDT::I64)]
    fn scale_integer_positive_factor(#[case] dt: TensorDT) {
        let (mut raw_bytes, len) = match dt {
            TensorDT::U8 => {
                let data = vec![0u8, 1, 10, 100];
                bytes!(data)
            }

            TensorDT::I8 => {
                let data = vec![-10i8, 0, 5, 50];
                bytes!(data)
            }

            TensorDT::I32 => {
                let data = vec![-10i32, 0, 5, 100];
                bytes!(data)
            }

            TensorDT::I64 => {
                let data = vec![-10i64, 0, 5, 100];
                bytes!(data)
            }

            _ => unreachable!(),
        };

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), dt);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        Scale::new(2.0).apply(&mut tensor).unwrap();

        match dt {
            TensorDT::U8 => {
                assert_result!(raw_bytes, vec![0u8, 2, 20, 200], u8);
            }

            TensorDT::I8 => {
                assert_result!(raw_bytes, vec![-20i8, 0, 10, 100], i8);
            }

            TensorDT::I32 => {
                assert_result!(raw_bytes, vec![-20i32, 0, 10, 200], i32);
            }

            TensorDT::I64 => {
                assert_result!(raw_bytes, vec![-20i64, 0, 10, 200], i64);
            }

            _ => unreachable!(),
        }
    }

    #[rstest]
    #[case(TensorDT::I8)]
    #[case(TensorDT::I32)]
    #[case(TensorDT::I64)]
    fn scale_integer_negative_factor(#[case] dt: TensorDT) {
        let (mut raw_bytes, len) = match dt {
            TensorDT::I8 => {
                let data = vec![-10i8, 0, 5];
                bytes!(data)
            }

            TensorDT::I32 => {
                let data = vec![-10i32, 0, 5];
                bytes!(data)
            }

            TensorDT::I64 => {
                let data = vec![-10i64, 0, 5];
                bytes!(data)
            }

            _ => unreachable!(),
        };

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), dt);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        Scale::new(-2.0).apply(&mut tensor).unwrap();

        match dt {
            TensorDT::I8 => {
                assert_result!(raw_bytes, vec![20i8, 0, -10], i8);
            }

            TensorDT::I32 => {
                assert_result!(raw_bytes, vec![20i32, 0, -10], i32);
            }

            TensorDT::I64 => {
                assert_result!(raw_bytes, vec![20i64, 0, -10], i64);
            }

            _ => unreachable!(),
        }
    }

    #[rstest]
    #[case(TensorDT::U8)]
    #[case(TensorDT::I8)]
    #[case(TensorDT::I32)]
    #[case(TensorDT::I64)]
    fn scale_integer_zero_factor(#[case] dt: TensorDT) {
        let (mut raw_bytes, len) = match dt {
            TensorDT::U8 => {
                let data = vec![1u8, 10, 100];
                bytes!(data)
            }

            TensorDT::I8 => {
                let data = vec![-10i8, 0, 10];
                bytes!(data)
            }

            TensorDT::I32 => {
                let data = vec![-100i32, 0, 100];
                bytes!(data)
            }

            TensorDT::I64 => {
                let data = vec![-100i64, 0, 100];
                bytes!(data)
            }

            _ => unreachable!(),
        };

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), dt);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        Scale::new(0.0).apply(&mut tensor).unwrap();

        match dt {
            TensorDT::U8 => {
                assert_result!(raw_bytes, vec![0u8, 0, 0], u8);
            }

            TensorDT::I8 => {
                assert_result!(raw_bytes, vec![0i8, 0, 0], i8);
            }

            TensorDT::I32 => {
                assert_result!(raw_bytes, vec![0i32, 0, 0], i32);
            }

            TensorDT::I64 => {
                assert_result!(raw_bytes, vec![0i64, 0, 0], i64);
            }

            _ => unreachable!(),
        }
    }

    // ============================================================
    // FRACTIONAL FACTOR FOR INTEGER TENSORS
    // ============================================================

    #[rstest]
    #[case(TensorDT::U8)]
    #[case(TensorDT::I8)]
    #[case(TensorDT::I32)]
    #[case(TensorDT::I64)]
    fn integer_scale_rejects_fractional_factor(#[case] dt: TensorDT) {
        let (mut raw_bytes, len) = match dt {
            TensorDT::U8 => {
                let data = vec![1u8, 2, 3];
                bytes!(data)
            }

            TensorDT::I8 => {
                let data = vec![1i8, 2, 3];
                bytes!(data)
            }

            TensorDT::I32 => {
                let data = vec![1i32, 2, 3];
                bytes!(data)
            }

            TensorDT::I64 => {
                let data = vec![1i64, 2, 3];
                bytes!(data)
            }

            _ => unreachable!(),
        };

        let original = raw_bytes.clone();

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), dt);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        let result = Scale::new(2.5).apply(&mut tensor);

        assert!(matches!(
            result,
            Err(TransformError::ScalarConversion(
                ScalarConversionError::FractionalValue
            ))
        ));

        assert_eq!(raw_bytes, original);
    }

    // ============================================================
    // OVERFLOW
    // ============================================================

    #[test]
    fn scale_u8_overflow_does_not_modify_tensor() {
        let data = vec![10u8, 127, 200];
        let expected = data.clone();

        let (mut raw_bytes, len) = bytes!(data);

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), TensorDT::U8);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        let result = Scale::new(2u8).apply(&mut tensor);

        assert!(matches!(result, Err(TransformError::Overflow)));

        assert_result!(raw_bytes, expected, u8);
    }

    #[test]
    fn scale_i8_positive_overflow_does_not_modify_tensor() {
        let data = vec![10i8, 64, 100];
        let expected = data.clone();

        let (mut raw_bytes, len) = bytes!(data);

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), TensorDT::I8);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        let result = Scale::new(2i8).apply(&mut tensor);

        assert!(matches!(result, Err(TransformError::Overflow)));

        assert_result!(raw_bytes, expected, i8);
    }

    #[test]
    fn scale_i8_negative_overflow_does_not_modify_tensor() {
        let data = vec![-100i8, -64, 10];
        let expected = data.clone();

        let (mut raw_bytes, len) = bytes!(data);

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), TensorDT::I8);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        let result = Scale::new(2i8).apply(&mut tensor);

        assert!(matches!(result, Err(TransformError::Overflow)));

        assert_result!(raw_bytes, expected, i8);
    }

    #[test]
    fn scale_i32_overflow_does_not_modify_tensor() {
        let data = vec![i32::MAX, 10];
        let expected = data.clone();

        let (mut raw_bytes, len) = bytes!(data);

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), TensorDT::I32);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        let result = Scale::new(2i32).apply(&mut tensor);

        assert!(matches!(result, Err(TransformError::Overflow)));

        assert_result!(raw_bytes, expected, i32);
    }

    #[test]
    fn scale_i64_overflow_does_not_modify_tensor() {
        let data = vec![i64::MAX, 10];
        let expected = data.clone();

        let (mut raw_bytes, len) = bytes!(data);

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), TensorDT::I64);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        let result = Scale::new(2i64).apply(&mut tensor);

        assert!(matches!(result, Err(TransformError::Overflow)));

        assert_result!(raw_bytes, expected, i64);
    }

    // ============================================================
    // FACTOR CONVERSION OVERFLOW
    // ============================================================

    #[test]
    fn scale_u8_rejects_out_of_range_factor() {
        let data = vec![1u8, 2, 3];
        let expected = data.clone();

        let (mut raw_bytes, len) = bytes!(data);

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), TensorDT::U8);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        let result = Scale::new(300i32).apply(&mut tensor);

        assert!(result.is_err());

        assert_result!(raw_bytes, expected, u8);
    }

    #[test]
    fn scale_i8_rejects_out_of_range_factor() {
        let data = vec![1i8, 2, 3];
        let expected = data.clone();

        let (mut raw_bytes, len) = bytes!(data);

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), TensorDT::I8);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        let result = Scale::new(200i32).apply(&mut tensor);

        assert!(result.is_err());

        assert_result!(raw_bytes, expected, i8);
    }

    #[test]
    fn scale_i32_rejects_out_of_range_factor() {
        let data = vec![1i32, 2, 3];
        let expected = data.clone();

        let (mut raw_bytes, len) = bytes!(data);

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), TensorDT::I32);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        let result = Scale::new(i64::MAX).apply(&mut tensor);

        assert!(result.is_err());

        assert_result!(raw_bytes, expected, i32);
    }
}
