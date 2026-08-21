use super::{Scalar, TensorViewMut, Transform, TransformError, scalar::is_zero::IsZero};

pub struct Standardize {
    mean: Scalar,
    std: Scalar,
}

impl Standardize {
    pub fn new<T: Into<Scalar>, M: Into<Scalar>>(mean: T, std: M) -> Result<Self, TransformError> {
        let mean = mean.into();
        let std = std.into();
        if !mean.is_finite() || !std.is_finite() || std == 0.into() {
            return Err(TransformError::InvalidValue);
        }

        Ok(Self { mean, std })
    }
}

impl Transform for Standardize {
    fn apply(&self, tensor: &mut TensorViewMut) -> Result<(), TransformError> {
        macro_rules! standardize {
            ($ty:ty, $t:expr) => {{
                let mean: $ty = self.mean.try_into()?;
                let std: $ty = self.std.try_into()?;
                if std.eq_zero() {
                    return Err(TransformError::InvalidValue);
                }
                $t.map_inplace(|x| *x = (*x - mean) / std);
            }};
        }
        match tensor {
            TensorViewMut::BF16(t) => standardize!(half::bf16, t),
            TensorViewMut::F16(t) => standardize!(half::f16, t),
            TensorViewMut::F32(t) => standardize!(f32, t),
            TensorViewMut::F64(t) => standardize!(f64, t),
            _ => {
                return Err(TransformError::UnsupportedDtype);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dataset::item::{TensorBatchLayout, TensorDT};
    use rstest::rstest;

    macro_rules! assert_invalid_value {
        ($expr:expr) => {
            assert!(matches!($expr, Err(TransformError::InvalidValue)));
        };
    }

    #[rstest]
    #[case(TensorDT::F16)]
    #[case(TensorDT::BF16)]
    #[case(TensorDT::F32)]
    #[case(TensorDT::F64)]
    fn standardize_float_tensors(#[case] dt: TensorDT) {
        let (mut raw_bytes, len) = match dt {
            TensorDT::F16 => {
                let data = vec![
                    half::f16::from_f32(1.0),
                    half::f16::from_f32(3.0),
                    half::f16::from_f32(5.0),
                ];
                let len = data.len();
                (bytemuck::pod_collect_to_vec(&data), len)
            }

            TensorDT::BF16 => {
                let data = vec![
                    half::bf16::from_f32(1.0),
                    half::bf16::from_f32(3.0),
                    half::bf16::from_f32(5.0),
                ];
                let len = data.len();
                (bytemuck::pod_collect_to_vec(&data), len)
            }

            TensorDT::F32 => {
                let data = vec![1.0f32, 3.0, 5.0];
                let len = data.len();
                (bytemuck::pod_collect_to_vec(&data), len)
            }

            TensorDT::F64 => {
                let data = vec![1.0f64, 3.0, 5.0];
                let len = data.len();
                (bytemuck::pod_collect_to_vec(&data), len)
            }

            _ => unreachable!(),
        };

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), dt);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        let transform = Standardize::new(Scalar::from(3.0f64), Scalar::from(2.0f64)).unwrap();

        transform.apply(&mut tensor).unwrap();

        match dt {
            TensorDT::F16 => {
                let result: Vec<half::f16> = bytemuck::pod_collect_to_vec(&raw_bytes);

                let expected = vec![
                    half::f16::from_f32(-1.0),
                    half::f16::from_f32(0.0),
                    half::f16::from_f32(1.0),
                ];

                assert_eq!(result, expected);
            }

            TensorDT::BF16 => {
                let result: Vec<half::bf16> = bytemuck::pod_collect_to_vec(&raw_bytes);

                let expected = vec![
                    half::bf16::from_f32(-1.0),
                    half::bf16::from_f32(0.0),
                    half::bf16::from_f32(1.0),
                ];

                assert_eq!(result, expected);
            }

            TensorDT::F32 => {
                let result: Vec<f32> = bytemuck::pod_collect_to_vec(&raw_bytes);

                assert_eq!(result, vec![-1.0, 0.0, 1.0]);
            }

            TensorDT::F64 => {
                let result: Vec<f64> = bytemuck::pod_collect_to_vec(&raw_bytes);

                assert_eq!(result, vec![-1.0, 0.0, 1.0]);
            }

            _ => unreachable!(),
        }
    }

    #[test]
    fn standardize_accepts_mixed_scalar_types() {
        let data = vec![1.0f32, 3.0, 5.0];
        let mut raw_bytes = bytemuck::pod_collect_to_vec(&data);

        let layout = TensorBatchLayout::new(vec![data.len()].into(), vec![1].into(), TensorDT::F32);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        let transform = Standardize::new(Scalar::from(3i32), Scalar::from(2i8)).unwrap();

        transform.apply(&mut tensor).unwrap();

        let result: Vec<f32> = bytemuck::pod_collect_to_vec(&raw_bytes);

        assert_eq!(result, vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn standardize_negative_std() {
        let data = vec![1.0f32, 3.0, 5.0];
        let mut raw_bytes = bytemuck::pod_collect_to_vec(&data);

        let layout = TensorBatchLayout::new(vec![data.len()].into(), vec![1].into(), TensorDT::F32);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        let transform = Standardize::new(Scalar::from(3.0), Scalar::from(-2.0)).unwrap();

        transform.apply(&mut tensor).unwrap();

        let result: Vec<f32> = bytemuck::pod_collect_to_vec(&raw_bytes);

        assert_eq!(result, vec![1.0, -0.0, -1.0]);
    }

    #[rstest]
    #[case(Scalar::from(0u8))]
    #[case(Scalar::from(0i8))]
    #[case(Scalar::from(0i32))]
    #[case(Scalar::from(0i64))]
    #[case(Scalar::from(0.0f32))]
    #[case(Scalar::from(0.0f64))]
    #[case(Scalar::from(half::f16::from_f32(0.0)))]
    #[case(Scalar::from(half::bf16::from_f32(0.0)))]
    fn standardize_rejects_zero_std(#[case] std: Scalar) {
        assert_invalid_value!(Standardize::new(Scalar::from(0.0), std));
    }

    #[rstest]
    #[case(Scalar::from(f32::NAN))]
    #[case(Scalar::from(f64::NAN))]
    #[case(Scalar::from(half::f16::NAN))]
    #[case(Scalar::from(half::bf16::NAN))]
    fn standardize_rejects_nan_mean(#[case] mean: Scalar) {
        assert_invalid_value!(Standardize::new(mean, Scalar::from(1.0),));
    }

    #[rstest]
    #[case(Scalar::from(f32::NAN))]
    #[case(Scalar::from(f64::NAN))]
    #[case(Scalar::from(half::f16::NAN))]
    #[case(Scalar::from(half::bf16::NAN))]
    fn standardize_rejects_nan_std(#[case] std: Scalar) {
        assert_invalid_value!(Standardize::new(Scalar::from(0.0), std,));
    }

    #[rstest]
    #[case(Scalar::from(f32::INFINITY))]
    #[case(Scalar::from(f32::NEG_INFINITY))]
    #[case(Scalar::from(f64::INFINITY))]
    #[case(Scalar::from(f64::NEG_INFINITY))]
    #[case(Scalar::from(half::f16::INFINITY))]
    #[case(Scalar::from(half::f16::NEG_INFINITY))]
    #[case(Scalar::from(half::bf16::INFINITY))]
    #[case(Scalar::from(half::bf16::NEG_INFINITY))]
    fn standardize_rejects_infinite_mean(#[case] mean: Scalar) {
        assert_invalid_value!(Standardize::new(mean, Scalar::from(1.0),));
    }

    #[rstest]
    #[case(Scalar::from(f32::INFINITY))]
    #[case(Scalar::from(f32::NEG_INFINITY))]
    #[case(Scalar::from(f64::INFINITY))]
    #[case(Scalar::from(f64::NEG_INFINITY))]
    #[case(Scalar::from(half::f16::INFINITY))]
    #[case(Scalar::from(half::f16::NEG_INFINITY))]
    #[case(Scalar::from(half::bf16::INFINITY))]
    #[case(Scalar::from(half::bf16::NEG_INFINITY))]
    fn standardize_rejects_infinite_std(#[case] std: Scalar) {
        assert_invalid_value!(Standardize::new(Scalar::from(0.0), std,));
    }

    #[rstest]
    #[case(TensorDT::U8)]
    #[case(TensorDT::I8)]
    #[case(TensorDT::I32)]
    #[case(TensorDT::I64)]
    fn standardize_rejects_integer_tensors(#[case] dt: TensorDT) {
        macro_rules! make_data {
            ($ty:ty) => {{
                let data = vec![1 as $ty, 2 as $ty, 3 as $ty];
                bytemuck::pod_collect_to_vec(&data)
            }};
        }

        let mut raw_bytes = match dt {
            TensorDT::U8 => make_data!(u8),
            TensorDT::I8 => make_data!(i8),
            TensorDT::I32 => make_data!(i32),
            TensorDT::I64 => make_data!(i64),
            _ => unreachable!(),
        };

        let original = raw_bytes.clone();

        let layout = TensorBatchLayout::new(vec![3].into(), vec![1].into(), dt);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        let transform = Standardize::new(Scalar::from(2.0), Scalar::from(1.0)).unwrap();

        let result = transform.apply(&mut tensor);

        assert!(matches!(result, Err(TransformError::UnsupportedDtype)));

        assert_eq!(raw_bytes, original);
    }

    #[test]
    fn standardize_fails_when_std_becomes_zero_after_conversion() {
        let transform = Standardize::new(Scalar::from(0.0f64), Scalar::from(1e-100f64)).unwrap();

        let data = vec![1.0f32, 2.0f32];
        let mut raw_bytes = bytemuck::pod_collect_to_vec(&data);
        let original = raw_bytes.clone();

        let layout = TensorBatchLayout::new(vec![data.len()].into(), vec![1].into(), TensorDT::F32);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        let result = transform.apply(&mut tensor);

        assert!(matches!(result, Err(TransformError::InvalidValue)));

        drop(tensor);
        assert_eq!(raw_bytes, original);
    }
}
