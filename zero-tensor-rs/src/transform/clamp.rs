use crate::transform::{IntoScalarOption, Scalar, ScalarConversionError};

use super::{TensorViewMut, Transform, TransformError};

pub enum IntRoundingMode {
    Error,
    Round,
    Floor,
    Ceil,
}

pub enum OverflowMode {
    Error,
    Clamp,
}

pub struct Clamp {
    min: Option<Scalar>,
    max: Option<Scalar>,
    int_rounding_mode: IntRoundingMode,
    overflow_mode: OverflowMode,
}

impl Clamp {
    pub fn new<T: IntoScalarOption>(min: T, max: T) -> Result<Self, TransformError> {
        let min = min.into_scalar_option();
        let max = max.into_scalar_option();

        if min.is_some_and(|x| x.is_nan()) || max.is_some_and(|x| x.is_nan()) {
            return Err(TransformError::InvalidValue);
        }
        if let (Some(mx), Some(mn)) = (max, min)
            && mx < mn
        {
            return Err(TransformError::InvalidValue);
        }
        Ok(Self {
            min,
            max,
            int_rounding_mode: IntRoundingMode::Error,
            overflow_mode: OverflowMode::Error,
        })
    }

    pub fn int_rounding_mode(self, int_rounding_mode: IntRoundingMode) -> Self {
        Self {
            int_rounding_mode,
            ..self
        }
    }

    pub fn overflow_mode(self, overflow_mode: OverflowMode) -> Self {
        Self {
            overflow_mode,
            ..self
        }
    }

    fn scalar_to_f64(value: Scalar) -> Result<f64, TransformError> {
        value.try_into().map_err(Into::into)
    }

    fn rounded_scalar(&self, value: Scalar) -> Result<Scalar, TransformError> {
        let value = Self::scalar_to_f64(value)?;

        let value = match self.int_rounding_mode {
            IntRoundingMode::Error => {
                return Err(ScalarConversionError::FractionalValue.into());
            }
            IntRoundingMode::Round => value.round(),
            IntRoundingMode::Floor => value.floor(),
            IntRoundingMode::Ceil => value.ceil(),
        };

        Ok(Scalar::F64(value))
    }

    fn resolve_int<T>(&self, value: Scalar, min: T, max: T) -> Result<T, TransformError>
    where
        T: TryFrom<Scalar, Error = ScalarConversionError> + Copy + Into<Scalar>,
    {
        match value.try_into() {
            Ok(value) => Ok(value),

            Err(ScalarConversionError::FractionalValue) => {
                let rounded = self.rounded_scalar(value)?;

                match rounded.try_into() {
                    Ok(value) => Ok(value),

                    Err(ScalarConversionError::Overflow) => {
                        let rounded_f64 = Self::scalar_to_f64(rounded)?;

                        match self.overflow_mode {
                            OverflowMode::Error => Err(ScalarConversionError::Overflow.into()),
                            OverflowMode::Clamp => {
                                let min_scalar: Scalar = min.into();

                                if Scalar::F64(rounded_f64) < min_scalar {
                                    Ok(min)
                                } else {
                                    Ok(max)
                                }
                            }
                        }
                    }

                    Err(e) => Err(e.into()),
                }
            }

            Err(ScalarConversionError::Overflow) => match self.overflow_mode {
                OverflowMode::Error => Err(ScalarConversionError::Overflow.into()),

                OverflowMode::Clamp => {
                    let source = Self::scalar_to_f64(value)?;

                    let min_scalar: Scalar = min.into();
                    let max_scalar: Scalar = max.into();

                    let min_value = Self::scalar_to_f64(min_scalar)?;
                    let max_value = Self::scalar_to_f64(max_scalar)?;

                    if source < min_value {
                        Ok(min)
                    } else if source > max_value {
                        Ok(max)
                    } else {
                        unreachable!("conversion overflow without crossing target bounds");
                    }
                }
            },

            Err(e) => Err(e.into()),
        }
    }

    fn resolve_f32(&self, value: Scalar) -> Result<f32, TransformError> {
        match value.try_into() {
            Ok(value) => Ok(value),

            Err(ScalarConversionError::Overflow) => match self.overflow_mode {
                OverflowMode::Error => Err(ScalarConversionError::Overflow.into()),
                OverflowMode::Clamp => {
                    let value = Self::scalar_to_f64(value)?;

                    if value < f32::MIN as f64 {
                        Ok(f32::MIN)
                    } else {
                        Ok(f32::MAX)
                    }
                }
            },

            Err(e) => Err(e.into()),
        }
    }

    fn resolve_f64(&self, value: Scalar) -> Result<f64, TransformError> {
        Ok(value.try_into()?)
    }

    fn resolve_f16(&self, value: Scalar) -> Result<half::f16, TransformError> {
        match value.try_into() {
            Ok(value) => Ok(value),

            Err(ScalarConversionError::Overflow) => match self.overflow_mode {
                OverflowMode::Error => Err(ScalarConversionError::Overflow.into()),
                OverflowMode::Clamp => {
                    let value = Self::scalar_to_f64(value)?;

                    if value < half::f16::MIN.to_f64() {
                        Ok(half::f16::MIN)
                    } else {
                        Ok(half::f16::MAX)
                    }
                }
            },

            Err(e) => Err(e.into()),
        }
    }

    fn resolve_bf16(&self, value: Scalar) -> Result<half::bf16, TransformError> {
        match value.try_into() {
            Ok(value) => Ok(value),

            Err(ScalarConversionError::Overflow) => match self.overflow_mode {
                OverflowMode::Error => Err(ScalarConversionError::Overflow.into()),
                OverflowMode::Clamp => {
                    let value = Self::scalar_to_f64(value)?;

                    if value < half::bf16::MIN.to_f64() {
                        Ok(half::bf16::MIN)
                    } else {
                        Ok(half::bf16::MAX)
                    }
                }
            },

            Err(e) => Err(e.into()),
        }
    }
}

impl Transform for Clamp {
    type Error = TransformError;

    fn apply(&self, tensor: &mut TensorViewMut) -> Result<(), Self::Error> {
        macro_rules! match_max_min {
            ($max:expr, $min:expr, $t:expr) => {
                match ($min, $max) {
                    (Some(min), Some(max)) => {
                        $t.map_inplace(|x| {
                            *x = (*x).clamp(min, max);
                        });
                    }

                    (Some(min), None) => {
                        $t.map_inplace(|x| {
                            *x = (*x).max(min);
                        });
                    }

                    (None, Some(max)) => {
                        $t.map_inplace(|x| {
                            *x = (*x).min(max);
                        });
                    }

                    (None, None) => {}
                }
            };
        }

        match tensor {
            TensorViewMut::U8(t) => {
                let min = self
                    .min
                    .map(|x| self.resolve_int(x, u8::MIN, u8::MAX))
                    .transpose()?;

                let max = self
                    .max
                    .map(|x| self.resolve_int(x, u8::MIN, u8::MAX))
                    .transpose()?;

                match_max_min!(max, min, t);
            }

            TensorViewMut::I8(t) => {
                let min = self
                    .min
                    .map(|x| self.resolve_int(x, i8::MIN, i8::MAX))
                    .transpose()?;

                let max = self
                    .max
                    .map(|x| self.resolve_int(x, i8::MIN, i8::MAX))
                    .transpose()?;

                match_max_min!(max, min, t);
            }

            TensorViewMut::I32(t) => {
                let min = self
                    .min
                    .map(|x| self.resolve_int(x, i32::MIN, i32::MAX))
                    .transpose()?;

                let max = self
                    .max
                    .map(|x| self.resolve_int(x, i32::MIN, i32::MAX))
                    .transpose()?;

                match_max_min!(max, min, t);
            }

            TensorViewMut::I64(t) => {
                let min = self
                    .min
                    .map(|x| self.resolve_int(x, i64::MIN, i64::MAX))
                    .transpose()?;

                let max = self
                    .max
                    .map(|x| self.resolve_int(x, i64::MIN, i64::MAX))
                    .transpose()?;

                match_max_min!(max, min, t);
            }

            TensorViewMut::F32(t) => {
                let min = self.min.map(|x| self.resolve_f32(x)).transpose()?;

                let max = self.max.map(|x| self.resolve_f32(x)).transpose()?;

                match_max_min!(max, min, t);
            }

            TensorViewMut::F64(t) => {
                let min = self.min.map(|x| self.resolve_f64(x)).transpose()?;

                let max = self.max.map(|x| self.resolve_f64(x)).transpose()?;

                match_max_min!(max, min, t);
            }

            TensorViewMut::F16(t) => {
                let min = self.min.map(|x| self.resolve_f16(x)).transpose()?;

                let max = self.max.map(|x| self.resolve_f16(x)).transpose()?;

                match_max_min!(max, min, t);
            }

            TensorViewMut::BF16(t) => {
                let min = self.min.map(|x| self.resolve_bf16(x)).transpose()?;

                let max = self.max.map(|x| self.resolve_bf16(x)).transpose()?;

                match_max_min!(max, min, t);
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

    fn make_tensor_f32(data: &mut [f32]) -> TensorViewMut<'_> {
        let len = data.len();
        let raw_bytes = bytemuck::cast_slice_mut(data);

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), TensorDT::F32);

        layout.try_view_mut(raw_bytes).unwrap()
    }

    fn make_tensor_f64(data: &mut [f64]) -> TensorViewMut<'_> {
        let len = data.len();
        let raw_bytes = bytemuck::cast_slice_mut(data);

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), TensorDT::F64);

        layout.try_view_mut(raw_bytes).unwrap()
    }

    fn make_tensor_i8(data: &mut [i8]) -> TensorViewMut<'_> {
        let len = data.len();
        let raw_bytes = bytemuck::cast_slice_mut(data);

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), TensorDT::I8);

        layout.try_view_mut(raw_bytes).unwrap()
    }

    fn make_tensor_u8(data: &mut [u8]) -> TensorViewMut<'_> {
        let len = data.len();
        let raw_bytes = bytemuck::cast_slice_mut(data);

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), TensorDT::U8);

        layout.try_view_mut(raw_bytes).unwrap()
    }

    // ---------------------------------------------------------
    // Constructor
    // ---------------------------------------------------------

    #[test]
    fn constructor_accepts_valid_bounds() {
        assert!(Clamp::new(0.0, 10.0).is_ok());
        assert!(Clamp::new(Some(10.0), Some(10.0)).is_ok());
        assert!(Clamp::new(None::<f64>, Some(10.0)).is_ok());
        assert!(Clamp::new(Some(0.0), None::<f64>).is_ok());
        assert!(Clamp::new(None::<f64>, None::<f64>).is_ok());
    }

    #[test]
    fn constructor_rejects_min_greater_than_max() {
        let result = Clamp::new(Some(10.0), Some(0.0));

        assert!(matches!(result, Err(TransformError::InvalidValue)));
    }

    #[test]
    fn constructor_rejects_nan_min() {
        let result = Clamp::new(Some(f64::NAN), Some(10.0));

        assert!(matches!(result, Err(TransformError::InvalidValue)));
    }

    #[test]
    fn constructor_rejects_nan_max() {
        let result = Clamp::new(Some(0.0), Some(f64::NAN));

        assert!(matches!(result, Err(TransformError::InvalidValue)));
    }

    // ---------------------------------------------------------
    // F32
    // ---------------------------------------------------------

    #[test]
    fn clamp_f32_both_bounds() {
        let mut data = vec![-10.0, 0.0, 5.0, 10.0, 20.0];

        {
            let mut tensor = make_tensor_f32(&mut data);

            Clamp::new(Some(0.0), Some(10.0))
                .unwrap()
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, vec![0.0, 0.0, 5.0, 10.0, 10.0]);
    }

    #[test]
    fn clamp_f32_min_only() {
        let mut data = vec![-10.0, 0.0, 5.0];

        {
            let mut tensor = make_tensor_f32(&mut data);

            Clamp::new(Some(0.0), None::<f64>)
                .unwrap()
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, vec![0.0, 0.0, 5.0]);
    }

    #[test]
    fn clamp_f32_max_only() {
        let mut data = vec![0.0, 5.0, 10.0, 20.0];

        {
            let mut tensor = make_tensor_f32(&mut data);

            Clamp::new(None::<f64>, Some(10.0))
                .unwrap()
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, vec![0.0, 5.0, 10.0, 10.0]);
    }

    #[test]
    fn clamp_f32_values_inside_bounds_are_unchanged() {
        let mut data = vec![1.0, 2.5, 5.0, 9.99];

        let original = data.clone();

        {
            let mut tensor = make_tensor_f32(&mut data);

            Clamp::new(Some(0.0), Some(10.0))
                .unwrap()
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, original);
    }

    #[test]
    fn clamp_f32_equal_bounds() {
        let mut data = vec![-10.0, 0.0, 5.0, 10.0];

        {
            let mut tensor = make_tensor_f32(&mut data);

            Clamp::new(Some(5.0), Some(5.0))
                .unwrap()
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, vec![5.0, 5.0, 5.0, 5.0]);
    }

    // ---------------------------------------------------------
    // F64
    // ---------------------------------------------------------

    #[test]
    fn clamp_f64() {
        let mut data = vec![-100.0, -1.5, 0.0, 1.5, 100.0];

        {
            let mut tensor = make_tensor_f64(&mut data);

            Clamp::new(Some(-1.0), Some(1.0))
                .unwrap()
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, vec![-1.0, -1.0, 0.0, 1.0, 1.0]);
    }

    // ---------------------------------------------------------
    // Integer: Error rounding mode
    // ---------------------------------------------------------

    #[rstest]
    #[case(0.5, 10.0)]
    #[case(0.0, 10.5)]
    #[case(-0.5, 10.0)]
    #[case(-10.5, 0.0)]
    fn integer_fractional_bounds_rejected(#[case] min: f64, #[case] max: f64) {
        let mut data = vec![0i8, 5, 10];

        let mut tensor = make_tensor_i8(&mut data);

        let result = Clamp::new(Some(min), Some(max)).unwrap().apply(&mut tensor);

        assert!(matches!(
            result,
            Err(TransformError::ScalarConversion(
                ScalarConversionError::FractionalValue
            ))
        ));
    }

    // ---------------------------------------------------------
    // Integer: exact bounds
    // ---------------------------------------------------------

    #[test]
    fn clamp_i8() {
        let mut data = vec![-128, -100, 0, 100, 127];

        {
            let mut tensor = make_tensor_i8(&mut data);

            Clamp::new(Some(-50.0), Some(50.0))
                .unwrap()
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, vec![-50, -50, 0, 50, 50]);
    }

    #[test]
    fn clamp_u8() {
        let mut data = vec![0, 10, 100, 200, 255];

        {
            let mut tensor = make_tensor_u8(&mut data);

            Clamp::new(Some(50.0), Some(200.0))
                .unwrap()
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, vec![50, 50, 100, 200, 200]);
    }

    // ---------------------------------------------------------
    // Integer rounding
    // ---------------------------------------------------------

    #[rstest]
    #[case(IntRoundingMode::Floor, vec![0, 1, 5, 5])]
    #[case(IntRoundingMode::Ceil,  vec![1, 1, 5, 6])]
    #[case(IntRoundingMode::Round, vec![1, 1, 5, 6])]
    fn integer_rounding_modes(#[case] mode: IntRoundingMode, #[case] expected: Vec<i8>) {
        let mut data = vec![0i8, 1, 5, 10];

        {
            let mut tensor = make_tensor_i8(&mut data);

            Clamp::new(Some(0.5), Some(5.5))
                .unwrap()
                .int_rounding_mode(mode)
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, expected);
    }

    // ---------------------------------------------------------
    // Integer overflow
    // ---------------------------------------------------------

    #[test]
    fn integer_bound_overflow_returns_error() {
        let mut data = vec![0i8, 10, 100];

        {
            let mut tensor = make_tensor_i8(&mut data);

            let result = Clamp::new(Some(-200.0), Some(200.0))
                .unwrap()
                .apply(&mut tensor);

            assert!(matches!(
                result,
                Err(TransformError::ScalarConversion(
                    ScalarConversionError::Overflow
                ))
            ));
        }

        assert_eq!(data, vec![0, 10, 100]);
    }

    #[test]
    fn integer_bound_overflow_can_be_clamped() {
        let mut data = vec![-128, -10, 0, 10, 127];

        {
            let mut tensor = make_tensor_i8(&mut data);

            Clamp::new(Some(-200.0), Some(200.0))
                .unwrap()
                .overflow_mode(OverflowMode::Clamp)
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, vec![-128, -10, 0, 10, 127]);
    }

    #[test]
    fn integer_overflow_clamps_only_outside_side() {
        let mut data = vec![-128, -100, 0, 100, 127];

        {
            let mut tensor = make_tensor_i8(&mut data);

            Clamp::new(Some(-200.0), Some(100.0))
                .unwrap()
                .overflow_mode(OverflowMode::Clamp)
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, vec![-128, -100, 0, 100, 100]);
    }

    // ---------------------------------------------------------
    // Modification / error atomicity
    // ---------------------------------------------------------

    #[test]
    fn clamp_modifies_only_values_outside_bounds() {
        let mut data = vec![-10.0, 1.0, 5.0, 9.0, 20.0];

        {
            let mut tensor = make_tensor_f32(&mut data);

            Clamp::new(Some(0.0), Some(10.0))
                .unwrap()
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, vec![0.0, 1.0, 5.0, 9.0, 10.0]);
    }

    #[test]
    fn failed_clamp_does_not_modify_tensor() {
        let mut data = vec![-10i8, 0, 10, 100];

        let original = data.clone();

        {
            let mut tensor = make_tensor_i8(&mut data);

            let result = Clamp::new(Some(-200.0), Some(200.0))
                .unwrap()
                .apply(&mut tensor);

            assert!(matches!(
                result,
                Err(TransformError::ScalarConversion(
                    ScalarConversionError::Overflow
                ))
            ));
        }

        assert_eq!(data, original);
    }

    // ---------------------------------------------------------
    // No-op
    // ---------------------------------------------------------

    #[test]
    fn clamp_without_bounds_is_noop() {
        let mut data = vec![-100.0f32, 0.0, 100.0];

        let original = data.clone();

        {
            let mut tensor = make_tensor_f32(&mut data);

            Clamp::new(None::<f64>, None::<f64>)
                .unwrap()
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, original);
    }

    #[test]
    fn clamp_f16_bound_overflow() {
        let mut data = vec![
            half::f16::from_f32(-1.0),
            half::f16::from_f32(0.0),
            half::f16::from_f32(1.0),
        ];

        {
            let l = data.len();
            let raw_bytes = bytemuck::cast_slice_mut(&mut data);

            let layout = TensorBatchLayout::new(vec![l].into(), vec![1].into(), TensorDT::F16);

            let mut tensor = layout.try_view_mut(raw_bytes).unwrap();

            Clamp::new(Some(-1e10), Some(1e10))
                .unwrap()
                .overflow_mode(OverflowMode::Clamp)
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(
            data,
            vec![
                half::f16::from_f32(-1.0),
                half::f16::from_f32(0.0),
                half::f16::from_f32(1.0),
            ]
        );
    }
}
