use super::{Scalar, Transform, TransformError};
use crate::{core::dataset::item::TensorViewMut, transform::ScalarConversionError};

pub enum OverflowMode {
    Error,
    Wrapping,
}

pub struct Add {
    value: Scalar,
    overflow: OverflowMode,
}

impl Add {
    pub fn new<T: Into<Scalar>>(value: T) -> Self {
        Self {
            value: value.into(),
            overflow: OverflowMode::Error,
        }
    }

    pub fn arith_overflow(self, overflow: OverflowMode) -> Self {
        Self { overflow, ..self }
    }
}

impl Transform for Add {
    fn apply(&self, tensor: &mut TensorViewMut) -> Result<(), TransformError> {
        if let Ok(u) = <Scalar as TryInto<u8>>::try_into(self.value)
            && u == 0
        {
            return Ok(());
        }

        macro_rules! add_int {
            ($ty:ty, $t:expr) => {{
                let h: $ty = self.value.try_into()?;

                if matches!(self.overflow, OverflowMode::Error) {
                    for x in $t.iter() {
                        x.checked_add(h).ok_or(TransformError::Overflow)?;
                    }
                    $t.map_inplace(|x| *x = unsafe { x.unchecked_add(h) });
                } else {
                    $t.map_inplace(|x| *x = x.wrapping_add(h));
                }
            }};
        }

        macro_rules! add {
            ($ty:ty, $t:expr) => {{
                let h: $ty = self.value.try_into()?;
                $t.map_inplace(|x| *x += h);
            }};
        }

        match tensor {
            TensorViewMut::BF16(t) => add!(half::bf16, t),
            TensorViewMut::F16(t) => add!(half::f16, t),
            TensorViewMut::U8(t) => {
                let val: i32 = self.value.try_into()?;
                if val < -(u8::MAX as i32) || val > (u8::MAX as i32) {
                    return Err(ScalarConversionError::Overflow.into());
                }
                if val < 0 {
                    let v = (-val) as u8;
                    match self.overflow {
                        OverflowMode::Error => {
                            for x in t.iter() {
                                x.checked_sub(v).ok_or(TransformError::Overflow)?;
                            }
                            t.map_inplace(|x| *x = unsafe { x.unchecked_sub(v) });
                        }
                        OverflowMode::Wrapping => {
                            t.map_inplace(|x| *x = x.wrapping_sub(v));
                        }
                    }
                } else {
                    let v = val as u8;
                    match self.overflow {
                        OverflowMode::Error => {
                            for x in t.iter() {
                                x.checked_add(v).ok_or(TransformError::Overflow)?;
                            }
                            t.map_inplace(|x| *x = unsafe { x.unchecked_add(v) });
                        }
                        OverflowMode::Wrapping => {
                            t.map_inplace(|x| *x = x.wrapping_add(v));
                        }
                    }
                }
            }
            TensorViewMut::I8(t) => add_int!(i8, t),
            TensorViewMut::I32(t) => add_int!(i32, t),
            TensorViewMut::I64(t) => add_int!(i64, t),
            TensorViewMut::F32(t) => add!(f32, t),
            TensorViewMut::F64(t) => add!(f64, t),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;
    use crate::core::dataset::item::{TensorBatchLayout, TensorDT, TensorViewMut};
    use rstest::rstest;

    fn make_tensor_f32(data: &mut [f32]) -> TensorViewMut<'_> {
        let l = data.len();
        let raw_bytes = bytemuck::cast_slice_mut(data);

        let layout = TensorBatchLayout::new(vec![l].into(), vec![1].into(), TensorDT::F32);

        layout.try_view_mut(raw_bytes).unwrap()
    }

    fn make_tensor_i8(data: &mut [i8]) -> TensorViewMut<'_> {
        let l = data.len();
        let raw_bytes = bytemuck::cast_slice_mut(data);

        let layout = TensorBatchLayout::new(vec![l].into(), vec![1].into(), TensorDT::I8);

        layout.try_view_mut(raw_bytes).unwrap()
    }

    fn make_tensor_i32(data: &mut [i32]) -> TensorViewMut<'_> {
        let l = data.len();
        let raw_bytes = bytemuck::cast_slice_mut(data);

        let layout = TensorBatchLayout::new(vec![l].into(), vec![1].into(), TensorDT::I32);

        layout.try_view_mut(raw_bytes).unwrap()
    }

    fn make_tensor_u8(data: &mut [u8]) -> TensorViewMut<'_> {
        let l = data.len();
        let raw_bytes = bytemuck::cast_slice_mut(data);

        let layout = TensorBatchLayout::new(vec![l].into(), vec![1].into(), TensorDT::U8);

        layout.try_view_mut(raw_bytes).unwrap()
    }

    // ------------------------------------------------------------
    // Basic
    // ------------------------------------------------------------

    #[test]
    fn add_f32() {
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0];

        {
            let mut tensor = make_tensor_f32(&mut data);

            Add::new(2.0).apply(&mut tensor).unwrap();
        }

        assert_eq!(data, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn add_f32_negative() {
        let mut data = vec![10.0f32, 20.0, 30.0];

        {
            let mut tensor = make_tensor_f32(&mut data);

            Add::new(-5.0).apply(&mut tensor).unwrap();
        }

        assert_eq!(data, vec![5.0, 15.0, 25.0]);
    }

    #[test]
    fn add_zero_does_not_modify_tensor() {
        let mut data = vec![1.0f32, 2.0, 3.0];
        let original = data.clone();

        {
            let mut tensor = make_tensor_f32(&mut data);

            Add::new(0.0).apply(&mut tensor).unwrap();
        }

        assert_eq!(data, original);
    }

    // ------------------------------------------------------------
    // Floating point dtypes
    // ------------------------------------------------------------

    #[test]
    fn add_f64() {
        let mut data = vec![1.0f64, 2.0, 3.0];

        let raw_bytes = bytemuck::cast_slice_mut(&mut data);

        let layout = TensorBatchLayout::new(vec![3].into(), vec![1].into(), TensorDT::F64);

        let mut tensor = layout.try_view_mut(raw_bytes).unwrap();

        Add::new(0.5).apply(&mut tensor).unwrap();

        assert_eq!(data, vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn add_f16() {
        let mut data = vec![
            half::f16::from_f32(1.0),
            half::f16::from_f32(2.0),
            half::f16::from_f32(3.0),
        ];

        let raw_bytes = bytemuck::cast_slice_mut(&mut data);

        let layout = TensorBatchLayout::new(vec![3].into(), vec![1].into(), TensorDT::F16);

        let mut tensor = layout.try_view_mut(raw_bytes).unwrap();

        Add::new(0.5).apply(&mut tensor).unwrap();

        let actual: Vec<f32> = data.iter().map(|x| x.to_f32()).collect();

        assert_eq!(actual, vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn add_bf16() {
        let mut data = vec![
            half::bf16::from_f32(1.0),
            half::bf16::from_f32(2.0),
            half::bf16::from_f32(3.0),
        ];

        let raw_bytes = bytemuck::cast_slice_mut(&mut data);

        let layout = TensorBatchLayout::new(vec![3].into(), vec![1].into(), TensorDT::BF16);

        let mut tensor = layout.try_view_mut(raw_bytes).unwrap();

        Add::new(0.5).apply(&mut tensor).unwrap();

        let actual: Vec<f32> = data.iter().map(|x| x.to_f32()).collect();

        assert_eq!(actual, vec![1.5, 2.5, 3.5]);
    }

    // ------------------------------------------------------------
    // Integer dtypes
    // ------------------------------------------------------------

    #[test]
    fn add_i8() {
        let mut data = vec![-10i8, 0, 10];

        {
            let mut tensor = make_tensor_i8(&mut data);

            Add::new(5.0).apply(&mut tensor).unwrap();
        }

        assert_eq!(data, vec![-5, 5, 15]);
    }

    #[test]
    fn add_i32() {
        let mut data = vec![-100i32, 0, 100];

        {
            let mut tensor = make_tensor_i32(&mut data);

            Add::new(50.0).apply(&mut tensor).unwrap();
        }

        assert_eq!(data, vec![-50, 50, 150]);
    }

    #[test]
    fn add_u8() {
        let mut data = vec![0u8, 10, 100];

        {
            let mut tensor = make_tensor_u8(&mut data);

            Add::new(20.0).apply(&mut tensor).unwrap();
        }

        assert_eq!(data, vec![20, 30, 120]);
    }

    // ------------------------------------------------------------
    // Invalid values
    // ------------------------------------------------------------

    #[rstest]
    #[case(f64::NAN)]
    #[case(f64::INFINITY)]
    #[case(f64::NEG_INFINITY)]
    fn integer_add_rejects_non_finite(#[case] value: f64) {
        let mut data = vec![1i32, 2, 3];

        {
            let mut tensor = make_tensor_i32(&mut data);

            let result = Add::new(value).apply(&mut tensor);

            assert!(matches!(
                result,
                Err(TransformError::ScalarConversion(
                    ScalarConversionError::InvalidValue
                ))
            ));
        }

        assert_eq!(data, vec![1, 2, 3]);
    }

    #[test]
    fn integer_add_rejects_fractional_value() {
        let mut data = vec![1i32, 2, 3];
        let original = data.clone();

        {
            let mut tensor = make_tensor_i32(&mut data);

            let result = Add::new(1.5).apply(&mut tensor);

            assert!(matches!(
                result,
                Err(TransformError::ScalarConversion(
                    ScalarConversionError::FractionalValue
                ))
            ));
        }

        assert_eq!(data, original);
    }

    // ------------------------------------------------------------
    // Value outside dtype range
    // ------------------------------------------------------------

    #[test]
    fn add_value_outside_dtype_range() {
        let mut data = vec![1u8, 2, 3];
        let original = data.clone();

        {
            let mut tensor = make_tensor_u8(&mut data);

            let result = Add::new(256).apply(&mut tensor);

            assert!(matches!(
                result,
                Err(TransformError::ScalarConversion(
                    ScalarConversionError::Overflow
                ))
            ));
        }

        assert_eq!(data, original);
    }

    #[test]
    fn add_negative_value_u8() {
        let mut data = vec![1u8, 2, 3];
        let exp: Vec<u8> = data.iter().map(|&x| x - 1).collect();
        {
            let mut tensor = make_tensor_u8(&mut data);

            let result = Add::new(-1).apply(&mut tensor);
            assert!(result.is_ok());
        }

        assert_eq!(data, exp);
    }

    // ------------------------------------------------------------
    // Arithmetic overflow
    // ------------------------------------------------------------

    #[test]
    fn add_positive_arithmetic_overflow() {
        let mut data = vec![100i8, 120, 127];
        let original = data.clone();

        {
            let mut tensor = make_tensor_i8(&mut data);

            let result = Add::new(10.0).apply(&mut tensor);

            assert!(matches!(result, Err(TransformError::Overflow)));
        }

        // Important: no partial modification.
        assert_eq!(data, original);
    }

    #[test]
    fn add_negative_arithmetic_overflow() {
        let mut data = vec![-100i8, -120, -128];
        let original = data.clone();

        {
            let mut tensor = make_tensor_i8(&mut data);

            let result = Add::new(-10.0).apply(&mut tensor);

            assert!(matches!(result, Err(TransformError::Overflow)));
        }

        assert_eq!(data, original);
    }

    // ------------------------------------------------------------
    // Wrapping
    // ------------------------------------------------------------

    #[test]
    fn add_wrapping_positive_overflow() {
        let mut data = vec![127i8, 126, 100];

        {
            let mut tensor = make_tensor_i8(&mut data);

            Add::new(1.0)
                .arith_overflow(OverflowMode::Wrapping)
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, vec![-128, 127, 101]);
    }

    #[test]
    fn add_wrapping_negative_overflow() {
        let mut data = vec![-128i8, -127, -100];

        {
            let mut tensor = make_tensor_i8(&mut data);

            Add::new(-1.0)
                .arith_overflow(OverflowMode::Wrapping)
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, vec![127, -128, -101]);
    }

    // ------------------------------------------------------------
    // All-or-nothing behavior
    // ------------------------------------------------------------

    #[test]
    fn overflow_does_not_partially_modify_tensor() {
        let mut data = vec![1i8, 2, 127, 4, 5];
        let original = data.clone();

        {
            let mut tensor = make_tensor_i8(&mut data);

            let result = Add::new(1.0).apply(&mut tensor);

            assert!(matches!(result, Err(TransformError::Overflow)));
        }

        assert_eq!(data, original);
    }

    // ------------------------------------------------------------
    // Zero-copy
    // ------------------------------------------------------------

    #[test]
    fn add_modifies_original_buffer() {
        let mut data = vec![10i32, 20, 30];

        {
            let mut tensor = make_tensor_i32(&mut data);

            Add::new(5.0).apply(&mut tensor).unwrap();
        }

        assert_eq!(data, vec![15, 25, 35]);
    }

    #[test]
    fn add_exact_positive_boundary() {
        let mut data = vec![120i8];

        {
            let mut tensor = make_tensor_i8(&mut data);

            Add::new(7.0).apply(&mut tensor).unwrap();
        }

        assert_eq!(data[0], 127);
    }

    #[test]
    fn add_exact_negative_boundary() {
        let mut data = vec![-120i8];

        {
            let mut tensor = make_tensor_i8(&mut data);

            Add::new(-7.0).apply(&mut tensor).unwrap();
        }

        assert_eq!(data, vec![-127]);
    }

    #[test]
    fn test_atomic_overflow() {
        let mut data = vec![1i8, 2, 127, 4];
        let original = data.clone();

        let mut tensor = make_tensor_i8(&mut data);

        let result = Add::new(1i8).apply(&mut tensor);

        assert!(matches!(result, Err(TransformError::Overflow)));
        assert_eq!(data, original);
    }
}
