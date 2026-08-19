use super::{Transform, TransformError};
use crate::core::dataset::item::TensorViewMut;

pub enum OverflowMode {
    Error,
    Wrapping,
}

pub struct Add {
    value: f64,
    overflow: OverflowMode,
}

impl Add {
    pub fn new(value: f64) -> Self {
        Self {
            value,
            overflow: OverflowMode::Error,
        }
    }

    pub fn arith_overflow(self, overflow: OverflowMode) -> Self {
        Self { overflow, ..self }
    }
}

fn is_float_int(val: f64) -> bool {
    val.fract() == 0.0
}

impl Transform for Add {
    type Error = TransformError;

    fn apply(&self, tensor: &mut TensorViewMut) -> Result<(), Self::Error> {
        macro_rules! int_add {
            ($t:ty, $value:expr, $tensor:expr, $overflow:expr) => {{
                if !$value.is_finite() {
                    return Err(TransformError::InvalidValue);
                }

                if !is_float_int($value) {
                    return Err(TransformError::InvalidValue);
                }
                if $value < <$t>::MIN as f64 || $value > <$t>::MAX as f64 {
                    println!("It is cast overflow");
                    return Err(TransformError::Overflow);
                }

                let v = $value as $t;
                if matches!($overflow, OverflowMode::Wrapping) {
                    $tensor.map_inplace(|x| *x = x.wrapping_add(v));
                } else {
                    for x in $tensor.iter() {
                        println!("{:?}, {}, {}", x.checked_add(v), x, v);
                        x.checked_add(v).ok_or(TransformError::Overflow)?;
                    }

                    $tensor.map_inplace(|x| *x = unsafe { x.unchecked_add(v) });
                }
                Ok(())
            }};
        }
        if self.value == 0.0 {
            return Ok(());
        }
        match tensor {
            TensorViewMut::BF16(t) => {
                t.map_inplace(|x| *x += half::bf16::from_f64(self.value));
            }
            TensorViewMut::F32(t) => {
                t.map_inplace(|x| *x += self.value as f32);
            }
            TensorViewMut::F64(t) => {
                t.map_inplace(|x| *x += self.value);
            }
            TensorViewMut::U8(tens) => return int_add!(u8, self.value, tens, self.overflow),
            TensorViewMut::F16(t) => {
                t.map_inplace(|x| *x += half::f16::from_f64(self.value));
            }
            TensorViewMut::I8(tens) => return int_add!(i8, self.value, tens, self.overflow),
            TensorViewMut::I32(tens) => {
                return int_add!(i32, self.value, tens, self.overflow);
            }
            TensorViewMut::I64(tens) => {
                return int_add!(i64, self.value, tens, self.overflow);
            }
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

            assert!(matches!(result, Err(TransformError::InvalidValue)));
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

            assert!(matches!(result, Err(TransformError::InvalidValue)));
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

            let result = Add::new(256.0).apply(&mut tensor);

            assert!(matches!(result, Err(TransformError::Overflow)));
        }

        assert_eq!(data, original);
    }

    #[test]
    fn add_negative_value_outside_u8_range() {
        let mut data = vec![1u8, 2, 3];
        let original = data.clone();

        {
            let mut tensor = make_tensor_u8(&mut data);

            let result = Add::new(-1.0).apply(&mut tensor);

            assert!(matches!(result, Err(TransformError::Overflow)));
        }

        assert_eq!(data, original);
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
}
