use crate::transform::{Scalar, scalar::IntoScalarOption};

use super::{TensorViewMut, Transform, TransformError, helpers::is_float_int};

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
    min: Option<f64>,
    max: Option<f64>,
    int_rounding_mode: IntRoundingMode,
    overflow_mode: OverflowMode,
}

impl Clamp {
    pub fn new<T: Into<Option<f64>>>(min: T, max: T) -> Result<Self, TransformError> {
        let min = min.into();
        let max = max.into();

        if min.is_some_and(|x| x.is_nan()) || max.is_some_and(|x| x.is_nan()) {
            return Err(TransformError::InvalidValue);
        }
        if let (Some(mx), Some(mn)) = (max, min) {
            if mx < mn {
                return Err(TransformError::InvalidValue);
            }
        }
        Ok(Self {
            min: min,
            max: max,
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

    fn check_int(&self) -> Result<(), TransformError> {
        match self.int_rounding_mode {
            IntRoundingMode::Error => {
                if let Some(max) = self.max {
                    if !is_float_int(max) {
                        return Err(TransformError::InvalidValue);
                    }
                }
                if let Some(min) = self.min {
                    if !is_float_int(min) {
                        return Err(TransformError::InvalidValue);
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

impl Transform for Clamp {
    type Error = TransformError;

    fn apply(&self, tensor: &mut TensorViewMut) -> Result<(), Self::Error> {
        macro_rules! halfed_check_overflow {
            ($t:ty) => {{
                let new_max = if let Some(max) = self.max {
                    if max > <$t>::MAX.to_f64() {
                        match self.overflow_mode {
                            OverflowMode::Error => {
                                return Err(TransformError::InvalidValue);
                            }
                            OverflowMode::Clamp => Some(<$t>::MAX.to_f64()),
                        }
                    } else if max < <$t>::MIN.to_f64() {
                        match self.overflow_mode {
                            OverflowMode::Error => {
                                return Err(TransformError::InvalidValue);
                            }
                            OverflowMode::Clamp => Some(<$t>::MIN.to_f64()),
                        }
                    } else {
                        self.max
                    }
                } else {
                    self.max
                };
                let new_min = if let Some(min) = self.min {
                    if min > <$t>::MAX.to_f64() {
                        match self.overflow_mode {
                            OverflowMode::Error => {
                                return Err(TransformError::InvalidValue);
                            }
                            OverflowMode::Clamp => Some(<$t>::MAX.to_f64()),
                        }
                    } else if min < <$t>::MIN.to_f64() {
                        match self.overflow_mode {
                            OverflowMode::Error => {
                                return Err(TransformError::InvalidValue);
                            }
                            OverflowMode::Clamp => Some(<$t>::MIN.to_f64()),
                        }
                    } else {
                        self.min
                    }
                } else {
                    self.min
                };
                (new_min, new_max)
            }};
        }

        macro_rules! halfed_clamp {
            ($tensor:expr, $t:ty) => {{
                let (mn, mx) = halfed_check_overflow!($t);
                $tensor.map_inplace(|x| {
                    let x64 = x.to_f64();
                    if let Some(min) = mn {
                        if x64 < min {
                            *x = <$t>::from_f64(min);
                        }
                    }
                    if let Some(max) = mx {
                        if x64 > max {
                            *x = <$t>::from_f64(max);
                        }
                    }
                })
            }};
        }

        macro_rules! check_overflow {
            ($t:ty) => {{
                let new_max = if let Some(max) = self.max {
                    if max > <$t>::MAX as f64 {
                        match self.overflow_mode {
                            OverflowMode::Error => {
                                return Err(TransformError::InvalidValue);
                            }
                            OverflowMode::Clamp => Some(<$t>::MAX as f64),
                        }
                    } else if max < <$t>::MIN as f64 {
                        match self.overflow_mode {
                            OverflowMode::Error => {
                                return Err(TransformError::InvalidValue);
                            }
                            OverflowMode::Clamp => Some(<$t>::MIN as f64),
                        }
                    } else {
                        self.max
                    }
                } else {
                    self.max
                };
                let new_min = if let Some(min) = self.min {
                    if min > <$t>::MAX as f64 {
                        match self.overflow_mode {
                            OverflowMode::Error => {
                                return Err(TransformError::InvalidValue);
                            }
                            OverflowMode::Clamp => Some(<$t>::MAX as f64),
                        }
                    } else if min < <$t>::MIN as f64 {
                        match self.overflow_mode {
                            OverflowMode::Error => {
                                return Err(TransformError::InvalidValue);
                            }
                            OverflowMode::Clamp => Some(<$t>::MIN as f64),
                        }
                    } else {
                        self.min
                    }
                } else {
                    self.min
                };
                (new_min, new_max)
            }};
        }

        macro_rules! int_clamp {
            ($tensor:expr, $t:ty) => {{
                Self::check_int(&self)?;
                let (mn, mx) = check_overflow!($t);
                $tensor.map_inplace(|x| {
                    let xto = (*x).clone() as f64;
                    if let Some(max) = mx {
                        if xto > max {
                            if is_float_int(max) {
                                *x = max as $t;
                            } else {
                                let v = match self.int_rounding_mode {
                                    IntRoundingMode::Round => max.round(),
                                    IntRoundingMode::Ceil => max.ceil(),
                                    IntRoundingMode::Floor => max.floor(),
                                    _ => {
                                        unreachable!();
                                    }
                                };
                                *x = v as $t;
                            }
                        }
                    }
                    if let Some(min) = mn {
                        if xto < min {
                            if is_float_int(min) {
                                *x = min as $t;
                            } else {
                                let val = match self.int_rounding_mode {
                                    IntRoundingMode::Round => min.round(),
                                    IntRoundingMode::Ceil => min.ceil(),
                                    IntRoundingMode::Floor => min.floor(),
                                    _ => {
                                        unreachable!();
                                    }
                                };
                                *x = val as $t;
                            }
                        }
                    }
                });
                return Ok(());
            }};
        }

        match tensor {
            TensorViewMut::BF16(t) => halfed_clamp!(t, half::bf16),
            TensorViewMut::F16(t) => halfed_clamp!(t, half::f16),
            TensorViewMut::F32(t) => {
                let (mn, mx) = check_overflow!(f32);
                t.map_inplace(|x| {
                    let x64 = *x as f64;
                    if let Some(min) = mn {
                        if x64 < min {
                            *x = min as f32;
                        }
                    }
                    if let Some(max) = mx {
                        if x64 > max {
                            *x = max as f32;
                        }
                    }
                });
            }
            TensorViewMut::F64(t) => {
                t.map_inplace(|x| {
                    let x64 = *x;
                    if let Some(min) = self.min {
                        if x64 < min {
                            *x = min;
                        }
                    }
                    if let Some(max) = self.max {
                        if x64 > max {
                            *x = max;
                        }
                    }
                });
            }
            TensorViewMut::U8(t) => {
                int_clamp!(t, u8)
            }
            TensorViewMut::I8(t) => {
                int_clamp!(t, i8)
            }
            TensorViewMut::I32(t) => {
                int_clamp!(t, i32)
            }
            TensorViewMut::I64(t) => {
                int_clamp!(t, i64)
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

        assert!(matches!(result, Err(TransformError::InvalidValue)));
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

            assert!(matches!(result, Err(TransformError::InvalidValue)));
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

            assert!(matches!(result, Err(TransformError::InvalidValue)));
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
