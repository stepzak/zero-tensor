use super::{ScalarConversionError, helpers::is_float_int};
pub mod cmp;
pub mod ops;

#[derive(Clone, Copy, Debug)]
pub enum Scalar {
    U8(u8),
    I8(i8),
    I32(i32),
    I64(i64),
    BF16(half::bf16),
    F16(half::f16),
    F32(f32),
    F64(f64),
}

pub trait IntoScalarOption {
    fn into_scalar_option(self) -> Option<Scalar>;
}

macro_rules! int_to_int {
    ($value:expr, $target:ty) => {{
        if $value as i128 > <$target>::MAX as i128 || ($value as i128) < <$target>::MIN as i128 {
            Err(ScalarConversionError::Overflow)
        } else {
            Ok($value as $target)
        }
    }};
}

macro_rules! float_to_int {
    ($f:expr, $tyfrom:ty) => {{
        let f_64 = $f as f64;
        if !f_64.is_finite() {
            Err(ScalarConversionError::InvalidValue)
        } else if !is_float_int(f_64) {
            Err(ScalarConversionError::FractionalValue)
        } else if f_64 > <$tyfrom>::MAX as f64 || f_64 < <$tyfrom>::MIN as f64 {
            Err(ScalarConversionError::Overflow)
        } else {
            Ok(f_64 as $tyfrom)
        }
    }};
}

macro_rules! impl_int_try_from {
    ($ty:ty) => {
        impl TryFrom<Scalar> for $ty {
            type Error = ScalarConversionError;

            fn try_from(value: Scalar) -> Result<$ty, Self::Error> {
                match value {
                    Scalar::U8(i) => int_to_int!(i, $ty),
                    Scalar::I8(i) => int_to_int!(i, $ty),
                    Scalar::I32(i) => int_to_int!(i, $ty),
                    Scalar::I64(i) => int_to_int!(i, $ty),
                    Scalar::BF16(h) => float_to_int!(h.to_f64(), $ty),
                    Scalar::F16(h) => float_to_int!(h.to_f64(), $ty),
                    Scalar::F32(f) => float_to_int!(f, $ty),
                    Scalar::F64(f) => float_to_int!(f, $ty),
                }
            }
        }
    };
}

macro_rules! impl_from {
    ($ty:ty, $variant:ident) => {
        impl From<$ty> for Scalar {
            fn from(value: $ty) -> Self {
                Self::$variant(value)
            }
        }

        impl IntoScalarOption for $ty {
            fn into_scalar_option(self) -> Option<Scalar> {
                Some(Scalar::$variant(self))
            }
        }

        impl IntoScalarOption for Option<$ty> {
            fn into_scalar_option(self) -> Option<Scalar> {
                self.map(Scalar::$variant)
            }
        }
    };
}

macro_rules! float_to_half {
    ($f:expr, $tyfrom:ty) => {{
        let i = $f as f64;
        if i.is_nan() || i.is_infinite() {
            Ok(<$tyfrom>::from_f64(i))
        } else if i < <$tyfrom>::MIN.to_f64() || i > <$tyfrom>::MAX.to_f64() {
            Err(ScalarConversionError::Overflow)
        } else {
            Ok(<$tyfrom>::from_f64(i))
        }
    }};
}

macro_rules! impl_half_try_from {
    ($ty:ty) => {
        impl TryFrom<Scalar> for $ty {
            type Error = ScalarConversionError;

            fn try_from(value: Scalar) -> Result<$ty, Self::Error> {
                match value {
                    Scalar::U8(i) => float_to_half!(i, $ty),
                    Scalar::I8(i) => float_to_half!(i, $ty),
                    Scalar::I32(i) => float_to_half!(i, $ty),
                    Scalar::I64(i) => float_to_half!(i, $ty),
                    Scalar::BF16(h) => Ok(<$ty>::from_f32(h.to_f32())),
                    Scalar::F16(h) => Ok(<$ty>::from_f32(h.to_f32())),
                    Scalar::F32(f) => float_to_half!(f, $ty),
                    Scalar::F64(f) => float_to_half!(f, $ty),
                }
            }
        }
    };
}

macro_rules! int_to_float {
    ($i:expr, $tyfrom:ty) => {
        if $i as i128 > <$tyfrom>::MAX as i128 || ($i as i128) < <$tyfrom>::MIN as i128 {
            Err(ScalarConversionError::Overflow)
        } else {
            Ok($i as $tyfrom)
        }
    };
}

macro_rules! impl_float_try_from {
    ($ty:ty) => {
        impl TryFrom<Scalar> for $ty {
            type Error = ScalarConversionError;

            fn try_from(value: Scalar) -> Result<$ty, Self::Error> {
                match value {
                    Scalar::U8(i) => int_to_float!(i, $ty),
                    Scalar::I8(i) => int_to_float!(i, $ty),
                    Scalar::I32(i) => int_to_float!(i, $ty),
                    Scalar::I64(i) => int_to_float!(i, $ty),
                    Scalar::BF16(h) => Ok(h.to_f32() as $ty),
                    Scalar::F16(h) => Ok(h.to_f32() as $ty),
                    Scalar::F32(f) => Ok(f as $ty),
                    Scalar::F64(f) => Ok(f as $ty),
                }
            }
        }
    };
}

impl_from! {u8, U8}
impl_from! {i8, I8}
impl_from! {i32, I32}
impl_from! {i64, I64}
impl_from! {half::bf16, BF16}
impl_from! {half::f16, F16}
impl_from! {f32, F32}
impl_from! {f64, F64}

impl_int_try_from! {u8}
impl_int_try_from! {i8}
impl_int_try_from! {i32}
impl_int_try_from! {i64}
impl_half_try_from! {half::bf16}
impl_half_try_from! {half::f16}
impl_float_try_from! {f32}
impl_float_try_from! {f64}

macro_rules! float_fn {
    ($fn_name:ident, $ret:ty, $fb:expr) => {
        pub fn $fn_name(self) -> $ret {
            match self {
                Scalar::BF16(f) => f.$fn_name(),
                Scalar::F16(f) => f.$fn_name(),
                Scalar::F32(f) => f.$fn_name(),
                Scalar::F64(f) => f.$fn_name(),
                _ => $fb,
            }
        }
    };
}

impl Scalar {
    pub fn to_f32_lossy(self) -> f32 {
        match self {
            Scalar::U8(v) => v as f32,
            Scalar::I8(v) => v as f32,
            Scalar::I32(v) => v as f32,
            Scalar::I64(v) => v as f32,
            Scalar::F32(v) => v,
            Scalar::F64(v) => v as f32,
            Scalar::BF16(v) => v.to_f32(),
            Scalar::F16(v) => v.to_f32(),
        }
    }

    pub fn to_f64_lossy(self) -> f64 {
        match self {
            Scalar::U8(v) => v as f64,
            Scalar::I8(v) => v as f64,
            Scalar::I32(v) => v as f64,
            Scalar::I64(v) => v as f64,
            Scalar::F32(v) => v as f64,
            Scalar::F64(v) => v,
            Scalar::BF16(v) => v.to_f64(),
            Scalar::F16(v) => v.to_f64(),
        }
    }

    float_fn! {is_nan, bool, false}
    float_fn! {is_finite, bool, true}
    float_fn! {is_infinite, bool, false}
}

#[cfg(test)]
mod tests;
