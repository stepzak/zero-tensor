use std::assert_matches;

use super::*;
use rstest::rstest;

// ============================================================
// From<T> for Scalar
// ============================================================

#[test]
fn from_integer_types() {
    assert!(matches!(Scalar::from(42u8), Scalar::U8(42)));
    assert!(matches!(Scalar::from(-42i8), Scalar::I8(-42)));
    assert!(matches!(Scalar::from(-42i32), Scalar::I32(-42)));
    assert!(matches!(Scalar::from(-42i64), Scalar::I64(-42)));
}

#[test]
fn from_float_types() {
    assert!(matches!(Scalar::from(1.5f32), Scalar::F32(v) if v == 1.5));
    assert!(matches!(Scalar::from(1.5f64), Scalar::F64(v) if v == 1.5));
}

#[test]
fn from_half_types() {
    let bf16 = half::bf16::from_f32(1.5);
    let f16 = half::f16::from_f32(1.5);

    assert!(matches!(Scalar::from(bf16), Scalar::BF16(v) if v == bf16));
    assert!(matches!(Scalar::from(f16), Scalar::F16(v) if v == f16));
}

// ============================================================
// Integer -> Integer
// ============================================================

#[rstest]
#[case(0i64, 0i32)]
#[case(127i64, 127i32)]
#[case(-128i64, -128i32)]
#[case(0i64, 0i32)]
#[case(i32::MAX as i64, i32::MAX)]
#[case(i32::MIN as i64, i32::MIN)]
fn integer_to_integer_success(#[case] input: i64, #[case] expected: i32) {
    let result = i32::try_from(Scalar::I64(input)).unwrap();
    assert_eq!(result, expected);
}

#[test]
fn i8_boundaries() {
    assert_eq!(i8::try_from(Scalar::I32(i8::MIN as i32)).unwrap(), i8::MIN);

    assert_eq!(i8::try_from(Scalar::I32(i8::MAX as i32)).unwrap(), i8::MAX);
}

#[test]
fn i8_overflow_positive() {
    let result = i8::try_from(Scalar::I32(i8::MAX as i32 + 1));

    assert_matches!(result, Err(ScalarConversionError::Overflow));
}

#[test]
fn i8_overflow_negative() {
    let result = i8::try_from(Scalar::I32(i8::MIN as i32 - 1));

    assert_matches!(result, Err(ScalarConversionError::Overflow));
}

#[test]
fn u8_overflow_negative() {
    let result = u8::try_from(Scalar::I8(-1));

    assert_matches!(result, Err(ScalarConversionError::Overflow));
}

#[test]
fn u8_overflow_positive() {
    let result = u8::try_from(Scalar::I32(256));

    assert_matches!(result, Err(ScalarConversionError::Overflow));
}

#[test]
fn u8_boundaries() {
    assert_eq!(u8::try_from(Scalar::I32(0)).unwrap(), 0);
    assert_eq!(u8::try_from(Scalar::I32(255)).unwrap(), 255);
}

// ============================================================
// Float -> Integer
// ============================================================

#[rstest]
#[case(0.0)]
#[case(1.0)]
#[case(-1.0)]
#[case(127.0)]
#[case(-128.0)]
fn float_to_i8_success(#[case] value: f64) {
    let result = i8::try_from(Scalar::F64(value)).unwrap();
    assert_eq!(result as f64, value);
}

#[test]
fn float_to_integer_rejects_fractional_value() {
    assert_matches!(
        i32::try_from(Scalar::F64(1.5)),
        Err(ScalarConversionError::FractionalValue)
    );

    assert_matches!(
        i32::try_from(Scalar::F32(-1.5)),
        Err(ScalarConversionError::FractionalValue)
    );
}

#[test]
fn float_to_integer_exact_boundary() {
    assert_eq!(i8::try_from(Scalar::F64(i8::MAX as f64)).unwrap(), i8::MAX);

    assert_eq!(i8::try_from(Scalar::F64(i8::MIN as f64)).unwrap(), i8::MIN);
}

#[test]
fn float_to_integer_overflow_positive() {
    assert_matches!(
        i8::try_from(Scalar::F64(128.0)),
        Err(ScalarConversionError::Overflow)
    );
}

#[test]
fn float_to_integer_overflow_negative() {
    assert_matches!(
        i8::try_from(Scalar::F64(-129.0)),
        Err(ScalarConversionError::Overflow)
    );
}

#[test]
fn float_to_integer_nan() {
    assert_matches!(
        i32::try_from(Scalar::F64(f64::NAN)),
        Err(ScalarConversionError::InvalidValue)
    );
}

#[test]
fn float_to_integer_positive_infinity() {
    assert_matches!(
        i32::try_from(Scalar::F64(f64::INFINITY)),
        Err(ScalarConversionError::InvalidValue)
    );
}

#[test]
fn float_to_integer_negative_infinity() {
    assert_matches!(
        i32::try_from(Scalar::F64(f64::NEG_INFINITY)),
        Err(ScalarConversionError::InvalidValue)
    );
}

// ============================================================
// Half -> Integer
// ============================================================

#[test]
fn f16_to_integer() {
    assert_eq!(
        i32::try_from(Scalar::F16(half::f16::from_f32(42.0))).unwrap(),
        42
    );
}

#[test]
fn bf16_to_integer() {
    assert_eq!(
        i32::try_from(Scalar::BF16(half::bf16::from_f32(-42.0))).unwrap(),
        -42
    );
}

#[test]
fn f16_to_integer_fractional() {
    assert_matches!(
        i32::try_from(Scalar::F16(half::f16::from_f32(1.5))),
        Err(ScalarConversionError::FractionalValue)
    );
}

#[test]
fn bf16_to_integer_fractional() {
    assert_matches!(
        i32::try_from(Scalar::BF16(half::bf16::from_f32(1.5))),
        Err(ScalarConversionError::FractionalValue)
    );
}

// ============================================================
// Integer -> Float
// ============================================================

#[test]
fn integer_to_f32() {
    assert_eq!(f32::try_from(Scalar::I32(42)).unwrap(), 42.0);

    assert_eq!(f32::try_from(Scalar::I8(-42)).unwrap(), -42.0);

    assert_eq!(f32::try_from(Scalar::U8(255)).unwrap(), 255.0);
}

#[test]
fn integer_to_f64() {
    assert_eq!(
        f64::try_from(Scalar::I64(-123456789)).unwrap(),
        -123456789.0
    );

    assert_eq!(f64::try_from(Scalar::U8(255)).unwrap(), 255.0);
}

#[test]
fn i64_to_f32_does_not_overflow() {
    let result = f32::try_from(Scalar::I64(i64::MAX));

    assert!(result.is_ok());
    assert!(result.unwrap().is_finite());
}

// ============================================================
// Float -> Float
// ============================================================

#[test]
fn f32_to_f64() {
    let value = 123.5f32;

    let result = f64::try_from(Scalar::F32(value)).unwrap();

    assert_eq!(result, value as f64);
}

#[test]
fn f64_to_f32() {
    let value = 123.5f64;

    let result = f32::try_from(Scalar::F64(value)).unwrap();

    assert_eq!(result, value as f32);
}

#[test]
fn f64_to_f32_special_values() {
    assert!(f32::try_from(Scalar::F64(f64::NAN)).unwrap().is_nan());

    assert_eq!(
        f32::try_from(Scalar::F64(f64::INFINITY)).unwrap(),
        f32::INFINITY
    );

    assert_eq!(
        f32::try_from(Scalar::F64(f64::NEG_INFINITY)).unwrap(),
        f32::NEG_INFINITY
    );
}

// ============================================================
// Float -> Half
// ============================================================

#[test]
fn f32_to_f16() {
    let value = 1.5f32;

    let result = half::f16::try_from(Scalar::F32(value)).unwrap();

    assert_eq!(result.to_f32(), value);
}

#[test]
fn f32_to_bf16() {
    let value = 1.5f32;

    let result = half::bf16::try_from(Scalar::F32(value)).unwrap();

    assert_eq!(result.to_f32(), value);
}

#[test]
fn f64_to_f16() {
    let value = 1.5f64;

    let result = half::f16::try_from(Scalar::F64(value)).unwrap();

    assert_eq!(result.to_f64(), value);
}

// ============================================================
// Integer -> Half
// ============================================================

#[test]
fn integer_to_f16() {
    let result = half::f16::try_from(Scalar::I32(42)).unwrap();

    assert_eq!(result.to_f32(), 42.0);
}

#[test]
fn integer_to_bf16() {
    let result = half::bf16::try_from(Scalar::I32(-42)).unwrap();

    assert_eq!(result.to_f32(), -42.0);
}

// ============================================================
// Half -> Float
// ============================================================

#[test]
fn f16_to_f32() {
    let value = half::f16::from_f32(1.5);

    let result = f32::try_from(Scalar::F16(value)).unwrap();

    assert_eq!(result, value.to_f32());
}

#[test]
fn bf16_to_f64() {
    let value = half::bf16::from_f32(1.5);

    let result = f64::try_from(Scalar::BF16(value)).unwrap();

    assert_eq!(result, value.to_f64());
}

// ============================================================
// Half -> Half
// ============================================================

#[test]
fn f16_to_f16() {
    let value = half::f16::from_f32(1.5);

    let result = half::f16::try_from(Scalar::F16(value)).unwrap();

    assert_eq!(result, value);
}

#[test]
fn bf16_to_bf16() {
    let value = half::bf16::from_f32(1.5);

    let result = half::bf16::try_from(Scalar::BF16(value)).unwrap();

    assert_eq!(result, value);
}

#[test]
fn f16_to_bf16() {
    let value = half::f16::from_f32(1.5);

    let result = half::bf16::try_from(Scalar::F16(value)).unwrap();

    assert_eq!(result.to_f32(), value.to_f32());
}

#[test]
fn bf16_to_f16() {
    let value = half::bf16::from_f32(1.5);

    let result = half::f16::try_from(Scalar::BF16(value)).unwrap();

    assert_eq!(result.to_f32(), value.to_f32());
}

// ============================================================
// Half overflow
// ============================================================

#[test]
fn f32_to_f16_overflow() {
    let value = half::f16::MAX.to_f64() * 2.0;

    assert_matches!(
        half::f16::try_from(Scalar::F64(value)),
        Err(ScalarConversionError::Overflow)
    );
}

#[test]
fn f32_to_bf16_overflow() {
    let value = half::bf16::MAX.to_f64() * 2.0;

    assert_matches!(
        half::bf16::try_from(Scalar::F64(value)),
        Err(ScalarConversionError::Overflow)
    );
}

// ============================================================
// Special float values -> half
// ============================================================

#[test]
fn nan_to_f16() {
    let result = half::f16::try_from(Scalar::F64(f64::NAN));

    assert!(result.is_ok());
    assert!(result.unwrap().is_nan());
}

#[test]
fn infinity_to_f16() {
    let result = half::f16::try_from(Scalar::F64(f64::INFINITY));
    println!("{result:?}");
    assert!(result.is_ok());
    assert!(result.unwrap().is_infinite());
}
