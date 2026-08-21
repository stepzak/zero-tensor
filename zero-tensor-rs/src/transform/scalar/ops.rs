use std::ops::{Neg, Not};

use super::Scalar;

macro_rules! promote_types {
    ($x:expr, $y:expr, $op:tt) => {
        match ($x, $y) {
            (Scalar::U8(a), Scalar::U8(b))     => Scalar::U8(a $op b),
            (Scalar::I8(a), Scalar::I8(b))     => Scalar::I8(a $op b),
            (Scalar::I32(a), Scalar::I32(b))   => Scalar::I32(a $op b),
            (Scalar::I64(a), Scalar::I64(b))   => Scalar::I64(a $op b),
            (Scalar::F32(a), Scalar::F32(b))   => Scalar::F32(a $op b),
            (Scalar::F64(a), Scalar::F64(b))   => Scalar::F64(a $op b),
            (Scalar::BF16(a), Scalar::BF16(b)) => Scalar::BF16(half::bf16::from_f32(a.to_f32() $op b.to_f32())),
            (Scalar::F16(a), Scalar::F16(b))   => Scalar::F16(half::f16::from_f32(a.to_f32() $op b.to_f32())),

            (Scalar::U8(a), Scalar::I8(b))     => Scalar::I32((a as i32) $op (b as i32)),
            (Scalar::I8(a), Scalar::U8(b))     => Scalar::I32((a as i32) $op (b as i32)),

            (Scalar::U8(a), Scalar::I32(b))     => Scalar::I32((a as i32) $op b),
            (Scalar::I32(a), Scalar::U8(b))     => Scalar::I32(a $op (b as i32)),

            (Scalar::U8(a), Scalar::I64(b))     => Scalar::I64((a as i64) $op b),
            (Scalar::I64(a), Scalar::U8(b))     => Scalar::I64(a $op (b as i64)),

            (Scalar::I8(a), Scalar::I32(b))     => Scalar::I32((a as i32) $op b),
            (Scalar::I32(a), Scalar::I8(b))     => Scalar::I32(a $op (b as i32)),

            (Scalar::I8(a), Scalar::I64(b))     => Scalar::I64((a as i64) $op b),
            (Scalar::I64(a), Scalar::I8(b))     => Scalar::I64(a $op (b as i64)),

            (Scalar::I32(a), Scalar::I64(b))    => Scalar::I64((a as i64) $op b),
            (Scalar::I64(a), Scalar::I32(b))    => Scalar::I64(a $op (b as i64)),

            (Scalar::I32(a), Scalar::F32(b))    => Scalar::F32((a as f32) $op b),
            (Scalar::F32(a), Scalar::I32(b))    => Scalar::F32(a $op (b as f32)),

            (Scalar::I64(a), Scalar::F32(b))    => Scalar::F64((a as f64) $op (b as f64)),
            (Scalar::F32(a), Scalar::I64(b))    => Scalar::F64((a as f64) $op (b as f64)),

            (Scalar::U8(a), Scalar::F32(b))     => Scalar::F32((a as f32) $op b),
            (Scalar::F32(a), Scalar::U8(b))     => Scalar::F32(a $op (b as f32)),

            (Scalar::F16(a), Scalar::F32(b))    => Scalar::F32(a.to_f32() $op b),
            (Scalar::F32(a), Scalar::F16(b))    => Scalar::F32(a $op b.to_f32()),

            (Scalar::BF16(a), Scalar::F32(b))   => Scalar::F32(a.to_f32() $op b),
            (Scalar::F32(a), Scalar::BF16(b))   => Scalar::F32(a $op b.to_f32()),

            (Scalar::F16(a), Scalar::BF16(b))   => Scalar::F32(a.to_f32() $op b.to_f32()),
            (Scalar::BF16(a), Scalar::F16(b))   => Scalar::F32(a.to_f32() $op b.to_f32()),

            (Scalar::F64(a), any)               => Scalar::F64(a $op any.to_f64_lossy()),
            (any, Scalar::F64(b))               => Scalar::F64(any.to_f64_lossy() $op b),

            (a, b)                              => Scalar::F32(a.to_f32_lossy() $op b.to_f32_lossy()),
        }
    };
}

impl Neg for Scalar {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Scalar::U8(val) => Scalar::I32(-(val as i32)),

            Scalar::I8(val) => Scalar::I8(-val),
            Scalar::I32(val) => Scalar::I32(-val),
            Scalar::I64(val) => Scalar::I64(-val),
            Scalar::F32(val) => Scalar::F32(-val),
            Scalar::F64(val) => Scalar::F64(-val),

            Scalar::BF16(val) => Scalar::BF16(half::bf16::from_f32(-val.to_f32())),
            Scalar::F16(val) => Scalar::F16(half::f16::from_f32(-val.to_f32())),
        }
    }
}

impl Not for Scalar {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            Scalar::U8(val) => Scalar::U8(!val),
            Scalar::I8(val) => Scalar::I8(!val),
            Scalar::I32(val) => Scalar::I32(!val),
            Scalar::I64(val) => Scalar::I64(!val),

            float_scalar => float_scalar,
        }
    }
}

macro_rules! generate_math_ops {
    ($trait_name:ident, $method_name:ident, $op:tt) => {
        impl std::ops::$trait_name for Scalar {
            type Output = Self;

            #[inline]
            fn $method_name(self, rhs: Self) -> Self::Output {
                promote_types!(self, rhs, $op)
            }
        }
    };
}

generate_math_ops!(Add, add, +);
generate_math_ops!(Sub, sub, -);
generate_math_ops!(Mul, mul, *);
generate_math_ops!(Div, div, /);

#[cfg(test)]
mod tests {
    use super::*;

    fn f16(v: f32) -> half::f16 {
        half::f16::from_f32(v)
    }

    fn bf16(v: f32) -> half::bf16 {
        half::bf16::from_f32(v)
    }

    // ------------------------------------------------------------
    // Same type
    // ------------------------------------------------------------

    #[test]
    fn same_type_add() {
        assert!(matches!(Scalar::U8(2) + Scalar::U8(3), Scalar::U8(5)));

        assert!(matches!(Scalar::I8(2) + Scalar::I8(3), Scalar::I8(5)));

        assert!(matches!(Scalar::I32(2) + Scalar::I32(3), Scalar::I32(5)));

        assert!(matches!(Scalar::I64(2) + Scalar::I64(3), Scalar::I64(5)));

        assert!(matches!(
            Scalar::F32(2.0) + Scalar::F32(3.0),
            Scalar::F32(x) if x == 5.0
        ));

        assert!(matches!(
            Scalar::F64(2.0) + Scalar::F64(3.0),
            Scalar::F64(x) if x == 5.0
        ));

        assert!(matches!(
            Scalar::F16(f16(2.0)) + Scalar::F16(f16(3.0)),
            Scalar::F16(x) if x == f16(5.0)
        ));

        assert!(matches!(
            Scalar::BF16(bf16(2.0)) + Scalar::BF16(bf16(3.0)),
            Scalar::BF16(x) if x == bf16(5.0)
        ));
    }

    #[test]
    fn same_type_sub() {
        assert!(matches!(Scalar::I32(10) - Scalar::I32(3), Scalar::I32(7)));

        assert!(matches!(
            Scalar::F32(10.0) - Scalar::F32(3.0),
            Scalar::F32(x) if x == 7.0
        ));
    }

    #[test]
    fn same_type_mul() {
        assert!(matches!(Scalar::I64(6) * Scalar::I64(7), Scalar::I64(42)));

        assert!(matches!(
            Scalar::F64(1.5) * Scalar::F64(2.0),
            Scalar::F64(x) if x == 3.0
        ));
    }

    #[test]
    fn same_type_div() {
        assert!(matches!(Scalar::I32(12) / Scalar::I32(3), Scalar::I32(4)));

        assert!(matches!(
            Scalar::F64(12.0) / Scalar::F64(4.0),
            Scalar::F64(x) if x == 3.0
        ));
    }

    // ------------------------------------------------------------
    // Integer promotion
    // ------------------------------------------------------------

    #[test]
    fn u8_i8_promotes_to_i32() {
        assert!(matches!(Scalar::U8(10) + Scalar::I8(-3), Scalar::I32(7)));

        assert!(matches!(Scalar::I8(-3) + Scalar::U8(10), Scalar::I32(7)));
    }

    #[test]
    fn u8_i32_promotes_to_i32() {
        assert!(matches!(Scalar::U8(10) + Scalar::I32(20), Scalar::I32(30)));

        assert!(matches!(Scalar::I32(20) + Scalar::U8(10), Scalar::I32(30)));
    }

    #[test]
    fn u8_i64_promotes_to_i64() {
        assert!(matches!(Scalar::U8(10) + Scalar::I64(20), Scalar::I64(30)));

        assert!(matches!(Scalar::I64(20) + Scalar::U8(10), Scalar::I64(30)));
    }

    #[test]
    fn i8_i32_promotes_to_i32() {
        assert!(matches!(Scalar::I8(-10) + Scalar::I32(20), Scalar::I32(10)));

        assert!(matches!(Scalar::I32(20) + Scalar::I8(-10), Scalar::I32(10)));
    }

    #[test]
    fn i8_i64_promotes_to_i64() {
        assert!(matches!(Scalar::I8(-10) + Scalar::I64(20), Scalar::I64(10)));

        assert!(matches!(Scalar::I64(20) + Scalar::I8(-10), Scalar::I64(10)));
    }

    #[test]
    fn i32_i64_promotes_to_i64() {
        assert!(matches!(Scalar::I32(10) + Scalar::I64(20), Scalar::I64(30)));

        assert!(matches!(Scalar::I64(20) + Scalar::I32(10), Scalar::I64(30)));
    }

    // ------------------------------------------------------------
    // F32 special promotion
    // ------------------------------------------------------------

    #[test]
    fn i32_f32_promotes_to_f32() {
        assert!(matches!(
            Scalar::I32(2) + Scalar::F32(0.5),
            Scalar::F32(x) if x == 2.5
        ));

        assert!(matches!(
            Scalar::F32(0.5) + Scalar::I32(2),
            Scalar::F32(x) if x == 2.5
        ));
    }

    #[test]
    fn i64_f32_promotes_to_f64() {
        assert!(matches!(
            Scalar::I64(2) + Scalar::F32(0.5),
            Scalar::F64(x) if x == 2.5
        ));

        assert!(matches!(
            Scalar::F32(0.5) + Scalar::I64(2),
            Scalar::F64(x) if x == 2.5
        ));
    }

    #[test]
    fn u8_f32_promotes_to_f32() {
        assert!(matches!(
            Scalar::U8(2) + Scalar::F32(0.5),
            Scalar::F32(x) if x == 2.5
        ));

        assert!(matches!(
            Scalar::F32(0.5) + Scalar::U8(2),
            Scalar::F32(x) if x == 2.5
        ));
    }

    // ------------------------------------------------------------
    // Half precision promotion
    // ------------------------------------------------------------

    #[test]
    fn f16_f32_promotes_to_f32() {
        assert!(matches!(
            Scalar::F16(f16(2.0)) + Scalar::F32(0.5),
            Scalar::F32(x) if x == 2.5
        ));

        assert!(matches!(
            Scalar::F32(0.5) + Scalar::F16(f16(2.0)),
            Scalar::F32(x) if x == 2.5
        ));
    }

    #[test]
    fn bf16_f32_promotes_to_f32() {
        assert!(matches!(
            Scalar::BF16(bf16(2.0)) + Scalar::F32(0.5),
            Scalar::F32(x) if x == 2.5
        ));

        assert!(matches!(
            Scalar::F32(0.5) + Scalar::BF16(bf16(2.0)),
            Scalar::F32(x) if x == 2.5
        ));
    }

    #[test]
    fn f16_bf16_promotes_to_f32() {
        assert!(matches!(
            Scalar::F16(f16(2.0)) + Scalar::BF16(bf16(0.5)),
            Scalar::F32(x) if x == 2.5
        ));

        assert!(matches!(
            Scalar::BF16(bf16(0.5)) + Scalar::F16(f16(2.0)),
            Scalar::F32(x) if x == 2.5
        ));
    }

    // ------------------------------------------------------------
    // F64 always wins
    // ------------------------------------------------------------

    #[test]
    fn f64_wins_over_integer() {
        assert!(matches!(
            Scalar::F64(0.5) + Scalar::I32(2),
            Scalar::F64(x) if x == 2.5
        ));

        assert!(matches!(
            Scalar::I32(2) + Scalar::F64(0.5),
            Scalar::F64(x) if x == 2.5
        ));
    }

    #[test]
    fn f64_wins_over_f32() {
        assert!(matches!(
            Scalar::F64(0.5) + Scalar::F32(2.0),
            Scalar::F64(x) if x == 2.5
        ));

        assert!(matches!(
            Scalar::F32(2.0) + Scalar::F64(0.5),
            Scalar::F64(x) if x == 2.5
        ));
    }

    #[test]
    fn f64_wins_over_f16() {
        assert!(matches!(
            Scalar::F16(f16(2.0)) + Scalar::F64(0.5),
            Scalar::F64(x) if x == 2.5
        ));
    }

    #[test]
    fn f64_wins_over_bf16() {
        assert!(matches!(
            Scalar::BF16(bf16(2.0)) + Scalar::F64(0.5),
            Scalar::F64(x) if x == 2.5
        ));
    }

    // ------------------------------------------------------------
    // Fallback -> F32
    // ------------------------------------------------------------

    #[test]
    fn fallback_integer_f16_promotes_to_f32() {
        assert!(matches!(
            Scalar::I8(2) + Scalar::F16(f16(0.5)),
            Scalar::F32(x) if x == 2.5
        ));
    }

    #[test]
    fn fallback_integer_bf16_promotes_to_f32() {
        assert!(matches!(
            Scalar::I32(2) + Scalar::BF16(bf16(0.5)),
            Scalar::F32(x) if x == 2.5
        ));
    }

    // ------------------------------------------------------------
    // Operand order for non-commutative operators
    // ------------------------------------------------------------

    #[test]
    fn subtraction_preserves_operand_order() {
        assert!(matches!(Scalar::I32(10) - Scalar::U8(3), Scalar::I32(7)));

        assert!(matches!(Scalar::U8(3) - Scalar::I32(10), Scalar::I32(-7)));
    }

    #[test]
    fn division_preserves_operand_order() {
        assert!(matches!(Scalar::I32(20) / Scalar::U8(4), Scalar::I32(5)));

        assert!(matches!(
            Scalar::F32(10.0) / Scalar::I64(2),
            Scalar::F64(x) if x == 5.0
        ));
    }

    #[test]
    fn mixed_multiplication_works() {
        assert!(matches!(
            Scalar::I32(3) * Scalar::F32(2.5),
            Scalar::F32(x) if x == 7.5
        ));
    }

    #[test]
    fn neg_u8_promotes_to_i32() {
        assert_eq!(-Scalar::U8(42), Scalar::I32(-42));

        assert_eq!(-Scalar::U8(0), Scalar::I32(0));

        assert_eq!(-Scalar::U8(u8::MAX), Scalar::I32(-(u8::MAX as i32)));
    }

    #[test]
    fn neg_signed_integers() {
        assert_eq!(-Scalar::I8(42), Scalar::I8(-42));

        assert_eq!(-Scalar::I32(-42), Scalar::I32(42));

        assert_eq!(-Scalar::I64(1_000_000), Scalar::I64(-1_000_000));
    }

    #[test]
    fn neg_floats() {
        assert_eq!(-Scalar::F32(1.5), Scalar::F32(-1.5));

        assert_eq!(-Scalar::F64(-42.25), Scalar::F64(42.25));
    }

    #[test]
    fn neg_half_floats() {
        let bf16 = Scalar::BF16(half::bf16::from_f32(1.5));
        let f16 = Scalar::F16(half::f16::from_f32(-2.25));

        assert_eq!(-bf16, Scalar::BF16(half::bf16::from_f32(-1.5)));

        assert_eq!(-f16, Scalar::F16(half::f16::from_f32(2.25)));
    }

    #[test]
    fn neg_negative_zero() {
        let result = -Scalar::F32(0.0);

        match result {
            Scalar::F32(value) => {
                assert_eq!(value, 0.0);
                assert!(value.is_sign_negative());
            }
            _ => panic!("expected Scalar::F32"),
        }
    }

    #[test]
    fn not_u8() {
        assert_eq!(!Scalar::U8(0b0000_1111), Scalar::U8(0b1111_0000));

        assert_eq!(!Scalar::U8(0), Scalar::U8(u8::MAX));

        assert_eq!(!Scalar::U8(u8::MAX), Scalar::U8(0));
    }

    #[test]
    fn not_i8() {
        assert_eq!(!Scalar::I8(0), Scalar::I8(!0i8));

        assert_eq!(!Scalar::I8(42), Scalar::I8(!42i8));

        assert_eq!(!Scalar::I8(-1), Scalar::I8(!-1i8));
    }

    #[test]
    fn not_i32() {
        assert_eq!(!Scalar::I32(0), Scalar::I32(!0i32));

        assert_eq!(!Scalar::I32(0x0F0F_0F0F), Scalar::I32(!0x0F0F_0F0Fi32));
    }

    #[test]
    fn not_i64() {
        assert_eq!(!Scalar::I64(0), Scalar::I64(!0i64));

        assert_eq!(!Scalar::I64(42), Scalar::I64(!42i64));

        assert_eq!(!Scalar::I64(-1), Scalar::I64(!-1i64));
    }

    #[test]
    fn not_floats_are_identity() {
        assert_eq!(!Scalar::F32(1.5), Scalar::F32(1.5));

        assert_eq!(!Scalar::F64(-42.25), Scalar::F64(-42.25));
    }

    #[test]
    fn not_half_floats_are_identity() {
        let bf16 = half::bf16::from_f32(1.5);
        let f16 = half::f16::from_f32(-2.25);

        assert_eq!(!Scalar::BF16(bf16), Scalar::BF16(bf16));

        assert_eq!(!Scalar::F16(f16), Scalar::F16(f16));
    }

    #[test]
    fn not_integers_is_involution() {
        let u8_value = Scalar::U8(42);
        assert_eq!(!(!u8_value), Scalar::U8(42));

        let i8_value = Scalar::I8(-42);
        assert_eq!(!(!i8_value), Scalar::I8(-42));

        let i32_value = Scalar::I32(123_456);
        assert_eq!(!(!i32_value), Scalar::I32(123_456));

        let i64_value = Scalar::I64(-987_654_321);
        assert_eq!(!(!i64_value), Scalar::I64(-987_654_321));
    }
}
