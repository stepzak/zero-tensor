use std::cmp::Ordering;

use super::Scalar;

macro_rules! promote_partial_cmp {
    ($x:expr, $y:expr) => {
        match ($x, $y) {
            (Scalar::U8(a), Scalar::U8(b)) => a.partial_cmp(b),
            (Scalar::I8(a), Scalar::I8(b)) => a.partial_cmp(b),
            (Scalar::I32(a), Scalar::I32(b)) => a.partial_cmp(b),
            (Scalar::I64(a), Scalar::I64(b)) => a.partial_cmp(b),

            (Scalar::F32(a), Scalar::F32(b)) => a.partial_cmp(b),
            (Scalar::F64(a), Scalar::F64(b)) => a.partial_cmp(b),

            (a, b) => a.to_f64_lossy().partial_cmp(&b.to_f64_lossy()),
        }
    };
}

impl PartialOrd for Scalar {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        promote_partial_cmp!(self, other)
    }
}

impl PartialEq for Scalar {
    fn eq(&self, other: &Self) -> bool {
        self.partial_cmp(other) == Some(Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use super::Scalar;

    #[test]
    fn same_integer_types() {
        assert_eq!(
            Scalar::U8(1).partial_cmp(&Scalar::U8(2)),
            Some(Ordering::Less)
        );
        assert_eq!(
            Scalar::I8(2).partial_cmp(&Scalar::I8(1)),
            Some(Ordering::Greater)
        );
        assert_eq!(Scalar::I32(42), Scalar::I32(42));
        assert_eq!(
            Scalar::I64(-10).partial_cmp(&Scalar::I64(-5)),
            Some(Ordering::Less)
        );
    }

    #[test]
    fn same_float_types() {
        assert_eq!(Scalar::F32(1.5), Scalar::F32(1.5));
        assert!(Scalar::F32(1.0) < Scalar::F32(2.0));

        assert_eq!(Scalar::F64(-1.0), Scalar::F64(-1.0));
        assert!(Scalar::F64(10.0) > Scalar::F64(5.0));
    }

    #[test]
    fn same_half_types() {
        let a = Scalar::F16(half::f16::from_f32(1.5));
        let b = Scalar::F16(half::f16::from_f32(2.5));

        assert!(a < b);

        let a = Scalar::BF16(half::bf16::from_f32(1.5));
        let b = Scalar::BF16(half::bf16::from_f32(1.5));

        assert_eq!(a, b);
    }

    #[test]
    fn mixed_integer_types() {
        assert_eq!(Scalar::U8(42), Scalar::I8(42));
        assert_eq!(
            Scalar::I8(-1).partial_cmp(&Scalar::U8(1)),
            Some(Ordering::Less)
        );

        assert_eq!(Scalar::U8(42), Scalar::I32(42));
        assert_eq!(
            Scalar::I32(-10).partial_cmp(&Scalar::U8(0)),
            Some(Ordering::Less)
        );

        assert_eq!(Scalar::I32(100), Scalar::I64(100));
        assert!(Scalar::I64(101) > Scalar::I32(100));
    }

    #[test]
    fn integer_and_float_comparison() {
        assert_eq!(Scalar::I32(42), Scalar::F32(42.0));
        assert!(Scalar::I32(41) < Scalar::F32(42.0));

        assert_eq!(Scalar::U8(255), Scalar::F32(255.0));
        assert!(Scalar::F32(255.0) > Scalar::I32(100));

        assert_eq!(Scalar::I64(42), Scalar::F64(42.0));
        assert!(Scalar::I64(-1) < Scalar::F64(0.0));
    }

    #[test]
    fn half_and_f32_comparison() {
        let f16 = Scalar::F16(half::f16::from_f32(1.5));
        let f32 = Scalar::F32(1.5);

        assert_eq!(f16, f32);

        let bf16 = Scalar::BF16(half::bf16::from_f32(2.0));
        let f32 = Scalar::F32(3.0);

        assert!(bf16 < f32);
    }

    #[test]
    fn half_and_integer_comparison() {
        let f16 = Scalar::F16(half::f16::from_f32(42.0));
        assert_eq!(f16, Scalar::I32(42));

        let bf16 = Scalar::BF16(half::bf16::from_f32(-10.0));
        assert!(bf16 < Scalar::I64(-5));
    }

    #[test]
    fn f64_promotes_all_other_types() {
        assert_eq!(Scalar::F64(42.0), Scalar::U8(42));
        assert_eq!(Scalar::F64(-42.0), Scalar::I8(-42));
        assert_eq!(Scalar::F64(42.0), Scalar::I32(42));
        assert_eq!(Scalar::F64(42.0), Scalar::I64(42));

        assert_eq!(Scalar::F64(1.5), Scalar::F16(half::f16::from_f32(1.5)));

        assert_eq!(Scalar::F64(1.5), Scalar::BF16(half::bf16::from_f32(1.5)));
    }

    #[test]
    fn comparison_is_symmetric() {
        let pairs = [
            (Scalar::U8(42), Scalar::I8(42)),
            (Scalar::I32(-10), Scalar::I64(-10)),
            (Scalar::I32(100), Scalar::F32(100.0)),
            (Scalar::I64(100), Scalar::F64(100.0)),
            (
                Scalar::F16(half::f16::from_f32(2.0)),
                Scalar::BF16(half::bf16::from_f32(2.0)),
            ),
        ];

        for (a, b) in pairs {
            assert_eq!(a == b, b == a);
            assert_eq!(a.partial_cmp(&b), b.partial_cmp(&a).map(Ordering::reverse));
        }
    }

    #[test]
    fn nan_is_not_equal_to_anything() {
        let nan32 = Scalar::F32(f32::NAN);

        assert_ne!(nan32, Scalar::F32(f32::NAN));
        assert_ne!(nan32, Scalar::F32(1.0));
        assert_ne!(Scalar::F32(1.0), nan32);
    }

    #[test]
    fn nan_has_no_ordering() {
        let nan32 = Scalar::F32(f32::NAN);
        let value32 = Scalar::F32(1.0);

        assert_eq!(nan32.partial_cmp(&value32), None);
        assert_eq!(value32.partial_cmp(&nan32), None);

        let nan64 = Scalar::F64(f64::NAN);

        assert_eq!(nan64.partial_cmp(&Scalar::I64(1)), None);
        assert_eq!(Scalar::I64(1).partial_cmp(&nan64), None);
    }

    #[test]
    fn infinity_comparison() {
        assert!(Scalar::F32(f32::INFINITY) > Scalar::F32(1.0));
        assert!(Scalar::F64(f64::NEG_INFINITY) < Scalar::I32(0));

        assert_eq!(Scalar::F32(f32::INFINITY), Scalar::F64(f64::INFINITY));

        assert!(Scalar::F64(f64::NEG_INFINITY) < Scalar::F16(half::f16::from_f32(-100.0)));
    }

    #[test]
    fn transitivity_for_ordered_values() {
        let a = Scalar::I8(-1);
        let b = Scalar::F32(0.0);
        let c = Scalar::F64(1.0);

        assert!(a < b);
        assert!(b < c);
        assert!(a < c);
    }

    #[test]
    fn precision_edge_i64_and_f32() {
        let a = Scalar::I64(16_777_217);
        let b = Scalar::F32(16_777_216.0);

        assert!(a > b);
        assert!(b < a);
        assert_ne!(a, b);
    }
}
