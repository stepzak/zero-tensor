pub trait IsZero {
    fn eq_zero(self) -> bool;
}

macro_rules! impl_is_zero {
    ($($ty:ty),* $(,)?) => {
        $(
            impl IsZero for $ty {
                #[inline]
                fn eq_zero(self) -> bool {
                    self == 0 as $ty
                }
            }
        )*
    };
}

impl_is_zero!(u8, i8, i32, i64, f32, f64);

impl IsZero for half::f16 {
    #[inline]
    fn eq_zero(self) -> bool {
        self == half::f16::ZERO
    }
}

impl IsZero for half::bf16 {
    #[inline]
    fn eq_zero(self) -> bool {
        self == half::bf16::ZERO
    }
}
