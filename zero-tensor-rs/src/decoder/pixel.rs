use bytemuck::Pod;

pub trait Pixel: Pod + Copy + Send + Sync + 'static {
    fn from_u8(value: u8) -> Self;

    fn size_of() -> usize {
        size_of::<Self>()
    }
}

impl Pixel for u8 {
    fn from_u8(value: u8) -> Self {
        value
    }
}

impl Pixel for i8 {
    fn from_u8(value: u8) -> Self {
        value as i8
    }
}

impl Pixel for i32 {
    fn from_u8(value: u8) -> Self {
        value as i32
    }
}

impl Pixel for i64 {
    fn from_u8(value: u8) -> Self {
        value as i64
    }
}

impl Pixel for half::f16 {
    fn from_u8(value: u8) -> Self {
        Self::from_f32(value as f32 / 255.0)
    }
}

impl Pixel for half::bf16 {
    fn from_u8(value: u8) -> Self {
        Self::from_f32(value as f32 / 255.0)
    }
}

impl Pixel for f32 {
    fn from_u8(value: u8) -> Self {
        value as f32 / 255.0
    }
}

impl Pixel for f64 {
    fn from_u8(value: u8) -> Self {
        value as f64 / 255.0
    }
}
