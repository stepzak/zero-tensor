use bytemuck::{NoUninit, Pod};

pub trait AugmentationItem: Copy + Send + Sync + Pod + NoUninit {}

impl AugmentationItem for u8 {}
impl AugmentationItem for i8 {}
impl AugmentationItem for i32 {}
impl AugmentationItem for i64 {}
impl AugmentationItem for half::bf16 {}
impl AugmentationItem for half::f16 {}
impl AugmentationItem for f32 {}
impl AugmentationItem for f64 {}
