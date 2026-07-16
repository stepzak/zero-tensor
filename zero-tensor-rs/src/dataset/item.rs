#[repr(u8)]
#[derive(Clone, Copy, std::fmt::Debug, PartialEq)]
pub enum TensorDT {
    F16,
    F32,
    F64,
    BF16,
    I8,
    I32,
    I64,
    U8,
}

pub type ShapeType = u32;
pub type StrideType = u32;

#[derive(Debug, Clone)]
pub struct TensorItemMeta {
    shape: Vec<ShapeType>,
    dt: TensorDT,
    strides: Vec<StrideType>,
}

impl TensorItemMeta {
    pub fn new(shape: Vec<ShapeType>, strides: Vec<StrideType>, dt: TensorDT) -> Self {
        TensorItemMeta { shape, strides, dt }
    }

    pub fn shape(&self) -> &[ShapeType] {
        &self.shape
    }

    pub fn dt(&self) -> TensorDT {
        self.dt
    }

    pub fn strides(&self) -> &[StrideType] {
        &self.strides
    }
}
