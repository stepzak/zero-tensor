use smallvec::SmallVec;

use crate::buffer::get_dt_size;

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

const MAX_NDIMS: usize = 8;

pub type ShapeVec = SmallVec<[ShapeType; MAX_NDIMS]>;
pub type StrideVec = SmallVec<[StrideType; MAX_NDIMS]>;

#[derive(Debug, Clone)]
pub struct TensorBatchLayout {
    shape: ShapeVec,
    dt: TensorDT,
    strides: StrideVec,
}

impl TensorBatchLayout {
    pub fn new(shape: ShapeVec, strides: StrideVec, dt: TensorDT) -> Self {
        TensorBatchLayout { shape, strides, dt }
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

    pub fn total_elements(&self) -> usize {
        self.shape.iter().product::<ShapeType>() as usize
    }

    pub fn total_bytes(&self) -> usize {
        self.total_elements() * get_dt_size(self.dt)
    }
}
