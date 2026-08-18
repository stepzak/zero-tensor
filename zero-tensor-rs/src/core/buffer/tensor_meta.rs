use std::{
    mem::offset_of,
    sync::atomic::{AtomicU8, Ordering},
};

use super::super::dataset::item::{ShapeType, TensorDT};

pub struct TensorDOffsets {
    shapes: usize,
    strides: usize,
    data: usize,
}

impl TensorDOffsets {
    pub fn new(shapes: usize, strides: usize, data: usize) -> Self {
        TensorDOffsets {
            shapes,
            strides,
            data,
        }
    }

    pub fn shapes(&self) -> usize {
        self.shapes
    }

    pub fn strides(&self) -> usize {
        self.strides
    }

    pub fn data(&self) -> usize {
        self.data
    }
}

#[repr(C, align(8))]
#[derive(Debug)]
pub struct TensorHeader {
    dt: TensorDT,
    ndims: u8,
    pub is_ready: AtomicU8,
}

impl Clone for TensorHeader {
    fn clone(&self) -> Self {
        Self {
            dt: self.dt,
            ndims: self.ndims,
            is_ready: self.is_ready.load(Ordering::Relaxed).into(),
        }
    }
}

impl TensorHeader {
    const DATA_ALIGNMENT: usize = 8;

    pub const fn dt_offset() -> usize {
        offset_of!(Self, dt)
    }

    pub const fn ndims_offset() -> usize {
        offset_of!(Self, ndims)
    }

    pub const fn is_ready_offset() -> usize {
        offset_of!(Self, is_ready)
    }

    pub fn new(dt: TensorDT, ndims: u8) -> Self {
        TensorHeader {
            dt,
            ndims,
            is_ready: 0.into(),
        }
    }

    pub fn dt(&self) -> TensorDT {
        self.dt
    }

    pub fn ndims(&self) -> u8 {
        self.ndims
    }

    #[inline]
    fn get_shape_strides_size(ndims: u8) -> usize {
        size_of::<ShapeType>() * ndims as usize
    }

    pub fn get_offsets(&self) -> TensorDOffsets {
        let ss_size = Self::get_shape_strides_size(self.ndims);
        let th_size = size_of::<Self>();

        let shapes_offset = th_size;
        let strides_offset = shapes_offset + ss_size;

        let data_offset =
            (strides_offset + ss_size + (Self::DATA_ALIGNMENT - 1)) & !(Self::DATA_ALIGNMENT - 1);

        TensorDOffsets::new(shapes_offset, strides_offset, data_offset)
    }
}
