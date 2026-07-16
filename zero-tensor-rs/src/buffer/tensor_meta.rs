use crate::dataset::item::TensorDT;

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
#[derive(Clone, Copy)]
pub struct TensorHeader {
    dt: TensorDT,
    ndims: u8,
}

impl TensorHeader {
    pub fn new(dt: TensorDT, ndims: u8) -> Self {
        TensorHeader { dt, ndims }
    }

    pub fn dt(&self) -> TensorDT {
        self.dt
    }

    pub fn ndims(&self) -> u8 {
        self.ndims
    }

    #[inline]
    fn get_shape_strides_size(ndims: u8) -> usize {
        size_of::<u32>() * ndims as usize
    }

    pub fn get_offsets(&self) -> TensorDOffsets {
        let ss_size = Self::get_shape_strides_size(self.ndims);
        let th_size = size_of::<Self>();

        let shapes_offset = th_size;
        let strides_offset = shapes_offset + ss_size;

        let data_offset = (strides_offset + ss_size + 7) & !7;

        TensorDOffsets::new(shapes_offset, strides_offset, data_offset)
    }
}
