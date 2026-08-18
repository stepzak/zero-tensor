use smallvec::SmallVec;

use super::super::buffer::get_dt_size;
use ndarray::{ArrayViewMutD, IxDyn, ShapeBuilder, ShapeError};

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

#[derive(Debug, thiserror::Error)]
pub enum TensorViewError {
    #[error("invalid byte buffer: {0}")]
    Cast(#[from(no_source)] bytemuck::PodCastError),

    #[error("invalid tensor shape: {0}")]
    Shape(#[from] ShapeError),
}

pub enum TensorViewMut<'a> {
    F16(ArrayViewMutD<'a, half::f16>),
    F32(ArrayViewMutD<'a, f32>),
    F64(ArrayViewMutD<'a, f64>),
    BF16(ArrayViewMutD<'a, half::bf16>),
    I8(ArrayViewMutD<'a, i8>),
    I32(ArrayViewMutD<'a, i32>),
    I64(ArrayViewMutD<'a, i64>),
    U8(ArrayViewMutD<'a, u8>),
}

pub type ShapeType = usize;
pub type StrideType = usize;

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

    pub fn try_view_mut<'a>(
        &self,
        raw_bytes: &'a mut [u8],
    ) -> Result<TensorViewMut<'a>, TensorViewError> {
        let layout = IxDyn(&self.shape).strides(IxDyn(self.strides()));

        match self.dt {
            TensorDT::I8 => {
                let typed_slice: &mut [i8] =
                    bytemuck::try_cast_slice_mut(raw_bytes).map_err(TensorViewError::Cast)?;
                let view = ArrayViewMutD::from_shape(layout, typed_slice)?;
                Ok(TensorViewMut::I8(view))
            }
            TensorDT::BF16 => {
                let typed_slice: &mut [half::bf16] =
                    bytemuck::try_cast_slice_mut(raw_bytes).map_err(TensorViewError::Cast)?;
                let view = ArrayViewMutD::from_shape(layout, typed_slice)?;
                Ok(TensorViewMut::BF16(view))
            }
            TensorDT::F32 => {
                let typed_slice: &mut [f32] =
                    bytemuck::try_cast_slice_mut(raw_bytes).map_err(TensorViewError::Cast)?;
                let view = ArrayViewMutD::from_shape(layout, typed_slice)?;
                Ok(TensorViewMut::F32(view))
            }
            TensorDT::F64 => {
                let typed_slice: &mut [f64] =
                    bytemuck::try_cast_slice_mut(raw_bytes).map_err(TensorViewError::Cast)?;
                let view = ArrayViewMutD::from_shape(layout, typed_slice)?;
                Ok(TensorViewMut::F64(view))
            }
            TensorDT::I32 => {
                let typed_slice: &mut [i32] =
                    bytemuck::try_cast_slice_mut(raw_bytes).map_err(TensorViewError::Cast)?;
                let view = ArrayViewMutD::from_shape(layout, typed_slice)?;
                Ok(TensorViewMut::I32(view))
            }
            TensorDT::I64 => {
                let typed_slice: &mut [i64] =
                    bytemuck::try_cast_slice_mut(raw_bytes).map_err(TensorViewError::Cast)?;
                let view = ArrayViewMutD::from_shape(layout, typed_slice)?;
                Ok(TensorViewMut::I64(view))
            }
            TensorDT::U8 => {
                let typed_slice = raw_bytes;
                let view = ArrayViewMutD::from_shape(layout, typed_slice)?;
                Ok(TensorViewMut::U8(view))
            }
            TensorDT::F16 => {
                let typed_slice: &mut [half::f16] =
                    bytemuck::try_cast_slice_mut(raw_bytes).map_err(TensorViewError::Cast)?;
                let view = ArrayViewMutD::from_shape(layout, typed_slice)?;
                Ok(TensorViewMut::F16(view))
            }
        }
    }
}
