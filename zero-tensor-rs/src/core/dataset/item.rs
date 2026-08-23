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

impl<'a> TensorViewMut<'a> {
    pub fn total_bytes(&self) -> usize {
        match self {
            TensorViewMut::F16(v) => v.len() * size_of::<half::f16>(),
            TensorViewMut::F32(v) => v.len() * size_of::<f32>(),
            TensorViewMut::F64(v) => v.len() * size_of::<f64>(),
            TensorViewMut::BF16(v) => v.len() * size_of::<half::bf16>(),
            TensorViewMut::I8(v) => v.len() * size_of::<i8>(),
            TensorViewMut::I32(v) => v.len() * size_of::<i32>(),
            TensorViewMut::I64(v) => v.len() * size_of::<i64>(),
            TensorViewMut::U8(v) => v.len() * size_of::<u8>(),
        }
    }
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
        self.shape.iter().product::<ShapeType>()
    }

    pub fn total_bytes(&self) -> usize {
        self.total_elements() * get_dt_size(self.dt)
    }

    fn try_view_mut_inner<'a>(
        &self,
        raw_bytes: &'a mut [u8],
        offset: usize,
    ) -> Result<TensorViewMut<'a>, TensorViewError> {
        let layout = IxDyn(&self.shape[offset..]).strides(IxDyn(&self.strides[offset..]));

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

    pub fn try_view_mut<'a>(
        &self,
        raw_bytes: &'a mut [u8],
    ) -> Result<TensorViewMut<'a>, TensorViewError> {
        self.try_view_mut_inner(raw_bytes, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::cast_slice_mut;

    #[test]
    fn view_contiguous_1d() {
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0];

        let raw_bytes = cast_slice_mut(&mut data);

        let layout = TensorBatchLayout::new(vec![4].into(), vec![1].into(), TensorDT::F32);

        let mut view = layout.try_view_mut(raw_bytes).unwrap();

        match &mut view {
            TensorViewMut::F32(v) => {
                assert_eq!(v[[0]], 1.0);
                assert_eq!(v[[1]], 2.0);
                assert_eq!(v[[2]], 3.0);
                assert_eq!(v[[3]], 4.0);

                v[[2]] = 42.0;
            }
            _ => panic!("expected F32 view"),
        }

        assert_eq!(data, vec![1.0, 2.0, 42.0, 4.0]);
    }

    #[test]
    fn view_contiguous_2d() {
        let mut data = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];

        let raw_bytes = cast_slice_mut(&mut data);

        let layout = TensorBatchLayout::new(vec![2, 3].into(), vec![3, 1].into(), TensorDT::F32);

        let mut view = layout.try_view_mut(raw_bytes).unwrap();

        match &mut view {
            TensorViewMut::F32(v) => {
                assert_eq!(v[[0, 0]], 0.0);
                assert_eq!(v[[0, 1]], 1.0);
                assert_eq!(v[[0, 2]], 2.0);

                assert_eq!(v[[1, 0]], 3.0);
                assert_eq!(v[[1, 1]], 4.0);
                assert_eq!(v[[1, 2]], 5.0);

                v[[1, 2]] = 123.0;
            }
            _ => panic!("expected F32 view"),
        }

        assert_eq!(data[5], 123.0);
    }

    #[test]
    fn view_strided() {
        let mut data = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];

        let raw_bytes = cast_slice_mut(&mut data);

        let layout = TensorBatchLayout::new(vec![2, 2].into(), vec![3, 1].into(), TensorDT::F32);

        let mut view = layout.try_view_mut(raw_bytes).unwrap();

        match &mut view {
            TensorViewMut::F32(v) => {
                assert_eq!(v[[0, 0]], 0.0);
                assert_eq!(v[[0, 1]], 1.0);

                assert_eq!(v[[1, 0]], 3.0);
                assert_eq!(v[[1, 1]], 4.0);

                v[[1, 0]] = 42.0;
            }
            _ => panic!("expected F32 view"),
        }

        assert_eq!(data, vec![0.0, 1.0, 2.0, 42.0, 4.0, 5.0]);
    }

    #[test]
    fn view_3d_contiguous() {
        let mut data: Vec<f32> = (0..24).map(|x| x as f32).collect();

        let raw_bytes = cast_slice_mut(&mut data);

        let layout =
            TensorBatchLayout::new(vec![2, 3, 4].into(), vec![12, 4, 1].into(), TensorDT::F32);

        let mut view = layout.try_view_mut(raw_bytes).unwrap();

        match &mut view {
            TensorViewMut::F32(v) => {
                assert_eq!(v[[0, 0, 0]], 0.0);
                assert_eq!(v[[0, 0, 3]], 3.0);

                assert_eq!(v[[0, 1, 0]], 4.0);
                assert_eq!(v[[0, 2, 0]], 8.0);

                assert_eq!(v[[1, 0, 0]], 12.0);
                assert_eq!(v[[1, 2, 3]], 23.0);

                v[[1, 2, 3]] = 999.0;
            }
            _ => panic!("expected F32 view"),
        }

        assert_eq!(data[23], 999.0);
    }

    #[test]
    fn view_mutation_is_zero_copy() {
        let mut data = vec![1.0f32, 2.0, 3.0];

        let raw_bytes = cast_slice_mut(&mut data);

        let layout = TensorBatchLayout::new(vec![3].into(), vec![1].into(), TensorDT::F32);

        let mut view = layout.try_view_mut(raw_bytes).unwrap();

        match &mut view {
            TensorViewMut::F32(v) => {
                v[[0]] = 100.0;
                v[[1]] = 200.0;
                v[[2]] = 300.0;
            }
            _ => panic!("expected F32 view"),
        }

        assert_eq!(data, vec![100.0, 200.0, 300.0]);
    }

    #[rstest::rstest]
    #[case(TensorDT::U8, 1)]
    #[case(TensorDT::I8, 1)]
    #[case(TensorDT::I32, 4)]
    #[case(TensorDT::I64, 8)]
    #[case(TensorDT::F16, 2)]
    #[case(TensorDT::BF16, 2)]
    #[case(TensorDT::F32, 4)]
    #[case(TensorDT::F64, 8)]
    fn view_correct_dtype(#[case] dt: TensorDT, #[case] element_size: usize) {
        let element_count = 4;
        let mut raw_bytes = vec![0u8; element_count * element_size];

        let layout = TensorBatchLayout::new(vec![element_count].into(), vec![1].into(), dt);

        let view = layout.try_view_mut(&mut raw_bytes).unwrap();

        match (dt, view) {
            (TensorDT::U8, TensorViewMut::U8(_))
            | (TensorDT::I8, TensorViewMut::I8(_))
            | (TensorDT::I32, TensorViewMut::I32(_))
            | (TensorDT::I64, TensorViewMut::I64(_))
            | (TensorDT::F16, TensorViewMut::F16(_))
            | (TensorDT::BF16, TensorViewMut::BF16(_))
            | (TensorDT::F32, TensorViewMut::F32(_))
            | (TensorDT::F64, TensorViewMut::F64(_)) => {}

            _ => panic!("wrong TensorViewMut variant for {:?}", dt),
        }
    }

    #[test]
    fn view_rejects_wrong_byte_length() {
        let mut raw_bytes = vec![0u8; 3];

        let layout = TensorBatchLayout::new(vec![1].into(), vec![1].into(), TensorDT::F32);

        assert!(layout.try_view_mut(&mut raw_bytes).is_err());
    }

    #[test]
    fn view_rejects_too_small_buffer() {
        let mut raw_bytes = vec![0u8; 8];

        let layout = TensorBatchLayout::new(vec![4].into(), vec![1].into(), TensorDT::F32);

        assert!(layout.try_view_mut(&mut raw_bytes).is_err());
    }

    #[test]
    fn view_nchw_layout() {
        let mut data: Vec<f32> = (0..24).map(|x| x as f32).collect();

        let raw_bytes = bytemuck::cast_slice_mut(&mut data);

        let layout = TensorBatchLayout::new(
            vec![2, 3, 2, 2].into(),
            vec![12, 4, 2, 1].into(),
            TensorDT::F32,
        );

        let mut view = layout.try_view_mut(raw_bytes).unwrap();

        match &mut view {
            TensorViewMut::F32(v) => {
                assert_eq!(v[[0, 0, 0, 0]], 0.0);
                assert_eq!(v[[0, 0, 0, 1]], 1.0);
                assert_eq!(v[[0, 0, 1, 0]], 2.0);

                assert_eq!(v[[0, 1, 0, 0]], 4.0);
                assert_eq!(v[[1, 0, 0, 0]], 12.0);
                assert_eq!(v[[1, 2, 1, 1]], 23.0);
            }
            _ => panic!("expected F32 view"),
        }
    }
}
