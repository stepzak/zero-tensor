use super::{Transform, error::TransformError};
use crate::{core::dataset::item::TensorViewMut};

pub struct Scale {
    value: f64
}

impl Scale {
    pub fn new(value: f64) -> Self {
        Scale { value }
    }
}

impl Transform for Scale {
    type Error = TransformError;

    fn apply(
        &self,
        tensor: &mut TensorViewMut
    ) -> Result<(), Self::Error>
    {
        match tensor {
            TensorViewMut::BF16(t) => {
                t.map_inplace(|x| *x *= half::bf16::from_f64(self.value));
            }
            TensorViewMut::F32(t) => {
                t.map_inplace(|x| *x *= self.value as f32);
            }
            TensorViewMut::F64(t) => {
                t.map_inplace(|x| *x *= self.value);
            }
            TensorViewMut::U8(_) => {
                return Err(TransformError::UnsupportedDtype);
            }
            TensorViewMut::F16(t) => {
                t.map_inplace(|x| *x *= half::f16::from_f64(self.value));
            }
            TensorViewMut::I8(_) => {
                return Err(TransformError::UnsupportedDtype);
            }
            TensorViewMut::I32(_) => {
                return Err(TransformError::UnsupportedDtype);
            }
            TensorViewMut::I64(_) => {
                return Err(TransformError::UnsupportedDtype);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::core::dataset::item::TensorBatchLayout;
    use super::*;

    #[test]
    fn scale_f32() {
        let mut data = vec![
            1.0f32,
            2.0,
            3.0,
            4.0,
        ];
        let l = data.len();
        let raw_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut data);

        let layout = TensorBatchLayout::new(vec![l].into(), vec![1].into(), crate::core::dataset::item::TensorDT::F32);
        let mut tensor = layout.try_view_mut(raw_bytes).unwrap();
        Scale::new(2.0)
            .apply(&mut tensor)
            .unwrap();

        assert_eq!(data, &[2.0, 4.0, 6.0, 8.0]);
    }

    //TODO: regression tests
}