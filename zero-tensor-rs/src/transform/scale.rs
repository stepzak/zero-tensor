use super::{Transform, error::TransformError};
use crate::core::dataset::item::TensorViewMut;

pub struct Scale {
    factor: f64,
}

impl Scale {
    pub fn new(factor: f64) -> Self {
        Scale { factor }
    }
}

impl Transform for Scale {
    type Error = TransformError;

    fn apply(&self, tensor: &mut TensorViewMut) -> Result<(), Self::Error> {
        match tensor {
            TensorViewMut::BF16(t) => {
                t.map_inplace(|x| *x *= half::bf16::from_f64(self.factor));
            }
            TensorViewMut::F32(t) => {
                t.map_inplace(|x| *x *= self.factor as f32);
            }
            TensorViewMut::F64(t) => {
                t.map_inplace(|x| *x *= self.factor);
            }
            TensorViewMut::U8(_) => {
                return Err(TransformError::UnsupportedDtype);
            }
            TensorViewMut::F16(t) => {
                t.map_inplace(|x| *x *= half::f16::from_f64(self.factor));
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
    use super::*;
    use crate::core::dataset::item::{TensorBatchLayout, TensorDT};
    use rstest::rstest;

    macro_rules! generate_bytes {
        ($ty:ty) => {{
            let data: Vec<$ty> = vec![0, 1];
            let l = data.len();
            let raw_bytes: Vec<u8> = bytemuck::pod_collect_to_vec(&data);
            (raw_bytes, l)
        }};
    }

    #[rstest]
    #[case(TensorDT::BF16)]
    #[case(TensorDT::F16)]
    #[case(TensorDT::F32)]
    #[case(TensorDT::F64)]
    fn scale(#[case] dt: TensorDT) {
        let factor = 2.0;
        let factor_f16 = half::f16::from_f64(factor);
        let default_vec = vec![half::f16::from_f64(1.0), half::f16::from_f64(2.0)];
        let doubled_vec: Vec<half::f16> = default_vec.iter().map(|x| *x * factor_f16).collect();
        let (mut raw_bytes, l) = match dt {
            TensorDT::BF16 => {
                let data: Vec<half::bf16> = default_vec
                    .iter()
                    .map(|x| half::bf16::from_f32((*x).to_f32()))
                    .collect();
                let l = data.len();
                let raw_bytes: Vec<u8> = bytemuck::pod_collect_to_vec(&data);
                (raw_bytes, l)
            }
            TensorDT::F16 => {
                let data = default_vec.clone();
                let l = data.len();
                let raw_bytes: Vec<u8> = bytemuck::pod_collect_to_vec(&data);
                (raw_bytes, l)
            }
            TensorDT::F32 => {
                let data: Vec<f32> = default_vec.iter().map(|x| x.to_f32()).collect();
                let l = data.len();
                let raw_bytes: Vec<u8> = bytemuck::pod_collect_to_vec(&data);
                (raw_bytes, l)
            }
            TensorDT::F64 => {
                let data: Vec<f64> = default_vec.iter().map(|x| x.to_f64()).collect();
                let l = data.len();
                let raw_bytes: Vec<u8> = bytemuck::pod_collect_to_vec(&data);
                (raw_bytes, l)
            }
            _ => unreachable!(),
        };

        let layout = TensorBatchLayout::new(vec![l].into(), vec![1].into(), dt);
        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();
        Scale::new(2.0).apply(&mut tensor).unwrap();

        match dt {
            TensorDT::BF16 => {
                let data_bf: Vec<half::bf16> = bytemuck::pod_collect_to_vec(&raw_bytes);
                let data: Vec<half::f16> = data_bf
                    .iter()
                    .map(|x| half::f16::from_f32(x.to_f32()))
                    .collect();
                assert_eq!(data, doubled_vec);
            }
            TensorDT::F16 => {
                let data: Vec<half::f16> = bytemuck::pod_collect_to_vec(&raw_bytes);
                assert_eq!(data, doubled_vec);
            }
            TensorDT::F32 => {
                let data_32: Vec<f32> = bytemuck::pod_collect_to_vec(&raw_bytes);
                let data: Vec<half::f16> =
                    data_32.iter().map(|x| half::f16::from_f32(*x)).collect();
                assert_eq!(data, doubled_vec);
            }
            TensorDT::F64 => {
                let data_64: Vec<f64> = bytemuck::pod_collect_to_vec(&raw_bytes);
                let data: Vec<half::f16> =
                    data_64.iter().map(|x| half::f16::from_f64(*x)).collect();
                assert_eq!(data, doubled_vec);
            }
            _ => unreachable!(),
        }
    }

    #[rstest]
    #[case(TensorDT::I32)]
    #[case(TensorDT::I8)]
    #[case(TensorDT::I64)]
    #[case(TensorDT::U8)]
    fn test_invalid_dt(#[case] dt: TensorDT) {
        let (mut raw_bytes, l) = match dt {
            TensorDT::I32 => generate_bytes!(i32),
            TensorDT::I8 => generate_bytes!(i8),
            TensorDT::I64 => generate_bytes!(i64),
            TensorDT::U8 => generate_bytes!(u8),
            _ => unreachable!(),
        };

        let raw_bytes_copy = raw_bytes.clone();

        let layout = TensorBatchLayout::new(vec![l].into(), vec![1].into(), dt);
        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();
        let res = Scale::new(2.0).apply(&mut tensor);
        assert!(matches!(res, Err(TransformError::UnsupportedDtype)));
        assert_eq!(raw_bytes, raw_bytes_copy);
    }
}
