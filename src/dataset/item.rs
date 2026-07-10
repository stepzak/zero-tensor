#[repr(u8)]
#[derive(Clone, Copy, std::fmt::Debug)]
pub enum TensorDT {
    F16,
    F32,
    F64,
    BF16,
    I8,
    I32,
    I64,
    B
}

#[derive(Debug, Clone)]
pub struct TensorItemMeta {
    shape: Vec<u32>,
    dt: TensorDT
}

impl TensorItemMeta {
    pub fn new(shape: Vec<u32>, dt: TensorDT) -> Self {
        TensorItemMeta { shape, dt }
    }

    pub fn shape(&self) -> &Vec<u32> {
        &self.shape
    }

    pub fn dt(&self) -> TensorDT {
        self.dt
    }
}