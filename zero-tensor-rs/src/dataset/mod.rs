pub mod item;

use item::TensorItemMeta;

pub type TensorBytes = Vec<u8>;

pub trait ZeroTensorDataset: Send + Sync {
    fn len(&self) -> usize;

    fn get_item(&self, index: usize) -> Option<(TensorBytes, TensorItemMeta)>;

    fn is_empty(&self) -> bool;
}
