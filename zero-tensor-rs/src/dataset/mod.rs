pub mod item;

use item::TensorItemMeta;

pub type TensorBytes = Vec<u8>;

pub trait ZeroTensorDataset: Send + Sync {
    fn len(&self) -> usize;

    fn get_item_into(&self, idx: usize, buf: &mut [u8]) -> Option<TensorItemMeta>;

    fn get_metadata(&self, idx: usize) -> Option<TensorItemMeta>;

    fn is_empty(&self) -> bool;
}
