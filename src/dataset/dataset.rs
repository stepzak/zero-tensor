use super::item::TensorItemMeta;

pub type TensorBytes = Vec<u8>;

pub trait ZeroTensorDataset: Send + Sync {
    fn len(&self) -> usize;

    fn get_item(&self, index: usize) -> Option<(TensorBytes, TensorItemMeta)>;

    ///Writes bytes of item of index {index} into the {tensor_buffer} and return (tensor meta, how many bytes are really written)
    fn get_item_into(
        &self,
        index: usize,
        tensor_buffer: &mut [u8],
    ) -> Option<(TensorItemMeta, usize)>;
}
