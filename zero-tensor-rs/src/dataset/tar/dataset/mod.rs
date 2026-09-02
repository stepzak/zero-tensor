pub mod item;
use std::marker::PhantomData;

pub use item::*;

pub struct TarDataset<'data, I: TarDatasetItem<'data>> {
    _marker: PhantomData<I>
}