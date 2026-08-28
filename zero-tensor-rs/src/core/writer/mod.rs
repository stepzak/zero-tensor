pub mod cache;
pub mod error;

use super::dataset::item::TensorBatchLayout;
use super::helpers::align_to;
use indexmap::IndexMap;

pub use cache::TensorWriterCache;
pub use error::*;

pub struct TensorWriter<'a, 'b, 'c> {
    slot_buffer: &'c mut [u8],
    cache: &'b mut TensorWriterCache<'a>,
}

impl<'a, 'b, 'c> TensorWriter<'a, 'b, 'c> {
    pub const ALIGNMENT: usize = 64;

    pub fn new(
        layouts: &IndexMap<&'a str, TensorBatchLayout>,
        slot_buffer: &'c mut [u8],
        cache: &'b mut TensorWriterCache<'a>,
    ) -> Result<Self, TensorWriterError> {
        cache.clear();

        let mut acc = 0;

        for (&k, v) in layouts {
            let size = align_to(v.total_bytes(), Self::ALIGNMENT);
            cache.insert(k, acc, size);
            acc += size;
        }

        if acc > slot_buffer.len() {
            return Err(TensorWriterError::BufferTooSmall {
                required: acc,
                available: slot_buffer.len(),
            });
        }

        Ok(TensorWriter { cache, slot_buffer })
    }

    pub fn get_offset_size(&self, key: &str) -> Option<(usize, usize)> {
        self.cache.get_offset_size(key)
    }

    pub fn write<F, E>(
        &mut self,
        key: &'a str,
        write_fn: F,
    ) -> Result<usize, TensorWriteError<'a, E>>
    where
        F: FnOnce(&mut [u8]) -> Result<usize, E>,
        E: std::error::Error,
    {
        let (offset, size) = self
            .cache
            .get_offset_size(key)
            .ok_or(TensorWriteError::UnknownKey(key))?;
        let idx = self.cache.slot_buffers().get_key_pos(key).unwrap();
        if self.cache.written()[idx] {
            return Err(TensorWriteError::KeyExists(key));
        }

        if offset + size > self.slot_buffer.len() {
            return Err(TensorWriteError::BufferOutOfBounds {
                key,
                offset: offset + size,
                total_size: self.slot_buffer.len(),
            });
        }

        let buf = &mut self.slot_buffer[offset..offset + size];

        let written = write_fn(buf).map_err(|e| TensorWriteError::DatasetError { source: e })?;
        if written > size {
            return Err(TensorWriteError::BufferOutOfBounds {
                key,
                offset: written,
                total_size: size,
            });
        }

        if written < size {
            buf[written..].fill(0);
        }

        self.cache.mark_written(key);
        Ok(written)
    }

    pub fn finalize(&self) -> Result<(), TensorWriterError> {
        if self.cache.is_fully_written() {
            return Ok(());
        }

        let missing = self.cache.get_missing_keys();

        if !missing.is_empty() {
            return Err(TensorWriterError::MissingKeys(missing));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests;
