pub mod error;

use indexmap::IndexMap;
use std::collections::HashSet;

use super::dataset::item::TensorBatchLayout;
use super::helpers::align_to;

pub use error::*;

type Offset = usize;
type Size = usize;

pub struct TensorWriter<'a> {
    slot_buffers: IndexMap<&'a str, (Offset, Size)>,
    slot_buffer: &'a mut [u8],
    written: HashSet<&'a str>,
}

impl<'a> TensorWriter<'a> {
    pub const ALIGNMENT: usize = 64;

    pub fn new(
        layouts: IndexMap<&'a str, TensorBatchLayout>,
        slot_buffer: &'a mut [u8],
    ) -> Result<Self, TensorWriterError<'a>> {
        let mut im = IndexMap::new();
        let mut acc = 0usize;

        for (k, v) in layouts {
            let size = align_to(v.total_bytes(), Self::ALIGNMENT);
            im.insert(k, (acc, size));
            acc += size;
        }

        if acc > slot_buffer.len() {
            return Err(TensorWriterError::BufferTooSmall {
                required: acc,
                available: slot_buffer.len(),
            });
        }

        let l = im.len();

        Ok(TensorWriter {
            slot_buffers: im,
            slot_buffer,
            written: HashSet::with_capacity(l),
        }) //TODO: think about smallvec if the keys are few
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
        if self.written.contains(key) {
            return Err(TensorWriteError::KeyExists(key));
        }

        let &(offset, size) = self
            .slot_buffers
            .get(key)
            .ok_or(TensorWriteError::UnknownKey(key))?;

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

        self.written.insert(key);
        Ok(written)
    }

    pub fn finalize(&self) -> Result<(), TensorWriterError<'a>> {
        if self.slot_buffers.len() == self.written.len() {
            return Ok(());
        }

        let missing: Vec<&str> = self
            .slot_buffers
            .keys()
            .copied()
            .filter(|k| !self.written.contains(k))
            .collect();

        if !missing.is_empty() {
            return Err(TensorWriterError::MissingKeys(missing));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests;
