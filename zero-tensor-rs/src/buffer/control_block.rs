use std::{
    mem::offset_of,
    sync::atomic::{AtomicU64, Ordering},
};

use crossbeam_utils::CachePadded;
use thiserror::Error;

#[repr(C, align(8))]
pub struct ZeroTensorControlBlock {
    pub head: CachePadded<AtomicU64>,
    pub tail: CachePadded<AtomicU64>,
    nslots: usize,
    slot_size: u32,
    is_running: AtomicU64,
}

#[derive(Debug, Error)]
pub enum ZTControlBlockError {
    #[error("nslots must be > 0")]
    InvalidNslots,

    #[error("slot_size must be > 0")]
    InvalidSlotSize,

    #[error("Invalid slot_size alignment. Got: {got} expected: {expected}")]
    InvalidSlotSizeAlignment { got: usize, expected: usize },
}

impl ZeroTensorControlBlock {
    pub const SIZE: usize = size_of::<Self>();
    pub const ALIGN: usize = align_of::<Self>();

    pub fn new(nslots: usize, slot_size: u32) -> Result<Self, ZTControlBlockError> {
        if nslots == 0 {
            return Err(ZTControlBlockError::InvalidNslots);
        }
        Self::validate_slot_size(slot_size as usize)?;

        Ok(Self {
            head: CachePadded::new(AtomicU64::new(0)),
            tail: CachePadded::new(AtomicU64::new(0)),
            nslots,
            slot_size,
            is_running: AtomicU64::new(1),
        })
    }

    fn warn_false_sharing(slot_size: usize, recommended: usize) {
        eprintln!(
            "[ZeroTensor] Warning: slot_size={} is not aligned to {} bytes. \
            Adjacent slots may share cache lines, causing false sharing. \
            Consider rounding up to {} bytes for optimal performance.",
            slot_size,
            recommended,
            (slot_size + recommended - 1) & !(recommended - 1)
        );
    }

    pub fn min_slot_alignment() -> usize {
        align_of::<AtomicU64>()
    }

    pub fn recommended_slot_alignment() -> usize {
        align_of::<CachePadded<AtomicU64>>()
    }

    pub fn slot_offset(slot_idx: usize, slot_size: usize) -> usize {
        Self::SIZE + slot_idx * slot_size
    }

    pub fn validate_slot_size(slot_size: usize) -> Result<(), ZTControlBlockError> {
        let min_align = Self::min_slot_alignment();
        let rec = Self::recommended_slot_alignment();
        if slot_size == 0 {
            return Err(ZTControlBlockError::InvalidSlotSize);
        }
        if !slot_size.is_multiple_of(min_align) {
            return Err(ZTControlBlockError::InvalidSlotSizeAlignment {
                got: slot_size,
                expected: min_align,
            });
        }
        if !slot_size.is_multiple_of(rec) {
            Self::warn_false_sharing(slot_size, rec);
        }

        Ok(())
    }

    pub fn nslots(&self) -> usize {
        self.nslots
    }

    pub fn slot_size(&self) -> u32 {
        self.slot_size
    }

    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire) == 1
    }

    pub fn stop(&mut self) {
        self.is_running.store(0, Ordering::Release)
    }

    pub fn nslots_offset() -> usize {
        offset_of!(Self, nslots)
    }

    pub fn slot_size_offset() -> usize {
        offset_of!(Self, slot_size)
    }

    pub fn is_running_offset() -> usize {
        offset_of!(Self, is_running)
    }

    pub fn is_running_size(&self) -> usize {
        size_of_val(&self.is_running)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_nslots() {
        let nslots = 0;
        let slot_size = 8;
        assert!(matches!(
            ZeroTensorControlBlock::new(nslots, slot_size),
            Err(ZTControlBlockError::InvalidNslots)
        ))
    }

    #[test]
    fn test_invalid_slot_size() {
        let nslots = 1;
        let slot_size = 0;
        assert!(matches!(
            ZeroTensorControlBlock::new(nslots, slot_size),
            Err(ZTControlBlockError::InvalidSlotSize)
        ))
    }

    #[test]
    fn test_invalid_align() {
        let slots = 1;
        let min_slot_size = ZeroTensorControlBlock::min_slot_alignment();
        let res = ZeroTensorControlBlock::new(slots, min_slot_size as u32 - 1);
        match res {
            Err(ZTControlBlockError::InvalidSlotSizeAlignment { got, expected }) => {
                assert_eq!(got, min_slot_size - 1);
                assert_eq!(expected, min_slot_size);
            }
            _ => {
                assert!(false)
            }
        }
    }

    #[test]
    fn test_ok() {
        let slots = 1;
        let slot_size = ZeroTensorControlBlock::recommended_slot_alignment();

        assert!(ZeroTensorControlBlock::new(slots, slot_size as u32).is_ok())
    }
}
