use std::sync::atomic::{AtomicBool, AtomicU64};

use crossbeam_utils::CachePadded;
use thiserror::Error;

#[repr(C, align(64))]
pub struct ControlBlock {
    pub head: CachePadded<AtomicU64>,
    pub tail: CachePadded<AtomicU64>,
    nslots: u64,
    slot_size: u64,
    is_running: AtomicBool
}

#[derive(Debug, Error)]
pub enum ZTControlBlockError {
    #[error("nslots must be > 0")]
    InvalidNslots,
    
    #[error("slot_size must be > 0")]
    InvalidSlotSize,
}

impl ControlBlock {
    pub fn new(nslots: u64, slot_size: u64) -> Result<Self, ZTControlBlockError> {
        if nslots == 0 {
            return Err(ZTControlBlockError::InvalidNslots);
        }
        if slot_size == 0 {
            return Err(ZTControlBlockError::InvalidSlotSize);
        }

        Ok(Self { head: CachePadded::new(AtomicU64::new(0)), tail: CachePadded::new(AtomicU64::new(0)), nslots, slot_size, is_running: AtomicBool::new(true) })
    }
}