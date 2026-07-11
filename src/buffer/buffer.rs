use std::{ffi::{self, CString, c_int, c_void}, ptr};
use thiserror::Error;

use libc::{mode_t, shm_open};

use crate::{buffer::tensor_meta::TensorHeader, dataset::item::TensorDT};

pub type ShapeType = u32;
pub type StrideType = u32;

pub struct ZeroTensorBuffer {
    addr: *mut u8,
    total_size: usize,
    fd: i32
}

#[derive(Error, Debug)]
pub enum ZTBufErr {
    #[error("{0}")]
    InvalidFilename(&'static str),

    #[error("shm_open failed and returned {0}")]
    ShmOpenFail(i32),

    #[error("ftruncate failed and returned {0}")]
    FtruncateFail(i32),

    #[error("mmap failed")]
    MmapFail,    
}

#[inline]
pub fn get_dt_size(dt: TensorDT) -> usize {
    match dt {
        TensorDT::B => size_of::<u8>(),
        TensorDT::BF16 => size_of::<i16>(),
        TensorDT::F16 => size_of::<i16>(),
        TensorDT::F32 => size_of::<f32>(),
        TensorDT::F64 => size_of::<f64>(),
        TensorDT::I32 => size_of::<i32>(),
        TensorDT::I64 => size_of::<i64>(),
        TensorDT::I8 => size_of::<i8>(),    
    }
}

impl ZeroTensorBuffer {
    fn open_shm(file_name: CString, oflag: c_int, mode: mode_t) -> Result<i32, ZTBufErr> {
        unsafe {
            let fd = shm_open(
                file_name.as_ptr(), 
                oflag, 
                mode);
            if fd < 0 {
                return Err(ZTBufErr::ShmOpenFail(fd));
            }
            return Ok(fd);
        }
    }

    fn ftrunc(fd: i32, length: i64) -> Result<i32, ZTBufErr> {
        let res = unsafe { libc::ftruncate(fd, length) };
        if res < 0 {
            unsafe { libc::close(fd) };
            return Err(ZTBufErr::FtruncateFail(res));
        }
        return Ok(res);
    }

    fn mmap(fd: i32, len: usize, prot: i32, flags: i32) -> Result<*mut c_void, ZTBufErr> {
        let addr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                len,
                prot,
                flags,
                fd,
                0
            )
        };
        if addr == libc::MAP_FAILED {
            return Err(ZTBufErr::MmapFail);
        }
        return Ok(addr);
    }

    pub fn new(name: &str, total_size: usize) -> Result<Self, ZTBufErr> {
        let fname = if name.starts_with('/') {
            name.to_string()
        } else {
            format!("/{}", name)
        };

        if fname[1..].contains('/') {
            return Err(ZTBufErr::InvalidFilename("name must not contain inner slashes"));
        }

        let cname = ffi::CString::new(fname).map_err(|_| {
           ZTBufErr::InvalidFilename("name contains internal zero byte")
        })?;

        let oflag = libc::O_CREAT | libc::O_RDWR;
        let mode = 0o666;

        let fd = Self::open_shm(cname, oflag, mode)?;
        let _ = Self::ftrunc(fd, total_size as i64 )?;
        let prot = libc::PROT_READ | libc::PROT_WRITE;
        let flags = libc::MAP_SHARED;
        let addr = Self::mmap(fd, total_size, prot, flags)? as *mut u8;

        Ok(
            ZeroTensorBuffer { addr, total_size, fd }
        )
    }


    ///Strides must be in bytes!
    pub fn write_tensor(&mut self, offset: usize, shape: &[ShapeType], strides: &[StrideType], dt: TensorDT, raw_data: &[u8]) {
        let ndims = shape.len() as u8;
        if strides.len() as u8 != ndims { todo!(); }
        let meta = TensorHeader::new(dt, ndims);
        let base = unsafe { self.addr.add(offset) };
        let offs = meta.get_offsets();

        let data_count: u32 = shape.iter().product();
        let data_size = get_dt_size(dt) * data_count as usize;
        assert!(offset + offs.data() + data_size <= self.total_size, "Buffer overflow");

        let header_ptr = base as *mut TensorHeader;
        unsafe { header_ptr.write(meta) };

        let shape_ptr = unsafe { base.add(offs.shapes()) as *mut ShapeType};
        unsafe { ptr::copy_nonoverlapping(shape.as_ptr(), shape_ptr, ndims as usize) };

        let strides_ptr = unsafe { base.add(offs.strides()) as *mut StrideType};
        unsafe { ptr::copy_nonoverlapping(strides.as_ptr(), strides_ptr, ndims as usize); }
        
        let data_ptr = unsafe { base.add(offs.data())};
        unsafe { ptr::copy_nonoverlapping(raw_data.as_ptr(), data_ptr, data_size);}
    }

    /// # Safety
    /// If (addr + slot_offset) is being read, the result might lead to Race Condition
    pub unsafe fn get_item_slice_mut(
        &mut self,
        slot_offset: usize,
        data_offset_in_slot: usize,
        len: usize
    ) -> &mut [u8] {
        assert!(slot_offset + data_offset_in_slot + len <= self.total_size, "Slice out of bounds");
        let ptr = unsafe { self.addr.add(slot_offset).add(data_offset_in_slot) };
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }
}

impl Drop for ZeroTensorBuffer {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.addr as *mut c_void, self.total_size);  
            libc::close(self.fd);  
        }
    }
}