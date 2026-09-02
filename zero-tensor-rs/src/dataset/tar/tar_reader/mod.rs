pub mod error;
pub(super) mod header;
pub use error::TarReaderError;
use std::{fs::File, path::Path};

pub(super) use header::*;
use memmap2::Mmap;

pub const MAX_PATH_LEN: usize = 260;

#[derive(Debug)]
pub struct TarRecord<'mmap, 'buf> {
    pub header: &'mmap TarHeader,
    pub name: &'buf str,
    pub data: &'mmap [u8],
}

pub struct TarReader {
    mmap: Mmap,
    offset: usize,
    eof: bool,
}

impl TarReader {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        mmap.advise(memmap2::Advice::Sequential)?;

        Ok(Self {
            mmap,
            offset: 0,
            eof: false,
        })
    }

    pub fn open_file<P: AsRef<Path>>(&mut self, path: P) -> Result<(), std::io::Error> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        mmap.advise(memmap2::Advice::Sequential)?;
        self.reset();
        Ok(())
    }

    pub fn reset(&mut self) {
        self.offset = 0;
        self.eof = false;
    }

    #[allow(dead_code)]
    pub fn is_eof(&self) -> bool {
        self.eof || self.offset >= self.mmap.len()
    }

    pub fn next_record<'mmap, 'buf>(
        &'mmap mut self,
        name_buf: &'buf mut [u8; MAX_PATH_LEN],
    ) -> Result<TarRecord<'mmap, 'buf>, TarReaderError>
    where
        'mmap: 'buf,
    {
        if self.eof {
            return Err(TarReaderError::EOF);
        }
        let mut gnu_long_name: Option<&'mmap str> = None;

        loop {
            let needed = self.offset + TAR_HEADER_SIZE;
            if needed > self.mmap.len() {
                return Err(TarReaderError::Overflow {
                    needed,
                    got: self.mmap.len(),
                });
            }

            let header = unsafe {
                TarHeader::from_mmap(self.mmap.as_ptr(), self.offset)
                    .ok_or(TarReaderError::HeaderError)?
            };

            if header.name.iter().all(|&x| x == 0) && header.typeflag == 0 {
                self.eof = true;
                return Err(TarReaderError::EOF);
            }

            let file_size = header.file_size().ok_or(TarReaderError::HeaderError)? as usize;
            let padding = file_size.next_multiple_of(TAR_HEADER_SIZE) - file_size;

            self.offset += TAR_HEADER_SIZE;

            if self.offset + file_size > self.mmap.len() {
                return Err(TarReaderError::Overflow {
                    needed: self.offset + file_size,
                    got: self.mmap.len(),
                });
            }

            if header.typeflag == b'L' {
                let name_bytes = &self.mmap[self.offset..self.offset + file_size];
                let end = name_bytes
                    .iter()
                    .position(|&x| x == 0)
                    .unwrap_or(name_bytes.len());
                if let Ok(s) = std::str::from_utf8(&name_bytes[..end]) {
                    gnu_long_name = Some(s);
                }

                self.offset += file_size + padding;
                continue;
            }

            if !header.is_regular_file() {
                self.offset += file_size + padding;
                continue;
            }

            let data = &self.mmap[self.offset..self.offset + file_size];

            self.offset += file_size + padding;

            let name: &'buf str = match gnu_long_name {
                Some(long_name) => long_name,
                None => header
                    .file_name_into(name_buf)
                    .ok_or(TarReaderError::HeaderError)?,
            };

            return Ok(TarRecord { header, name, data });
        }
    }
}

#[cfg(test)]
mod tests;
