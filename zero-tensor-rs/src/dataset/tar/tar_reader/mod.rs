use std::mem::offset_of;
use std::str;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TarHeader {
    pub name: [u8; 100],
    pub mode: [u8; 8],
    pub uid: [u8; 8],
    pub gid: [u8; 8],
    pub size: [u8; 12],
    pub mtime: [u8; 12],
    pub chksum: [u8; 8],
    pub typeflag: u8,
    pub linkname: [u8; 100],
    pub magic: [u8; 6],
    pub version: [u8; 2],
    pub uname: [u8; 32],
    pub gname: [u8; 32],
    pub devmajor: [u8; 8],
    pub devminor: [u8; 8],
    pub prefix: [u8; 155],
    _pad: [u8; 12],
}

const GNU_MAGIC: &[u8] = b"ustar ";
const POSIX_MAGIC: &[u8] = b"ustar\0";
const CHKSUM_LEFT: usize = offset_of!(TarHeader, chksum);
const CHKSUM_RIGHT: usize = CHKSUM_LEFT + size_of::<[u8; 8]>();

#[derive(Debug, PartialEq, Eq)]
pub enum TarType {
    Gnu,
    Posix,
    Unknown,
}

const _: () = assert!(std::mem::size_of::<TarHeader>() == 512);

impl TarHeader {
    fn parse_octal(bytes: &[u8]) -> Option<u64> {
        let end = bytes
            .iter()
            .position(|&x| x == 0 || x == b' ')
            .unwrap_or(bytes.len());

        let s = str::from_utf8(&bytes[..end]).ok()?;
        let s = s.trim_start();
        if s.is_empty() {
            return Some(0);
        }
        u64::from_str_radix(s, 8).ok()
    }

    fn parse_str(bytes: &[u8]) -> Option<&str> {
        let end = bytes.iter().position(|&x| x == 0).unwrap_or(bytes.len());
        str::from_utf8(&bytes[..end]).ok()
    }

    pub fn file_size(&self) -> Option<u64> {
        Self::parse_octal(&self.size)
    }

    pub fn file_mode(&self) -> Option<u64> {
        Self::parse_octal(&self.mode)
    }

    pub fn file_name(&self) -> Option<String> {
        let prefix = Self::parse_str(&self.prefix).unwrap_or("");
        let name = Self::parse_str(&self.name)?;
        if prefix.is_empty() {
            Some(name.to_string())
        } else {
            Some(format!("{prefix}/{name}"))
        }
    }

    pub fn get_type(&self) -> TarType {
        if &self.magic == GNU_MAGIC {
            TarType::Gnu
        } else if &self.magic == POSIX_MAGIC {
            TarType::Posix
        } else {
            TarType::Unknown
        }
    }

    pub fn is_regular_file(&self) -> bool {
        self.typeflag == 0 || self.typeflag == b'0'
    }

    pub fn is_directory(&self) -> bool {
        self.typeflag == b'5'
    }

    pub fn checksum_valid(&self) -> bool {
        let stored = Self::parse_octal(&self.chksum).unwrap_or(0);
        let (unsigned_sum, signed_sum) = self.compute_checksums();
        stored == unsigned_sum || stored == signed_sum
    }

    fn as_bytes(&self) -> &[u8; 512] {
        let ptr = self as *const Self as *const [u8; 512];
        unsafe { &*ptr }
    }

    fn compute_checksums(&self) -> (u64, u64) {
        let header_bytes = self.as_bytes();
        let mut unsigned_sum: u64 = 0;
        let mut signed_sum: i64 = 0;

        for (i, &byte) in header_bytes.iter().enumerate() {
            if i >= CHKSUM_LEFT && i < CHKSUM_RIGHT {
                unsigned_sum += b' ' as u64;
                signed_sum += b' ' as i64;
            } else {
                unsigned_sum += byte as u64;
                signed_sum += byte as i8 as i64;
            }
        }

        (unsigned_sum, signed_sum as u64)
    }
}
