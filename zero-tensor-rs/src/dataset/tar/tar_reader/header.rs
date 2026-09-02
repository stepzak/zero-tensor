use std::io::{Cursor, Write};
use std::mem::offset_of;
use std::str;

#[repr(C, align(512))]
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

pub const TAR_HEADER_SIZE: usize = 512;
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

const _: () = assert!(std::mem::size_of::<TarHeader>() == TAR_HEADER_SIZE);

impl TarHeader {
    /// # Safety
    /// offset must be a multiple of 512 and ptr must be a valid mmap
    pub unsafe fn from_mmap<'mmap>(ptr: *const u8, offset: usize) -> Option<&'mmap TarHeader> {
        if offset % TAR_HEADER_SIZE != 0 {
            return Self::cold_alignment_error();
        }
        unsafe { Some(&*(ptr.add(offset) as *const TarHeader)) }
    }

    #[cold]
    #[inline(never)]
    fn cold_alignment_error() -> Option<&'static TarHeader> {
        None
    }

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

    /// Warning: ignores `prefix`. Use only if filename length if less then 100
    pub fn name_ref(&self) -> Option<&str> {
        Self::parse_str(&self.name)
    }

    pub fn prefix_ref(&self) -> Option<&str> {
        Self::parse_str(&self.prefix)
    }

    pub fn file_name_into<'buf>(&self, buf: &'buf mut [u8; 260]) -> Option<&'buf str> {
        let name_str = self.name_ref()?;
        let prefix_ref = self.prefix_ref().unwrap_or("");
        let mut cursor = Cursor::new(&mut buf[..]);
        if prefix_ref.is_empty() {
            write!(cursor, "{}", name_str).ok()?;
        } else {
            write!(cursor, "{}/{}", prefix_ref, name_str).ok()?;
        }
        let len = cursor.position();
        str::from_utf8(&buf[..len as usize]).ok()
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

    pub fn as_bytes(&self) -> &[u8; TAR_HEADER_SIZE] {
        let ptr = self as *const Self as *const [u8; TAR_HEADER_SIZE];
        unsafe { &*ptr }
    }

    pub fn compute_checksums(&self) -> (u64, u64) {
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

#[cfg(test)]
mod tests {
    use super::*;
    #[repr(align(512))]
    struct AlignedBuffer {
        pub data: [u8; 1024],
    }

    fn create_test_header() -> TarHeader {
        let mut header = TarHeader {
            name: [0; 100],
            mode: [0; 8],
            uid: [0; 8],
            gid: [0; 8],
            size: [0; 12],
            mtime: [0; 12],
            chksum: [0; 8],
            typeflag: 0,
            linkname: [0; 100],
            magic: [0; 6],
            version: [0; 2],
            uname: [0; 32],
            gname: [0; 32],
            devmajor: [0; 8],
            devminor: [0; 8],
            prefix: [0; 155],
            _pad: [0; 12],
        };

        let name = b"test.txt\0";
        header.name[..name.len()].copy_from_slice(name);

        let size = b"00000002000\0";
        header.size[..size.len()].copy_from_slice(size);

        let mode = b"0000644\0";
        header.mode[..mode.len()].copy_from_slice(mode);

        header.magic.copy_from_slice(POSIX_MAGIC);

        header.typeflag = b'0';

        header
    }

    #[test]
    fn test_parse_octal_normal() {
        let bytes = b"0000644\0";
        assert_eq!(TarHeader::parse_octal(bytes), Some(0o644));
    }

    #[test]
    fn test_parse_octal_with_spaces() {
        let bytes = b"0000644 ";
        assert_eq!(TarHeader::parse_octal(bytes), Some(0o644));
    }

    #[test]
    fn test_parse_octal_zero() {
        let bytes = b"00000000000\0";
        assert_eq!(TarHeader::parse_octal(bytes), Some(0));
    }

    #[test]
    fn test_parse_octal_empty() {
        let bytes = b"\0\0\0\0\0\0\0\0\0\0\0\0";
        assert_eq!(TarHeader::parse_octal(bytes), Some(0));
    }

    #[test]
    fn test_parse_octal_invalid() {
        let bytes = b"invalid\0";
        assert_eq!(TarHeader::parse_octal(bytes), None);
    }

    #[test]
    fn test_parse_str_normal() {
        let bytes = b"test.txt\0";
        assert_eq!(TarHeader::parse_str(bytes), Some("test.txt"));
    }

    #[test]
    fn test_parse_str_no_null() {
        let bytes = b"test.txt";
        assert_eq!(TarHeader::parse_str(bytes), Some("test.txt"));
    }

    #[test]
    fn test_parse_str_empty() {
        let bytes = b"\0";
        assert_eq!(TarHeader::parse_str(bytes), Some(""));
    }

    #[test]
    fn test_parse_str_invalid_utf8() {
        let bytes = [0xFF, 0xFE, 0x00];
        assert_eq!(TarHeader::parse_str(&bytes), None);
    }

    #[test]
    fn test_file_name_without_prefix() {
        let header = create_test_header();
        assert_eq!(header.file_name(), Some("test.txt".to_string()));
    }

    #[test]
    fn test_file_name_with_prefix() {
        let mut header = create_test_header();
        let prefix = b"dir/subdir\0";
        header.prefix[..prefix.len()].copy_from_slice(prefix);
        assert_eq!(header.file_name(), Some("dir/subdir/test.txt".to_string()));
    }

    #[test]
    fn test_file_size() {
        let header = create_test_header();
        assert_eq!(header.file_size(), Some(0o2000));
    }

    #[test]
    fn test_file_mode() {
        let header = create_test_header();
        assert_eq!(header.file_mode(), Some(0o644));
    }

    #[test]
    fn test_get_type_posix() {
        let header = create_test_header();
        assert_eq!(header.get_type(), TarType::Posix);
    }

    #[test]
    fn test_get_type_gnu() {
        let mut header = create_test_header();
        header.magic.copy_from_slice(GNU_MAGIC);
        assert_eq!(header.get_type(), TarType::Gnu);
    }

    #[test]
    fn test_get_type_unknown() {
        let mut header = create_test_header();
        header.magic = [0; 6];
        assert_eq!(header.get_type(), TarType::Unknown);
    }

    #[test]
    fn test_is_regular_file() {
        let header = create_test_header();
        assert!(header.is_regular_file());
    }

    #[test]
    fn test_is_regular_file_null_typeflag() {
        let mut header = create_test_header();
        header.typeflag = 0;
        assert!(header.is_regular_file());
    }

    #[test]
    fn test_is_directory() {
        let mut header = create_test_header();
        header.typeflag = b'5';
        assert!(header.is_directory());
    }

    #[test]
    fn test_is_not_directory() {
        let header = create_test_header();
        assert!(!header.is_directory());
    }

    #[test]
    fn test_checksum_valid() {
        let mut header = create_test_header();

        let (unsigned_sum, _) = header.compute_checksums();

        let chksum_str = format!("{:06o}\0", unsigned_sum);
        header.chksum[..chksum_str.len()].copy_from_slice(chksum_str.as_bytes());

        assert!(header.checksum_valid());
    }

    #[test]
    fn test_checksum_invalid() {
        let header = create_test_header();
        assert!(!header.checksum_valid());
    }

    #[test]
    fn test_header_size() {
        assert_eq!(std::mem::size_of::<TarHeader>(), 512);
    }

    #[test]
    fn test_as_bytes() {
        let header = create_test_header();
        let bytes = header.as_bytes();
        assert_eq!(bytes.len(), 512);

        assert_eq!(&bytes[0..8], b"test.txt");
    }

    #[test]
    fn test_compute_checksums_consistency() {
        let header = create_test_header();
        let (unsigned_sum, signed_sum) = header.compute_checksums();

        assert_eq!(unsigned_sum, signed_sum);
    }

    #[test]
    fn test_full_header_roundtrip() {
        let mut header = create_test_header();

        let name = b"document.pdf\0";
        header.name[..name.len()].copy_from_slice(name);

        let prefix = b"archive/docs\0";
        header.prefix[..prefix.len()].copy_from_slice(prefix);

        let size = b"00000010000\0";
        header.size[..size.len()].copy_from_slice(size);

        let mode = b"0000755\0";
        header.mode[..mode.len()].copy_from_slice(mode);

        header.typeflag = b'0';

        let (unsigned_sum, _) = header.compute_checksums();
        let chksum_str = format!("{:06o}\0", unsigned_sum);
        header.chksum[..chksum_str.len()].copy_from_slice(chksum_str.as_bytes());

        assert_eq!(
            header.file_name(),
            Some("archive/docs/document.pdf".to_string())
        );
        assert_eq!(header.file_size(), Some(0o10000));
        assert_eq!(header.file_mode(), Some(0o755));
        assert!(header.is_regular_file());
        assert!(!header.is_directory());
        assert_eq!(header.get_type(), TarType::Posix);
        assert!(header.checksum_valid());
    }

    #[test]
    fn test_from_mmap_valid_offset_zero() {
        let mut aligned = AlignedBuffer { data: [0u8; 1024] };
        let buf = &mut aligned.data;
        buf[100..106].copy_from_slice(POSIX_MAGIC);
        buf[156] = b'0';

        let ptr = buf.as_ptr();
        let header = unsafe { TarHeader::from_mmap(ptr, 0) };

        assert!(header.is_some());
        let h = header.unwrap();
        assert_eq!(h.name_ref(), Some(""));
        assert!(h.is_regular_file());
    }

    #[test]
    fn test_from_mmap_valid_offset_512() {
        let mut aligned = AlignedBuffer { data: [0u8; 1024] };
        let buf = &mut aligned.data;
        let name = b"second_header.dat\0";
        buf[512..512 + name.len()].copy_from_slice(name);
        buf[512 + 100..512 + 106].copy_from_slice(POSIX_MAGIC);
        buf[512 + 156] = b'0';

        let ptr = buf.as_ptr();
        let header = unsafe { TarHeader::from_mmap(ptr, 512) };

        assert!(header.is_some());
        assert_eq!(header.unwrap().name_ref(), Some("second_header.dat"));
    }

    #[test]
    fn test_from_mmap_invalid_offset() {
        let buf = [0u8; 512];
        let ptr = buf.as_ptr();

        let header = unsafe { TarHeader::from_mmap(ptr, 100) };
        assert!(header.is_none());

        let header = unsafe { TarHeader::from_mmap(ptr, 511) };
        assert!(header.is_none());
    }

    #[test]
    fn test_name_ref_and_prefix_ref() {
        let mut header = create_test_header();

        let prefix = b"src/utils\0";
        header.prefix[..prefix.len()].copy_from_slice(prefix);

        assert_eq!(header.name_ref(), Some("test.txt"));
        assert_eq!(header.prefix_ref(), Some("src/utils"));
    }

    #[test]
    fn test_file_name_into_with_prefix() {
        let mut header = create_test_header();
        header.prefix[..7].copy_from_slice(b"my_dir\0");

        let mut buf = [0u8; 260];
        let result = header.file_name_into(&mut buf);

        assert_eq!(result, Some("my_dir/test.txt"));
    }

    #[test]
    fn test_file_name_into_without_prefix() {
        let header = create_test_header();

        let mut buf = [0u8; 260];
        let result = header.file_name_into(&mut buf);

        assert_eq!(result, Some("test.txt"));
    }
}
