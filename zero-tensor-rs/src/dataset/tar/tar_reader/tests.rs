use super::*;
use std::io::Write;
use tempfile::NamedTempFile;

const POSIX_MAGIC: &[u8] = b"ustar\0";

fn create_test_tar_entry(name: &str, data: &[u8], typeflag: u8) -> Vec<u8> {
    let mut entry = Vec::new();

    let mut header_uninit = std::mem::MaybeUninit::<TarHeader>::uninit();
    let header_ptr = header_uninit.as_mut_ptr() as *mut u8;

    unsafe {
        std::ptr::write_bytes(header_ptr, 0, 512);

        let name_bytes = name.as_bytes();
        let name_len = name_bytes.len().min(100);
        std::ptr::copy_nonoverlapping(name_bytes.as_ptr(), header_ptr, name_len);

        let size_str = format!("{:06o}\0", data.len());
        std::ptr::copy_nonoverlapping(size_str.as_ptr(), header_ptr.add(124), size_str.len());

        std::ptr::copy_nonoverlapping(POSIX_MAGIC.as_ptr(), header_ptr.add(257), 6);

        header_ptr.add(156).write(typeflag);

        let header = &*header_uninit.as_ptr();
        let (checksum, _) = header.compute_checksums();
        let chksum_str = format!("{:06o}\0 ", checksum);
        std::ptr::copy_nonoverlapping(chksum_str.as_ptr(), header_ptr.add(148), chksum_str.len());
    }

    let header_bytes =
        unsafe { std::slice::from_raw_parts(header_uninit.as_ptr() as *const u8, 512) };
    entry.extend_from_slice(header_bytes);

    entry.extend_from_slice(data);

    let padding = data.len().next_multiple_of(512) - data.len();
    entry.extend(std::iter::repeat(0u8).take(padding));

    entry
}

fn create_gnu_long_name_entry(long_name: &str) -> Vec<u8> {
    let name_bytes = long_name.as_bytes();
    create_test_tar_entry("././@LongLink", name_bytes, b'L')
}

fn create_temp_tar(entries: &[Vec<u8>]) -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();
    for entry in entries {
        file.write_all(entry).unwrap();
    }

    file.write_all(&[0u8; 1024]).unwrap();
    file.flush().unwrap();
    file
}

#[test]
fn test_read_single_file() {
    let data = b"Hello, World!";
    let entry = create_test_tar_entry("test.txt", data, b'0');
    let file = create_temp_tar(&[entry]);

    let mut reader = TarReader::open(file.path()).unwrap();
    let mut name_buf = [0u8; MAX_PATH_LEN];

    let record = reader.next_record(&mut name_buf).unwrap();
    assert_eq!(record.name, "test.txt");
    assert_eq!(record.data, data);

    let result = reader.next_record(&mut name_buf);
    assert!(matches!(result, Err(TarReaderError::EOF)));
}

#[test]
fn test_read_multiple_files() {
    let entry1 = create_test_tar_entry("file1.txt", b"Content 1", b'0');
    let entry2 = create_test_tar_entry("file2.txt", b"Content 2", b'0');
    let entry3 = create_test_tar_entry("file3.txt", b"Content 3", b'0');

    let file = create_temp_tar(&[entry1, entry2, entry3]);
    let mut reader = TarReader::open(file.path()).unwrap();
    let mut name_buf = [0u8; MAX_PATH_LEN];

    let record1 = reader.next_record(&mut name_buf).unwrap();
    assert_eq!(record1.name, "file1.txt");
    assert_eq!(record1.data, b"Content 1");

    let record2 = reader.next_record(&mut name_buf).unwrap();
    assert_eq!(record2.name, "file2.txt");
    assert_eq!(record2.data, b"Content 2");

    let record3 = reader.next_record(&mut name_buf).unwrap();
    assert_eq!(record3.name, "file3.txt");
    assert_eq!(record3.data, b"Content 3");
}

#[test]
fn test_skip_directories() {
    let dir_entry = create_test_tar_entry("mydir/", &[], b'5');
    let file_entry = create_test_tar_entry("mydir/file.txt", b"data", b'0');

    let file = create_temp_tar(&[dir_entry, file_entry]);
    let mut reader = TarReader::open(file.path()).unwrap();
    let mut name_buf = [0u8; MAX_PATH_LEN];

    let record = reader.next_record(&mut name_buf).unwrap();
    assert_eq!(record.name, "mydir/file.txt");
    assert_eq!(record.data, b"data");
}

#[test]
fn test_gnu_long_name() {
    let long_name = "very/long/path/that/exceeds/100/characters/limit/in/tar/header/and/needs/gnu/extension/to/work/properly.txt";
    let long_name_entry = create_gnu_long_name_entry(long_name);
    let file_entry = create_test_tar_entry("././@LongLink", b"file content", b'0');

    let file = create_temp_tar(&[long_name_entry, file_entry]);
    let mut reader = TarReader::open(file.path()).unwrap();
    let mut name_buf = [0u8; MAX_PATH_LEN];

    let record = reader.next_record(&mut name_buf).unwrap();
    assert_eq!(record.name, long_name);
    assert_eq!(record.data, b"file content");
}

#[test]
fn test_padding_calculation() {
    let data = vec![0xAB; 100];
    let entry = create_test_tar_entry("padded.txt", &data, b'0');

    let file = create_temp_tar(&[entry]);
    let mut reader = TarReader::open(file.path()).unwrap();
    let mut name_buf = [0u8; MAX_PATH_LEN];

    let record = reader.next_record(&mut name_buf).unwrap();
    assert_eq!(record.name, "padded.txt");
    assert_eq!(record.data.len(), 100);
    assert_eq!(record.data[0], 0xAB);
    assert_eq!(record.data[99], 0xAB);
}

#[test]
fn test_exact_512_bytes() {
    let data = vec![0xCD; 512];
    let entry = create_test_tar_entry("exact512.bin", &data, b'0');

    let file = create_temp_tar(&[entry]);
    let mut reader = TarReader::open(file.path()).unwrap();
    let mut name_buf = [0u8; MAX_PATH_LEN];

    let record = reader.next_record(&mut name_buf).unwrap();
    assert_eq!(record.name, "exact512.bin");
    assert_eq!(record.data.len(), 512);
}

#[test]
fn test_reset_reader() {
    let entry = create_test_tar_entry("test.txt", b"data", b'0');
    let file = create_temp_tar(&[entry]);

    let mut reader = TarReader::open(file.path()).unwrap();
    let mut name_buf = [0u8; MAX_PATH_LEN];

    let record1 = reader.next_record(&mut name_buf).unwrap();
    assert_eq!(record1.name, "test.txt");

    reader.reset();

    let record2 = reader.next_record(&mut name_buf).unwrap();
    assert_eq!(record2.name, "test.txt");
}

#[test]
fn test_is_eof() {
    let entry = create_test_tar_entry("test.txt", b"data", b'0');
    let file = create_temp_tar(&[entry]);

    let mut reader = TarReader::open(file.path()).unwrap();
    let mut name_buf = [0u8; MAX_PATH_LEN];

    assert!(!reader.is_eof());

    let record = reader.next_record(&mut name_buf).unwrap();
    assert_eq!(record.name, "test.txt");
    let rec = reader.next_record(&mut name_buf);
    assert!(matches!(rec, Err(TarReaderError::EOF)));
    assert!(reader.is_eof());
}

#[test]
fn test_multiple_gnu_long_names() {
    let long_name1 = "first/very/long/path/that/exceeds/100/characters/limit/in/tar/header/and/needs/gnu/extension/to/work/properly.txt";
    let long_name2 = "second/another/very/long/path/that/exceeds/100/characters/limit/in/tar/header/and/needs/gnu/extension/to/work/properly.txt";

    let long_name_entry1 = create_gnu_long_name_entry(long_name1);
    let file_entry1 = create_test_tar_entry("././@LongLink", b"content1", b'0');

    let long_name_entry2 = create_gnu_long_name_entry(long_name2);
    let file_entry2 = create_test_tar_entry("././@LongLink", b"content2", b'0');

    let file = create_temp_tar(&[long_name_entry1, file_entry1, long_name_entry2, file_entry2]);

    let mut reader = TarReader::open(file.path()).unwrap();
    let mut name_buf = [0u8; MAX_PATH_LEN];

    let record1 = reader.next_record(&mut name_buf).unwrap();
    assert_eq!(record1.name, long_name1);
    assert_eq!(record1.data, b"content1");

    let record2 = reader.next_record(&mut name_buf).unwrap();
    assert_eq!(record2.name, long_name2);
    assert_eq!(record2.data, b"content2");
}

#[test]
fn test_empty_tar() {
    let file = create_temp_tar(&[]);
    let mut reader = TarReader::open(file.path()).unwrap();
    let mut name_buf = [0u8; MAX_PATH_LEN];

    let result = reader.next_record(&mut name_buf);
    assert!(matches!(result, Err(TarReaderError::EOF)));
}
