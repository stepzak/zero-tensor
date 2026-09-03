use super::*;
use crate::core::dataset::ZeroTensorDataset;
use crate::core::dataset::item::{TensorBatchLayout, TensorDT};
use crate::core::producer::epoch_context::EpochContext;
use crate::core::writer::TensorWriteError;
use rand::SeedableRng;
use rand::rngs::SmallRng;
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

fn create_temp_tar(entries: &[Vec<u8>]) -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();
    for entry in entries {
        file.write_all(entry).unwrap();
    }

    file.write_all(&[0u8; 1024]).unwrap();
    file.flush().unwrap();
    file
}

struct TestProcessor;

impl<'data> TarRecordProcessor<'data> for TestProcessor {
    type Error = std::io::Error;

    fn get_layout(
        &self,
        _filename: &str,
        _header: &TarHeader,
        data: &[u8],
    ) -> Result<IndexMap<&'data str, TensorBatchLayout>, Self::Error> {
        let mut map = IndexMap::new();
        map.insert(
            "data",
            TensorBatchLayout::new(vec![data.len()].into(), vec![1].into(), TensorDT::U8),
        );
        Ok(map)
    }

    fn write_into<'layout, 'b, 'c>(
        &self,
        _filename: &str,
        data: &[u8],
        writer: &mut crate::core::writer::TensorWriter<'layout, 'b, 'c>,
    ) -> Result<(), Self::Error> {
        writer
            .write("data", |buf| {
                let len = data.len().min(buf.len());
                buf[..len].copy_from_slice(&data[..len]);
                Ok(len)
            })
            .map_err(|_: TensorWriteError<Self::Error>| {
                std::io::Error::from(std::io::ErrorKind::InvalidData)
            })?;
        Ok(())
    }
}

fn create_shard_with_files(files: &[(&str, &[u8])]) -> NamedTempFile {
    let mut entries = Vec::new();
    for (name, data) in files {
        entries.push(create_test_tar_entry(name, data, b'0'));
    }
    create_temp_tar(&entries)
}

#[test]
fn test_basic_creation() {
    let shard1 = create_shard_with_files(&[("file1.txt", b"data1"), ("file2.txt", b"data2")]);

    let dataset = TarDataset::new(
        vec![shard1.path().to_path_buf()],
        10,
        None::<fn(&PathBuf) -> Result<usize, std::io::Error>>,
        TestProcessor,
        SmallRng::seed_from_u64(42),
    )
    .unwrap();

    assert_eq!(dataset.total_samples, 2);
    assert_eq!(dataset.buffer_cap, 10);
}

#[test]
fn test_count_items_multiple_shards() {
    let shard1 = create_shard_with_files(&[("file1.txt", b"data1"), ("file2.txt", b"data2")]);
    let shard2 = create_shard_with_files(&[
        ("file3.txt", b"data3"),
        ("file4.txt", b"data4"),
        ("file5.txt", b"data5"),
    ]);

    let dataset = TarDataset::new(
        vec![shard1.path().to_path_buf(), shard2.path().to_path_buf()],
        10,
        None::<fn(&PathBuf) -> Result<usize, std::io::Error>>,
        TestProcessor,
        SmallRng::seed_from_u64(42),
    )
    .unwrap();

    assert_eq!(dataset.total_samples, 5);
}

#[test]
fn test_custom_shard_size_fn() {
    let shard1 = create_shard_with_files(&[("file1.txt", b"data1")]);
    let shard2 = create_shard_with_files(&[("file2.txt", b"data2")]);

    let custom_fn = |_path: &PathBuf| -> Result<usize, std::io::Error> { Ok(100) };

    let dataset = TarDataset::new(
        vec![shard1.path().to_path_buf(), shard2.path().to_path_buf()],
        10,
        Some(custom_fn),
        TestProcessor,
        SmallRng::seed_from_u64(42),
    )
    .unwrap();

    assert_eq!(dataset.total_samples, 200);
}

#[test]
fn test_buffer_initialization() {
    let shard = create_shard_with_files(&[
        ("file1.txt", b"data1"),
        ("file2.txt", b"data2"),
        ("file3.txt", b"data3"),
    ]);

    let dataset = TarDataset::new(
        vec![shard.path().to_path_buf()],
        2,
        None::<fn(&PathBuf) -> Result<usize, std::io::Error>>,
        TestProcessor,
        SmallRng::seed_from_u64(42),
    )
    .unwrap();
    let ctx = EpochContext {
        epoch: 0,
        shuffle: true,
    };
    dataset.next_epoch(&ctx).unwrap();

    let cell0 = dataset.shuffle_buffer[0].lock();
    assert_eq!(cell0.filename, "file1.txt");
    assert_eq!(cell0.data, b"data1");

    let cell1 = dataset.shuffle_buffer[1].lock();
    assert_eq!(cell1.filename, "file2.txt");
    assert_eq!(cell1.data, b"data2");
}

#[test]
fn test_shard_transition() {
    let shard1 = create_shard_with_files(&[("file1.txt", b"data1")]);
    let shard2 = create_shard_with_files(&[("file2.txt", b"data2")]);

    let dataset = TarDataset::new(
        vec![shard1.path().to_path_buf(), shard2.path().to_path_buf()],
        2,
        None::<fn(&PathBuf) -> Result<usize, std::io::Error>>,
        TestProcessor,
        SmallRng::seed_from_u64(42),
    )
    .unwrap();

    let ctx = EpochContext {
        epoch: 0,
        shuffle: false,
    };
    dataset.next_epoch(&ctx).unwrap();
    let cell0 = dataset.shuffle_buffer[0].lock();

    assert_eq!(cell0.filename, "file1.txt");

    let cell1 = dataset.shuffle_buffer[1].lock();
    assert_eq!(cell1.filename, "file2.txt");
}

#[test]
fn test_exhausted_state() {
    let shard = create_shard_with_files(&[("file1.txt", b"data1")]);

    let dataset = TarDataset::new(
        vec![shard.path().to_path_buf()],
        5,
        None::<fn(&PathBuf) -> Result<usize, std::io::Error>>,
        TestProcessor,
        SmallRng::seed_from_u64(42),
    )
    .unwrap();

    let ctx = EpochContext {
        epoch: 0,
        shuffle: true,
    };
    dataset.next_epoch(&ctx).unwrap();
    assert!(dataset.exhausted.load(Ordering::Acquire));

    let cell0 = dataset.shuffle_buffer[0].lock();
    assert_eq!(cell0.filename, "file1.txt");

    let cell1 = dataset.shuffle_buffer[1].lock();
    assert!(cell1.data.is_empty());
}

#[test]
fn test_shuffle_deterministic() {
    let shard1 = create_shard_with_files(&[("file1.txt", b"data1"), ("file2.txt", b"data2")]);
    let shard2 = create_shard_with_files(&[("file3.txt", b"data3"), ("file4.txt", b"data4")]);

    let shards = vec![shard1.path().to_path_buf(), shard2.path().to_path_buf()];

    let dataset1 = TarDataset::new(
        shards.clone(),
        4,
        None::<fn(&PathBuf) -> Result<usize, std::io::Error>>,
        TestProcessor,
        SmallRng::seed_from_u64(42),
    )
    .unwrap();

    let dataset2 = TarDataset::new(
        shards,
        4,
        None::<fn(&PathBuf) -> Result<usize, std::io::Error>>,
        TestProcessor,
        SmallRng::seed_from_u64(42),
    )
    .unwrap();

    let ctx = EpochContext {
        epoch: 0,
        shuffle: true,
    };
    dataset1.next_epoch(&ctx).unwrap();
    dataset2.next_epoch(&ctx).unwrap();

    let shards1 = dataset1.shards.read();
    let shards2 = dataset2.shards.read();
    assert_eq!(*shards1, *shards2);
}

#[test]
fn test_no_shuffle() {
    let shard1 = create_shard_with_files(&[("file1.txt", b"data1")]);
    let shard2 = create_shard_with_files(&[("file2.txt", b"data2")]);

    let shards = vec![shard1.path().to_path_buf(), shard2.path().to_path_buf()];

    let dataset = TarDataset::new(
        shards.clone(),
        2,
        None::<fn(&PathBuf) -> Result<usize, std::io::Error>>,
        TestProcessor,
        SmallRng::seed_from_u64(42),
    )
    .unwrap();
    let ctx = EpochContext {
        epoch: 0,
        shuffle: false,
    };

    dataset.next_epoch(&ctx).unwrap();

    let current_shards = dataset.shards.read();
    assert_eq!(*current_shards, shards);
}

#[test]
fn test_next_epoch_resets_state() {
    let shard = create_shard_with_files(&[("file1.txt", b"data1"), ("file2.txt", b"data2")]);

    let dataset = TarDataset::new(
        vec![shard.path().to_path_buf()],
        2,
        None::<fn(&PathBuf) -> Result<usize, std::io::Error>>,
        TestProcessor,
        SmallRng::seed_from_u64(42),
    )
    .unwrap();

    let ctx = EpochContext {
        epoch: 0,
        shuffle: false,
    };
    dataset.next_epoch(&ctx).unwrap();

    let cell0 = dataset.shuffle_buffer[0].lock();
    assert!(!cell0.data.is_empty(), "Cell 0 should be filled");
    drop(cell0);

    let cell1 = dataset.shuffle_buffer[1].lock();
    assert!(!cell1.data.is_empty(), "Cell 1 should be filled");
    drop(cell1);

    let ctx2 = EpochContext {
        epoch: 1,
        shuffle: false,
    };
    dataset.next_epoch(&ctx2).unwrap();

    assert_eq!(dataset.current_shard_idx.load(Ordering::Relaxed), 0);

    let cell0 = dataset.shuffle_buffer[0].lock();
    assert!(!cell0.data.is_empty(), "Cell 0 should be refilled");
}

#[test]
fn test_empty_shards_error() {
    let result = TarDataset::new(
        vec![],
        10,
        None::<fn(&PathBuf) -> Result<usize, std::io::Error>>,
        TestProcessor,
        SmallRng::seed_from_u64(42),
    );

    assert!(matches!(result, Err(TarDatasetError::Empty)));
}

#[test]
fn test_buffer_capacity_larger_than_data() {
    let shard = create_shard_with_files(&[("file1.txt", b"data1")]);

    let dataset: TarDataset<'_, TestProcessor, SmallRng> = TarDataset::new(
        vec![shard.path().to_path_buf()],
        100,
        None::<fn(&PathBuf) -> Result<usize, std::io::Error>>,
        TestProcessor,
        SmallRng::seed_from_u64(42),
    )
    .unwrap();
    let ctx = EpochContext {
        epoch: 0,
        shuffle: true,
    };
    dataset.next_epoch(&ctx).unwrap();
    let cell0 = dataset.shuffle_buffer[0].lock();
    assert_eq!(cell0.filename, "file1.txt");

    for i in 1..100 {
        let cell = dataset.shuffle_buffer[i].lock();
        assert!(cell.data.is_empty());
    }
}
