use super::*;
use std::error::Error;

use super::super::dataset::item::{TensorBatchLayout, TensorDT};
use smallvec::smallvec;

fn mock_layout(target_bytes: usize) -> TensorBatchLayout {
    TensorBatchLayout::new(smallvec![target_bytes], smallvec![1], TensorDT::U8)
}

#[derive(Debug)]
struct DummyDatasetError;
impl std::fmt::Display for DummyDatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "dummy dataset error")
    }
}
impl Error for DummyDatasetError {}

#[test]
fn test_writer_happy_path_and_alignment() {
    let mut layouts = IndexMap::new();
    layouts.insert("a", mock_layout(10));
    layouts.insert("b", mock_layout(20));

    let mut buffer = vec![0u8; 128];
    let mut writer = TensorWriter::new(layouts, &mut buffer).unwrap();
    let written_a = writer
        .write("a", |buf| -> Result<usize, std::io::Error> {
            buf[0..5].copy_from_slice(b"hello");
            Ok(5)
        })
        .unwrap();
    assert_eq!(written_a, 5);

    let written_b = writer
        .write("b", |buf| -> Result<usize, std::io::Error> {
            buf[0..20].copy_from_slice(b"world12345world12345");
            Ok(20)
        })
        .unwrap();
    assert_eq!(written_b, 20);

    writer.finalize().unwrap();

    assert_eq!(&buffer[0..5], b"hello");
    assert_eq!(&buffer[5..64], &[0u8; 59]);

    assert_eq!(&buffer[64..84], b"world12345world12345");
    assert_eq!(&buffer[84..128], &[0u8; 44]);
}

#[test]
fn test_writer_buffer_too_small_on_creation() {
    let mut layouts = IndexMap::new();
    layouts.insert("a", mock_layout(100));

    let mut buffer = vec![0u8; 64];
    let result = TensorWriter::new(layouts, &mut buffer);

    assert!(matches!(
        result,
        Err(TensorWriterError::BufferTooSmall {
            required: 128,
            available: 64
        })
    ));
}

#[test]
fn test_writer_unknown_key() {
    let mut layouts = IndexMap::new();
    layouts.insert("a", mock_layout(10));
    let mut buffer = vec![0u8; 64];
    let mut writer = TensorWriter::new(layouts, &mut buffer).unwrap();

    let result = writer.write("unknown_key", |_| -> Result<usize, std::io::Error> {
        Ok(0)
    });
    assert!(matches!(
        result,
        Err(TensorWriteError::UnknownKey("unknown_key"))
    ));
}

#[test]
fn test_writer_key_exists_duplicate_write() {
    let mut layouts = IndexMap::new();
    layouts.insert("a", mock_layout(10));
    let mut buffer = vec![0u8; 64];
    let mut writer = TensorWriter::new(layouts, &mut buffer).unwrap();

    writer
        .write("a", |_| -> Result<usize, std::io::Error> { Ok(5) })
        .unwrap();

    let result = writer.write("a", |_| -> Result<usize, std::io::Error> { Ok(5) });
    assert!(matches!(result, Err(TensorWriteError::KeyExists("a"))));
}

#[test]
fn test_writer_closure_lies_about_written_bytes() {
    let mut layouts = IndexMap::new();
    layouts.insert("a", mock_layout(10));
    let mut buffer = vec![0u8; 64];
    let mut writer = TensorWriter::new(layouts, &mut buffer).unwrap();

    let result = writer.write("a", |_| -> Result<usize, std::io::Error> { Ok(100) });
    println!("{result:?}");
    assert!(matches!(
        result,
        Err(TensorWriteError::BufferOutOfBounds {
            key: "a",
            offset: 100,
            total_size: 64
        })
    ));
}

#[test]
fn test_writer_propagates_dataset_error() {
    let mut layouts = IndexMap::new();
    layouts.insert("a", mock_layout(10));
    let mut buffer = vec![0u8; 64];
    let mut writer = TensorWriter::new(layouts, &mut buffer).unwrap();

    let result = writer.write("a", |_| Err(DummyDatasetError));

    assert!(matches!(result, Err(TensorWriteError::DatasetError { .. })));
}

#[test]
fn test_finalize_missing_keys() {
    let mut layouts = IndexMap::new();
    layouts.insert("a", mock_layout(10));
    layouts.insert("b", mock_layout(10));
    layouts.insert("c", mock_layout(10));

    let mut buffer = vec![0u8; 192];
    let mut writer = TensorWriter::new(layouts, &mut buffer).unwrap();

    writer
        .write("a", |_| -> Result<usize, std::io::Error> { Ok(10) })
        .unwrap();
    writer
        .write("c", |_| -> Result<usize, std::io::Error> { Ok(10) })
        .unwrap();

    let result = writer.finalize();
    assert!(
        matches!(result, Err(TensorWriterError::MissingKeys(ref keys)) if keys.contains(&"b") && keys.len() == 1)
    );
}

#[test]
fn test_finalize_success_fast_path() {
    let mut layouts = IndexMap::new();
    layouts.insert("a", mock_layout(10));
    layouts.insert("b", mock_layout(10));

    let mut buffer = vec![0u8; 128];
    let mut writer = TensorWriter::new(layouts, &mut buffer).unwrap();

    writer
        .write("a", |_| -> Result<usize, std::io::Error> { Ok(10) })
        .unwrap();
    writer
        .write("b", |_| -> Result<usize, std::io::Error> { Ok(10) })
        .unwrap();

    assert!(writer.finalize().is_ok());
}
