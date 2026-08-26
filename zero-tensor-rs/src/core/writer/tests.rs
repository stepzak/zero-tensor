use super::*;
use crate::core::dataset::item::{TensorBatchLayout, TensorDT};
use indexmap::IndexMap;
use smallvec::smallvec;

fn mock_layout(total_bytes: usize) -> TensorBatchLayout {
    TensorBatchLayout::new(smallvec![total_bytes], smallvec![1], TensorDT::U8)
}

fn mock_layout_nd(shape: Vec<usize>, dt: TensorDT) -> TensorBatchLayout {
    let strides: Vec<usize> = {
        let mut s = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            s[i] = s[i + 1] * shape[i + 1];
        }
        s
    };
    TensorBatchLayout::new(shape.into(), strides.into(), dt)
}

#[derive(Debug)]
struct DummyError;
impl std::fmt::Display for DummyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "dummy error")
    }
}
impl std::error::Error for DummyError {}

#[test]
fn test_single_tensor_happy_path() {
    let mut layouts = IndexMap::new();
    layouts.insert("data", mock_layout(100));

    let mut buffer = vec![0u8; 192];

    let mut cache = TensorWriterCache::with_capacity(1);
    let mut writer = TensorWriter::new(&layouts, &mut buffer, &mut cache).unwrap();

    assert_eq!(writer.data_offset(), 64);

    let written = writer
        .write("data", |buf| -> Result<usize, std::io::Error> {
            buf[..100].fill(42);
            Ok(100)
        })
        .unwrap();

    assert_eq!(written, 100);
    writer.finalize().unwrap();

    assert_eq!(&buffer[64..164], &[42u8; 100]);
    assert_eq!(&buffer[164..192], &[0u8; 28]);
}

#[test]
fn test_multiple_tensors_happy_path() {
    let mut layouts = IndexMap::new();
    layouts.insert("image", mock_layout(100));
    layouts.insert("label", mock_layout(10));

    let mut buffer = vec![0u8; 320];
    {
        let mut cache = TensorWriterCache::with_capacity(2);
        let mut writer = TensorWriter::new(&layouts, &mut buffer, &mut cache).unwrap();

        assert_eq!(writer.data_offset(), 128);

        writer
            .write("image", |buf| -> Result<usize, std::io::Error> {
                buf[..100].fill(1);
                Ok(100)
            })
            .unwrap();

        writer
            .write("label", |buf| -> Result<usize, std::io::Error> {
                buf[..10].fill(2);
                Ok(10)
            })
            .unwrap();

        writer.finalize().unwrap();
    }
    assert_eq!(&buffer[128..228], &[1u8; 100]);
    assert_eq!(&buffer[256..266], &[2u8; 10]);
}

#[test]
fn test_metadata_alignment() {
    let mut layouts = IndexMap::new();
    layouts.insert("a", mock_layout(10));
    layouts.insert("b", mock_layout(20));
    layouts.insert("c", mock_layout(30));

    let mut buffer = vec![0u8; 1024];
    let mut cache = TensorWriterCache::with_capacity(3);
    let writer = TensorWriter::new(&layouts, &mut buffer, &mut cache).unwrap();

    assert_eq!(writer.data_offset(), 192);

    let (off_a, _) = writer.get_offset_size("a").unwrap();
    let (off_b, _) = writer.get_offset_size("b").unwrap();
    let (off_c, _) = writer.get_offset_size("c").unwrap();

    assert_eq!(off_a, 192);
    assert_eq!(off_b, 192 + 64);
    assert_eq!(off_c, 192 + 64 + 64);
}

#[test]
fn test_data_alignment() {
    let mut layouts = IndexMap::new();
    layouts.insert("small", mock_layout(10));
    layouts.insert("medium", mock_layout(100));
    layouts.insert("large", mock_layout(1000));

    let mut buffer = vec![0u8; 4096];
    let mut cache = TensorWriterCache::with_capacity(3);
    let writer = TensorWriter::new(&layouts, &mut buffer, &mut cache).unwrap();

    let (off_small, size_small) = writer.get_offset_size("small").unwrap();
    let (off_medium, size_medium) = writer.get_offset_size("medium").unwrap();
    let (off_large, size_large) = writer.get_offset_size("large").unwrap();

    assert_eq!(off_small % 64, 0);
    assert_eq!(off_medium % 64, 0);
    assert_eq!(off_large % 64, 0);

    assert_eq!(size_small % 64, 0);
    assert_eq!(size_medium % 64, 0);
    assert_eq!(size_large % 64, 0);

    assert_eq!(size_small, 64);
    assert_eq!(size_medium, 128);
    assert_eq!(size_large, 1024);
}

#[test]
fn test_buffer_too_small() {
    let mut layouts = IndexMap::new();
    layouts.insert("data", mock_layout(100));

    let mut buffer = vec![0u8; 100];
    let mut cache = TensorWriterCache::with_capacity(1);
    let result = TensorWriter::new(&layouts, &mut buffer, &mut cache);

    assert!(matches!(
        result,
        Err(TensorWriterError::BufferTooSmall {
            required: 192,
            available: 100
        })
    ));
}

#[test]
fn test_unknown_key() {
    let mut layouts = IndexMap::new();
    layouts.insert("data", mock_layout(100));

    let mut buffer = vec![0u8; 192];
    let mut cache = TensorWriterCache::with_capacity(1);
    let mut writer = TensorWriter::new(&layouts, &mut buffer, &mut cache).unwrap();

    let result = writer.write("unknown", |_| -> Result<usize, std::io::Error> { Ok(0) });

    assert!(matches!(
        result,
        Err(TensorWriteError::UnknownKey("unknown"))
    ));
}

#[test]
fn test_key_exists() {
    let mut layouts = IndexMap::new();
    layouts.insert("data", mock_layout(100));

    let mut buffer = vec![0u8; 192];
    let mut cache = TensorWriterCache::with_capacity(1);
    let mut writer = TensorWriter::new(&layouts, &mut buffer, &mut cache).unwrap();

    writer
        .write("data", |_| -> Result<usize, std::io::Error> { Ok(100) })
        .unwrap();
    let result = writer.write("data", |_| -> Result<usize, std::io::Error> { Ok(100) });

    assert!(matches!(result, Err(TensorWriteError::KeyExists("data"))));
}

#[test]
fn test_buffer_overflow() {
    let mut layouts = IndexMap::new();
    layouts.insert("data", mock_layout(100));

    let mut buffer = vec![0u8; 192];
    let mut cache = TensorWriterCache::with_capacity(1);
    let mut writer = TensorWriter::new(&layouts, &mut buffer, &mut cache).unwrap();

    let result = writer.write("data", |_| -> Result<usize, std::io::Error> { Ok(200) });

    assert!(matches!(
        result,
        Err(TensorWriteError::BufferOutOfBounds {
            key: "data",
            offset: 200,
            total_size: 128
        })
    ));
}

#[test]
fn test_dataset_error_propagation() {
    let mut layouts = IndexMap::new();
    layouts.insert("data", mock_layout(100));

    let mut buffer = vec![0u8; 192];
    let mut cache = TensorWriterCache::with_capacity(1);
    let mut writer = TensorWriter::new(&layouts, &mut buffer, &mut cache).unwrap();

    let result = writer.write("data", |_| Err(DummyError));

    assert!(matches!(result, Err(TensorWriteError::DatasetError { .. })));
}

#[test]
fn test_finalize_all_written() {
    let mut layouts = IndexMap::new();
    layouts.insert("a", mock_layout(10));
    layouts.insert("b", mock_layout(20));

    let mut buffer = vec![0u8; 320];
    let mut cache = TensorWriterCache::with_capacity(2);
    let mut writer = TensorWriter::new(&layouts, &mut buffer, &mut cache).unwrap();

    writer
        .write("a", |_| -> Result<usize, std::io::Error> { Ok(10) })
        .unwrap();
    writer
        .write("b", |_| -> Result<usize, std::io::Error> { Ok(20) })
        .unwrap();

    assert!(writer.finalize().is_ok());
}

#[test]
fn test_finalize_missing_keys() {
    let mut layouts = IndexMap::new();
    layouts.insert("a", mock_layout(10));
    layouts.insert("b", mock_layout(20));
    layouts.insert("c", mock_layout(30));

    let mut buffer = vec![0u8; 512];
    let mut cache = TensorWriterCache::with_capacity(3);
    let mut writer = TensorWriter::new(&layouts, &mut buffer, &mut cache).unwrap();

    writer
        .write("a", |_| -> Result<usize, std::io::Error> { Ok(10) })
        .unwrap();
    writer
        .write("c", |_| -> Result<usize, std::io::Error> { Ok(30) })
        .unwrap();

    let result = writer.finalize();
    assert!(
        matches!(result, Err(TensorWriterError::MissingKeys(ref keys)) if keys.contains(&"b".to_string()) && keys.len() == 1)
    );
}

#[test]
fn test_finalize_fast_path() {
    let mut layouts = IndexMap::new();
    layouts.insert("a", mock_layout(10));
    layouts.insert("b", mock_layout(20));

    let mut buffer = vec![0u8; 320];
    let mut cache = TensorWriterCache::with_capacity(2);
    let mut writer = TensorWriter::new(&layouts, &mut buffer, &mut cache).unwrap();

    writer
        .write("a", |_| -> Result<usize, std::io::Error> { Ok(10) })
        .unwrap();
    writer
        .write("b", |_| -> Result<usize, std::io::Error> { Ok(20) })
        .unwrap();

    assert!(writer.finalize().is_ok());
}

#[test]
fn test_zero_padding_partial_write() {
    let mut layouts = IndexMap::new();
    layouts.insert("data", mock_layout(100));

    let mut buffer = vec![0u8; 192];

    let mut cache = TensorWriterCache::with_capacity(1);
    let mut writer = TensorWriter::new(&layouts, &mut buffer, &mut cache).unwrap();

    writer
        .write("data", |buf| -> Result<usize, std::io::Error> {
            buf[..50].fill(99);
            Ok(50)
        })
        .unwrap();

    writer.finalize().unwrap();

    assert_eq!(&buffer[64..114], &[99u8; 50]);
    assert_eq!(&buffer[114..192], &[0u8; 78]);
}

#[test]
fn test_zero_padding_full_write() {
    let mut layouts = IndexMap::new();
    layouts.insert("data", mock_layout(64));

    let mut buffer = vec![0u8; 128];
    let mut cache = TensorWriterCache::with_capacity(1);
    let mut writer = TensorWriter::new(&layouts, &mut buffer, &mut cache).unwrap();

    writer
        .write("data", |buf| -> Result<usize, std::io::Error> {
            buf.fill(77);
            Ok(64)
        })
        .unwrap();

    writer.finalize().unwrap();
    assert_eq!(&buffer[64..128], &[77u8; 64]);
}

#[test]
fn test_different_dtypes() {
    let mut layouts = IndexMap::new();
    layouts.insert("f32", mock_layout_nd(vec![4], TensorDT::F32));
    layouts.insert("i32", mock_layout_nd(vec![4], TensorDT::I32));
    layouts.insert("u8", mock_layout_nd(vec![4], TensorDT::U8));
    layouts.insert("i64", mock_layout_nd(vec![4], TensorDT::I64));

    let mut buffer = vec![0u8; 2048];
    let mut cache = TensorWriterCache::with_capacity(4);
    let mut writer = TensorWriter::new(&layouts, &mut buffer, &mut cache).unwrap();

    writer
        .write("f32", |buf| -> Result<usize, std::io::Error> {
            let floats: &mut [f32] = bytemuck::cast_slice_mut(&mut buf[..16]);
            floats.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
            Ok(16)
        })
        .unwrap();

    writer
        .write("i32", |buf| -> Result<usize, std::io::Error> {
            let ints: &mut [i32] = bytemuck::cast_slice_mut(&mut buf[..16]);
            ints.copy_from_slice(&[10, 20, 30, 40]);
            Ok(16)
        })
        .unwrap();

    writer
        .write("u8", |buf| -> Result<usize, std::io::Error> {
            buf[..4].copy_from_slice(&[1, 2, 3, 4]);
            Ok(4)
        })
        .unwrap();

    writer
        .write("i64", |buf| -> Result<usize, std::io::Error> {
            let ints: &mut [i64] = bytemuck::cast_slice_mut(&mut buf[..32]);
            ints.copy_from_slice(&[100, 200, 300, 400]);
            Ok(32)
        })
        .unwrap();

    writer.finalize().unwrap();
}

#[test]
fn test_different_ndims() {
    let mut layouts = IndexMap::new();
    layouts.insert("1d", mock_layout_nd(vec![10], TensorDT::F32));
    layouts.insert("2d", mock_layout_nd(vec![3, 4], TensorDT::F32));
    layouts.insert("3d", mock_layout_nd(vec![2, 3, 4], TensorDT::F32));
    layouts.insert("4d", mock_layout_nd(vec![2, 3, 4, 5], TensorDT::F32));

    let mut buffer = vec![0u8; 4096];
    let mut cache = TensorWriterCache::with_capacity(4);
    let writer = TensorWriter::new(&layouts, &mut buffer, &mut cache).unwrap();

    let (off_1d, _) = writer.get_offset_size("1d").unwrap();
    let (off_2d, _) = writer.get_offset_size("2d").unwrap();
    let (off_3d, _) = writer.get_offset_size("3d").unwrap();
    let (off_4d, _) = writer.get_offset_size("4d").unwrap();

    assert_eq!(off_1d % 64, 0);
    assert_eq!(off_2d % 64, 0);
    assert_eq!(off_3d % 64, 0);
    assert_eq!(off_4d % 64, 0);
}

#[test]
fn test_cache_reuse() {
    let mut layouts = IndexMap::new();
    layouts.insert("data", mock_layout(100));

    let mut buffer = vec![0u8; 192];
    let mut cache = TensorWriterCache::with_capacity(1);

    {
        let mut writer = TensorWriter::new(&layouts, &mut buffer, &mut cache).unwrap();
        writer
            .write("data", |_| -> Result<usize, std::io::Error> { Ok(100) })
            .unwrap();
        writer.finalize().unwrap();
    }

    {
        let mut writer = TensorWriter::new(&layouts, &mut buffer, &mut cache).unwrap();
        writer
            .write("data", |_| -> Result<usize, std::io::Error> { Ok(100) })
            .unwrap();
        writer.finalize().unwrap();
    }
}

#[test]
fn test_cache_clear() {
    let mut cache = TensorWriterCache::with_capacity(3);
    cache.insert("a", 0, 100);
    cache.insert("b", 100, 200);
    cache.mark_written("a");

    assert_eq!(cache.written().len(), 2);
    assert!(cache.written()[0]);
    assert!(!cache.written()[1]);

    cache.clear();

    assert_eq!(cache.slot_buffers().len(), 0);
    assert_eq!(cache.written().len(), 0);
}

#[test]
fn test_realistic_image_dataset() {
    let mut layouts = IndexMap::new();
    layouts.insert("image", mock_layout_nd(vec![3, 224, 224], TensorDT::U8));
    layouts.insert("mask", mock_layout_nd(vec![224, 224], TensorDT::U8));
    layouts.insert("label", mock_layout_nd(vec![1], TensorDT::I32));

    let size = 200960;
    let mut buffer = vec![0u8; size];
    let mut cache = TensorWriterCache::with_capacity(3);
    let mut writer = TensorWriter::new(&layouts, &mut buffer, &mut cache).unwrap();

    writer
        .write("image", |buf| -> Result<usize, std::io::Error> {
            buf[..150528].fill(128);
            Ok(150528)
        })
        .unwrap();

    writer
        .write("mask", |buf| -> Result<usize, std::io::Error> {
            buf[..50176].fill(255);
            Ok(50176)
        })
        .unwrap();

    writer
        .write("label", |buf| -> Result<usize, std::io::Error> {
            let label: &mut [i32] = bytemuck::cast_slice_mut(&mut buf[..4]);
            label[0] = 42;
            Ok(4)
        })
        .unwrap();

    writer.finalize().unwrap();

    let (off_image, _) = writer.get_offset_size("image").unwrap();
    let (off_mask, _) = writer.get_offset_size("mask").unwrap();
    let (off_label, _) = writer.get_offset_size("label").unwrap();
    assert_eq!(buffer[off_image], 128);
    assert_eq!(buffer[off_mask], 255);
    let label: i32 = bytemuck::pod_read_unaligned(&buffer[off_label..off_label + 4]);
    assert_eq!(label, 42);
}
