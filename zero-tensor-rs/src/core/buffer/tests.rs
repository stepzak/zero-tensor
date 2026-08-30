use super::*;
use crate::core::dataset::item::{ShapeVec, StrideVec, TensorBatchLayout, TensorDT};
use indexmap::IndexMap;
use std::mem::size_of;

#[test]
fn test_filename_too_long() {
    let fname = "a".repeat(300);
    let res = ZeroTensorBuffer::new(&fname, 64, 1);
    assert!(matches!(res, Err(ZTBufErr::InvalidFilename(_))));
}

#[test]
fn test_filename_nullbyte() {
    let fname = "abcd\0efgh";
    let res = ZeroTensorBuffer::new(&fname, 64, 1);
    assert!(matches!(res, Err(ZTBufErr::InvalidFilename(_))));
}

#[test]
fn test_filename_slash() {
    let fname = "abcd/efgh";
    let res = ZeroTensorBuffer::new(&fname, 64, 1);
    assert!(matches!(res, Err(ZTBufErr::InvalidFilename(_))));
}

#[test]
fn test_buf_overflow() {
    let mut buf = ZeroTensorBuffer::new("zt_buf_test", 256, 1).unwrap();
    let shape = vec![100, 100];
    let strides = vec![100, 1];
    let data = vec![0u8; 40000];
    let res = buf.write_tensor(0, &shape, &strides, TensorDT::F32, &data);
    assert!(matches!(res, Err(ZTBufErr::BufferOverflow(_, _))));
}

#[test]
fn test_shape_stride_mismatch() {
    let mut buf = ZeroTensorBuffer::new("zt_buf_test", 4096, 2).unwrap();
    let shape = vec![100, 100];
    let strides = vec![100, 1, 2];
    let data = vec![0u8; 10000 * 4];
    let res = buf.write_tensor(0, &shape, &strides, TensorDT::F32, &data);
    assert!(matches!(res, Err(ZTBufErr::InvalidShape(3, 2))))
}

fn make_layout(shape: &[usize], dtype: TensorDT) -> TensorBatchLayout {
    let mut shape_vec = ShapeVec::new();
    shape_vec.extend_from_slice(shape);

    let mut stride_vec = StrideVec::new();
    let mut stride = 1usize;
    for &dim in shape.iter().rev() {
        stride_vec.insert(0, stride);
        stride *= dim;
    }

    TensorBatchLayout::new(shape_vec, stride_vec, dtype)
}

fn make_layouts(
    pairs: &[(&'static str, &[usize], TensorDT)],
) -> IndexMap<&'static str, TensorBatchLayout> {
    let mut map = IndexMap::new();
    for &(key, shape, dt) in pairs {
        map.insert(key, make_layout(shape, dt));
    }
    map
}

#[test]
fn test_align_to_exact_multiple() {
    assert_eq!(align_to(0, 8), 0);
    assert_eq!(align_to(8, 8), 8);
    assert_eq!(align_to(16, 8), 16);
    assert_eq!(align_to(64, 64), 64);
}

#[test]
fn test_align_to_rounds_up() {
    assert_eq!(align_to(1, 8), 8);
    assert_eq!(align_to(7, 8), 8);
    assert_eq!(align_to(9, 8), 16);
    assert_eq!(align_to(100, 64), 128);
}

#[test]
fn test_align_to_different_alignments() {
    assert_eq!(align_to(5, 4), 8);
    assert_eq!(align_to(5, 16), 16);
    assert_eq!(align_to(17, 16), 32);
}

#[test]
fn test_slot_size_is_aligned_to_64() {
    let layouts = make_layouts(&[
        ("image", &[3, 224, 224], TensorDT::F32),
        ("label", &[], TensorDT::I64),
    ]);

    for batch_size in [1, 2, 7, 32, 64, 100, 127, 256] {
        let size = ZeroTensorBuffer::calculate_slot_size(&layouts, batch_size);
        assert_eq!(
            size % 64,
            0,
            "Slot size {} for batch_size={} is not aligned to 64",
            size,
            batch_size
        );
    }
}

#[test]
fn test_slot_size_grows_with_batch_size() {
    let layouts = make_layouts(&[("image", &[3, 64, 64], TensorDT::F32)]);

    let size_1 = ZeroTensorBuffer::calculate_slot_size(&layouts, 1);
    let size_10 = ZeroTensorBuffer::calculate_slot_size(&layouts, 10);
    let size_100 = ZeroTensorBuffer::calculate_slot_size(&layouts, 100);

    assert!(size_1 < size_10);
    assert!(size_10 < size_100);
}

#[test]
fn test_slot_size_batch_size_linear_growth() {
    let layouts = make_layouts(&[("image", &[3, 32, 32], TensorDT::F32)]);

    let size_1 = ZeroTensorBuffer::calculate_slot_size(&layouts, 1);
    let size_2 = ZeroTensorBuffer::calculate_slot_size(&layouts, 2);
    let size_10 = ZeroTensorBuffer::calculate_slot_size(&layouts, 10);
    let size_100 = ZeroTensorBuffer::calculate_slot_size(&layouts, 100);

    let delta_1_2 = size_2 - size_1;
    let delta_1_10 = size_10 - size_1;
    let delta_1_100 = size_100 - size_1;

    assert!(
        delta_1_10 >= 8 * delta_1_2 && delta_1_10 <= 10 * delta_1_2 + 64,
        "Non-linear growth: delta_1_2={}, delta_1_10={}",
        delta_1_2,
        delta_1_10
    );

    assert!(
        delta_1_100 >= 98 * delta_1_2 && delta_1_100 <= 100 * delta_1_2 + 64,
        "Non-linear growth: delta_1_2={}, delta_1_100={}",
        delta_1_2,
        delta_1_100
    );
}

#[test]
fn test_slot_size_at_least_header() {
    let layouts = IndexMap::new();
    let size = ZeroTensorBuffer::calculate_slot_size(&layouts, 1);
    assert!(
        size >= size_of::<TensorHeader>() as u64,
        "Slot size {} is smaller than header size {}",
        size,
        size_of::<TensorHeader>()
    );
}

#[test]
fn test_slot_size_more_tensors_bigger_slot() {
    let layouts_1 = make_layouts(&[("image", &[3, 64, 64], TensorDT::F32)]);
    let layouts_2 = make_layouts(&[
        ("image", &[3, 64, 64], TensorDT::F32),
        ("label", &[], TensorDT::I64),
    ]);
    let layouts_3 = make_layouts(&[
        ("image", &[3, 64, 64], TensorDT::F32),
        ("label", &[], TensorDT::I64),
        ("mask", &[64, 64], TensorDT::U8),
    ]);

    let size_1 = ZeroTensorBuffer::calculate_slot_size(&layouts_1, 32);
    let size_2 = ZeroTensorBuffer::calculate_slot_size(&layouts_2, 32);
    let size_3 = ZeroTensorBuffer::calculate_slot_size(&layouts_3, 32);

    assert!(size_1 < size_2, "Adding label should increase slot size");
    assert!(size_2 < size_3, "Adding mask should increase slot size");
}

#[test]
fn test_slot_size_bigger_dtype_bigger_slot() {
    let layouts_f32 = make_layouts(&[("data", &[100], TensorDT::F32)]);
    let layouts_f64 = make_layouts(&[("data", &[100], TensorDT::F64)]);

    let size_f32 = ZeroTensorBuffer::calculate_slot_size(&layouts_f32, 32);
    let size_f64 = ZeroTensorBuffer::calculate_slot_size(&layouts_f64, 32);

    assert!(
        size_f64 > size_f32,
        "F64 should require more space than F32: f32={}, f64={}",
        size_f32,
        size_f64
    );
}

#[test]
fn test_slot_size_sufficient_for_data() {
    let layouts = make_layouts(&[
        ("image", &[3, 224, 224], TensorDT::F32),
        ("label", &[], TensorDT::I64),
    ]);
    let batch_size = 32;

    let size = ZeroTensorBuffer::calculate_slot_size(&layouts, batch_size);

    let mut min_data_size: usize = size_of::<TensorHeader>();
    for layout in layouts.values() {
        min_data_size += layout.total_bytes() * batch_size;
        min_data_size += size_of_val(layout);
    }

    assert!(
        size >= min_data_size as u64,
        "Slot size {} is smaller than minimum required {}",
        size,
        min_data_size
    );
}

#[test]
fn test_slot_size_empty_layouts() {
    let layouts: IndexMap<&str, TensorBatchLayout> = IndexMap::new();
    let size = ZeroTensorBuffer::calculate_slot_size(&layouts, 32);

    assert!(size >= size_of::<TensorHeader>() as u64);
    assert_eq!(size % 64, 0);
}

#[test]
fn test_slot_size_batch_size_zero() {
    let layouts = make_layouts(&[("image", &[3, 64, 64], TensorDT::F32)]);
    let size = ZeroTensorBuffer::calculate_slot_size(&layouts, 0);

    assert!(size >= size_of::<TensorHeader>() as u64);
    assert_eq!(size % 64, 0);
}

#[test]
fn test_slot_size_scalar_tensor() {
    let layouts = make_layouts(&[("label", &[], TensorDT::I64)]);
    let size = ZeroTensorBuffer::calculate_slot_size(&layouts, 32);

    assert!(size >= size_of::<TensorHeader>() as u64);
    assert_eq!(size % 64, 0);
}

#[test]
fn test_slot_size_large_batch() {
    let layouts = make_layouts(&[
        ("image", &[3, 224, 224], TensorDT::F32),
        ("label", &[], TensorDT::I64),
    ]);
    let size = ZeroTensorBuffer::calculate_slot_size(&layouts, 10000);

    assert!(size > 0);
    assert_eq!(size % 64, 0);

    assert!(size > 1_000_000_000);
}

#[test]
fn test_slot_size_deterministic() {
    let layouts = make_layouts(&[
        ("image", &[3, 224, 224], TensorDT::F32),
        ("label", &[], TensorDT::I64),
    ]);

    let size1 = ZeroTensorBuffer::calculate_slot_size(&layouts, 32);
    let size2 = ZeroTensorBuffer::calculate_slot_size(&layouts, 32);
    let size3 = ZeroTensorBuffer::calculate_slot_size(&layouts, 32);

    assert_eq!(size1, size2);
    assert_eq!(size2, size3);
}

#[test]
fn test_slot_size_realistic_imagenet_config() {
    let layouts = make_layouts(&[
        ("image", &[3, 224, 224], TensorDT::F32),
        ("label", &[], TensorDT::I64),
    ]);
    let batch_size = 32;

    let size = ZeroTensorBuffer::calculate_slot_size(&layouts, batch_size);

    let expected_min = 19_000_000u64;
    let expected_max = 25_000_000u64;

    assert!(
        size >= expected_min && size <= expected_max,
        "ImageNet slot size {} is outside expected range [{}, {}]",
        size,
        expected_min,
        expected_max
    );
}
