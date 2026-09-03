use rand::SeedableRng;
use rand::rngs::StdRng;
use std::fs::File;
use std::path::PathBuf;
use tempfile::tempdir;
use turbojpeg::Image;
use turbojpeg::PixelFormat::RGB;

use crate::core::dataset::ZeroTensorDataset;
use crate::core::dataset::item::TensorDT;
use crate::core::producer::epoch_context::EpochContext;
use crate::core::writer::{TensorWriter, TensorWriterCache};
use crate::dataset::tar::TarDataset;
use crate::dataset::tar::processors::{TarJpegProcessor, TarJpegProcessorError};
use crate::dataset::tar::tar_reader::TarHeader;

#[test]
fn test_tar_jpeg_pipeline_with_real_writer() {
    let mock_jpeg_bytes = {
        let mut compressor = turbojpeg::Compressor::new().unwrap();
        let raw_pixels = vec![255u8; 16 * 16 * 3];
        let image = Image {
            height: 16,
            width: 16,
            pitch: 16 * 3,
            format: RGB,
            pixels: raw_pixels.as_slice(),
        };
        compressor.compress_to_vec(image).unwrap()
    };
    let tmp_dir = tempdir().unwrap();
    let shard_path = tmp_dir.path().join("shard_000.tar");
    let file = File::create(&shard_path).unwrap();
    let mut builder = tar::Builder::new(file);

    let mut header1 = tar::Header::new_gnu();
    let img_name1 = "class_7/image_001.jpg";
    header1.set_path(img_name1).unwrap();
    header1.set_size(mock_jpeg_bytes.len() as u64);
    header1.set_cksum();
    builder
        .append(&header1, mock_jpeg_bytes.as_slice())
        .unwrap();

    let mut header2 = tar::Header::new_gnu();
    let img_name2 = "class_42/image_002.jpg";
    header2.set_path(img_name2).unwrap();
    header2.set_size(mock_jpeg_bytes.len() as u64);
    header2.set_cksum();
    builder
        .append(&header2, mock_jpeg_bytes.as_slice())
        .unwrap();

    builder.finish().unwrap();

    let label_extractor = |filename: &str| -> i64 {
        if let Some(start) = filename.find("class_") {
            let rest = &filename[start + 6..];
            if let Some(end) = rest.find('/') {
                return rest[..end].parse().unwrap_or(0);
            }
        }
        0
    };

    let processor = TarJpegProcessor::<u8, _>::new(None, label_extractor).unwrap();

    let shard_paths = vec![shard_path];
    let buffer_capacity = 2;
    fn total_samples(_p: &PathBuf) -> Result<usize, TarJpegProcessorError> {
        Ok(2)
    }
    let rng = StdRng::seed_from_u64(42);

    let dataset = TarDataset::new(
        shard_paths,
        buffer_capacity,
        Some(total_samples),
        processor,
        rng,
    )
    .unwrap();

    let epoch_ctx = EpochContext {
        shuffle: false,
        epoch: 0,
    };
    dataset.next_epoch(&epoch_ctx).unwrap();

    let idxs = vec![0, 1];
    let layouts = dataset.dynamic_layouts(&idxs).unwrap();

    assert!(layouts.contains_key("image"));
    assert!(layouts.contains_key("label"));

    let img_layout = layouts.get("image").unwrap();
    assert_eq!(img_layout.shape(), &[3, 16, 16]);
    assert_eq!(img_layout.dt(), TensorDT::U8);

    let required_bytes: usize = layouts.values().map(|v| (v.total_bytes() + 63) & !63).sum();

    let mut raw_slot_buffer = vec![0u8; required_bytes];
    let mut writer_cache = TensorWriterCache::with_capacity(2);

    let mut writer = TensorWriter::new(&layouts, &mut raw_slot_buffer, &mut writer_cache).unwrap();

    dataset.write_item_into(0, &mut writer).unwrap();

    assert!(writer.finalize().is_ok());
    let (label_offset, _label_size) = writer.get_offset_size("label").unwrap();
    let (img_offset, img_size) = writer.get_offset_size("image").unwrap();
    drop(writer);

    assert!(
        raw_slot_buffer[img_offset..img_offset + img_size]
            .iter()
            .any(|&x| x != 0)
    );

    let label_bytes: [u8; 8] = (&raw_slot_buffer[label_offset..label_offset + 8])
        .try_into()
        .unwrap();

    let parsed_label = i64::from_le_bytes(label_bytes);

    assert!(parsed_label == 7 || parsed_label == 42);
}

#[test]
fn test_cold_alignment_error_handling() {
    let dummy_data = [0u8; 1024];
    let ptr = dummy_data.as_ptr();

    let bad_offset = 13;
    let result = unsafe { TarHeader::from_mmap(ptr, bad_offset) };

    assert!(result.is_none());
}
