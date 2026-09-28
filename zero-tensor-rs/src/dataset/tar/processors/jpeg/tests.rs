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

#[rstest::rstest]
#[case(false)]
#[case(true)]
fn mixed_size_images_use_batch_padding(#[case] normalize: bool) {
    use crate::augmentation::{AugmentationPipeline, default::normalize::Normalize};
    let dir = tempdir().unwrap();
    let path = dir.path().join("mixed.tar");
    let mut archive = tar::Builder::new(File::create(&path).unwrap());
    let mut references = Vec::new();
    for (idx, (width, height)) in [(4, 2), (2, 4)].into_iter().enumerate() {
        let pixels: Vec<u8> = (0..width * height)
            .flat_map(|i| [200, i as u8 * 10, 30])
            .collect();
        let jpeg = turbojpeg::Compressor::new()
            .unwrap()
            .compress_to_vec(Image {
                pixels: pixels.as_slice(),
                width,
                height,
                pitch: width * 3,
                format: RGB,
            })
            .unwrap();
        references.push(turbojpeg::decompress(&jpeg, RGB).unwrap());
        let mut header = tar::Header::new_gnu();
        header.set_size(jpeg.len() as u64);
        header.set_cksum();
        archive
            .append_data(&mut header, format!("{idx}.jpg"), jpeg.as_slice())
            .unwrap();
    }
    archive.finish().unwrap();
    let augmentation = if normalize {
        Some(
            AugmentationPipeline::new()
                .then(Normalize::<f32>::new(vec![0.1, 0.2, 0.3], vec![0.5; 3]).unwrap())
                .unwrap(),
        )
    } else {
        None
    };
    let dataset = TarDataset::new(
        vec![path],
        2,
        None::<fn(&PathBuf) -> Result<usize, TarJpegProcessorError>>,
        TarJpegProcessor::<f32, _>::new(augmentation, |_| 7).unwrap(),
        StdRng::seed_from_u64(42),
    )
    .unwrap();
    dataset
        .next_epoch(&EpochContext {
            epoch: 0,
            shuffle: false,
        })
        .unwrap();
    let layouts = dataset.dynamic_layouts(&[0, 1]).unwrap();
    assert_eq!(layouts["image"].shape(), &[3, 4, 4]);
    let size: usize = layouts
        .values()
        .map(|v| v.total_bytes().next_multiple_of(64))
        .sum();
    for (idx, reference) in references.iter().enumerate() {
        let mut buffer = aligned_vec::AVec::<u8>::with_capacity(64, size);
        buffer.resize(size, 0xff);
        let mut cache = TensorWriterCache::with_capacity(2);
        let mut writer = TensorWriter::new(&layouts, &mut buffer, &mut cache).unwrap();
        dataset.write_item_into(idx, &mut writer).unwrap();
        writer.finalize().unwrap();
        let (offset, _) = writer.get_offset_size("image").unwrap();
        drop(writer);
        let actual: &[f32] = bytemuck::cast_slice(&buffer[offset..offset + 3 * 4 * 4 * 4]);
        for c in 0..3 {
            for y in 0..4 {
                for x in 0..4 {
                    let expected = if y < reference.height && x < reference.width {
                        let value =
                            reference.pixels[y * reference.pitch + x * 3 + c] as f32 / 255.0;
                        if normalize {
                            (value - [0.1, 0.2, 0.3][c]) * 2.0
                        } else {
                            value
                        }
                    } else {
                        0.0
                    };
                    assert!(
                        (actual[c * 16 + y * 4 + x] - expected).abs() < 1e-6,
                        "image={idx}, channel={c}, y={y}, x={x}"
                    );
                }
            }
        }
    }
}
