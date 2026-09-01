use crate::augmentation::default::flip::RandomHorizontalFlip;

use super::*;
use tempfile::TempDir;
use turbojpeg::Compressor;

fn save_test_jpeg(path: &Path, width: usize, height: usize) {
    let mut compressor = Compressor::new().unwrap();
    let mut pixels = vec![0u8; width * height * 3];

    for y in 0..height {
        for x in 0..width {
            pixels[(y * width + x) * 3] = ((x + 1) % 256) as u8;
            pixels[(y * width + x) * 3 + 1] = 128;
            pixels[(y * width + x) * 3 + 2] = 64;
        }
    }

    let image = turbojpeg::Image {
        pixels: pixels.as_slice(),
        width,
        pitch: width * 3,
        height,
        format: turbojpeg::PixelFormat::RGB,
    };

    let jpeg_data = compressor.compress_to_vec(image).unwrap();
    std::fs::write(path, jpeg_data).unwrap();
}

#[test]
fn test_jpeg_folder_dataset_e2e() {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();
    let class_a = root.join("class_a");
    let class_b = root.join("class_b");
    std::fs::create_dir(&class_a).unwrap();
    std::fs::create_dir(&class_b).unwrap();

    save_test_jpeg(&class_a.join("img1.jpg"), 64, 64);
    save_test_jpeg(&class_b.join("img2.jpg"), 100, 80);

    let label_fn = |path: &Path| {
        path.parent()
            .unwrap()
            .file_name()
            .unwrap()
            .to_str()
            .map(|name| if name == "class_a" { 0 } else { 1 })
    };

    let dataset = JpegFolderDataset::new(root, label_fn).unwrap();
    assert_eq!(dataset.len(), 2);

    let layouts = dataset.dynamic_layouts(&[0, 1]).unwrap();
    let img_layout = layouts.get("image").unwrap();
    assert_eq!(img_layout.shape(), &[3, 80, 100]);

    let max_elements = 3 * 80 * 100;
    let mut mock_buf = vec![0f32; max_elements];

    let bytes_written = dataset.inner_write(0, &mut mock_buf).unwrap();

    let expected_bytes = 100 * 80 * 3 * 4;
    assert_eq!(bytes_written, expected_bytes);

    let f32_view = mock_buf;

    assert!(
        f32_view[0] > 0.0,
        "Red channel of first pixel should be > 0"
    );
    assert!(
        (f32_view[1] - 128.0 / 255.0).abs() < 0.01,
        "Green channel mismatch"
    );

    let padding_start_idx = 64 * 3;
    assert_eq!(
        f32_view[padding_start_idx], 0.0,
        "Right padding should be zeroed"
    );
    assert_eq!(
        f32_view[padding_start_idx + 1],
        0.0,
        "Right padding should be zeroed"
    );
    assert_eq!(
        f32_view[padding_start_idx + 2],
        0.0,
        "Right padding should be zeroed"
    );

    let bottom_padding_idx = (64 * 100 + 0) * 3;
    assert_eq!(
        f32_view[bottom_padding_idx], 0.0,
        "Bottom padding should be zeroed"
    );
}

#[test]
fn test_augmentation_with_large_image() {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();
    let class_dir = root.join("class_a");
    std::fs::create_dir(&class_dir).unwrap();

    save_test_jpeg(&class_dir.join("big.jpg"), 1000, 1000);

    let pipeline = AugmentationPipeline::<f32>::new()
        .then(RandomHorizontalFlip::new(1.0).unwrap())
        .unwrap();

    let dataset = JpegFolderDataset::<f32>::new(root, |_| Some(0))
        .unwrap()
        .with_augmentation(pipeline);

    let _ = dataset.dynamic_layouts(&[0]).unwrap();
    let max_elements = 3 * 1000 * 1000;
    let mut mock_buf = vec![0u8; max_elements * 4];

    let bytes = dataset
        .inner_write(0, bytemuck::cast_slice_mut(&mut mock_buf))
        .unwrap();
    assert!(bytes > 0);
}
