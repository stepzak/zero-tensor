use std::fs;
use std::path::Path;
use std::time::Instant;
use turbojpeg::{Compressor, Image, PixelFormat};
use zero_tensor_lib::augmentation::AugmentationPipeline;
use zero_tensor_lib::augmentation::default::RandomCrop;
use zero_tensor_lib::augmentation::default::flip::RandomHorizontalFlip;
use zero_tensor_lib::augmentation::default::normalize::Normalize;
use zero_tensor_lib::augmentation::default::resize::Resize;
use zero_tensor_lib::core::dataset::ZeroTensorDataset;

use zero_tensor_lib::core::producer::ZeroTensorProducerBuilder;
use zero_tensor_lib::dataset::image::JpegFolderDataset;

fn generate_dataset(dir: &Path, num_images: usize, num_classes: usize) {
    println!(
        "[Rust] Generating {} images in {} classes...",
        num_images, num_classes
    );

    if dir.exists() {
        fs::remove_dir_all(dir).expect("Failed to clean dataset directory");
    }
    fs::create_dir_all(dir).expect("Failed to create dataset directory");

    let mut compressor = Compressor::new().expect("Failed to create compressor");

    for class_id in 0..num_classes {
        fs::create_dir_all(dir.join(format!("class_{:03}", class_id))).unwrap();
    }

    for i in 0..num_images {
        let width = 100 + (i * 37 % 301);
        let height = 100 + (i * 53 % 301);

        let mut pixels = vec![128u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                pixels[(y * width + x) * 3] = (x % 256) as u8;
            }
        }

        let image = Image {
            pixels: pixels.as_slice(),
            width,
            pitch: width * 3,
            height,
            format: PixelFormat::RGB,
        };

        let jpeg_data = compressor.compress_to_vec(image).unwrap();
        let class_id = i % num_classes;
        fs::write(
            dir.join(format!("class_{:03}/img_{:05}.jpg", class_id, i)),
            &jpeg_data,
        )
        .unwrap();
    }

    println!("[Rust] Dataset generated at: {:?}", dir);
}

fn main() {
    let num_images = 2000;
    let num_classes = 10;
    let batch_size = 32;

    let dataset_dir = dirs::cache_dir()
        .expect("Failed to get cache directory")
        .join("zero_tensor_bench");

    println!("[Rust] Dataset directory: {:?}", dataset_dir);

    let socket_path = "/tmp/zt_bench.sock";
    let shm_name = "zt_bench";
    let num_slots = 16;

    generate_dataset(&dataset_dir, num_images, num_classes);

    println!("[Rust] Initializing JpegFolderDataset...");
    let start_init = Instant::now();

    let label_fn = |path: &Path| {
        path.parent()
            .and_then(|p| p.file_name())
            .and_then(|name| name.to_str())
            .and_then(|name| {
                name.strip_prefix("class_")
                    .and_then(|s| s.parse::<i64>().ok())
            })
    };

    let pipeline = AugmentationPipeline::<f32>::new()
        .then(Resize::new(256, 256))
        .unwrap()
        .then(RandomCrop::new(224, 224))
        .unwrap()
        .then(RandomHorizontalFlip::new(0.5).unwrap())
        .unwrap()
        .then(Normalize::imagenet())
        .unwrap();

    let dataset =
        JpegFolderDataset::<f32>::new(&dataset_dir, label_fn).expect("Failed to create dataset");

    let dataset = dataset.with_augmentation(pipeline);

    let builder = ZeroTensorProducerBuilder::from_dataset(
        &dataset,
        shm_name,
        socket_path,
        batch_size,
        num_images,
    )
    .expect("Failed to create builder");

    let slot_size = builder.slot_size;
    let mut producer = builder
        .num_slots(num_slots)
        .build()
        .expect("Failed to build producer");

    println!(
        "[Rust] Slot size: {} bytes ({:.2} MB)",
        slot_size,
        slot_size as f64 / 1024.0 / 1024.0
    );

    let init_time = start_init.elapsed();
    println!(
        "[Rust] Dataset initialized in {:.2}s ({} images)",
        init_time.as_secs_f32(),
        dataset.len()
    );

    println!("[Rust] Creating Producer...");

    println!("[Rust] Producer created. Waiting for Consumer to connect...");
    println!("[Rust] Socket: {}", socket_path);
    println!("[Rust] SHM: {}", shm_name);
    println!(
        "[Rust] Streaming {} batches of size {}...",
        num_images / batch_size,
        batch_size
    );

    let start_stream = Instant::now();
    producer
        .start_streaming(&dataset, batch_size)
        .expect("Streaming failed");
    let stream_time = start_stream.elapsed();

    println!(
        "[Rust] Streaming completed in {:.2}s",
        stream_time.as_secs_f32()
    );
    println!("[Rust] Done!");

    fs::remove_dir_all(&dataset_dir).ok();
}
