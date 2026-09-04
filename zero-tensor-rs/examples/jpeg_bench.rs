use std::path::Path;
use std::time::Instant;

use zero_tensor_lib::augmentation::AugmentationPipeline;
use zero_tensor_lib::augmentation::default::crop::RandomCrop;
use zero_tensor_lib::augmentation::default::flip::RandomHorizontalFlip;
use zero_tensor_lib::augmentation::default::normalize::Normalize;
use zero_tensor_lib::augmentation::default::resize::Resize;
use zero_tensor_lib::core::dataset::ZeroTensorDataset;
use zero_tensor_lib::core::producer::ZeroTensorProducerBuilder;
use zero_tensor_lib::dataset::image::JpegFolderDataset;

fn main() {
    let batch_size = 32;
    let dataset_dir = dirs::cache_dir()
        .expect("Failed to get cache directory")
        .join("zero_tensor_bench");

    println!("[Rust] Dataset directory: {:?}", dataset_dir);

    if !dataset_dir.exists() {
        eprintln!("[Rust] ERROR: Dataset not found at {:?}", dataset_dir);
        eprintln!("[Rust] Please run Python generator first:");
        eprintln!("[Rust]   cd zero-tensor-py && uv run python benchmarks/generate_dataset.py");
        std::process::exit(1);
    }

    let socket_path = "/tmp/zt_bench.sock";
    let shm_name = "zt_bench";
    let num_slots = 16;

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

    let init_time = start_init.elapsed();
    println!(
        "[Rust] Dataset initialized in {:.2}s ({} images)",
        init_time.as_secs_f32(),
        dataset.len()
    );

    let builder = ZeroTensorProducerBuilder::from_dataset(
        &dataset,
        shm_name,
        socket_path,
        batch_size,
        dataset.len(),
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
    println!("[Rust] Producer created. Waiting for Consumer to connect...");
    println!("[Rust] Socket: {}", socket_path);
    println!("[Rust] SHM: {}", shm_name);
    println!("[Rust] Streaming batches of size {}...", batch_size);

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
}
