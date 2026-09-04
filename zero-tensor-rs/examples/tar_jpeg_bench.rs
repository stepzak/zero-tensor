use std::path::PathBuf;
use std::time::Instant;

use rand::SeedableRng;
use rand::rngs::StdRng;

use zero_tensor_lib::augmentation::AugmentationPipeline;
use zero_tensor_lib::augmentation::default::crop::RandomCrop;
use zero_tensor_lib::augmentation::default::flip::RandomHorizontalFlip;
use zero_tensor_lib::augmentation::default::normalize::Normalize;
use zero_tensor_lib::augmentation::default::resize::Resize;
use zero_tensor_lib::core::dataset::ZeroTensorDataset;
use zero_tensor_lib::core::producer::ZeroTensorProducerBuilder;
use zero_tensor_lib::dataset::tar::TarDataset;
use zero_tensor_lib::dataset::tar::processors::{TarJpegProcessor, TarJpegProcessorError};

fn main() {
    let batch_size = 32;
    let buffer_cap = 1024;
    let num_slots = 16;

    let tar_dir = dirs::cache_dir()
        .expect("Failed to get cache directory")
        .join("zero_tensor_bench")
        .join("tar");

    println!("[Rust] TAR directory: {:?}", tar_dir);

    if !tar_dir.exists() {
        eprintln!("[Rust] ERROR: TAR dataset not found at {:?}", tar_dir);
        eprintln!("[Rust] Please run Python generator first:");
        eprintln!(
            "[Rust]   cd zero-tensor-py && uv run python benchmarks/generate_dataset.py --format tar"
        );
        std::process::exit(1);
    }

    let mut shard_paths: Vec<PathBuf> = std::fs::read_dir(&tar_dir)
        .expect("Failed to read TAR directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "tar"))
        .map(|e| e.path())
        .collect();
    shard_paths.sort();

    if shard_paths.is_empty() {
        eprintln!("[Rust] ERROR: No .tar files found in {:?}", tar_dir);
        std::process::exit(1);
    }

    println!("[Rust] Found {} shards", shard_paths.len());

    let socket_path = "/tmp/zt_tar_bench.sock";
    let shm_name = "zt_tar_bench";

    let label_fn = |filename: &str| -> i64 {
        if let Some(start) = filename.find("class_") {
            let rest = &filename[start + 6..];
            if let Some(end) = rest.find('/') {
                return rest[..end].parse().unwrap_or(0);
            }
        }
        0
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

    println!("[Rust] Initializing TarDataset...");
    let start_init = Instant::now();

    let processor = TarJpegProcessor::<f32, _>::new(Some(pipeline), label_fn)
        .expect("Failed to create processor");

    let rng = StdRng::seed_from_u64(42);
    let dataset = TarDataset::new(
        shard_paths,
        buffer_cap,
        None::<fn(&PathBuf) -> Result<usize, TarJpegProcessorError>>,
        processor,
        rng,
    )
    .expect("Failed to create dataset");

    let init_time = start_init.elapsed();
    println!(
        "[Rust] TarDataset initialized in {:.2}s (total_samples={})",
        init_time.as_secs_f32(),
        dataset.total_epoch_len()
    );

    let builder = ZeroTensorProducerBuilder::new(32 * 1024 * 1024, shm_name, socket_path);

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
