use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::Instant;

use rand::SeedableRng;
use turbojpeg::{Compressor, Image, PixelFormat};

use zero_tensor_lib::augmentation::AugmentationPipeline;
use zero_tensor_lib::augmentation::default::crop::RandomCrop;
use zero_tensor_lib::augmentation::default::flip::RandomHorizontalFlip;
use zero_tensor_lib::augmentation::default::normalize::Normalize;
use zero_tensor_lib::augmentation::default::resize::Resize;
use zero_tensor_lib::core::dataset::ZeroTensorDataset;
use zero_tensor_lib::core::producer::ZeroTensorProducerBuilder;
use zero_tensor_lib::dataset::tar::TarDataset;
use zero_tensor_lib::dataset::tar::processors::TarJpegProcessor;

const NUM_SHARDS: usize = 4;
const IMAGES_PER_SHARD: usize = 2500;
const NUM_CLASSES: usize = 10;
const BATCH_SIZE: usize = 32;
const BUFFER_CAP: usize = 1024;

fn generate_jpeg(width: usize, height: usize, seed: usize) -> Vec<u8> {
    let mut compressor = Compressor::new().expect("Failed to create compressor");
    let mut pixels = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            pixels[idx] = ((x + seed * 17) % 256) as u8;
            pixels[idx + 1] = ((y + seed * 31) % 256) as u8;
            pixels[idx + 2] = ((x + y + seed * 53) % 256) as u8;
        }
    }
    let image = Image {
        pixels: pixels.as_slice(),
        width,
        pitch: width * 3,
        height,
        format: PixelFormat::RGB,
    };
    compressor.compress_to_vec(image).unwrap()
}

fn generate_tar_shards(dataset_dir: &Path) -> Vec<PathBuf> {
    if dataset_dir.exists() {
        fs::remove_dir_all(dataset_dir).expect("Failed to clean dataset directory");
    }
    fs::create_dir_all(dataset_dir).expect("Failed to create dataset directory");

    let mut shard_paths = Vec::new();

    for shard_idx in 0..NUM_SHARDS {
        let shard_path = dataset_dir.join(format!("shard_{:03}.tar", shard_idx));
        let file = File::create(&shard_path).expect("Failed to create shard file");
        let mut builder = tar::Builder::new(file);

        for img_idx in 0..IMAGES_PER_SHARD {
            let global_idx = shard_idx * IMAGES_PER_SHARD + img_idx;
            let class_id = global_idx % NUM_CLASSES;

            let width = 200 + (global_idx * 37 % 301);
            let height = 200 + (global_idx * 53 % 301);

            let jpeg_data = generate_jpeg(width, height, global_idx);
            let filename = format!("class_{:03}/img_{:05}.jpg", class_id, global_idx);

            let mut header = tar::Header::new_gnu();
            header.set_path(&filename).unwrap();
            header.set_size(jpeg_data.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();

            builder
                .append(&header, jpeg_data.as_slice())
                .expect("Failed to append to tar");
        }

        builder.finish().expect("Failed to finish tar");
        shard_paths.push(shard_path);
    }

    shard_paths
}

fn main() {
    let dataset_dir = dirs::cache_dir()
        .expect("Failed to get cache directory")
        .join("zero_tensor_tar_bench");

    println!("[Rust] Dataset directory: {:?}", dataset_dir);

    let socket_path = "/tmp/zt_tar_bench.sock";
    let shm_name = "zt_tar_bench";
    let num_slots = 16;

    println!(
        "[Rust] Generating {} shards with {} images each ({} classes)...",
        NUM_SHARDS, IMAGES_PER_SHARD, NUM_CLASSES
    );
    let gen_start = Instant::now();
    let shard_paths = generate_tar_shards(&dataset_dir);
    println!(
        "[Rust] Generated {} shards in {:.2}s",
        shard_paths.len(),
        gen_start.elapsed().as_secs_f32()
    );

    let pipeline = AugmentationPipeline::<f32>::new()
        .then(Resize::new(256, 256))
        .unwrap()
        .then(RandomCrop::new(224, 224))
        .unwrap()
        .then(RandomHorizontalFlip::new(0.5).unwrap())
        .unwrap()
        .then(Normalize::imagenet())
        .unwrap();

    let label_fn = |filename: &str| -> i64 {
        if let Some(start) = filename.find("class_") {
            let rest = &filename[start + 6..];
            if let Some(end) = rest.find('/') {
                return rest[..end].parse().unwrap_or(0);
            }
        }
        0
    };

    println!("[Rust] Initializing TarDataset...");
    let init_start = Instant::now();

    let processor = TarJpegProcessor::<f32, _>::new(Some(pipeline), label_fn)
        .expect("Failed to create processor");

    let rng = rand::rngs::StdRng::seed_from_u64(42);
    let dataset = TarDataset::new(
        shard_paths,
        BUFFER_CAP,
        None::<fn(&PathBuf) -> Result<usize, _>>,
        processor,
        rng,
    )
    .expect("Failed to create dataset");

    let init_time = init_start.elapsed();
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
    println!("[Rust] Socket: {}", socket_path);
    println!("[Rust] SHM: {}", shm_name);
    println!("[Rust] Streaming batches of size {}...", BATCH_SIZE);

    let stream_start = Instant::now();
    producer
        .start_streaming(&dataset, BATCH_SIZE)
        .expect("Streaming failed");

    let stream_time = stream_start.elapsed();
    println!(
        "[Rust] Streaming completed in {:.2}s",
        stream_time.as_secs_f32()
    );
    println!("[Rust] Done!");

    fs::remove_dir_all(&dataset_dir).ok();
}
