use std::path::Path;
use zero_tensor_lib::{
    dataset::{
        ZeroTensorDataset,
        item::{ShapeType, TensorDT, TensorItemMeta},
    },
    producer::ZeroTensorProducer,
};

const BATCH_SIZE: usize = 32;
const CHANNELS: ShapeType = 3;
const HEIGHT: ShapeType = 512;
const WIDTH: ShapeType = 512;
const STEPS: usize = 50;

struct BenchDataset {
    raw_item_size: usize,
}

impl ZeroTensorDataset for BenchDataset {
    fn len(&self) -> usize {
        BATCH_SIZE * STEPS
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get_item(&self, _idx: usize) -> Option<(Vec<u8>, TensorItemMeta)> {
        let shape = vec![CHANNELS, HEIGHT, WIDTH];
        let strides = vec![HEIGHT * WIDTH, WIDTH, 1];
        let meta = TensorItemMeta::new(shape, strides, TensorDT::F32);

        let mut data = vec![0u8; self.raw_item_size];
        for (i, slot) in data.iter_mut().enumerate() {
            *slot = ((_idx + i) % 255) as u8;
        }

        Some((data, meta))
    }
}

fn main() {
    let socket_path = Path::new("/tmp/zt_bench.sock");
    let shm_name = "zt_bench";

    if socket_path.exists() {
        let _ = std::fs::remove_file(socket_path);
    }

    let item_elements = CHANNELS * HEIGHT * WIDTH;
    let raw_item_size = item_elements * 4;

    let slot_size = (raw_item_size * BATCH_SIZE as u32) + 4096;

    println!("[Rust Bench] Initializing ZeroTensorProducer...");
    println!(" -> SHM Name: {}", shm_name);
    println!(
        " -> Slot Size: {:.2} MB",
        slot_size as f64 / 1024.0 / 1024.0
    );

    let mut producer = ZeroTensorProducer::new(
        STEPS,
        slot_size as usize,
        shm_name,
        socket_path,
        None,
        None,
        false,
    )
    .expect("Failed to create producer");

    let dataset = BenchDataset {
        raw_item_size: raw_item_size as usize,
    };

    println!("[Rust Bench] Ready! Waiting for Python consumer to connect...");

    producer
        .start_streaming(&dataset, BATCH_SIZE)
        .expect("Streaming failed");

    println!("[Rust Bench] Finished streaming. Cleaning up...");
    if socket_path.exists() {
        let _ = std::fs::remove_file(socket_path);
    }
}
