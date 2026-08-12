use std::path::Path;
use zero_tensor_lib::{
    dataset::{
        ZeroTensorDataset,
        item::{ShapeType, TensorBatchLayout, TensorDT},
    },
    producer::ZeroTensorProducerBuilder,
};

const BATCH_SIZE: usize = 32;
const CHANNELS: ShapeType = 3;
const HEIGHT: ShapeType = 512;
const WIDTH: ShapeType = 512;
const STEPS: usize = 400;
const NSLOTS: usize = 24;

struct BenchDataset {
    raw_item_size: usize,
    meta: TensorBatchLayout,
    source_buffer: Vec<u8>,
}

impl BenchDataset {
    fn new(raw_item_size: usize) -> Self {
        let shape = vec![CHANNELS, HEIGHT, WIDTH];
        let strides = vec![HEIGHT * WIDTH, WIDTH, 1];
        let meta = TensorBatchLayout::new(shape.into(), strides.into(), TensorDT::F32);
        let mut source = vec![0u8; raw_item_size];
        fastrand::Rng::new().fill(&mut source);

        Self {
            raw_item_size,
            meta,
            source_buffer: source,
        }
    }
}

impl ZeroTensorDataset for BenchDataset {
    type Error = std::io::Error;

    fn len(&self) -> usize {
        BATCH_SIZE * STEPS
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get_batch_layout(&self, _idxs: &[usize]) -> Result<TensorBatchLayout, Self::Error> {
        Ok(self.meta.clone())
    }

    fn write_item_into(&self, idx: usize, buf: &mut [u8]) -> Result<(), Self::Error> {
        let target = &mut buf[..self.raw_item_size];
        target.copy_from_slice(&self.source_buffer[..self.raw_item_size]);

        for byte in target.iter_mut() {
            *byte = byte.wrapping_add(idx as u8);
        }

        Ok(())
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

    let mut producer = ZeroTensorProducerBuilder::new(slot_size, shm_name, socket_path)
        .num_slots(NSLOTS)
        .build()
        .expect("Failed to create producer");

    let dataset = BenchDataset::new(raw_item_size as usize);

    println!("[Rust Bench] Ready! Waiting for Python consumer to connect...");

    producer
        .start_streaming(&dataset, BATCH_SIZE)
        .expect("Streaming failed");

    println!("[Rust Bench] Finished streaming");
}
