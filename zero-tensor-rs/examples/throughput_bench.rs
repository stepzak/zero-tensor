use indexmap::IndexMap;
use std::path::Path;
use zero_tensor_lib::core::{
    dataset::{
        ZeroTensorDataset,
        item::{ShapeType, TensorBatchLayout, TensorDT},
    },
    producer::ZeroTensorProducerBuilder,
};

const BATCH_SIZE: usize = 48;
const CHANNELS: ShapeType = 3;
const HEIGHT: ShapeType = 512;
const WIDTH: ShapeType = 512;
const STEPS: u64 = 600;
const NSLOTS: u64 = 32;

struct BenchDataset {
    raw_item_size: usize,
    meta: IndexMap<&'static str, TensorBatchLayout>,
    source_buffer: Vec<u8>,
}

impl BenchDataset {
    fn new(raw_item_size: usize) -> Self {
        let shape = vec![CHANNELS, HEIGHT, WIDTH];
        let strides = vec![HEIGHT * WIDTH, WIDTH, 1];
        let layout = TensorBatchLayout::new(shape.into(), strides.into(), TensorDT::F32);
        let mut meta = IndexMap::new();
        meta.insert("data", layout);
        let mut source = vec![0u8; raw_item_size];
        fastrand::Rng::new().fill(&mut source);

        Self {
            raw_item_size,
            meta,
            source_buffer: source,
        }
    }
}

impl<'a> ZeroTensorDataset<'a> for BenchDataset {
    type Error = std::io::Error;

    fn len(&self) -> usize {
        BATCH_SIZE * STEPS as usize
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn static_layouts(&self) -> Option<&IndexMap<&'static str, TensorBatchLayout>> {
        Some(&self.meta)
    }

    fn write_item_into<'layout, 'b, 'c>(
        &self,
        _idx: usize,
        writer: &mut zero_tensor_lib::core::writer::TensorWriter<'layout, 'b, 'c>,
    ) -> Result<(), Self::Error> {
        let _ = writer.write("data", |buf: &mut [u8]| -> Result<usize, std::io::Error> {
            let target = &mut buf[..self.raw_item_size];
            target.copy_from_slice(&self.source_buffer[..self.raw_item_size]);
            Ok(self.raw_item_size)
        });
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
    let dataset = BenchDataset::new(raw_item_size);

    let builder = ZeroTensorProducerBuilder::from_dataset(
        &dataset,
        shm_name,
        socket_path,
        BATCH_SIZE,
        BATCH_SIZE,
    )
    .expect("Failed to create builder");

    let slot_size = builder.slot_size;

    println!("[Rust Bench] Initializing ZeroTensorProducer...");
    println!(" -> SHM Name: {}", shm_name);
    println!(
        " -> Slot Size: {:.2} MB",
        slot_size as f64 / 1024.0 / 1024.0
    );
    println!(
        " -> Total SHM: {:.2} MB",
        (slot_size * NSLOTS) as f64 / 1024.0 / 1024.0
    );

    let mut producer = builder
        .num_slots(NSLOTS)
        .build()
        .expect("Failed to create producer");

    println!("[Rust Bench] Ready! Waiting for Python consumer to connect...");

    producer
        .start_streaming(&dataset, BATCH_SIZE)
        .expect("Streaming failed");

    println!("[Rust Bench] Finished streaming");
}
