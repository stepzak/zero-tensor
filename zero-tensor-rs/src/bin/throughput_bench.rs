use indexmap::IndexMap;
use std::path::Path;
use zero_tensor_lib::core::{
    buffer::tensor_meta::TensorHeader,
    dataset::{
        ZeroTensorDataset,
        item::{ShapeType, StrideType, TensorBatchLayout, TensorDT},
    },
    producer::ZeroTensorProducerBuilder,
    writer::TensorWriter,
};

const BATCH_SIZE: usize = 48;
const CHANNELS: ShapeType = 3;
const HEIGHT: ShapeType = 512;
const WIDTH: ShapeType = 512;
const STEPS: u64 = 200;
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
    let raw_item_size = item_elements as usize * 4;

    let ndims = 3;
    let tensor_header_size = size_of::<TensorHeader>();
    let shape_stride_size = size_of::<ShapeType>() + size_of::<StrideType>();
    let per_tensor_meta = tensor_header_size + ndims * shape_stride_size;
    let metadata_size_per_item =
        (per_tensor_meta + TensorWriter::ALIGNMENT - 1) & !(TensorWriter::ALIGNMENT - 1);

    let data_size_per_item =
        (raw_item_size + TensorWriter::ALIGNMENT - 1) & !(TensorWriter::ALIGNMENT - 1);
    let element_size = metadata_size_per_item + data_size_per_item;

    let slot_size = (element_size * BATCH_SIZE) as u64;

    println!("[Rust Bench] Initializing ZeroTensorProducer...");
    println!(" -> SHM Name: {}", shm_name);
    println!(
        " -> Metadata size per item: {} bytes",
        metadata_size_per_item
    );
    println!(" -> Data size per item: {} bytes", data_size_per_item);
    println!(" -> Element size per item: {} bytes", element_size);
    println!(
        " -> Slot Size: {:.2} MB",
        slot_size as f64 / 1024.0 / 1024.0
    );
    println!(
        " -> Total SHM: {:.2} MB",
        (slot_size * NSLOTS) as f64 / 1024.0 / 1024.0
    );

    let mut producer = ZeroTensorProducerBuilder::new(slot_size, shm_name, socket_path)
        .num_slots(NSLOTS)
        .build()
        .expect("Failed to create producer");

    let dataset = BenchDataset::new(raw_item_size);

    println!("[Rust Bench] Ready! Waiting for Python consumer to connect...");

    producer
        .start_streaming(&dataset, BATCH_SIZE)
        .expect("Streaming failed");

    println!("[Rust Bench] Finished streaming");
}
