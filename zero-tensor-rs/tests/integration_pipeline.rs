use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;
use tempfile::tempdir;
use zero_tensor_lib::core::{
    buffer::{ZeroTensorBuffer, control_block::ZeroTensorControlBlock, tensor_meta::TensorHeader},
    dataset::{
        ZTDatasetError, ZeroTensorDataset,
        item::{ShapeVec, StrideVec, TensorBatchLayout, TensorDT},
    },
    producer::ZeroTensorProducerBuilder,
};
use zero_tensor_lib::pipeline::Pipeline;
use zero_tensor_lib::transform::{Add, Scale};

#[derive(Debug)]
struct TestError(String);

impl std::fmt::Display for TestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for TestError {}

impl ZTDatasetError for TestError {
    fn index(&self) -> Option<usize> {
        None
    }
}

struct SimpleF32Dataset {
    len: usize,
}

impl SimpleF32Dataset {
    fn new(len: usize) -> Self {
        Self { len }
    }
}

impl ZeroTensorDataset for SimpleF32Dataset {
    type Error = TestError;

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn get_batch_layout(&self, _idxs: &[usize]) -> Result<TensorBatchLayout, Self::Error> {
        Ok(TensorBatchLayout::new(
            ShapeVec::from_slice(&[4]),
            StrideVec::from_slice(&[1]),
            TensorDT::F32,
        ))
    }

    fn write_item_into(&self, idx: usize, buf: &mut [u8]) -> Result<usize, Self::Error> {
        let f32_buf = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut f32, 4) };
        let mut bytes_written = 0usize;
        for i in 0..4 {
            f32_buf[i] = (idx + i) as f32;
            bytes_written += size_of::<f32>();
        }
        Ok(bytes_written)
    }
}

fn spawn_consumer(
    socket_path: PathBuf,
    shm_name: String,
    slot_size: u64,
    nslots: u64,
    expected_batches: usize,
) -> thread::JoinHandle<Vec<Vec<f32>>> {
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(100));

        let mut stream = UnixStream::connect(&socket_path).expect("Consumer: connect failed");
        stream
            .write_all(b"START\n")
            .expect("Consumer: write START failed");

        let mut handshake = Vec::new();
        let mut buf = [0u8; 4096];
        loop {
            let n = stream
                .read(&mut buf)
                .expect("Consumer: read handshake failed");
            handshake.extend_from_slice(&buf[..n]);
            if handshake.contains(&b'\n') {
                break;
            }
        }

        let total_size = ZeroTensorControlBlock::SIZE as u64 + nslots * slot_size;
        let mut buffer = ZeroTensorBuffer::open(&shm_name, total_size as usize)
            .expect("Consumer: open SHM failed");

        let mut tail = 0u64;
        let mut results = Vec::with_capacity(expected_batches);

        for _ in 0..expected_batches {
            loop {
                let head = buffer.control_block().head.load(Ordering::Acquire);
                if head > tail {
                    break;
                }
                thread::sleep(Duration::from_micros(100));
            }

            let slot_idx = (tail % nslots) as usize;
            let slot_offset = ZeroTensorControlBlock::slot_offset(slot_idx, slot_size as usize);

            loop {
                let slot_bytes = buffer
                    .get_slot_slice(slot_offset, slot_size as usize)
                    .expect("get_slot_slice failed");
                let header_ptr = slot_bytes.as_ptr() as *const TensorHeader;
                let is_ready = unsafe { (*header_ptr).is_ready.load(Ordering::Acquire) };
                if is_ready == 1 {
                    break;
                }
                thread::sleep(Duration::from_micros(100));
            }

            let slot_bytes = buffer
                .get_slot_slice(slot_offset, slot_size as usize)
                .expect("get_slot_slice failed");
            let header_ptr = slot_bytes.as_ptr() as *const TensorHeader;
            let header = unsafe { &*header_ptr };
            let offs = header.get_offsets();

            let data_ptr = unsafe { slot_bytes.as_ptr().add(offs.data()) as *const f32 };
            let data_slice = unsafe { std::slice::from_raw_parts(data_ptr, 16) };
            results.push(data_slice.to_vec());

            tail += 1;
            buffer.control_block().tail.store(tail, Ordering::Release);
        }

        buffer.control_block_mut().stop();

        thread::sleep(Duration::from_millis(100));

        let _ = stream.shutdown(std::net::Shutdown::Both);

        results
    })
}

#[test]
fn test_pipeline_applies_single_transform_to_items() {
    let dir = tempdir().unwrap();
    let socket_path = dir.path().join("pipeline_single.sock");
    let shm_name = "zt_pipeline_single";

    let batch_size = 4;
    let steps = 1;
    let dataset = SimpleF32Dataset::new(batch_size * steps);

    let slot_size = 4096u64;
    let nslots = 2u64;

    let pipeline = Pipeline::new().then(Scale::new(2.0f32));

    let mut producer = ZeroTensorProducerBuilder::new(slot_size, shm_name, &socket_path)
        .num_slots(nslots)
        .pipeline(pipeline)
        .build()
        .expect("Failed to build producer");

    let consumer_handle = spawn_consumer(
        socket_path.clone(),
        shm_name.to_string(),
        slot_size,
        nslots,
        steps,
    );

    let result = producer.start_streaming(&dataset, batch_size);

    assert!(
        result.is_ok()
            || matches!(
                result,
                Err(zero_tensor_lib::core::producer::ZTProducerErr::IoError(_))
            ),
        "Unexpected result: {:?}",
        result
    );

    let results = consumer_handle.join().expect("Consumer panicked");

    assert_eq!(results.len(), 1);
    let batch = &results[0];

    assert_eq!(&batch[0..4], &[0.0, 2.0, 4.0, 6.0]);
    assert_eq!(&batch[4..8], &[2.0, 4.0, 6.0, 8.0]);
    assert_eq!(&batch[8..12], &[4.0, 6.0, 8.0, 10.0]);
    assert_eq!(&batch[12..16], &[6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn test_pipeline_applies_multiple_transforms_in_order() {
    let dir = tempdir().unwrap();
    let socket_path = dir.path().join("pipeline_multi.sock");
    let shm_name = "zt_pipeline_multi";

    let batch_size = 4;
    let steps = 1;
    let dataset = SimpleF32Dataset::new(batch_size * steps);

    let slot_size = 4096u64;
    let nslots = 2u64;

    let pipeline = Pipeline::new()
        .then(Scale::new(2.0f32))
        .then(Add::new(10.0f32));

    let mut producer = ZeroTensorProducerBuilder::new(slot_size, shm_name, &socket_path)
        .num_slots(nslots)
        .pipeline(pipeline)
        .build()
        .expect("Failed to build producer");

    let consumer_handle = spawn_consumer(
        socket_path.clone(),
        shm_name.to_string(),
        slot_size,
        nslots,
        steps,
    );

    let result = producer.start_streaming(&dataset, batch_size);
    assert!(
        result.is_ok()
            || matches!(
                result,
                Err(zero_tensor_lib::core::producer::ZTProducerErr::IoError(_))
            ),
        "Unexpected result: {:?}",
        result
    );

    let results = consumer_handle.join().expect("Consumer panicked");

    assert_eq!(results.len(), 1);
    let batch = &results[0];

    assert_eq!(&batch[0..4], &[10.0, 12.0, 14.0, 16.0]);
    assert_eq!(&batch[4..8], &[12.0, 14.0, 16.0, 18.0]);
    assert_eq!(&batch[8..12], &[14.0, 16.0, 18.0, 20.0]);
    assert_eq!(&batch[12..16], &[16.0, 18.0, 20.0, 22.0]);
}

#[test]
fn test_empty_pipeline_does_not_modify_data() {
    let dir = tempdir().unwrap();
    let socket_path = dir.path().join("pipeline_empty.sock");
    let shm_name = "zt_pipeline_empty";

    let batch_size = 4;
    let steps = 1;
    let dataset = SimpleF32Dataset::new(batch_size * steps);

    let slot_size = 4096u64;
    let nslots = 2u64;

    let pipeline = Pipeline::new();

    let mut producer = ZeroTensorProducerBuilder::new(slot_size, shm_name, &socket_path)
        .num_slots(nslots)
        .pipeline(pipeline)
        .build()
        .expect("Failed to build producer");

    let consumer_handle = spawn_consumer(
        socket_path.clone(),
        shm_name.to_string(),
        slot_size,
        nslots,
        steps,
    );

    let result = producer.start_streaming(&dataset, batch_size);
    assert!(
        result.is_ok()
            || matches!(
                result,
                Err(zero_tensor_lib::core::producer::ZTProducerErr::IoError(_))
            ),
        "Unexpected result: {:?}",
        result
    );

    let results = consumer_handle.join().expect("Consumer panicked");

    assert_eq!(results.len(), 1);
    let batch = &results[0];

    assert_eq!(&batch[0..4], &[0.0, 1.0, 2.0, 3.0]);
    assert_eq!(&batch[4..8], &[1.0, 2.0, 3.0, 4.0]);
}
