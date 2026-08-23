use std::process::Command;
use std::thread;
use std::time::Duration;
use tempfile::tempdir;
use zero_tensor_lib::core::{
    buffer::get_dt_size,
    dataset::{
        ZTDatasetError, ZeroTensorDataset,
        item::{ShapeType, ShapeVec, StrideVec, TensorBatchLayout, TensorDT},
    },
    producer::ZeroTensorProducerBuilder,
};

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

struct DynamicDataset {
    shapes: Vec<(ShapeType, ShapeType)>, // (H, W)
}

impl DynamicDataset {
    fn new(num_items: usize) -> Self {
        let mut rng = fastrand::Rng::new();
        let shapes = (0..num_items)
            .map(|_| (rng.usize(2..6), rng.usize(2..6)))
            .collect();
        Self { shapes }
    }
}

impl ZeroTensorDataset for DynamicDataset {
    type Error = TestError;

    fn len(&self) -> usize {
        self.shapes.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get_batch_layout(&self, indices: &[usize]) -> Result<TensorBatchLayout, Self::Error> {
        if indices.is_empty() {
            return Err(TestError("Empty batch".into()));
        }

        let (max_h, max_w) = indices
            .iter()
            .map(|&i| self.shapes[i])
            .fold((0, 0), |(mh, mw), (h, w)| (mh.max(h), mw.max(w)));

        let mut shape = ShapeVec::new();
        shape.push(max_h);
        shape.push(max_w);

        let mut strides = StrideVec::new();
        strides.push(max_w);
        strides.push(1);

        Ok(TensorBatchLayout::new(shape, strides, TensorDT::F32))
    }

    fn write_item_into(&self, idx: usize, buf: &mut [u8]) -> Result<usize, Self::Error> {
        let (h, w) = self.shapes[idx];
        let total_els = (h * w) as usize;

        let f32_buf =
            unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut f32, total_els) };
        let mut bytes_written = 0;
        for r in 0..h {
            for c in 0..w {
                f32_buf[(r * w + c) as usize] = (r * 10 + c + idx * 100) as f32;
                bytes_written += size_of::<f32>();
            }
        }
        Ok(bytes_written)
    }
}

#[test]
fn test_dynamic_batching_e2e() {
    let dir = tempdir().unwrap();
    let socket_path = dir.path().join("dyn_test.sock");
    let shm_name = "zt_dyn_integration";

    let batch_size = 4;
    let steps = 3;
    let dataset = DynamicDataset::new(batch_size * steps);

    let max_item_bytes = (5 * 5 * get_dt_size(TensorDT::F32)) as usize;
    let slot_size = (max_item_bytes * batch_size) + 4096;

    let mut producer = ZeroTensorProducerBuilder::new(slot_size as u64, shm_name, &socket_path)
        .num_slots(3)
        .build()
        .expect("Failed to init producer");

    let consumer_socket = socket_path.clone();
    let consumer_shm = shm_name.to_string();

    let python_handle = thread::spawn(move || {
        thread::sleep(Duration::from_millis(200));

        let root_dir = std::env::current_dir()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let python_project_dir = root_dir.join("zero-tensor-py");
        let consumer_script = root_dir.join("zero-tensor-rs/tests/integration_consumer.py");
        let python_path = python_project_dir.join("src");

        let status = Command::new("uv")
            .arg("--directory")
            .arg(&python_project_dir)
            .arg("run")
            .arg("python3")
            .arg(&consumer_script)
            .arg(&consumer_socket)
            .arg(&consumer_shm)
            .arg(slot_size.to_string())
            .arg(batch_size.to_string())
            .arg(steps.to_string())
            .env("PYTHONPATH", python_path)
            .status()
            .expect("Failed to execute python consumer");

        assert!(
            status.success(),
            "Python consumer failed with status: {:?}",
            status
        );
    });

    producer
        .start_streaming(&dataset, batch_size)
        .expect("Streaming failed");
    python_handle.join().expect("Consumer thread panicked");
}
