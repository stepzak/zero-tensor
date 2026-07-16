use std::process::Command;
use std::thread;
use std::time::Duration;
use tempfile::tempdir;

use zero_tensor_lib::{
    dataset::ZeroTensorDataset,
    dataset::item::{TensorDT, TensorItemMeta},
    producer::ZeroTensorProducer,
};

struct NonContiguousMockDataset {
    len: usize,
}

impl ZeroTensorDataset for NonContiguousMockDataset {
    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len != 0
    }

    fn get_item(&self, idx: usize) -> Option<(Vec<u8>, TensorItemMeta)> {
        if idx >= self.len {
            return None;
        }

        let shape = vec![2, 3];
        let strides = vec![3, 1];
        let dt = TensorDT::F32;

        let total_elements = 6;
        let mut raw_data = vec![0u8; total_elements * 4];

        let f32_slice = unsafe {
            std::slice::from_raw_parts_mut(raw_data.as_mut_ptr() as *mut f32, total_elements)
        };

        f32_slice[0] = idx as f32;
        f32_slice[3] = idx as f32 + 0.5;

        let meta = TensorItemMeta::new(shape, strides, dt);
        Some((raw_data, meta))
    }
}

#[test]
fn test_rust_producer_python_consumer_e2e() {
    let dir = tempdir().unwrap();
    let socket_path = dir.path().join("integration_test.sock");
    let shm_name = "zt_integration_test_shm";

    let batch_size = 2;
    let steps = 4;
    let slot_size = 4096;

    let dataset = NonContiguousMockDataset {
        len: batch_size * steps,
    };

    let mut producer = ZeroTensorProducer::new(steps, slot_size, shm_name, &socket_path, None)
        .expect("Failed to initialize Rust producer");

    let consumer_socket = socket_path.clone();
    let consumer_shm = shm_name.to_string();

    let python_handle = thread::spawn(move || {
        thread::sleep(Duration::from_millis(50));

        let root_dir = std::env::current_dir()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();

        let python_project_dir = root_dir.join("zero-tensor-py");
        let consumer_script_path = root_dir.join("zero-tensor-rs/tests/integration_consumer.py");
        let python_path = python_project_dir.join("src");

        let status = Command::new("uv")
            .arg("--directory")
            .arg(&python_project_dir)
            .arg("run")
            .arg("python3")
            .arg(&consumer_script_path)
            .arg(&consumer_socket)
            .arg(&consumer_shm)
            .arg(slot_size.to_string())
            .arg(steps.to_string())
            .env("PYTHONPATH", python_path)
            .status()
            .expect("Failed to execute python command via uv");

        assert!(status.success(), "Python consumer exited with error status");

        assert!(status.success(), "Python consumer exited with error status");
    });

    producer
        .start_streaming(&dataset, batch_size)
        .expect("Streaming failed");

    python_handle
        .join()
        .expect("Python consumer thread panicked");
}
