use std::process::Command;
use std::thread;
use std::time::Duration;
use tempfile::tempdir;

use zero_tensor_lib::{
    buffer::get_dt_size,
    dataset::{
        ZeroTensorDataset,
        item::{TensorDT, TensorItemMeta},
    },
    producer::ZeroTensorProducerBuilder,
};

struct MockDataset {
    len: usize,
    meta: TensorItemMeta,
}

impl MockDataset {
    pub fn new(len: usize) -> Self {
        let shape = vec![2, 3];
        let strides = vec![3, 1];
        let dt = TensorDT::F32;

        let meta = TensorItemMeta::new(shape, strides, dt);

        Self { len, meta }
    }
}

impl ZeroTensorDataset for MockDataset {
    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn get_metadata(&self, _idx: usize) -> Option<TensorItemMeta> {
        Some(self.meta.clone())
    }

    fn get_item_into(&self, idx: usize, buf: &mut [u8]) -> Option<TensorItemMeta> {
        if idx >= self.len {
            return None;
        }
        let meta = self.get_metadata(idx)?;
        let total_elements = meta.shape().iter().product::<u32>() as usize;
        let total_bytes = total_elements * get_dt_size(meta.dt());

        if buf.len() < total_bytes {
            return None;
        }
        match meta.dt() {
            TensorDT::F32 => {
                let f32_slice = unsafe {
                    std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut f32, total_elements)
                };
                (0..total_elements).for_each(|i| {
                    f32_slice[i] = idx as f32 + i as f32 * 0.5;
                });
            }
            _ => {
                buf[..total_bytes].fill(0);
            }
        }

        Some(meta)
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

    let dataset = MockDataset::new(batch_size * steps);

    let mut producer = ZeroTensorProducerBuilder::new(steps, slot_size, shm_name, &socket_path)
        .build()
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
