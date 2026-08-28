use indexmap::IndexMap;
use std::process::Command;
use std::thread;
use std::time::Duration;
use tempfile::tempdir;
use zero_tensor_lib::core::{
    dataset::{
        ZeroTensorDataset,
        item::{TensorBatchLayout, TensorDT},
    },
    producer::ZeroTensorProducerBuilder,
};

struct VerificationDataset {
    total_items: usize,
}

impl VerificationDataset {
    fn new(total_items: usize) -> Self {
        Self { total_items }
    }
}

impl<'a> ZeroTensorDataset<'a> for VerificationDataset {
    type Error = std::io::Error;

    fn len(&self) -> usize {
        self.total_items
    }

    fn is_empty(&self) -> bool {
        false
    }

    fn static_layouts(&self) -> Option<&IndexMap<&'static str, TensorBatchLayout>> {
        use std::sync::OnceLock;
        static LAYOUTS: OnceLock<IndexMap<&'static str, TensorBatchLayout>> = OnceLock::new();
        Some(LAYOUTS.get_or_init(|| {
            let mut map = IndexMap::new();
            map.insert(
                "img",
                TensorBatchLayout::new(vec![2, 2].into(), vec![2, 1].into(), TensorDT::F32),
            );
            map.insert(
                "lbl",
                TensorBatchLayout::new(vec![1].into(), vec![1].into(), TensorDT::I32),
            );
            map
        }))
    }

    fn write_item_into<'layout, 'b, 'c>(
        &self,
        idx: usize,
        writer: &mut zero_tensor_lib::core::writer::TensorWriter<'layout, 'b, 'c>,
    ) -> Result<(), Self::Error> {
        writer
            .write("img", |buf| -> Result<usize, Self::Error> {
                let floats: &mut [f32] = bytemuck::cast_slice_mut(&mut buf[..16]);
                for i in 0..4 {
                    floats[i] = (idx * 10 + i) as f32;
                }
                Ok(16)
            })
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        writer
            .write("lbl", |buf| -> Result<usize, Self::Error> {
                let ints: &mut [i32] = bytemuck::cast_slice_mut(&mut buf[..4]);
                ints[0] = (idx * 100) as i32;
                Ok(4)
            })
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        Ok(())
    }
}

#[test]
fn test_e2e_batch_verification() {
    let dir = tempdir().unwrap();
    let socket_path = dir.path().join("e2e_verify.sock");
    let shm_name = "zt_e2e_verify_shm";

    let batch_size = 4;
    let total_items = 12;
    let dataset = VerificationDataset::new(total_items);

    let slot_size = 4096u64;

    let mut producer = ZeroTensorProducerBuilder::new(slot_size, shm_name, &socket_path)
        .num_slots(4)
        .build()
        .expect("Failed to init producer");

    let consumer_socket = socket_path.clone();
    let consumer_shm = shm_name.to_string();

    let consumer_handle = thread::spawn(move || {
        thread::sleep(Duration::from_millis(100));

        let root_dir = std::env::current_dir()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();

        let python_project_dir = root_dir.join("zero-tensor-py");
        let consumer_script = root_dir.join("zero-tensor-rs/tests/integration_multiple_tensors.py");
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
            .arg((total_items / batch_size).to_string())
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

    consumer_handle.join().expect("Consumer thread panicked");

    println!("E2E Batch Verification PASSED");
}
