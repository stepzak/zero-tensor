use crate::core::writer::TensorWriter;
use crate::core::dataset::item::{TensorBatchLayout, TensorDT};
use indexmap::IndexMap;
use tempfile::tempdir;
use std::thread;
use std::time::Duration;

use super::*;

struct FailingDataset;

impl<'a> ZeroTensorDataset<'a> for FailingDataset {
    type Error = std::io::Error;

    fn len(&self) -> usize {
        10
    }

    fn is_empty(&self) -> bool {
        false
    }

    fn dynamic_layouts(
        &self,
        _idxs: &[usize],
    ) -> Result<IndexMap<&'a str, TensorBatchLayout>, Self::Error> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Simulated dataset error",
        ))
    }

    fn write_item_into<'layout, 'b, 'c>(
        &self, 
        _idx: usize, 
        _writer: &mut TensorWriter<'layout, 'b, 'c>
    ) -> Result<(), Self::Error>
    {
        Ok(())
    }
}

#[test]
fn test_dataset_failure() {
    let dir = tempdir().unwrap();
    let socket_path = dir.path().join("zero_tensor.sock");
    let shm_name = "zt_test_buffer_fail";

    let batch_size = 2;
    let dataset = FailingDataset;

    let mut producer = ZeroTensorProducerBuilder::new(2048, shm_name, &socket_path)
        .build()
        .expect("Failed to init producer");

    let consumer_socket = socket_path.clone();
    let consumer_shm_name = shm_name.to_string();

    let consumer_handle = thread::spawn(move || {
        thread::sleep(Duration::from_millis(20));

        let mut stream = match UnixStream::connect(&consumer_socket) {
            Ok(s) => s,
            Err(_) => return, 
        };

        let _ = ZeroTensorBuffer::open(&consumer_shm_name, 2048 * 2)
            .expect("Consumer failed to open SHM");

        let _ = stream.write_all(b"START\n");

        let mut buf = [0; 16];
        let res = stream.read(&mut buf);
        assert!(
            res.is_ok()
                || matches!(
                    res.unwrap_err().kind(),
                    std::io::ErrorKind::UnexpectedEof
                        | std::io::ErrorKind::ConnectionReset
                        | std::io::ErrorKind::BrokenPipe
                )
        );
    });

    let res = producer.start_streaming(&dataset, batch_size);
    let _ = consumer_handle.join();

    assert!(res.is_err(), "Expected dataset error, got {:?}", res);
}


#[test]
fn test_raii_producer_cleans_up_socket_on_drop() {
    let dir = tempdir().unwrap();
    let sock_path = dir.path().join("integration_test.sock");
    let shm_name = "zt_integration_test_shm";

    {
        let _ = ZeroTensorProducerBuilder::new(4096, shm_name, &sock_path)
            .overwrite_socket(true)
            .read_timeout(1000)
            .build()
            .expect("Failed to create producer");
    }

    assert!(
        !sock_path.exists(),
        "Socket file must be unlinked after producer is dropped"
    );

    #[cfg(target_os = "linux")]
    {
        use std::path::PathBuf;
        let shm_path = PathBuf::from(format!("/dev/shm/{}", shm_name));
        assert!(
            !shm_path.exists(),
            "Shared memory segment should be unlinked on drop"
        );
    }
}

#[test]
fn test_raii_cleanup_on_panic() {
    let dir = tempdir().unwrap();
    let sock_path = dir.path().join("integration_test_panic.sock");
    let shm_name = "zt_integration_test_shm_panic";

    let handle = std::thread::spawn({
        let sock_path = sock_path.clone();
        move || {
            let _producer = ZeroTensorProducerBuilder::new(4096, shm_name, &sock_path)
                .read_timeout(1000)
                .build()
                .unwrap();

            assert!(sock_path.exists());
            panic!("Simulated worker panic inside task!");
        }
    });

    let _ = handle.join();

    assert!(
        !sock_path.exists(),
        "Socket should be cleaned up even after panic unwinding"
    );
}

#[test]
fn test_producer_detects_dead_consumer() {
    let dir = tempdir().unwrap();
    let sock_path = dir.path().join("death_test.sock");
    let shm_name = "zt_death_test_shm";

    struct TinyDataset;
    
    impl<'a> ZeroTensorDataset<'a> for TinyDataset {
        type Error = std::io::Error;

        fn len(&self) -> usize {
            100
        }

        fn is_empty(&self) -> bool {
            false
        }

        fn dynamic_layouts(
            &self,
            _idxs: &[usize],
        ) -> Result<IndexMap<&'a str, TensorBatchLayout>, Self::Error>
        {
             let mut layouts = IndexMap::new();
            layouts.insert(
                "data",
                TensorBatchLayout::new(vec![4].into(), vec![1].into(), TensorDT::F32),
            );
            Ok(layouts)
        }

        fn write_item_into<'layout, 'b, 'c>(
            &self, 
            _idx: usize, 
            writer: &mut TensorWriter<'layout, 'b, 'c>
        ) -> Result<(), Self::Error>
        {
            writer
                .write("data", |buf| {
                    buf[..16].fill(0);
                    Ok::<usize, std::io::Error>(16)
                })
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
            Ok(())
        }
    }

    let mut producer = ZeroTensorProducerBuilder::new(4096, shm_name, &sock_path)
        .num_slots(2)
        .read_timeout(200)
        .build()
        .expect("failed to build producer");

    let consumer_sock_path = sock_path.clone();
    let consumer_handle = thread::spawn(move || {
        let mut stream = loop {
            match UnixStream::connect(&consumer_sock_path) {
                Ok(s) => break s,
                Err(_) => thread::sleep(Duration::from_millis(20)),
            }
        };
        stream.write_all(b"START\n").unwrap();
        let mut buf = [0u8; 4096];
        let mut handshake = Vec::new();
        loop {
            let n = stream.read(&mut buf).unwrap();
            if n == 0 {
                break;
            }
            handshake.extend_from_slice(&buf[..n]);
            if handshake.contains(&b'\n') {
                break;
            }
        }
        assert!(String::from_utf8_lossy(&handshake).starts_with("ZT"));

        drop(stream);
    });

    let start = std::time::Instant::now();
    let result = producer.start_streaming(&TinyDataset, 1);
    let elapsed = start.elapsed();

    consumer_handle
        .join()
        .expect("fake consumer thread panicked");

    assert!(
        result.is_err(),
        "producer should return an error on consumer disconnect, not hang or return Ok(()) silently"
    );
    assert!(
        elapsed < Duration::from_secs(3),
        "producer took too long ({:?}) to notice the dead consumer",
        elapsed
    );

    drop(producer);

    assert!(
        !sock_path.exists(),
        "socket should be cleaned up (Drop must have run)"
    );
    #[cfg(target_os = "linux")]
    {
        use std::path::PathBuf;
        let shm_path = PathBuf::from(format!("/dev/shm/{}", shm_name));
        assert!(
            !shm_path.exists(),
            "shm segment should be cleaned up (Drop must have run)"
        );
    }
}

#[test]
fn test_multi_tensor_dataset() {
    let dir = tempdir().unwrap();
    let sock_path = dir.path().join("multi_tensor.sock");
    let shm_name = "zt_multi_tensor_shm";

    struct MultiTensorDataset;
    
    impl<'a> ZeroTensorDataset<'a> for MultiTensorDataset {
        type Error = std::io::Error;

        fn len(&self) -> usize {
            10
        }

        fn is_empty(&self) -> bool {
            false
        }

        fn dynamic_layouts(
            &self,
            _idxs: &[usize],
        ) -> Result<IndexMap<&'a str, TensorBatchLayout>, Self::Error> {
            let mut layouts = IndexMap::new();
            layouts.insert(
                "image",
                TensorBatchLayout::new(vec![4].into(), vec![1].into(), TensorDT::F32),
            );
            layouts.insert(
                "label",
                TensorBatchLayout::new(vec![1].into(), vec![1].into(), TensorDT::I32),
            );
            Ok(layouts)
        }

        fn write_item_into<'layout, 'b, 'c>(
            &self, 
            idx: usize, 
            writer: &mut TensorWriter<'layout, 'b, 'c>
        ) -> Result<(), Self::Error>
        {
            writer
                .write("image", |buf| {
                    let floats: &mut [f32] = bytemuck::cast_slice_mut(&mut buf[..16]);
                    for (i, f) in floats.iter_mut().enumerate() {
                        *f = (idx * 4 + i) as f32;
                    }
                    Ok::<usize, std::io::Error>(16)
                })
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

            writer
                .write("label", |buf| {
                    let ints: &mut [i32] = bytemuck::cast_slice_mut(&mut buf[..4]);
                    ints[0] = idx as i32;
                    Ok::<usize, std::io::Error>(4)
                })
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

            Ok(())
        }
    }

    let mut producer = ZeroTensorProducerBuilder::new(4096, shm_name, &sock_path)
        .num_slots(2)
        .read_timeout(1000)
        .build()
        .expect("failed to build producer");

    let consumer_sock_path = sock_path.clone();
    let consumer_handle = thread::spawn(move || {
        let mut stream = loop {
            match UnixStream::connect(&consumer_sock_path) {
                Ok(s) => break s,
                Err(_) => thread::sleep(Duration::from_millis(20)),
            }
        };
        stream.write_all(b"START\n").unwrap();

        let mut buf = [0u8; 4096];
        let mut handshake = Vec::new();
        loop {
            let n = stream.read(&mut buf).unwrap();
            if n == 0 {
                break;
            }
            handshake.extend_from_slice(&buf[..n]);
            if handshake.contains(&b'\n') {
                break;
            }
        }

        let handshake_str = String::from_utf8_lossy(&handshake).to_string();
        assert!(handshake_str.starts_with("ZT"));
        assert!(
            handshake_str.contains("image") && handshake_str.contains("label"),
            "Handshake should contain tensor keys, got: {}",
            handshake_str
        );

        drop(stream);
    });

    let result = producer.start_streaming(&MultiTensorDataset, 2);
    let _ = consumer_handle.join();

    let _ = result;
}