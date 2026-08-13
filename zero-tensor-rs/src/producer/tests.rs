use crate::dataset::item::TensorBatchLayout;
use tempfile::tempdir;

use super::*;

struct FailingDataset;

impl ZeroTensorDataset for FailingDataset {
    type Error = std::io::Error;

    fn len(&self) -> usize {
        10
    }
    fn is_empty(&self) -> bool {
        false
    }

    fn get_batch_layout(&self, _idxs: &[usize]) -> Result<TensorBatchLayout, Self::Error> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Simulated dataset error",
        ))
    }

    fn write_item_into(&self, _idx: usize, _buf: &mut [u8]) -> Result<(), Self::Error> {
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
        let mut stream =
            UnixStream::connect(&consumer_socket).expect("Consumer failed to connect to socket");

        let _ = ZeroTensorBuffer::open(&consumer_shm_name, 2048 * 2)
            .expect("Consumer failed to open SHM");

        stream.write_all(b"START\n").expect("Write err");

        let mut buf = [0; 16];
        let res = stream.read(&mut buf);
        assert!(res.is_ok() || res.unwrap_err().kind() == std::io::ErrorKind::UnexpectedEof);
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
    let sock_path = dir.path().join("integration_test.sock");
    let shm_name = "zt_integration_test_shm";

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
    impl ZeroTensorDataset for TinyDataset {
        type Error = std::io::Error;
        fn len(&self) -> usize {
            100
        }
        fn is_empty(&self) -> bool {
            false
        }
        fn get_batch_layout(&self, _idxs: &[usize]) -> Result<TensorBatchLayout, Self::Error> {
            Ok(TensorBatchLayout::new(
                vec![4].into(),
                vec![1].into(),
                TensorDT::F32,
            ))
        }
        fn write_item_into(&self, _idx: usize, buf: &mut [u8]) -> Result<(), Self::Error> {
            buf[..16].fill(0);
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
            handshake.extend_from_slice(&buf[..n]);
            if handshake.contains(&b'\n') {
                break;
            }
        }
        assert!(String::from_utf8_lossy(&handshake).starts_with("ZT"));

        drop(stream);
    });

    let start = std::time::Instant::now();
    let result = producer.start_streaming(&TinyDataset, /*batch_size=*/ 1);
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
        let shm_path = PathBuf::from(format!("/dev/shm/{}", shm_name));
        assert!(
            !shm_path.exists(),
            "shm segment should be cleaned up (Drop must have run)"
        );
    }
}
