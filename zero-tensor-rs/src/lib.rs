pub mod buffer;
pub mod dataset;
pub mod producer;

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::os::unix::net::UnixStream;
    use std::path::PathBuf;
    use std::thread;
    use std::time::Duration;
    use tempfile::tempdir;

    use crate::buffer::{ZTBufErr, ZeroTensorBuffer};
    use crate::dataset::ZeroTensorDataset;
    use crate::dataset::item::{TensorBatchLayout, TensorDT};
    use crate::producer::ZeroTensorProducerBuilder;

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
    fn test_shuffle_determinism_with_seed() {
        let seed = Some(1337);
        let len = 1000;

        let mut indices1: Vec<usize> = (0..len).collect();
        let mut rng1 = fastrand::Rng::with_seed(seed.unwrap());
        rng1.shuffle(&mut indices1);

        let mut indices2: Vec<usize> = (0..len).collect();
        let mut rng2 = fastrand::Rng::with_seed(seed.unwrap());
        rng2.shuffle(&mut indices2);

        assert_eq!(
            indices1, indices2,
            "Shuffled indices must be identical with the same seed"
        );
    }

    #[test]
    fn test_shuffle_differs_across_epochs() {
        let base_seed = 42u64;
        let len = 100;

        let mut epoch0: Vec<usize> = (0..len).collect();
        fastrand::Rng::with_seed(base_seed).shuffle(&mut epoch0);

        let mut epoch1: Vec<usize> = (0..len).collect();
        fastrand::Rng::with_seed(base_seed + 1).shuffle(&mut epoch1);

        assert_ne!(
            epoch0, epoch1,
            "Epochs must have different shuffle patterns"
        );
    }

    #[test]
    fn test_filename_too_long() {
        let fname = "a".repeat(300);
        let res = ZeroTensorBuffer::new(&fname, 64, 1);
        assert!(matches!(res, Err(ZTBufErr::InvalidFilename(_))));
    }

    #[test]
    fn test_filename_nullbyte() {
        let fname = "abcd\0efgh";
        let res = ZeroTensorBuffer::new(&fname, 64, 1);
        assert!(matches!(res, Err(ZTBufErr::InvalidFilename(_))));
    }

    #[test]
    fn test_filename_slash() {
        let fname = "abcd/efgh";
        let res = ZeroTensorBuffer::new(&fname, 64, 1);
        assert!(matches!(res, Err(ZTBufErr::InvalidFilename(_))));
    }

    #[test]
    fn test_buf_overflow() {
        let mut buf = ZeroTensorBuffer::new("zt_buf_test", 256, 1).unwrap();
        let shape = vec![100, 100];
        let strides = vec![100, 1];
        let data = vec![0u8; 40000];
        let res = buf.write_tensor(0, &shape, &strides, TensorDT::F32, &data);
        assert!(matches!(res, Err(ZTBufErr::BufferOverflow(_, _))));
    }

    #[test]
    fn test_shape_stride_mismatch() {
        let mut buf = ZeroTensorBuffer::new("zt_buf_test", 4096, 2).unwrap();
        let shape = vec![100, 100];
        let strides = vec![100, 1, 2];
        let data = vec![0u8; 10000 * 4];
        let res = buf.write_tensor(0, &shape, &strides, TensorDT::F32, &data);
        assert!(matches!(res, Err(ZTBufErr::InvalidShape(3, 2))))
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
            let mut stream = UnixStream::connect(&consumer_socket)
                .expect("Consumer failed to connect to socket");

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
}
