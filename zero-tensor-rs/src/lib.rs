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

    use crate::buffer::tensor_meta::TensorHeader;
    use crate::buffer::{ZTBufErr, ZeroTensorBuffer};
    use crate::dataset::ZeroTensorDataset;
    use crate::dataset::item::{ShapeType, StrideType, TensorBatchLayout, TensorDT};
    use crate::producer::{CONSUMER_RESP_BUFFER, ZeroTensorProducerBuilder};

    struct MockDataset {
        len: usize,
        meta: TensorBatchLayout,
    }

    impl MockDataset {
        fn new(len: usize) -> Self {
            let shape = vec![2, 3];
            let strides = vec![3, 1];
            let dt = TensorDT::F32;
            let meta = TensorBatchLayout::new(shape.into(), strides.into(), dt);

            Self { len, meta }
        }
    }

    impl ZeroTensorDataset for MockDataset {
        type Error = std::io::Error;

        fn len(&self) -> usize {
            self.len
        }

        fn is_empty(&self) -> bool {
            self.len == 0
        }

        fn get_batch_layout(&self, _idxs: &[usize]) -> Result<TensorBatchLayout, Self::Error> {
            Ok(self.meta.clone())
        }

        fn write_item_into(&self, idx: usize, buf: &mut [u8]) -> Result<(), Self::Error> {
            if idx >= self.len {
                return Err(std::io::ErrorKind::InvalidData.into());
            }
            let meta = self.get_batch_layout(&[idx])?;
            let total_elements = meta.total_elements();
            let total_bytes = meta.total_bytes();

            match meta.dt() {
                TensorDT::F32 => {
                    let f32_slice = unsafe {
                        std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut f32, total_elements)
                    };

                    (0..total_elements).for_each(|i| {
                        f32_slice[i] = i as f32 * 0.5 + idx as f32;
                    });
                }
                _ => {
                    buf[..total_bytes].fill(0);
                }
            }

            Ok(())
        }
    }

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
    fn test_end_to_end_streaming() {
        let dir = tempdir().unwrap();
        let socket_path = dir.path().join("zero_tensor.sock");
        let shm_name = "zt_test_buffer";

        let batch_size = 2;
        let steps = 2;
        let slot_size = 2048;

        let dataset = MockDataset::new(batch_size * steps);

        let idxs: &[usize] = &[0];
        let meta = dataset.get_batch_layout(idxs).unwrap();

        let mut producer = ZeroTensorProducerBuilder::new(slot_size, shm_name, &socket_path)
            .num_slots(2)
            .build()
            .expect("Failed to init producer");

        let consumer_socket = socket_path.clone();
        let consumer_shm_name = shm_name.to_string();

        let consumer_handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));

            let mut stream = match UnixStream::connect(&consumer_socket) {
                Ok(s) => s,
                Err(e) => panic!(
                    "Consumer failed to connect to socket {:?}: {}",
                    consumer_socket, e
                ),
            };

            let consumer_buffer = match ZeroTensorBuffer::open(&consumer_shm_name, slot_size * 2) {
                Ok(b) => b,
                Err(e) => panic!("Consumer failed to open SHM {}: {}", consumer_shm_name, e),
            };

            let mut sock_buf = [0; CONSUMER_RESP_BUFFER];

            for step in 0..steps {
                let n = match stream.read(&mut sock_buf) {
                    Ok(n) if n > 0 => n,
                    Ok(_) => panic!("Producer closed connection unexpectedly at step {}", step),
                    Err(e) => panic!("Failed to read from socket at step {}: {}", step, e),
                };

                let msg = std::str::from_utf8(&sock_buf[..n])
                    .unwrap_or_else(|_| panic!("Invalid UTF-8 message at step {}", step));

                assert!(msg.starts_with("READY"), "Expected READY, got: {}", msg);
                let offset: usize = msg
                    .trim_end()
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| panic!("Failed to parse offset from: {}", msg));

                let expected_offset = (step % 2) * slot_size;
                assert_eq!(offset, expected_offset, "Offset mismatch at step {}", step);

                let slot_bytes = consumer_buffer
                    .get_slot_slice(offset, slot_size)
                    .expect("Failed to get slot slice");

                let header_ptr = slot_bytes.as_ptr() as *const TensorHeader;
                let header = unsafe { &*header_ptr };
                let offs = header.get_offsets();

                assert_eq!(header.ndims(), 3, "NDims mismatch");
                assert_eq!(header.dt(), TensorDT::F32, "DataType mismatch");

                let shape_ptr =
                    unsafe { slot_bytes.as_ptr().add(offs.shapes()) as *const ShapeType };
                let read_shape = unsafe { std::slice::from_raw_parts(shape_ptr, 3) };
                assert_eq!(
                    read_shape,
                    &[batch_size as ShapeType, 2, 3],
                    "Shape mismatch"
                );

                let strides_ptr =
                    unsafe { slot_bytes.as_ptr().add(offs.strides()) as *const StrideType };
                let read_strides = unsafe { std::slice::from_raw_parts(strides_ptr, 3) };
                assert_eq!(read_strides, &[24, 12, 4], "Strides mismatch");

                let data_ptr = unsafe { slot_bytes.as_ptr().add(offs.data()) as *const f32 };
                let total_els = meta.shape().iter().product::<ShapeType>() as usize;

                for i in 0..total_els {
                    let expected_val = i as f32 * 0.5 + (step * batch_size) as f32;
                    let actual_val = unsafe { *data_ptr.add(i) };
                    assert_eq!(
                        actual_val, expected_val,
                        "Data mismatch at index {} in step {}",
                        i, step
                    );
                }

                thread::sleep(Duration::from_millis(5));

                if let Err(e) = stream.write_all(b"RELEASE\n") {
                    panic!("Failed to send RELEASE at step {}: {}", step, e);
                }
                if let Err(e) = stream.flush() {
                    panic!("Failed to flush at step {}: {}", step, e);
                }
            }

            if let Err(e) = stream.write_all(b"STOP\n") {
                eprintln!("Warning: Failed to send STOP: {}", e);
            } else {
                let _ = stream.flush();
            }
        });

        let _ = producer.start_streaming(&dataset, batch_size);
        consumer_handle.join().expect("Consumer thread panicked");
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
        let res = ZeroTensorBuffer::new(&fname, 1);
        assert!(matches!(res, Err(ZTBufErr::InvalidFilename(_))));
    }

    #[test]
    fn test_filename_nullbyte() {
        let fname = "abcd\0efgh";
        let res = ZeroTensorBuffer::new(&fname, 1);
        assert!(matches!(res, Err(ZTBufErr::InvalidFilename(_))));
    }

    #[test]
    fn test_filename_slash() {
        let fname = "abcd/efgh";
        let res = ZeroTensorBuffer::new(&fname, 1);
        assert!(matches!(res, Err(ZTBufErr::InvalidFilename(_))));
    }

    #[test]
    fn test_buf_overflow() {
        let mut buf = ZeroTensorBuffer::new("zt_buf_test", 1024).unwrap();
        let shape = vec![100, 100];
        let strides = vec![100, 1];
        let data = vec![0u8; 40000];
        let res = buf.write_tensor(0, &shape, &strides, TensorDT::F16, &data);
        assert!(matches!(res, Err(ZTBufErr::BufferOverflow(1024, _))));
    }

    #[test]
    fn test_shape_stride_mismatch() {
        let mut buf = ZeroTensorBuffer::new("zt_buf_test", 1024).unwrap();
        let shape = vec![100, 100];
        let strides = vec![100, 1, 2];
        let data = vec![0u8; 10000 * 4];
        let res = buf.write_tensor(0, &shape, &strides, TensorDT::F16, &data);
        assert!(matches!(res, Err(ZTBufErr::InvalidShape(3, 2))))
    }

    #[test]
    fn test_dataset_failure() {
        let dir = tempdir().unwrap();
        let socket_path = dir.path().join("zero_tensor.sock");
        let shm_name = "zt_test_buffer";

        let batch_size = 2;
        let steps = 2;
        let slot_size = 2048;

        let dataset = FailingDataset;
        let mut producer = ZeroTensorProducerBuilder::new(slot_size, shm_name, &socket_path)
            .build()
            .expect("Failed to init producer");

        let consumer_socket = socket_path.clone();
        let consumer_shm_name = shm_name.to_string();

        let consumer_handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));

            let mut stream = UnixStream::connect(&consumer_socket)
                .expect("Consumer failed to connect to socket");

            let _ = ZeroTensorBuffer::open(&consumer_shm_name, slot_size * 2)
                .expect("Consumer failed to open SHM");

            let mut sock_buf = [0; CONSUMER_RESP_BUFFER];

            for _ in 0..steps {
                let _ = stream
                    .read(&mut sock_buf)
                    .expect("Failed to read from socket");
            }
        });

        let res = producer.start_streaming(&dataset, batch_size);
        let _ = consumer_handle.join();
        assert!(res.is_err())
    }
}
