pub mod buffer;
pub mod dataset;
pub mod producer;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use std::thread;
    use std::time::Duration;
    use std::io::{Read, Write};
    use std::os::unix::net::UnixStream;
    use tempfile::tempdir;

    use crate::buffer::buffer::ZeroTensorBuffer;
use crate::dataset::dataset::ZeroTensorDataset;
    use crate::dataset::item::{TensorItemMeta, TensorDT, ShapeType, StrideType};
    use crate::buffer::tensor_meta::TensorHeader;
    use crate::producer::producer::{CONSUMER_RESP_BUFFER, ZeroTensorProducer};

    struct NonContiguousMockDataset {
        len: usize,
    }

    impl ZeroTensorDataset for NonContiguousMockDataset {
        fn len(&self) -> usize {
            self.len
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
    fn test_end_to_end_streaming() {
        let dir = tempdir().unwrap();
        let socket_path = dir.path().join("zero_tensor.sock");
        let shm_name = "zt_test_buffer";

        let batch_size = 2;
        let steps = 2;
        let slot_size = 2048;

        let dataset = NonContiguousMockDataset { len: batch_size * steps };

        let mut producer = ZeroTensorProducer::new(
            steps,
            slot_size,
            shm_name,
            &socket_path,
            None
        ).expect("Failed to init producer");

        let consumer_socket = socket_path.clone();
        let consumer_shm_name = shm_name.to_string();

        let consumer_handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));

            let mut stream = UnixStream::connect(&consumer_socket)
                .expect("Consumer failed to connect to socket");

            let consumer_buffer = ZeroTensorBuffer::open(&consumer_shm_name, slot_size * 2)
                .expect("Consumer failed to open SHM");

            let mut sock_buf = [0; CONSUMER_RESP_BUFFER];

            for step in 0..steps {
                let n = stream.read(&mut sock_buf).expect("Failed to read from socket");
                let msg = std::str::from_utf8(&sock_buf[..n]).unwrap();
                assert!(msg.starts_with("READY"));

                let offset: usize = msg
                    .trim_end()
                    .split_whitespace()
                    .nth(1)
                    .unwrap()
                    .parse()
                    .expect("Failed to parse offset");

                let expected_offset = (step % 2) * slot_size;
                assert_eq!(offset, expected_offset);

                let slot_bytes = consumer_buffer.get_slot_slice(offset, slot_size);

                let header_ptr = slot_bytes.as_ptr() as *const TensorHeader;
                let header = unsafe { &*header_ptr };
                let offs = header.get_offsets();

                assert_eq!(header.ndims(), 3);
                assert_eq!(header.dt(), TensorDT::F32);

                let shape_ptr = unsafe { slot_bytes.as_ptr().add(offs.shapes()) as *const ShapeType };
                let read_shape = unsafe { std::slice::from_raw_parts(shape_ptr, 3) };
                assert_eq!(read_shape, &[batch_size as ShapeType, 2, 3]);

                let strides_ptr = unsafe { slot_bytes.as_ptr().add(offs.strides()) as *const StrideType };
                let read_strides = unsafe { std::slice::from_raw_parts(strides_ptr, 3) };
                
                assert_eq!(read_strides, &[24, 12, 4]);

                let data_ptr = unsafe { slot_bytes.as_ptr().add(offs.data()) as *const f32 };
                
                let idx_0 = step * batch_size;

                assert_eq!(unsafe { *data_ptr }, idx_0 as f32);
                assert_eq!(unsafe { *data_ptr.add(3) }, idx_0 as f32 + 0.5);

                thread::sleep(Duration::from_millis(10));

                stream.write_all(b"RELEASE\n").expect("Failed to write RELEASE");
                stream.flush().unwrap();
            }
        });

        producer.start_streaming(&dataset, batch_size).expect("Streaming failed");

        consumer_handle.join().expect("Consumer thread panicked");
    }
}