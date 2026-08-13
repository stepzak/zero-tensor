use super::*;

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
