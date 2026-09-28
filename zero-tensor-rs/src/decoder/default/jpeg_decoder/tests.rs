use std::assert_matches;

use crate::decoder::*;

use super::*;
use turbojpeg::{Compressor, Image as TjImage, PixelFormat};

fn create_test_jpeg(width: usize, height: usize) -> Vec<u8> {
    let mut compressor = Compressor::new().expect("Failed to create compressor");

    let mut pixels = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            pixels[idx] = (x % 256) as u8;
            pixels[idx + 1] = (y % 256) as u8;
            pixels[idx + 2] = 128;
        }
    }

    let image = TjImage {
        pixels: pixels.as_slice(),
        width,
        pitch: width * 3,
        height,
        format: PixelFormat::RGB,
    };

    compressor
        .compress_to_vec(image)
        .expect("Failed to compress JPEG")
}

#[test]
fn test_jpeg_info() {
    let jpeg_bytes = create_test_jpeg(64, 64);
    let decoder = JpegDecoder::new();

    let info = decoder.info(&jpeg_bytes).expect("Failed to read header");

    assert_eq!(info.width, 64);
    assert_eq!(info.height, 64);
    assert_eq!(info.channels, 3);
    assert!(matches!(info.format(), crate::decoder::ImageFormat::Jpeg));
}

#[test]
fn test_jpeg_decode_u8_planar() {
    let width = 32;
    let height = 32;
    let jpeg_bytes = create_test_jpeg(width, height);
    let decoder = JpegDecoder::new();

    let mut output = vec![0u8; width * height * 3];
    let info = decoder
        .decode(&jpeg_bytes, &mut output, None)
        .expect("Failed to decode");

    assert_eq!(info.width, width);
    assert_eq!(info.height, height);

    assert!(
        output.iter().any(|&x| x > 0),
        "Output should not be all zeros"
    );

    let idx = 20 * width + 10;
    assert_matches!(info.format(), ImageFormat::Jpeg);
    assert!((output[idx] as i32 - 10).abs() < 5, "R channel mismatch");
    assert!(
        (output[width * height + idx] as i32 - 20).abs() < 5,
        "G channel mismatch"
    );
    assert!(
        (output[2 * width * height + idx] as i32 - 128).abs() < 5,
        "B channel mismatch"
    );
}

#[test]
fn test_jpeg_decode_f32_thread_local_path() {
    let width = 16;
    let height = 16;
    let jpeg_bytes = create_test_jpeg(width, height);
    let decoder = JpegDecoder::new();

    let mut output = vec![0.0f32; width * height * 3];
    let info = decoder
        .decode(&jpeg_bytes, &mut output, None)
        .expect("Failed to decode");

    assert_eq!(info.width, width);
    assert_eq!(info.height, height);

    for (i, &val) in output.iter().enumerate() {
        assert!(
            val >= 0.0 && val <= 1.0,
            "Pixel {} value {} is out of [0.0, 1.0] range",
            i,
            val
        );
    }

    assert!(
        output.iter().any(|&x| x > 0.0),
        "Output should not be all zeros"
    );
}

#[test]
fn test_jpeg_decode_buffer_too_small() {
    let jpeg_bytes = create_test_jpeg(32, 32);
    let decoder = JpegDecoder::new();

    let required_size = 32 * 32 * 3;
    let mut output = vec![0u8; required_size / 2];

    let result = decoder.decode(&jpeg_bytes, &mut output, None);

    assert!(result.is_err());
    if let Err(DecodeError::BufferOverflow {
        available,
        requested,
    }) = result
    {
        assert_eq!(available, required_size / 2);
        assert_eq!(requested, required_size);
    } else {
        panic!("Expected BufferOverflow error, got: {:?}", result);
    }
}

#[test]
fn test_jpeg_decode_invalid_data() {
    let decoder = JpegDecoder::new();
    let mut output = vec![0u8; 100];

    let result = decoder.decode(b"this is not a jpeg file", &mut output, None);

    assert!(result.is_err(), "Should fail on invalid data");
}

#[test]
fn decode_chw_matches_rgb_reference_and_zeroes_each_plane_padding() {
    let (width, height) = (17, 9);
    let (stride, max_height) = (23, 12);
    let jpeg = create_test_jpeg(width, height);
    let reference = turbojpeg::decompress(&jpeg, PixelFormat::RGB).unwrap();
    let decoder = JpegDecoder::new();
    let plane_len = stride * max_height;
    let mut bytes = vec![99u8; 3 * plane_len + 1];
    let mut floats = vec![99.0f32; 3 * plane_len + 1];
    decoder
        .decode(&jpeg, &mut bytes, PaddingConfig::new(stride, max_height))
        .unwrap();
    decoder
        .decode(&jpeg, &mut floats, PaddingConfig::new(stride, max_height))
        .unwrap();
    for c in 0..3 {
        for y in 0..max_height {
            for x in 0..stride {
                let expected = if y < height && x < width {
                    reference.pixels[y * reference.pitch + x * 3 + c]
                } else {
                    0
                };
                let idx = c * plane_len + y * stride + x;
                assert_eq!(bytes[idx], expected, "channel={c}, y={y}, x={x}");
                assert_eq!(floats[idx], expected as f32 / 255.0);
            }
        }
    }
    assert_eq!(bytes[3 * plane_len], 99);
    assert_eq!(floats[3 * plane_len], 99.0);
}
