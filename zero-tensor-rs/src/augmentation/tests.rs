#[cfg(test)]
mod pipeline_tests {
    use crate::augmentation::default::crop::RandomCrop;
    use crate::augmentation::default::flip::RandomHorizontalFlip;
    use crate::augmentation::default::normalize::Normalize;
    use crate::augmentation::default::resize::Resize;
    use crate::augmentation::{AugmentationPipeline, ImageShape};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn create_test_image(c: usize, h: usize, w: usize) -> Vec<f32> {
        let mut img = vec![0.0f32; c * h * w];
        for channel in 0..c {
            let c_offset = channel * h * w;
            for y in 0..h {
                for x in 0..w {
                    img[c_offset + y * w + x] =
                        ((channel * 1000 + y * w + x) as f32) / (c * h * w) as f32;
                }
            }
        }
        img
    }

    #[test]
    fn test_pipeline_full_imagenet_style() {
        let pipeline = AugmentationPipeline::<f32>::new()
            .then(Resize::new(256, 256))
            .unwrap()
            .then(RandomCrop::new(224, 224))
            .unwrap()
            .then(RandomHorizontalFlip::new(0.5).unwrap())
            .unwrap()
            .then(Normalize::imagenet())
            .unwrap();

        let input = create_test_image(3, 300, 400);
        let mut output = vec![0.0f32; 3 * 224 * 224];

        let mut rng = StdRng::seed_from_u64(42);
        let shape = pipeline
            .apply(
                &input,
                ImageShape::new(3, 300, 400),
                &mut output,
                Some(&mut rng),
            )
            .unwrap();

        assert_eq!(shape, ImageShape::new(3, 224, 224));

        assert!(output.iter().any(|&x| x.abs() > 1e-6));

        assert!(output.iter().all(|&x| x.abs() < 100.0));
    }

    #[test]
    fn test_pipeline_only_size_preserving() {
        let pipeline = AugmentationPipeline::<f32>::new()
            .then(RandomHorizontalFlip::new(1.0).unwrap())
            .unwrap()
            .then(Normalize::new(vec![0.5, 0.5, 0.5], vec![0.5, 0.5, 0.5]).unwrap())
            .unwrap();
        let input = vec![1.0f32; 12];
        let mut output = vec![0.0f32; 12];

        let mut rng = StdRng::seed_from_u64(42);
        let shape = pipeline
            .apply(
                &input,
                ImageShape::new(3, 2, 2),
                &mut output,
                Some(&mut rng),
            )
            .unwrap();

        assert_eq!(shape, ImageShape::new(3, 2, 2));
    }

    #[test]
    fn test_pipeline_only_size_changing() {
        let pipeline = AugmentationPipeline::<f32>::new()
            .then(Resize::new(100, 100))
            .unwrap()
            .then(RandomCrop::new(50, 50))
            .unwrap();

        let input = create_test_image(3, 200, 200);
        let mut output = vec![0.0f32; 3 * 50 * 50];

        let mut rng = StdRng::seed_from_u64(42);
        let shape = pipeline
            .apply(
                &input,
                ImageShape::new(3, 200, 200),
                &mut output,
                Some(&mut rng),
            )
            .unwrap();

        assert_eq!(shape, ImageShape::new(3, 50, 50));
    }

    #[test]
    fn test_pipeline_empty() {
        let pipeline = AugmentationPipeline::<f32>::new();
        let input = vec![1.0f32; 12];
        let mut output = vec![0.0f32; 12];

        let shape = pipeline
            .apply(&input, ImageShape::new(3, 2, 2), &mut output, None)
            .unwrap();

        assert_eq!(shape, ImageShape::new(3, 2, 2));
        assert_eq!(input, output);
    }

    #[test]
    fn test_pipeline_invalid_order() {
        let result = AugmentationPipeline::<f32>::new()
            .then(Normalize::imagenet())
            .unwrap()
            .then(Resize::new(224, 224));

        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_valid_order() {
        let result = AugmentationPipeline::<f32>::new()
            .then(Resize::new(256, 256))
            .unwrap()
            .then(RandomCrop::new(224, 224))
            .unwrap()
            .then(RandomHorizontalFlip::new(0.5).unwrap())
            .unwrap()
            .then(Normalize::imagenet());

        assert!(result.is_ok());
    }

    #[test]
    fn test_pipeline_output_size() {
        let pipeline = AugmentationPipeline::<f32>::new()
            .then(Resize::new(256, 256))
            .unwrap()
            .then(RandomCrop::new(224, 224))
            .unwrap()
            .then(Normalize::imagenet())
            .unwrap();

        assert_eq!(pipeline.output_size(), Some((224, 224)));
    }

    #[test]
    fn test_pipeline_output_size_no_size_changing() {
        let pipeline = AugmentationPipeline::<f32>::new()
            .then(RandomHorizontalFlip::new(0.5).unwrap())
            .unwrap()
            .then(Normalize::imagenet())
            .unwrap();

        assert_eq!(pipeline.output_size(), None);
    }

    #[test]
    fn test_pipeline_deterministic_with_seed() {
        let pipeline = AugmentationPipeline::<f32>::new()
            .then(Resize::new(100, 100))
            .unwrap()
            .then(RandomHorizontalFlip::new(0.5).unwrap())
            .unwrap();

        let input = create_test_image(3, 200, 200);
        let mut output1 = vec![0.0f32; 3 * 100 * 100];
        let mut output2 = vec![0.0f32; 3 * 100 * 100];

        let mut rng1 = StdRng::seed_from_u64(42);
        pipeline
            .apply(
                &input,
                ImageShape::new(3, 200, 200),
                &mut output1,
                Some(&mut rng1),
            )
            .unwrap();

        let mut rng2 = StdRng::seed_from_u64(42);
        pipeline
            .apply(
                &input,
                ImageShape::new(3, 200, 200),
                &mut output2,
                Some(&mut rng2),
            )
            .unwrap();

        assert_eq!(output1, output2);
    }
    #[test]
    fn test_pipeline_with_larger_intermediate_size() {
        let pipeline = AugmentationPipeline::<f32>::new()
            .then(Resize::new(256, 256))
            .unwrap()
            .then(RandomCrop::new(224, 224))
            .unwrap()
            .then(Normalize::imagenet())
            .unwrap();

        assert_eq!(pipeline.max_intermediate_size(), Some((256, 256)));
        assert_eq!(pipeline.output_size(), Some((224, 224)));
    }

    #[test]
    fn test_large_input_small_output() {
        let pipeline = AugmentationPipeline::<f32>::new()
            .then(Resize::new(256, 256))
            .unwrap()
            .then(RandomCrop::new(224, 224))
            .unwrap();

        assert_eq!(pipeline.max_intermediate_size(), Some((256, 256)));
        assert_eq!(pipeline.output_size(), Some((224, 224)));
    }
}
