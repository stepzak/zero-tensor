use super::*;
use crate::{
    core::dataset::item::{TensorBatchLayout, TensorDT},
    transform::{Scale, TransformError},
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct MockTransform {
    calls: Arc<AtomicUsize>,
}

impl Transform for MockTransform {
    fn apply(&self, _: &mut TensorViewMut) -> Result<(), TransformError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct FailingTransform;

impl Transform for FailingTransform {
    fn apply(&self, _: &mut TensorViewMut) -> Result<(), TransformError> {
        Err(TransformError::InvalidValue)
    }
}

fn make_tensor() -> (Vec<u8>, TensorBatchLayout) {
    let data = vec![1.0f32, 2.0, 3.0];
    let raw_bytes = bytemuck::pod_collect_to_vec(&data);

    let layout = TensorBatchLayout::new(vec![data.len()].into(), vec![1].into(), TensorDT::F32);

    (raw_bytes, layout)
}

#[test]
fn empty_pipeline_succeeds() {
    let (mut raw_bytes, layout) = make_tensor();
    let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

    let pipeline = Pipeline::new();

    assert!(pipeline.exec(&mut tensor).is_ok());
}

#[test]
fn executes_all_steps() {
    let calls = Arc::new(AtomicUsize::new(0));

    let pipeline = Pipeline::new()
        .then(MockTransform {
            calls: Arc::clone(&calls),
        })
        .then(MockTransform {
            calls: Arc::clone(&calls),
        })
        .then(MockTransform {
            calls: Arc::clone(&calls),
        });

    let (mut raw_bytes, layout) = make_tensor();
    let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

    pipeline.exec(&mut tensor).unwrap();

    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[test]
fn stops_after_error() {
    let calls = Arc::new(AtomicUsize::new(0));

    let pipeline = Pipeline::new()
        .then(MockTransform {
            calls: Arc::clone(&calls),
        })
        .then(FailingTransform)
        .then(MockTransform {
            calls: Arc::clone(&calls),
        });

    let (mut raw_bytes, layout) = make_tensor();
    let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

    let result = pipeline.exec(&mut tensor);

    assert!(result.is_err());

    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn returns_correct_step_number() {
    let calls = Arc::new(AtomicUsize::new(0));

    let pipeline = Pipeline::new()
        .then(MockTransform {
            calls: Arc::clone(&calls),
        })
        .then(MockTransform {
            calls: Arc::clone(&calls),
        })
        .then(FailingTransform);

    let (mut raw_bytes, layout) = make_tensor();
    let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

    let error = pipeline.exec(&mut tensor).unwrap_err();

    assert_eq!(error.step, 3);
    assert!(matches!(error.error, TransformError::InvalidValue));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn first_step_error_has_step_one() {
    let pipeline = Pipeline::new().then(FailingTransform);

    let (mut raw_bytes, layout) = make_tensor();
    let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

    let error = pipeline.exec(&mut tensor).unwrap_err();

    assert_eq!(error.step, 1);
    assert!(matches!(error.error, TransformError::InvalidValue));
}

#[test]
fn applies_real_transforms_in_order() {
    let data = vec![1.0f32, 2.0, 3.0];
    let mut raw_bytes = bytemuck::pod_collect_to_vec(&data);

    let layout = TensorBatchLayout::new(vec![3].into(), vec![1].into(), TensorDT::F32);

    let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

    let pipeline = Pipeline::new().then(Scale::new(2.0)).then(Scale::new(3.0));

    pipeline.exec(&mut tensor).unwrap();

    let result: Vec<f32> = bytemuck::pod_collect_to_vec(&raw_bytes);

    assert_eq!(result, vec![6.0, 12.0, 18.0]);
}
