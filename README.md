# ZeroTensor

**Ultra-Fast Zero-Copy IPC Data Loader for PyTorch**

Break the PyTorch `DataLoader` bottleneck. ZeroTensor is a high-performance, lock-free inter-process communication (IPC) data transport built in Rust. It serves as a drop-in replacement for standard multiprocessing data loading, eliminating serialization overhead, runtime memory allocations, and kernel-space bottlenecks.

---

## Prerequisites

Current version only supports `Unix` system(for `/dev/shm`).

You also need to install `libturbojpeg0-dev` and `pkg-config`

## Key Features

-  **Blazing fast**: 30-34 GB/s sustained throughput (vs ~6-7 GB/s for PyTorch DataLoader)
-  **True zero-copy**: Consumer gets PyTorch tensors backed directly by shared memory via `torch.as_strided`
-  **Multi-tensor support**: Stream multiple named tensors (`image`, `mask`, `label`) in a single batch
-  **Dynamic batching**: Automatic padding to handle variable-size inputs
-  **Type-safe**: Full support for `f16`, `bf16`, `f32`, `f64`, `i8`, `i32`, `i64`, `u8`
-  **Clean IPC**: Unix domain socket for control plane + POSIX shared memory for data
-  **RAII cleanup**: Automatic socket/SHM cleanup on drop, even on panic or SIGINT

## Performance

### 1. Real-World Scenario (JPEG Decode + Augmentations)
*Pipeline: Decode → Resize(256) → RandomCrop(224) → RandomFlip(0.5) → Normalize*

| Loader | Throughput | Images/sec | Notes |
|--------|-----------|------------|-------|
| **ZeroTensor** | **~2.6 GB/s** | **~2200** | Rust parallel decode + SIMD augmentations |
| PyTorch DataLoader | ~0.8 GB/s | ~800 | 4 workers, `pin_memory`, `prefetch_factor=2` |

### 2. JPEG Decode with no augmentation

| Loader | Throughput | Images/sec | Notes |
|--------|-----------|------------|-------|
| **ZeroTensor** | **~11 GB/s** | **~6600** | Rust parallel decode |
| PyTorch DataLoader | ~2 GB/s | ~1200 | 4 workers, `pin_memory`, `prefetch_factor=2` |

### 3. Raw Synthetic Throughput (Pre-computed F32 Tensors)
*3×512×512 F32 tensors, batch size 48*

| Loader | Throughput | Notes |
|--------|-----------|-------|
| **ZeroTensor** | **30-34 GB/s** | Zero-copy, Rust producer |
| PyTorch DataLoader | 6-7 GB/s | Multiprocessing + pickle serialization + copy |

> *Note*: For maximum stable throughput, pin **only** the Python consumer to half of CPU cores. Let the Rust producer use all available cores for parallel augmentation

---

## The Problem with Standard Data Loading

When scaling up training, the standard PyTorch `DataLoader` (`num_workers > 0`) hits hard architectural limits:

1. **Pickle Serialization Overhead:** Tensors are serialized/deserialized through Unix pipes, consuming up to 30% of total CPU cycles.
2. **Page-Fault Storms:** Workers constantly allocate new memory blocks. The Linux kernel must constantly interrupt execution to map virtual addresses to physical pages.
3. **Zombie Shared Memory:** Dirty kills (`Ctrl+C`, OOM) leave orphaned `/dev/shm` blocks, leaking RAM until a server reboot.
4. **GIL Contention:** Python-based preprocessing in worker processes fights for the Global Interpreter Lock.

---

## How ZeroTensor Solves It

ZeroTensor decouples heavy I/O operations (parallel loading, decoding, preprocessing) in Rust from the Python training loop using an optimized ring buffer architecture.

```text
┌─────────────────────────────────────────────────────────────────┐
│                       RUST PRODUCER                             │
│  [Dataset Fetch] → [Rayon Parallel Write] → [SIMD Pipeline]     │
└────────────────────────────┬────────────────────────────────────┘
                             │ 
                 ZERO-COPY SHARED MEMORY (mmap)
                 (Pre-allocated, Pre-faulted, Lock-Free)
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                      PYTHON CONSUMER                            │
│  [Atomic Head Check] → [torch.as_strided View] → [GPU Transfer] │
└─────────────────────────────────────────────────────────────────┘
```

* **Lock-Free SPSC Ring Buffer**: Uses `CachePadded` atomics for true zero-lock concurrency between `Producer` and `Consumer`.
* **Pre-faulted Shared Memory**: Memory is mapped and "warmed up" once at startup. Zero runtime page faults.
* **Strict RAII Cleanup**: All sockets and `/dev/shm` segments are tied to Rust lifecycles. They are safely unlinked on panic, `SIGINT`, or normal drop.

## Quick Start
### 1. Rust Data Producer
Define your dataset using the `ZeroTensorDataset` trait. You can optionally attach a high-performance `Pipeline` to preprocess data in Rust.

**Data flow:**
1. Producer writes tensor data + metadata into a free slot in SHM
2. Producer sets `is_ready = 1` and increments `head`
3. Consumer polls `head`, reads metadata, creates zero-copy PyTorch tensors via `torch.as_strided`
4. Consumer increments `tail` after processing, freeing the slot

## Quick Start

### Producer (Rust, e.g. built-in JPEG Dataset)

```rust
use zero_tensor_lib::core::{dataset::ZeroTensorDataset, producer::ZeroTensorProducerBuilder};
use zero_tensor_lib::dataset::image::JpegFolderDataset;
use zero_tensor_lib::augmentation::{AugmentationPipeline, default::*};
use std::path::Path;

fn main() {
    let dataset_dir = Path::new("/path/to/imagenet");
    
    // 1. Define your augmentation pipeline (executed in Rust, may require preallocated buffers. Preallocated buffers do not affect hot loop speed)
    let pipeline = AugmentationPipeline::<f32>::new()
        .then(Resize::new(256, 256)).unwrap()
        .then(RandomCrop::new(224, 224)).unwrap()
        .then(RandomHorizontalFlip::new(0.5).unwrap()).unwrap()
        .then(Normalize::imagenet()).unwrap();

    // 2. Initialize the dataset (infers f32/u8 from generic type)
    let label_fn = |path: &Path| {
        path.parent()
            .and_then(|p| p.file_name())
            .and_then(|name| name.to_str())
            .and_then(|name| name.strip_prefix("class_").and_then(|s| s.parse::<i64>().ok()))
    };

    let dataset = JpegFolderDataset::<f32>::new(dataset_dir, label_fn)
        .expect("Failed to create dataset")
        .with_augmentation(pipeline); // Attach the pipeline

    // 3. Start streaming
    let mut producer = ZeroTensorProducerBuilder::from_dataset(
        &dataset,
        "zt_shared_buffer",    // SHM name
        "/tmp/zt.sock",         // Unix socket
        32,                     // batch_size
        None                    // probe_size (None = scan all to find max dimensions)
    ).unwrap().build().unwrap();

    println!("Waiting for Python consumer to connect...");
    producer.start_streaming(&dataset, 32).unwrap();
}
```

## 2. Python Training Consumer
Wrap your training loop with the Python context manager. Tensors are mapped from memory instantly with **zero-copy**.

```py
import torch
from zero_tensor_py import ZeroTensorConsumer

socket_path = "/tmp/zt.sock"
shm_name = "zt_shared_buffer"
device = torch.device("cuda" if torch.cuda.is_available() else " is cpu")

with ZeroTensorConsumer(socket_path, shm_name, prefetch_factor=12) as consumer:
    for epoch in range(5):
        for batch in consumer:
            # batch is a dict: {"image": Tensor, "label": Tensor}
            image = batch["image"]
            label = batch["label"]
            
            # CRITICAL: Use consumer.to_device for safe, non-blocking GPU transfers
            # It handles CUDA events to prevent the producer from overwriting SHM 
            # before the GPU has finished copying the data.
            inputs = consumer.to_device(image, device, non_blocking=True)
            targets = consumer.to_device(label, device, non_blocking=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

## Advanced: Custom Datasets & Multi-Tensor Support
If you have custom data formats (e.g., LMDB, custom binary), implement the ZeroTensorDataset trait directly. You can stream multiple named tensors (e.g., `image`, `label`) in a single batch.


```rust
use zero_tensor_lib::core::{
    dataset::{ZeroTensorDataset, item::{TensorBatchLayout, TensorDT}},
    writer::TensorWriter,
};
use indexmap::IndexMap;
use std::sync::OnceLock;

struct MyCustomDataset;

impl<'a> ZeroTensorDataset<'a> for MyCustomDataset {
    type Error = std::io::Error;

    fn len(&self) -> usize { 10_000 }

    fn static_layouts(&self) -> Option<&IndexMap<&'static str, TensorBatchLayout>> {
        static LAYOUTS: OnceLock<IndexMap<&'static str, TensorBatchLayout>> = OnceLock::new();
        Some(LAYOUTS.get_or_init(|| {
            let mut m = IndexMap::new();
            m.insert("image", TensorBatchLayout::new(vec![3, 224, 224].into(), vec![224*224, 224, 1].into(), TensorDT::F32));
            m.insert("label", TensorBatchLayout::new(vec![1].into(), vec![1].into(), TensorDT::I64));
            m.insert("mask",  TensorBatchLayout::new(vec![224, 224].into(), ...));

            m
        }))
    }

    fn write_item_into<'layout, 'b, 'c>(
        &self,
        idx: usize,
        writer: &mut TensorWriter<'layout, 'b, 'c>,
    ) -> Result<(), Self::Error> {
        // Write image
        writer.write("image", |buf| {
            let floats: &mut [f32] = bytemuck::cast_slice_mut(buf);
            // ... fill floats ...
            Ok(floats.len() * 4)
        }).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        
        // Write label
        writer.write("label", |buf| {
            let ints: &mut [i64] = bytemuck::cast_slice_mut(buf);
            ints[0] = idx as i64;
            Ok(8)
        }).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        writer.write("mask",  |buf| { /* ... */ Ok(size) })?;

        Ok(())
    }
}
```

Consumer receives a dictionary:

```python
batch = next(consumer)
image = batch["image"]  # [B, 3, 224, 224]
label = batch["label"]  # [B, 1]
mask = batch["mask"] # [B, 224, 224]
```

## Dynamic Batching
For variable-size inputs, implement ``dynamic_layouts()``:

```rust
fn static_layouts(&self) -> Option<&IndexMap<&'static str, TensorBatchLayout>> {
    static LAYOUTS: OnceLock<IndexMap<&str, TensorBatchLayout>> = OnceLock::new();
    Some(LAYOUTS.get_or_init(|| {
        let mut m = IndexMap::new();
        m.insert("image", TensorBatchLayout::new(vec![3, 224, 224].into(), ...));
        m.insert("mask",  TensorBatchLayout::new(vec![224, 224].into(), ...));
        m.insert("label", TensorBatchLayout::new(vec![1].into(), ...));
        m
    }))
}

fn write_item_into(&self, idx: usize, writer: &mut TensorWriter) -> Result<()> {
    writer.write("image", |buf| { /* ... */ Ok(size) })?;
    writer.write("mask",  |buf| { /* ... */ Ok(size) })?;
    writer.write("label", |buf| { /* ... */ Ok(size) })?;
    Ok(())
}
```

Smaller items are automatically zero-padded to the max size in the batch.

---

##  Safety & Cleanup
* **RAII**: ZeroTensorProducer cleans up socket and SHM on drop, even on panic
* **SIGINT**: Ctrl+C is handled gracefully via ctrlc crate
* **Dead consumer detection**: Producer detects if consumer disconnects and stops
* **Buffer overflow protection**: All writes are bounds-checked