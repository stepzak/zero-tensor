# ZeroTensor

**Ultra-Fast Zero-Copy IPC Data Loader for PyTorch**

Break the PyTorch `DataLoader` bottleneck. ZeroTensor is a high-performance, lock-free inter-process communication (IPC) data transport built in Rust. It serves as a drop-in replacement for standard multiprocessing data loading, eliminating serialization overhead, runtime memory allocations, and kernel-space bottlenecks.

---

## ✨ Key Features

- 🚀 **Blazing fast**: 24-26 GB/s sustained throughput (vs ~3-5 GB/s for PyTorch DataLoader)
- 🎯 **True zero-copy**: Consumer gets PyTorch tensors backed directly by shared memory via `torch.as_strided`
- 🔄 **Multi-tensor support**: Stream multiple named tensors (`image`, `mask`, `label`) in a single batch
- 🧠 **Dynamic batching**: Automatic padding to handle variable-size inputs
- 🛡️ **Type-safe**: Full support for `f16`, `bf16`, `f32`, `f64`, `i8`, `i32`, `i64`, `u8`
- 🎨 **Transform pipeline**: CPU transforms (scale, standardize, clamp, add) with parallel execution
- 🔌 **Clean IPC**: Unix domain socket for control plane + POSIX shared memory for data
- 🧹 **RAII cleanup**: Automatic socket/SHM cleanup on drop, even on panic or SIGINT

## 📊 Performance

Benchmarks on a single socket (Intel/AMD CPU, DDR4/DDR5):

| Loader | Throughput | Notes |
|--------|-----------|-------|
| **ZeroTensor** | **24-26 GB/s** | Zero-copy, Rust producer |
| PyTorch DataLoader (8 workers, pin_memory) | 3-5 GB/s | Multiprocessing + copy |

> Benchmarked with `3×512×512` F32 tensors, batch size 48, 200 steps. See `zero-tensor-rs/src/bin/throughput_bench.rs` and `zero-tensor-py/benchmarks/zt_bench.py`.

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

```rust

**Data flow:**
1. Producer writes tensor data + metadata into a free slot in SHM
2. Producer sets `is_ready = 1` and increments `head`
3. Consumer polls `head`, reads metadata, creates zero-copy PyTorch tensors via `torch.as_strided`
4. Consumer increments `tail` after processing, freeing the slot

## Quick Start

### Producer (Rust)

```rust
use zero_tensor_lib::core::{
    dataset::{ZeroTensorDataset, item::{TensorBatchLayout, TensorDT}},
    producer::ZeroTensorProducerBuilder,
    writer::TensorWriter,
};
use indexmap::IndexMap;

struct MyDataset;

impl<'a> ZeroTensorDataset<'a> for MyDataset {
    type Error = std::io::Error;

    fn len(&self) -> usize { 10_000 }

    fn static_layouts(&self) -> Option<&IndexMap<&'static str, TensorBatchLayout>> {
        use std::sync::OnceLock;
        static LAYOUTS: OnceLock<IndexMap<&'static str, TensorBatchLayout>> = OnceLock::new();
        Some(LAYOUTS.get_or_init(|| {
            let mut m = IndexMap::new();
            m.insert(
                "image",
                TensorBatchLayout::new(
                    vec![3, 224, 224].into(),
                    vec![224 * 224, 224, 1].into(),
                    TensorDT::F32,
                ),
            );
            m
        }))
    }

    fn write_item_into<'layout, 'b, 'c>(
        &self,
        idx: usize,
        writer: &mut TensorWriter<'layout, 'b, 'c>,
    ) -> Result<(), Self::Error> {
        writer.write("image", |buf| {
            let floats: &mut [f32] = bytemuck::cast_slice_mut(buf);
            // Fill your data here...
            for (i, x) in floats.iter_mut().enumerate() {
                *x = (idx * 1000 + i) as f32;
            }
            Ok(floats.len() * 4)
        }).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        Ok(())
    }
}

fn main() {
    let mut producer = ZeroTensorProducerBuilder::new(
        64 * 1024 * 1024,       // 64 MB per slot
        "my_shm",               // SHM name
        "/tmp/my_producer.sock" // Unix socket
    )
    .num_slots(8)
    .build()
    .unwrap();

    producer.start_streaming(&MyDataset, 32).unwrap(); // batch_size = 32
}
```

## 2. Python Training Consumer
Wrap your training loop with the Python context manager. Tensors are mapped from memory instantly with **zero-copy**.

```py
import torch
from zero_tensor_py import ZeroTensorConsumer

socket_path = "/tmp/zt.sock"
shm_name = "zt_shared_buffer"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with ZeroTensorConsumer(socket_path, shm_name) as consumer:
    for epoch in range(5):
        for batch in consumer:
            image = batch["image"]
            # IMPORTANT: do not use default batch.to() method, as it may lead to Race Condition
            inputs = consumer.to_device(image, device, non_blocking=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
```

## Multi-Tensor Datasets
Stream multiple tensors per sample (e.g., image + mask + label):

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

Consumer receives a dictionary:

```python
batch = next(consumer)
image = batch["image"]  # [B, 3, 224, 224]
mask  = batch["mask"]   # [B, 224, 224]
label = batch["label"]  # [B, 1]
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

--

##  Safety & Cleanup
* **RAII**: ZeroTensorProducer cleans up socket and SHM on drop, even on panic
* **SIGINT**: Ctrl+C is handled gracefully via ctrlc crate
* **Dead consumer detection**: Producer detects if consumer disconnects and stops
* **Buffer overflow protection**: All writes are bounds-checked