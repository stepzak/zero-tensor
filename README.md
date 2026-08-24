# ZeroTensor

**Ultra-Fast Zero-Copy IPC Data Loader for PyTorch**

Break the PyTorch `DataLoader` bottleneck. ZeroTensor is a high-performance, lock-free inter-process communication (IPC) data transport built in Rust. It serves as a drop-in replacement for standard multiprocessing data loading, eliminating serialization overhead, runtime memory allocations, and kernel-space bottlenecks.

---

## Performance at a Glance

*Environment: Synthetic dataset (3×1024×1024 F32 tensors), Batch Size 12, 200 Steps, Single-node CPU.*

| Framework / Configuration | Throughput | Page Faults | CPU Kernel Time |
| :--- | :---: | :---: | :---: |
| **PyTorch DataLoader** (Standard) | ~6.5 GB/s | Linear growth | ~90% |
| **ZeroTensor** (Pure IPC, No Pipeline) | **~22.0 GB/s** | **O(1) (Startup only)** | **~5%** |
| **ZeroTensor** (+ Rust SIMD Pipeline) | **~13.5 GB/s** | **O(1) (Startup only)** | **~15%** |

> **Why is ZeroTensor faster even with a pipeline?**  
> Standard PyTorch wastes cycles on Pickle serialization and IPC pipes. ZeroTensor applies transformations (like `Scale`, `Normalize`) directly in Rust using SIMD vectorization *before* the data ever touches Python, maintaining a massive throughput advantage.

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
* **In-Place Rust Transforms**: Apply `Scale`, `Add`, `Clamp`, or `Standardize` directly to the SHM buffer with automatic AVX2/AVX-512 vectorization.
* **Strict RAII Cleanup**: All sockets and `/dev/shm` segments are tied to Rust lifecycles. They are safely unlinked on panic, `SIGINT`, or normal drop.

## Quick Start
### 1. Rust Data Producer
Define your dataset using the `ZeroTensorDataset` trait. You can optionally attach a high-performance `Pipeline` to preprocess data in Rust.

```rust
use std::path::Path;
use zero_tensor_lib::{
    core::dataset::{
        item::{TensorDT, TensorBatchLayout},
        ZeroTensorDataset,
    },
    producer::ZeroTensorProducerBuilder,
    pipeline::Pipeline,
    transform::Scale,
};
use smallvec::smallvec;

struct MyDataset { /* Store metadata or source paths here */ }

impl ZeroTensorDataset for MyDataset {
    type Error = std::io::Error;

    fn len(&self) -> usize { 10_000 }
    fn is_empty(&self) -> bool { false }

    /// Returns the layout for a SINGLE item (Producer handles batch dimension automatically)
    fn get_batch_layout(&self, _indices: &[usize]) -> Result<TensorBatchLayout, Self::Error> {
        Ok(TensorBatchLayout::new(
            smallvec![3, 1024, 1024],          // Shape [C, H, W]
            smallvec![1024 * 1024, 1024, 1],   // Strides in elements
            TensorDT::F32
        ))
    }

    /// Writes raw item bytes directly into the pre-allocated slice
    fn write_item_into(&self, idx: usize, buf: &mut [u8]) -> Result<(), Self::Error> {
        // Write your decoded data (e.g., image pixels) into 'buf'
        // The buffer is already sized to the max shape of the batch
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = MyDataset {};
    let slot_size = 48 * 1024 * 1024; // 48 MB per slot

    // Optional: Add a high-performance preprocessing pipeline
    let pipeline = Pipeline::new().then(Scale::new(1.0 / 255.0)); // Normalize to [0, 1]

    let mut producer = ZeroTensorProducerBuilder::new(
        slot_size,
        "zt_shared_buffer",
        Path::new("/tmp/zt.sock"),
    )
    .num_slots(4)               // Use 3+ slots for pipeline efficiency
    .overwrite_socket(true)     // Safely clean up dead sockets on startup
    .shuffle(true)
    .seed(42)
    .pipeline(pipeline)         // Attach the pipeline
    .build()?;

    println!("Producer running... Waiting for Python consumer.");
    producer.start_streaming(&dataset, 12)?; // Batch size = 12

    Ok(())
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
            # IMPORTANT: do not use default batch.to() method, as it may lead to Race Condition
            inputs = consumer.to_device(batch, device, non_blocking=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
```

## Roadmap & Future Work

We are actively working on scaling ZeroTensor to support more complex deep learning workloads. Contributions are highly welcome!

[x] **In-Place Rust Dataset Pipeline**: Refactored core dataset traits to highly optimized in-place memory writes using zero-cost slicing.

[x] **Builder Pattern & Multi-Epoch Shuffling**: Integrated flexible producer initialization with configurable shuffling seeds and epoch signaling.

[x] **SIMD-Optimized Transforms**: Scale, Add, Clamp, and Standardize now utilize native SIMD vectorization (AVX2/AVX-512) via ndarray + rayon.

[x] **Selective Zeroing & Atomic Safety**: Guaranteed "all-or-nothing" atomicity for integer operations and prevention of data leakage between batches.

[] **Managed Consumer API**: Allow Python to automatically spawn and manage the Rust Producer process (similar to webdataset).

[] **GPU-Direct Support**: Direct SHM-to-GPU memory mapping via CUDA IPC to bypass CPU RAM entirely (targeting NVIDIA DALI-level performance).

[] **Distributed Training Support**: Multi-node broadcasting capabilities.