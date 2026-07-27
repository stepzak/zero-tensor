# ZeroTensor: Ultra-Fast IPC Data Loader for PyTorch

`ZeroTensor` is a high-performance, lightweight inter-process communication (IPC) data transport for PyTorch built in Rust. It serves as a drop-in replacement for the standard PyTorch `DataLoader` in high-throughput training scenarios, eliminating serialization overhead, runtime memory allocations, and kernel-space system call bottlenecks.

---

## Performance Benchmark (Total: 4800 MB Transferred)

## 🚀 Performance Benchmark (Total: 4800 MB Transferred)

*Environment: Synthetic dataset (3x512x512 F32 images), Batch Size 32, 50 Steps.*

| Metric | Standard PyTorch DataLoader | ZeroTensor IPC Loader | Improvement |
| :--- | :---: | :---: | :---: |
| **Throughput** | ~3.52 GB/s | **7.33+ GB/s** | **>2.0x Faster** |
| **Execution Time** | 1.33 s | **0.64 s** | **~52% Time Reduction** |
| **Page Faults** | Linear growth per batch | **O(1) (Startup only)** | **Eliminated runtime paging** |
| **Sys/CPU time** | 5.06s/13.27s (~38%) | 0.17s/2.2s(~8.5%) | **User-space dominant** |

> **Note:** The benchmark includes realistic CPU load (`copy_from_slice` + arithmetic) in the Rust producer to simulate real-world decoding/preprocessing. ZeroTensor maintains its lead even under heavy computational load due to its zero-copy architecture.

---

## The Problem

The standard PyTorch `DataLoader` using multiprocessing (`num_workers > 0`) hits severe performance walls due to Python and Linux kernel limitations:

1. **Page-Fault Storms:** PyTorch workers constantly allocate new memory blocks for each incoming batch. Under high throughput, this forces the Linux kernel to constantly interrupt execution to map virtual addresses to physical pages (hundreds of thousands of page-faults per second).
2. **Zombie Shared Memory:** If a PyTorch training run is dirty-killed (`Ctrl+C`, Out-Of-Memory, `kill -9`), orphaned shared memory blocks clutter `/dev/shm`, leaking RAM until a manual server reboot.
3. **Double Copy & Serialization:** Tensors are serialized/deserialized through Unix sockets or pipes, consuming up to 30% of total CPU cycles in kernel space (`sys` mode).

---

## Architectural Solutions of ZeroTensor

`ZeroTensor` decouples heavy I/O operations (parallel loading, decoding) in Rust from the Python-based model training loop using an optimized ring buffer.

* **Pre-allocated Ring Buffer (`mmap`):** Shared memory is mapped and "warmed up" once on startup. ZeroTensor maintains a fixed number of slots (`nslots`), avoiding dynamic allocations during the hot training loop.
* **Lock-Free Parallel Loading (Rayon):** The Rust producer utilizes a work-stealing thread pool to parallelize dataset loading, populating memory slots concurrently without expensive mutex locks.
* **Strict RAII Resource Management:** All temporary files, Unix sockets, and shared memory segments are tied to Rust's resource lifecycles (`Drop` trait). When the server drops, resources are safely unlinked and freed from `/dev/shm`.
* **Idempotent Socket Binding:** The custom `overwrite` flag allows the engine to safely clean up dead, non-responsive zombie sockets upon initialization without failing.

---

## Quick Start

### 1. Rust Data Producer


Define your dataset using the `ZeroTensorDataset` trait. Note the support for dynamic layouts via `get_batch_layout`.

```rust
use std::path::Path;
use zero_tensor_lib::{
    dataset::{
        item::{TensorDT, TensorBatchLayout},
        ZeroTensorDataset,
    },
    producer::ZeroTensorProducerBuilder,
};
use smallvec::smallvec;

struct MyDataset {
    // Store metadata or source paths here
}

impl ZeroTensorDataset for MyDataset {
    type Error = std::io::Error;

    fn len(&self) -> usize { 10000 }
    fn is_empty(&self) -> bool { false }

    /// Returns the layout for the ENTIRE batch (handling padding if needed)
    fn get_batch_layout(&self, indices: &[usize]) -> Result<TensorBatchLayout, Self::Error> {
        // Logic to find max H/W in indices and return padded layout
        // For fixed size, just return the static layout:
        Ok(TensorBatchLayout::new(
            smallvec![3, 512, 512], // Shape [C, H, W]
            smallvec![512*512, 512, 1], // Strides in elements
            TensorDT::F32
        ))
    }

    /// Writes raw item bytes directly into the pre-allocated slice
    fn write_item_into(&self, idx: usize, buf: &mut [u8]) -> Result<(), Self::Error> {
        // Write your data (e.g., image pixels) into 'buf'
        // The buffer is already sized to the max shape of the batch
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = MyDataset {};
    let slot_size = 32 * 1024 * 1024; // 32 MB per slot
    let steps = 100;

    let mut producer = ZeroTensorProducerBuilder::new(
        steps,
        slot_size,
        "zt_shared_buffer",
        Path::new("/tmp/zt.sock"),
    )
    .num_slots(3) // Use 3+ slots for pipeline efficiency
    .overwrite_socket(true)
    .shuffle(true)
    .seed(42)
    .build()?;

    println!("Producer running... Waiting for Python consumer.");
    producer.start_streaming(&dataset, 32)?; // Batch size = 32

    Ok(())
}
```

## 2. Python Training Consumer
Simply wrap your training loop with the Python context manager. Tensors are mapped from memory instantly with zero-copy.

```py
import torch
from zero_tensor_py import ZeroTensorConsumer

socket_path = "/tmp/zt.sock"
shm_name = "zt_shared_buffer"
slot_size = 32 * 1024 * 1024  # Must match producer slot size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Multi-epoch training loop
for epoch in range(5):
    with ZeroTensorConsumer(
        socket_path, shm_name, slot_size, nslots=3
    ) as consumer:
        for batch in consumer:
            inputs = batch.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
```
## System Profile Deep Dive
The telemetry captured via ``perf stat`` highlights why ZeroTensor outperforms traditional approaches:

### Allocation Complexity: *O(1)* vs *O(N)*
* **PyTorch** displays linear growth in page faults. Every batch requires new virtual memory mappings, and the first write to that memory triggers hardware page faults. 
* **ZeroTensor** is bounded at *O(1)*. The 75,540 page-fault count represents the initial import of PyTorch, NumPy, and the mapping of the ring-buffer at startup. During the entire training run, the page-fault count remains flat.

### Kernel vs User Space
* **PyTorch** spends 98% of its CPU runtime in kernel space (3.29s out of 3.33s elapsed time) resolving memory allocations and managing IPC file descriptors.
* **ZeroTensor** shifts the execution profile entirely to user space, spending only 0.17s in kernel space. Your CPU cores are dedicated to actual data processing, not OS housekeeping.

---

## Roadmap & TODO

We are actively working on scaling `ZeroTensor` to support more complex deep learning workloads. Contributions are highly welcome!

[x] **In-Place Rust Dataset Pipeline**: Refactored core dataset traits from dynamic heap allocations (Vec<u8>) to highly optimized in-place memory writes (write_item_into) using zero-cost slicing.

[x] **Builder Pattern & Multi-Epoch Shuffling**: Integrated flexible producer initialization with configurable shuffling seeds.

[ ] **Native Multi-Epoch Control Loop**: Support continuous connection handling across epochs via explicit EPOCH_DONE signaling.

[x] **Dynamic Tensor Shapes Support**: Implement elastic memory partitioning inside SHM slots for variable sequence length workloads (e.g., LLM tokenization, audio processing).