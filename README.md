# ZeroTensor: Ultra-Fast IPC Data Loader for PyTorch

`ZeroTensor` is a high-performance, lightweight inter-process communication (IPC) data transport for PyTorch built in Rust. It serves as a drop-in replacement for the standard PyTorch `DataLoader` in high-throughput training scenarios, eliminating serialization overhead, runtime memory allocations, and kernel-space system call bottlenecks.

---

## Performance Benchmark (Total: 4800 MB Transferred)

| Metric | Standard PyTorch DataLoader | ZeroTensor IPC Loader | System Impact |
| :--- | :---: | :---: | :--- |
| **Throughput** | 2.16 GB/s | **8.0+ GB/s** | **>3.5x Faster** |
| **Execution Time** | 2.17 s | **<0.60 s** | **Time reduced by ~75%** |
| **Absolute Page-Faults** | **1,625,422** | **75,540** | **21.5x Reduction** |
| **Page-Fault Rate** | ~241,926 / sec | **~33,192 / sec** | Dramatic reduction in OS paging overhead |
| **Kernel Space Time (`sys`)** | **3.29 s** | **0.17 s** | **19x Less OS kernel overhead** |
| **CPU Utilization (Python)** | 2.0 CPUs (100% Core + 100% Atom) | **1.0 CPU (User-space only)** | Minimizes scheduling noise, preserves cores |

> This benchmark measures **pure IPC transport and zero-copy slicing overhead** using a synthetic dataset. By removing heavy disk I/O boundaries, we isolate and expose the core architectural performance of both data loaders. Under these conditions, ZeroTensor demonstrates near-hardware memory bandwidth limits.

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

Define your dataset and spawn the streaming loop using the thread-safe `ZeroTensorProducer`:

```rust
use std::path::Path;
use zero_tensor_lib::{
    dataset::{
        item::{TensorDT, TensorItemMeta},
        ZeroTensorDataset,
    },
    producer::ZeroTensorProducerBuilder,
};

struct MyDataset {
    meta: TensorItemMeta,
}

impl ZeroTensorDataset for MyDataset {
    fn len(&self) -> usize { 10000 }
    fn is_empty(&self) -> bool { false }

    fn get_metadata(&self, _idx: usize) -> Option<TensorItemMeta> {
        Some(self.meta.clone())
    }

    fn get_item_into(&self, idx: usize, buf: &mut [u8]) -> Option<TensorItemMeta> {
        // Write raw item bytes directly into the pre-allocated slice
        // e.g., copy image bytes or raw f32 slices
        Some(self.meta.clone())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = MyDataset::new();
    let slot_size = 32 * 1024 * 1024; // 32 MB slot size
    let steps = 100;

    let mut producer = ZeroTensorProducerBuilder::new(
        steps,
        slot_size,
        "zt_shared_buffer",
        Path::new("/tmp/zt.sock"),
    )
    .num_slots(3)
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

[x] **In-Place Rust Dataset Pipeline**: Refactored core dataset traits from dynamic heap allocations (Vec<u8>) to highly optimized in-place memory writes (get_item_into) using zero-cost slicing.

[x] **Builder Pattern & Multi-Epoch Shuffling**: Integrated flexible producer initialization with configurable shuffling seeds.

[ ] **Native Multi-Epoch Control Loop**: Support continuous connection handling across epochs via explicit EPOCH_DONE signaling.

[ ] **Dynamic Tensor Shapes Support**: Implement elastic memory partitioning inside SHM slots for variable sequence length workloads (e.g., LLM tokenization, audio processing).