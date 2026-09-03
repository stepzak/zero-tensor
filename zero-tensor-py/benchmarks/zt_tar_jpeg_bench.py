import os
import time
from zero_tensor_py.consumer import ZeroTensorConsumer


def benchmark_zero_tensor():
    print("[Python] Starting ZeroTensor TAR Benchmark...")

    socket_path = "/tmp/zt_tar_bench.sock"
    shm_name = "zt_tar_bench"

    if not os.path.exists(socket_path):
        print("[Python] ERROR: Rust producer is not running!")
        print("[Python] Please run the Rust producer first:")
        print("  cargo run --release --example tar_jpeg_bench")
        return

    print("[Python] Connecting to Producer...")

    start_time = None
    total_bytes = 0
    batch_count = 0
    warmup_batches = 20
    target_batches = 150
    batch_size = 32

    with ZeroTensorConsumer(socket_path, shm_name, prefetch_factor=12) as consumer:
        while batch_count < warmup_batches + target_batches:
            for batch in consumer:
                if batch_count < warmup_batches:
                    batch_count += 1
                    continue

                if start_time is None:
                    start_time = time.perf_counter()
                    total_bytes = 0
                image = batch["image"]
                label = batch["label"]

                if total_bytes == 0:
                    batch_size = image.shape[0]

                total_bytes += image.nbytes + label.nbytes
                batch_count += 1

                _ = image.sum().item()

                if batch_count >= warmup_batches + target_batches:
                    break

    end_time = time.perf_counter()
    duration = end_time - start_time

    mb_total = total_bytes / (1024 ** 2)
    gb_total = total_bytes / (1024 ** 3)
    gb_per_sec = gb_total / duration if duration > 0 else 0.0
    batches_per_sec = target_batches / duration if duration > 0 else 0.0
    images_per_sec = target_batches * batch_size / duration if duration > 0 else 0.0

    print("\n" + "=" * 70)
    print("ZEROTENSOR TAR BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Batches processed:    {batch_count - warmup_batches}")
    print(f"Total time:           {duration:.4f}s")
    print(f"Throughput:           {gb_per_sec:.2f} GB/s")
    print(f"Batches/sec:          {batches_per_sec:.1f}")
    print(f"Images/sec:           {images_per_sec:.0f}")
    print(f"Total data:           {mb_total:.1f} MB")
    print("=" * 70)


if __name__ == "__main__":
    benchmark_zero_tensor()