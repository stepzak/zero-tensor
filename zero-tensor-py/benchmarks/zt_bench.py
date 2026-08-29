import os
import time

from zero_tensor_py.consumer import ZeroTensorConsumer


def benchmark_zero_tensor():
    print("[Bench] Starting ZeroTensor IPC Loader...")
    
    socket_path = "/tmp/zt_bench.sock"
    shm_name = "zt_bench"
    
    if not os.path.exists(socket_path):
        print("Skip: Rust producer is not running. Run the rust bench companion first!")
        return

    start_time = time.perf_counter()
    total_bytes = 0
    counter = 0
    with ZeroTensorConsumer(socket_path, shm_name, prefetch_factor=3) as consumer:
        for batch in consumer:
            batch = batch["data"]
            total_bytes += batch.nbytes
            counter += 1
            _ = batch[0, 0, 0, 0].item()
            
    end_time = time.perf_counter()
    duration = end_time - start_time
    mb_total = total_bytes / (1024 ** 2)
    gb_total = total_bytes / (1024 ** 3)
    
    if duration > 0:
        gb_per_sec = gb_total / duration
    else:
        gb_per_sec = 0.0
        
    print(f"ZeroTensor IPC: {duration:.4f}s ({gb_per_sec:.2f} GB/s) | Total: {mb_total:.1f} MB")

if __name__ == "__main__":
    benchmark_zero_tensor()
