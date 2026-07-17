import os
import time
import torch
from zero_tensor_py import ZeroTensorConsumer

import torch
import time

ELEMENT_SHAPE = (3, 512, 512) 
BATCH_SIZE = 32
PYTORCH_STEPS = 50 


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = torch.randn(ELEMENT_SHAPE, dtype=torch.float32)
        self.data.fill_(1.0)
    def __len__(self):
        return PYTORCH_STEPS * BATCH_SIZE 
    def __getitem__(self, idx):
        val = float(idx % 255)
        return self.data.clone().fill_(val)

def benchmark_standard_loader():
    dataset = DummyDataset()
    
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=2, 
        drop_last=True,
        prefetch_factor=1 
    ) 
    
    print("[Bench] Starting Standard PyTorch DataLoader (Safe Mode)...")
    start_time = time.perf_counter()
    total_bytes = 0
    
    for batch in loader:
        total_bytes += batch.nbytes
        _ = batch[0, 0, 0, 0].item()
        del batch 

    mb = total_bytes / (1024**2)
        
    end_time = time.perf_counter()
    duration = end_time - start_time
    gb_per_sec = (total_bytes / (1024**3)) / duration
    print(f"Standard PyTorch: {duration:.2f}s ({gb_per_sec:.2f} GB/s) | Total: {mb:.1f} MB")

def benchmark_zero_tensor():
    print("[Bench] Starting ZeroTensor IPC Loader...")
    
    socket_path = "/tmp/zt_bench.sock"
    shm_name = "zt_bench"
    slot_size = (32 * 3 * 512 * 512 * 4) + 4096 
    
    if not os.path.exists(socket_path):
        print("Skip: Rust producer is not running. Run the rust bench companion first!")
        return

    start_time = time.perf_counter()
    total_bytes = 0
    
    with ZeroTensorConsumer(socket_path, shm_name, slot_size, nslots=2) as consumer:
        for batch in consumer:
            total_bytes += batch.nbytes
            
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
    benchmark_standard_loader()
    benchmark_zero_tensor()