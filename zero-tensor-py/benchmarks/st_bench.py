import time
import torch


ELEMENT_SHAPE = (3, 512, 512) 
BATCH_SIZE = 48
PYTORCH_STEPS = 200 
NWORKERS = 8


class DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return PYTORCH_STEPS * BATCH_SIZE 
    def __getitems__(self, indices):
        batch = torch.empty((len(indices), *ELEMENT_SHAPE), dtype=torch.float32)
        for i, idx in enumerate(indices):
            batch[i].fill_(float(idx % 255))
        return batch

def benchmark_standard_loader():
    dataset = DummyDataset()
    
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NWORKERS, 
        persistent_workers=True,
        collate_fn=lambda batch: batch,
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

if __name__ == "__main__":
    benchmark_standard_loader()