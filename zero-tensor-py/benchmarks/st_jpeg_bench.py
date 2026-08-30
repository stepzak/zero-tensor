import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def pad_collate(batch):
    images, labels = zip(*batch)
    
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    
    padded_images = []
    for img in images:
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]
        padded = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), value=0)
        padded_images.append(padded)
    
    images = torch.stack(padded_images)
    labels = torch.tensor(labels)
    
    return images, labels


def benchmark_pytorch():
    dataset_dir = os.path.expanduser("~/.cache/zero_tensor_bench")
    
    if not os.path.exists(dataset_dir):
        print("[PyTorch] ERROR: Dataset not found!")
        return
    
    batch_size = 32
    warmup_batches = 5
    target_batches = 150
    
    print("[PyTorch] Initializing ImageFolder...")
    start_init = time.time()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
    
    init_time = time.time() - start_init
    print(f"[PyTorch] Dataset initialized in {init_time:.2f}s ({len(dataset)} images)")
    
    print(f"\n[PyTorch] Starting benchmark ({target_batches} batches of size {batch_size})...")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=pad_collate
    )
    
    start_bench = None
    total_bytes = 0
    batch_count = 0
    
    for images, labels in dataloader:
        if batch_count < warmup_batches:
            batch_count += 1
            continue
        
        if start_bench is None:
            start_bench = time.time()
            total_bytes = 0
        
        _ = images.sum().item()
        total_bytes += images.nbytes + labels.nbytes
        batch_count += 1
        
        if batch_count >= warmup_batches + target_batches:
            break
    
    end_time = time.time()
    duration = end_time - start_bench
    images_per_sec = target_batches * batch_size / duration
    mb_per_sec = total_bytes / (1024 * 1024) / duration
    gb_per_sec = total_bytes / (1024 ** 3) / duration
    
    print("\n" + "=" * 70)
    print("PYTORCH DATALOADER BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Batches processed:    {target_batches}")
    print(f"Total time:           {duration:.4f}s")
    print(f"Throughput:           {gb_per_sec:.2f} GB/s")
    print(f"Images/sec:           {images_per_sec:.0f}")
    print(f"Total data:           {total_bytes / (1024**2):.1f} MB")
    print("=" * 70)


if __name__ == "__main__":
    benchmark_pytorch()