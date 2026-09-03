import os
import time
import tarfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


class TarImageDataset(Dataset):
    def __init__(self, shard_paths, transform=None):
        self.samples = []  
        self.transform = transform

        for shard_path in shard_paths:
            with tarfile.open(shard_path, "r") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    if not member.name.endswith((".jpg", ".jpeg")):
                        continue

                    
                    label = 0
                    if "class_" in member.name:
                        try:
                            rest = member.name.split("class_")[1]
                            label = int(rest.split("/")[0])
                        except (IndexError, ValueError):
                            label = 0

                    f = tar.extractfile(member)
                    if f:
                        self.samples.append((f.read(), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        jpeg_bytes, label = self.samples[idx]
        img = Image.open(__import__("io").BytesIO(jpeg_bytes)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.int64)


def benchmark_pytorch_tar():
    dataset_dir = os.path.expanduser("~/.cache/zero_tensor_tar_bench")
    if not os.path.exists(dataset_dir):
        print("[PyTorch] ERROR: TAR shards not found!")
        print("[PyTorch] Run Rust producer first:")
        print("  cargo run --release --example tar_jpeg_bench")
        return

    shard_paths = sorted(Path(dataset_dir).glob("shard_*.tar"))
    if not shard_paths:
        print("[PyTorch] ERROR: No shards found in", dataset_dir)
        return

    print(f"[PyTorch] Found {len(shard_paths)} shards")

    batch_size = 32
    warmup_batches = 5
    target_batches = 150

    print("[PyTorch] Initializing TarImageDataset...")
    start_init = time.time()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = TarImageDataset([str(p) for p in shard_paths], transform=transform)
    init_time = time.time() - start_init
    print(f"[PyTorch] Dataset initialized in {init_time:.2f}s ({len(dataset)} images)")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    print(f"\n[PyTorch] Starting benchmark ({target_batches} batches)...")
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
    gb_per_sec = total_bytes / (1024 ** 3) / duration

    print("\n" + "=" * 70)
    print("PYTORCH TAR BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Batches processed:    {batch_count - warmup_batches}")
    print(f"Total time:           {duration:.4f}s")
    print(f"Throughput:           {gb_per_sec:.2f} GB/s")
    print(f"Images/sec:           {images_per_sec:.0f}")
    print(f"Total data:           {total_bytes / (1024 ** 2):.1f} MB")
    print("=" * 70)


if __name__ == "__main__":
    benchmark_pytorch_tar()