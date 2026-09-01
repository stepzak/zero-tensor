import os
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def benchmark_pytorch_with_augmentations():
    dataset_dir = os.path.expanduser("~/.cache/zero_tensor_bench")
    if not os.path.exists(dataset_dir):
        print("[PyTorch] ERROR: Dataset not found!")
        print("[PyTorch] Run Rust producer first to generate dataset:")
        print("  cargo run --release --example jpeg_bench")
        return

    batch_size = 32
    warmup_batches = 5
    target_batches = 150

    print("[PyTorch] Initializing ImageFolder with augmentations...")
    start_init = time.time()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
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
        persistent_workers=True,
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

        # Имитация потребления данных (как в Rust consumer)
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
    print("PYTORCH DATALOADER BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Batches processed:    {target_batches}")
    print(f"Total time:           {duration:.4f}s")
    print(f"Throughput:           {gb_per_sec:.2f} GB/s")
    print(f"Images/sec:           {images_per_sec:.0f}")
    print(f"Total data:           {total_bytes / (1024**2):.1f} MB")
    print("=" * 70)


if __name__ == "__main__":
    benchmark_pytorch_with_augmentations()