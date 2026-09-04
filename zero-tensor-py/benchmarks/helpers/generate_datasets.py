import argparse
import io
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image


def generate_jpeg_bytes(
    width: int, height: int, seed: int, quality: int = 85
) -> bytes:
    
    rng = np.random.RandomState(seed)
    
    
    pixels = rng.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
    
    img = Image.fromarray(pixels)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def generate_imagefolder(
    output_dir: Path,
    num_images: int,
    num_classes: int,
    seed: int,
    quality: int,
):
    
    if output_dir.exists():
        print(f"[Generate] Cleaning existing dataset at {output_dir}")
        import shutil
        shutil.rmtree(output_dir)

    print(
        f"[Generate] Creating ImageFolder: {num_images} images, "
        f"{num_classes} classes, seed={seed}"
    )

    images_per_class = num_images // num_classes
    remainder = num_images % num_classes

    global_idx = 0
    for class_id in range(num_classes):
        class_dir = output_dir / f"class_{class_id:03d}"
        class_dir.mkdir(parents=True, exist_ok=True)

        n = images_per_class + (1 if class_id < remainder else 0)
        for _ in range(n):
            
            width = 100 + (global_idx * 37 % 301)
            height = 100 + (global_idx * 53 % 301)

            jpeg_bytes = generate_jpeg_bytes(width, height, seed=global_idx, quality=quality)
            img_path = class_dir / f"img_{global_idx:05d}.jpg"
            img_path.write_bytes(jpeg_bytes)

            global_idx += 1

    print(f"[Generate] Done! Total images: {global_idx}")
    print(f"[Generate] Output: {output_dir}")


def generate_tar_shards(
    output_dir: Path,
    num_images: int,
    num_classes: int,
    num_shards: int,
    seed: int,
    quality: int,
):
    
    if output_dir.exists():
        print(f"[Generate] Cleaning existing dataset at {output_dir}")
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[Generate] Creating TAR shards: {num_images} images, "
        f"{num_classes} classes, {num_shards} shards, seed={seed}"
    )

    images_per_shard = num_images // num_shards
    remainder = num_images % num_shards

    global_idx = 0
    for shard_idx in range(num_shards):
        shard_path = output_dir / f"shard_{shard_idx:03d}.tar"
        n = images_per_shard + (1 if shard_idx < remainder else 0)

        with tarfile.open(shard_path, "w") as tar:
            for _ in range(n):
                width = 100 + (global_idx * 37 % 301)
                height = 100 + (global_idx * 53 % 301)

                jpeg_bytes = generate_jpeg_bytes(
                    width, height, seed=global_idx, quality=quality
                )

                class_id = global_idx % num_classes
                filename = f"class_{class_id:03d}/img_{global_idx:05d}.jpg"

                info = tarfile.TarInfo(name=filename)
                info.size = len(jpeg_bytes)
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(jpeg_bytes))

                global_idx += 1

        print(f"  Created {shard_path.name}: {n} images")

    print(f"[Generate] Done! Total images: {global_idx}")
    print(f"[Generate] Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate test datasets for ZeroTensor benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--format",
        choices=["imagefolder", "tar", "both"],
        default="imagefolder",
        help="Формат выходных данных (default: imagefolder)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path.home() / ".cache" / "zero_tensor_bench",
        help="Директория для выходных данных",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=5000,  
        help="Общее количество изображений (default: 5000)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Количество классов (default: 10)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=4,
        help="Количество TAR-шардов (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed для воспроизводимости (default: 42)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=85,
        help="JPEG качество 1-100 (default: 85)",
    )

    args = parser.parse_args()

    if args.format in ("imagefolder", "both"):
        generate_imagefolder(
            output_dir=args.output_dir,
            num_images=args.num_images,
            num_classes=args.num_classes,
            seed=args.seed,
            quality=args.quality,
        )

    if args.format in ("tar", "both"):
        tar_dir = args.output_dir / "tar" if args.format == "both" else args.output_dir
        generate_tar_shards(
            output_dir=tar_dir,
            num_images=args.num_images,
            num_classes=args.num_classes,
            num_shards=args.num_shards,
            seed=args.seed,
            quality=args.quality,
        )


if __name__ == "__main__":
    main()