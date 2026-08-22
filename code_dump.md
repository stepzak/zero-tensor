# Code Dump: zero-tensor-lib

**Generated:** 2026-08-23 01:39:17
**Files:** 32

---

## 📑 Table of Contents

- `.github/workflows/ci.yml`
- `Cargo.toml`
- `test.py`
- `zero-tensor-py/benchmarks/st_bench.py`
- `zero-tensor-py/benchmarks/zt_bench.py`
- `zero-tensor-py/pyproject.toml`
- `zero-tensor-py/src/zero_tensor_py/__init__.py`
- `zero-tensor-py/src/zero_tensor_py/consumer.py`
- `zero-tensor-py/src/zero_tensor_py/exceptions.py`
- `zero-tensor-py/src/zero_tensor_py/protocol.py`
- `zero-tensor-py/tests/test_consumer.py`
- `zero-tensor-rs/Cargo.toml`
- `zero-tensor-rs/src/bin/throughput_bench.rs`
- `zero-tensor-rs/src/lib.rs`
- `zero-tensor-rs/src/pipeline/error.rs`
- `zero-tensor-rs/src/pipeline/mod.rs`
- `zero-tensor-rs/src/pipeline/tests.rs`
- `zero-tensor-rs/src/transform/add.rs`
- `zero-tensor-rs/src/transform/clamp.rs`
- `zero-tensor-rs/src/transform/error.rs`
- `zero-tensor-rs/src/transform/helpers.rs`
- `zero-tensor-rs/src/transform/mod.rs`
- `zero-tensor-rs/src/transform/scalar/cmp.rs`
- `zero-tensor-rs/src/transform/scalar/is_zero.rs`
- `zero-tensor-rs/src/transform/scalar/mod.rs`
- `zero-tensor-rs/src/transform/scalar/ops.rs`
- `zero-tensor-rs/src/transform/scalar/tests.rs`
- `zero-tensor-rs/src/transform/scale.rs`
- `zero-tensor-rs/src/transform/standardize.rs`
- `zero-tensor-rs/tests/integration_consumer.py`
- `zero-tensor-rs/tests/integration_producer.rs`
- `zero-tensor-rs/tests/integration_signals.rs`

---

## 📄 Source Code

# .github/workflows/ci.yml (92 lines)
```yaml
name: CI

on:
  push:
    branches: [ "main", "master" ]
  pull_request:
    branches: [ "main", "master" ]

env:
  CARGO_TERM_COLOR: always

jobs:
  rust-checks:
    name: Rust Code Quality & Integration Tests
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Rust toolchain
        uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy, rustfmt

      - name: Enable Rust caching
        uses: Swatinem/rust-cache@v2

      - name: Set up UV
        uses: astral-sh/setup-uv@v5
        with: 
          enable-cache: true
          cache-dependency-glob: "zero-tensor-py/uv.lock"
          python-version: "3.11"

      - name: Install Python dependencies for E2E
        run: uv sync --locked --project zero-tensor-py

      - name: Check Rust formatting
        run: cargo fmt --check

      - name: Run Clippy lints
        run: cargo clippy -- -D warnings
      
      - name: Build Rust Dev
        run: cargo build

      - name: Cleanup IPC leftovers before tests
        run: rm -f /tmp/zt*.sock /dev/shm/zt* || true

      - name: Run Rust tests
        run: cargo test -- --nocapture

  python-checks:
    name: Python Quality & Tests (Py ${{ matrix.python-version }})
    runs-on: ubuntu-latest

    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Rust toolchain
        uses: dtolnay/rust-toolchain@stable

      - name: Enable Rust caching
        uses: Swatinem/rust-cache@v2

      - name: Set up UV
        uses: astral-sh/setup-uv@v5
        with: 
          enable-cache: true
          cache-dependency-glob: "zero-tensor-py/uv.lock"
          python-version: ${{ matrix.python-version }}

      - name: Install UV dependencies
        run: uv sync --locked --project zero-tensor-py

      - name: Run Ruff
        working-directory: ./zero-tensor-py
        run: uv run ruff check

      - name: Cleanup IPC leftovers before tests
        run: rm -f /tmp/zt*.sock /dev/shm/zt* || true

      - name: Run Pytest
        working-directory: ./zero-tensor-py
        run: uv run pytest
```

# Cargo.toml (5 lines)
```toml
[workspace]
resolver="2"
members=[
    "zero-tensor-rs"
]
```

# test.py (288 lines)
```python
#!/usr/bin/env python3
"""
Сборщик исходного кода проекта в один файл.
Оптимизирован для отправки в LLM для анализа/ревью.

Использование:
    python collect_code.py                      # собрать в code_dump.md
    python collect_code.py -o output.txt        # указать выходной файл
    python collect_code.py --format plain       # без markdown
    python collect_code.py --no-code            # только структура файлов
    python collect_code.py --ignore "*.lock"    # дополнительные паттерны
"""

import argparse
import os
import sys
import fnmatch
from pathlib import Path
from datetime import datetime

# Расширения текстовых файлов, которые стоит собирать
TEXT_EXTENSIONS = {
    # Rust
    ".rs", ".toml",
    # Python
    ".py", ".pyi",
    # Конфиги
    ".md", ".txt", ".ini", ".cfg", ".yaml", ".yml", ".json",
    # C/C++
    ".c", ".h", ".cpp", ".hpp", ".cc", ".hh",
    # Шелл
    ".sh", ".bash", ".zsh",
    # Docker
    "Dockerfile",
    # Прочее
    ".gitignore", ".dockerignore", ".env",
}

# Директории, которые всегда игнорируем
IGNORE_DIRS = {
    "target", "build", "dist", ".git", ".hg", ".svn",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".venv", "venv", "env", ".env",
    "node_modules", ".tox", ".nox",
    "htmlcov", ".coverage",
    ".idea", ".vscode",
}

# Маппинг расширений на языки для markdown-подсветки
LANG_MAP = {
    ".rs": "rust",
    ".py": "python",
    ".toml": "toml",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sh": "bash",
    ".bash": "bash",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".js": "javascript",
    ".ts": "typescript",
}


def is_text_file(path: Path) -> bool:
    """Проверяем, является ли файл текстовым по расширению или имени."""
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    if path.name in {"Dockerfile", "Makefile", "LICENSE", "README"}:
        return True
    return False


def should_ignore(path: Path, extra_ignore: list[str]) -> bool:
    """Проверяем, нужно ли игнорировать путь."""
    # Проверяем директорию
    for part in path.parts:
        if part in IGNORE_DIRS:
            return True
    
    # Проверяем дополнительные паттерны
    path_str = str(path)
    for pattern in extra_ignore:
        if fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(path.name, pattern):
            return True
    
    return False


def get_language(path: Path) -> str:
    """Определяем язык для markdown-подсветки."""
    if path.name == "Dockerfile":
        return "dockerfile"
    return LANG_MAP.get(path.suffix.lower(), "")


def collect_files(root: Path, extra_ignore: list[str]) -> list[Path]:
    """Собираем все текстовые файлы, исключая мусор."""
    files = []
    
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        
        if should_ignore(path, extra_ignore):
            continue
        
        if is_text_file(path):
            files.append(path)
    
    # Сортируем для предсказуемого порядка
    return sorted(files)


def read_file_safe(path: Path, max_size_mb: float) -> tuple[str, bool]:
    """
    Безопасно читаем файл.
    Возвращает (содержимое, was_truncated).
    """
    max_bytes = int(max_size_mb * 1024 * 1024)
    
    try:
        size = path.stat().st_size
        if size > max_bytes:
            return f"[FILE TOO LARGE: {size / 1024 / 1024:.2f} MB, limit {max_size_mb} MB]", True
        
        # Читаем с заменой не-UTF-8 символов
        content = path.read_text(encoding="utf-8", errors="replace")
        return content, False
    except Exception as e:
        return f"[ERROR READING FILE: {e}]", True


def format_header(path: Path, rel_path: Path, lines_count: int, truncated: bool) -> str:
    """Форматируем заголовок файла."""
    marker = " ⚠️ TRUNCATED" if truncated else ""
    return f"# {rel_path} ({lines_count} lines){marker}\n"


def collect_to_file(
    root: Path,
    output: Path,
    format_type: str,
    no_code: bool,
    extra_ignore: list[str],
    max_size_mb: float,
):
    """Основная функция сбора."""
    files = collect_files(root, extra_ignore)
    
    if not files:
        print(f"❌ No files found in {root}")
        return
    
    total_lines = 0
    total_chars = 0
    file_stats = []
    
    with output.open("w", encoding="utf-8") as out:
        # Заголовок дампа
        out.write(f"# Code Dump: {root.name}\n\n")
        out.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write(f"**Files:** {len(files)}\n\n")
        out.write("---\n\n")
        
        # Таблица содержания
        out.write("## 📑 Table of Contents\n\n")
        for f in files:
            rel = f.relative_to(root)
            out.write(f"- `{rel}`\n")
        out.write("\n---\n\n")
        
        # Содержимое файлов
        out.write("## 📄 Source Code\n\n")
        
        for f in files:
            rel = f.relative_to(root)
            content, truncated = read_file_safe(f, max_size_mb)
            lines = content.splitlines()
            lines_count = len(lines)
            total_lines += lines_count
            total_chars += len(content)
            
            file_stats.append((rel, lines_count, len(content), truncated))
            
            if format_type == "markdown":
                out.write(format_header(f, rel, lines_count, truncated))
                if not no_code:
                    lang = get_language(f)
                    out.write(f"```{lang}\n")
                    out.write(content)
                    if not content.endswith("\n"):
                        out.write("\n")
                    out.write("```\n\n")
            else:
                # Plain text формат
                out.write(f"{'=' * 80}\n")
                out.write(f"FILE: {rel} ({lines_count} lines)\n")
                out.write(f"{'=' * 80}\n")
                if not no_code:
                    out.write(content)
                    if not content.endswith("\n"):
                        out.write("\n")
                out.write("\n")
    
    # Статистика
    print(f"\n✅ Collected {len(files)} files into {output}")
    print(f"   Total lines: {total_lines:,}")
    print(f"   Total size:  {total_chars / 1024:.1f} KB")
    print(f"\n📊 Top 10 largest files:")
    
    top_files = sorted(file_stats, key=lambda x: x[2], reverse=True)[:10]
    for rel, lines, chars, truncated in top_files:
        marker = " ⚠️" if truncated else ""
        print(f"   {chars / 1024:6.1f} KB | {lines:5d} lines | {rel}{marker}")


def main():
    parser = argparse.ArgumentParser(
        description="Сбор исходного кода проекта в один файл",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Корневая директория проекта (по умолчанию: текущая)",
    )
    parser.add_argument(
        "-o", "--output",
        default="code_dump.md",
        help="Выходной файл (по умолчанию: code_dump.md)",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "plain"],
        default="markdown",
        help="Формат вывода (по умолчанию: markdown)",
    )
    parser.add_argument(
        "--no-code",
        action="store_true",
        help="Собрать только структуру файлов, без содержимого",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        help="Дополнительные паттерны игнорирования (можно указывать несколько раз)",
    )
    parser.add_argument(
        "--max-size",
        type=float,
        default=1.0,
        help="Максимальный размер файла в MB (по умолчанию: 1.0)",
    )
    
    args = parser.parse_args()
    
    root = Path(args.root).resolve()
    output = Path(args.output)
    
    if not root.exists():
        print(f"❌ Directory not found: {root}")
        sys.exit(1)
    
    if not root.is_dir():
        print(f"❌ Not a directory: {root}")
        sys.exit(1)
    
    print(f"🔍 Scanning {root}...")
    
    collect_to_file(
        root=root,
        output=output,
        format_type=args.format,
        no_code=args.no_code,
        extra_ignore=args.ignore,
        max_size_mb=args.max_size,
    )


if __name__ == "__main__":
    main()
```

# zero-tensor-py/benchmarks/st_bench.py (48 lines)
```python
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
```

# zero-tensor-py/benchmarks/zt_bench.py (39 lines)
```python
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
    with ZeroTensorConsumer(socket_path, shm_name) as consumer:
        for batch in consumer:
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
```

# zero-tensor-py/pyproject.toml (26 lines)
```toml
[project]
name = "zero-tensor-py"
version = "0.5.0"
description = "Add your description here"
readme = "README.md"
authors = [
    { name = "stepzak", email = "stepanzakatin@gmail.com" }
]
requires-python = ">=3.10"
dependencies = [
    "atomics>=1.0.3",
    "numpy>=2.2.6",
    "torch>=2.0.0",
]

[build-system]
requires = ["uv_build>=0.9.17,<0.10.0"]
build-backend = "uv_build"

[dependency-groups]
dev = [
    "pytest>=8.0.0",
    "black>=24.0.0",
    "ruff>=0.3.0",
    "py-spy>=0.4.2",
]
```

# zero-tensor-py/src/zero_tensor_py/__init__.py (5 lines)
```python
from .consumer import ZeroTensorConsumer

__all__ = [
    "ZeroTensorConsumer"
]
```

# zero-tensor-py/src/zero_tensor_py/consumer.py (239 lines)
```python
import gc
import mmap
import os
import select
import socket
import struct
from typing import Generator, Optional
import atomics

import torch
from zero_tensor_py.protocol import TensorHeaderParser
import zero_tensor_py.exceptions as zt_exc

VERSION = "0.5.0"
_CONTROL_START_MSG = b"START\n"
_CONTROL_STOP_MSG = b"STOP\n"
_CONTROL_NEXT_EPOCH_MSG = b"EPOCH_DONE\n"
_SOCK_WAIT_POLL_TIMEOUT = 0.00001
_PROTO_BEGIN_STR = "ZT"

class ZeroTensorConsumer:
    def __init__(self, socket_path: str, shm_name: str):
        self.socket_path = socket_path
        self.shm_name = os.path.join("/dev/shm", shm_name)
        self.slot_size = None
        self.nslots = None

        self.sock: Optional[socket.socket] = None
        self.shm_file = None
        self.mem: Optional[mmap.mmap] = None
        
        self.handshake_dict = {}
        self.cb_size = 0
        self.head_offset = 0
        self.tail_offset = 0
        self.is_running_offset = 0
        self.header_size = 0
        self.dt_offset = 0
        self.ndims_offset = 0
        self.is_ready_offset = 0
        self.shape_type_size = 0
        self._tail_view = None
        self._head_view = None
        self._is_running_view = None

    def _parse_handshake(self, handshake_str: str):
        parts = handshake_str.strip().split()
        if not parts or parts[0] != _PROTO_BEGIN_STR:
            raise zt_exc.ProtocolError(f"Invalid handshake protocol: {handshake_str}")
        if parts[1] != VERSION:
            raise zt_exc.ProtocolError(f"Invalid protocol version, consumer is {VERSION}, producer is {parts[1]}")
        for part in parts[2:]:
            if "=" in part:
                key, val = part.split("=", 1)
                try:
                    self.handshake_dict[key] = int(val)
                except ValueError:
                    raise zt_exc.MalformedMessageError(f"{key} did not have a valid value: {val}. Full str: {handshake_str}")
        try:
            self.cb_size = self.handshake_dict["cb_size"]
            self.head_offset = self.handshake_dict["head_offset"]
            self.tail_offset = self.handshake_dict["tail_offset"]
            self.is_running_offset = self.handshake_dict["is_running_offset"]
            
            self.header_size = self.handshake_dict["header_size"]
            self.dt_offset = self.handshake_dict["dt_offset"]
            self.ndims_offset = self.handshake_dict["ndims_offset"]
            self.is_ready_offset = self.handshake_dict["is_ready_offset"]
            self.shape_type_size = self.handshake_dict["shape_type_size"]
        except KeyError as e:
            missing = e.args[0]
            raise zt_exc.ProtocolError(f"Invalid handshake protocol, {missing} is missing. Full str: {handshake_str}")

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.sock.connect(self.socket_path)
            self.sock.sendall(_CONTROL_START_MSG)
            
            handshake_bytes = b""
            while b"\n" not in handshake_bytes:
                chunk = self.sock.recv(4096)
                if not chunk:
                    raise zt_exc.ZTConnectionError("Producer closed connection during handshake")
                handshake_bytes += chunk
            self._parse_handshake(handshake_bytes.decode('utf-8'))
            
            self.shm_file = open(self.shm_name, "r+b")
            
            cb_map = mmap.mmap(self.shm_file.fileno(), self.cb_size)
            nslots_offset = self.handshake_dict["nslots_offset"]
            slot_size_offset = self.handshake_dict["slot_size_offset"]
            
            self.nslots = struct.unpack_from("<I", cb_map, nslots_offset)[0]
            self.slot_size = struct.unpack_from("<I", cb_map, slot_size_offset)[0]
            cb_map.close()
            
            self.total_size = self.cb_size + (self.nslots * self.slot_size)
            self.mem = mmap.mmap(self.shm_file.fileno(), self.total_size)
            self._tail_view = memoryview(self.mem)[self.tail_offset:self.tail_offset+8]
            self._head_view = memoryview(self.mem)[self.head_offset:self.head_offset + 8]
            self._is_running_view = memoryview(self.mem)[self.is_running_offset:self.is_running_offset + 8]
            self.sock.setblocking(False)
            
        except Exception as e:
            self.close()
            raise e

    def close(self):
        if self.mem is not None and self.is_running_offset > 0:
            try:
                self._store_is_running(0)
            except Exception:
                pass

        for view in (self._tail_view, self._head_view, self._is_running_view):
            if view is not None:
                view.release()
        self._tail_view = self._head_view = self._is_running_view = None

        if self.mem is not None:
            try:
                self.mem.close()
            except BufferError:
                gc.collect()
                try:
                    self.mem.close()
                except BufferError:
                    pass
            self.mem = None
            
        if self.shm_file is not None:
            self.shm_file.close()
            self.shm_file = None

        if self.sock is not None:
            try:
                self.sock.sendall(_CONTROL_STOP_MSG)
            except (BrokenPipeError, OSError):
                pass 
            self.sock.close()
            self.sock = None

    def __enter__(self) -> "ZeroTensorConsumer":
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _load_head(self) -> int:
        with atomics.atomicview(buffer=self._head_view, atype=atomics.UINT) as a:
            return a.load(order=atomics.MemoryOrder.ACQUIRE)

    def _load_is_running(self) -> int:
        with atomics.atomicview(buffer=self._is_running_view, atype=atomics.UINT) as a:
            return a.load(order=atomics.MemoryOrder.ACQUIRE)
        
    def _load_tail(self) -> int:
        with atomics.atomicview(buffer = self._tail_view, atype = atomics.UINT) as a:
            return a.load(order = atomics.MemoryOrder.ACQUIRE)
    

    def _store_tail(self, tail: int) -> None:
        with atomics.atomicview(buffer=self._tail_view, atype=atomics.UINT) as a:
            a.store(tail, order=atomics.MemoryOrder.RELEASE)

    def _store_is_running(self, value: int) -> None:
        with atomics.atomicview(buffer=self._is_running_view, atype=atomics.UINT) as a:
            a.store(value, order=atomics.MemoryOrder.RELEASE)

    def _load_is_ready(self, slot_offset: int) -> bool:
        view = memoryview(self.mem)[slot_offset + self.is_ready_offset : slot_offset + self.is_ready_offset + 1]
        with atomics.atomicview(buffer=view, atype=atomics.UINT, width = 1) as a:
            return a.load(order=atomics.MemoryOrder.ACQUIRE) == 1
        
    def __iter__(self) -> Generator[torch.Tensor, None, None]:
        if self.sock is None or self.mem is None:
            raise RuntimeError("Consumer is not connected. Use 'with' or 'connect'")
        return self._iter_epoch()
    
    def _iter_epoch(self) -> Generator[torch.Tensor, None, None]:
        if self.mem is None:
            raise RuntimeError("Memory not mapped")
            
        tail = self._load_tail()
        
        while True:
            is_running = self._load_is_running()
            if is_running == 0:
                break
            head = self._load_head()
            while head <= tail:
                try:
                    readable, _, _ = select.select([self.sock], [], [], _SOCK_WAIT_POLL_TIMEOUT)
                    if readable:
                        is_running = self._load_is_running()
                        if is_running == 0:
                            return
                        chunk = self.sock.recv(1024)
                        if chunk == b"":
                            raise zt_exc.ZTConnectionError("Producer disconnected")
                        if _CONTROL_NEXT_EPOCH_MSG in chunk:
                            return
                    head = self._load_head()
                except BlockingIOError:
                    pass
                continue
                
            slot_idx = tail % self.nslots
            slot_offset = self.cb_size + (slot_idx * self.slot_size)
            
            while not self._load_is_ready(slot_offset):
                try:
                    readable, _, _ = select.select([self.sock], [], [], _SOCK_WAIT_POLL_TIMEOUT)
                    if readable:
                        is_running = self._load_is_running()
                        if is_running == 0:
                            return
                        chunk = self.sock.recv(1024)
                        if chunk == b"":
                            raise zt_exc.ZTConnectionError("Producer disconnected")
                        if _CONTROL_NEXT_EPOCH_MSG in chunk:
                            return
                except BlockingIOError:
                    pass
                
            shape, strides, dt, data_offset, data_size = TensorHeaderParser.parse_meta(
                self.mem, slot_offset, self.header_size, self.dt_offset, self.ndims_offset, self.shape_type_size
            )
            
            raw_view = memoryview(self.mem)[data_offset:data_offset + data_size]
            flat_tensor = torch.frombuffer(raw_view, dtype=dt)
            batch_tensor = torch.as_strided(flat_tensor, shape, strides)
            try:
                yield batch_tensor
            finally:    
                tail += 1
                self._store_tail(tail)
```

# zero-tensor-py/src/zero_tensor_py/exceptions.py (18 lines)
```python
class ZeroTensorError(Exception):
    def __init__(self, message):
        super().__init__(message)


class MalformedMessageError(ZeroTensorError):
    def __init__(self, message):
        super().__init__(message)


class ProtocolError(ZeroTensorError):
    def __init__(self, message):
        super().__init__(message)


class ZTConnectionError(ZeroTensorError):
    def __init__(self, message):
        super().__init__(message)
```

# zero-tensor-py/src/zero_tensor_py/protocol.py (65 lines)
```python
import struct
import torch
import zero_tensor_py.exceptions as exc

DT_F16: int = 0
DT_F32: int = 1
DT_F64: int = 2
DT_BF16: int = 3
DT_I8: int = 4
DT_I32: int = 5
DT_I64: int = 6
DT_U8: int = 7

DT_MAP: dict[int, torch.dtype] = {
    DT_U8: torch.uint8,
    DT_BF16: torch.bfloat16,
    DT_F16: torch.float16,
    DT_F32: torch.float32,
    DT_F64: torch.float64,
    DT_I32: torch.int32,
    DT_I64: torch.int64,
    DT_I8: torch.int8,
}

UNSIGNED_FORMATS = {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}


class TensorHeaderParser:
    """
    Parser TensorHeader from shared memory
    """

    @staticmethod
    def parse_meta(
        mmap_obj, 
        slot_offset: int, 
        header_size: int, 
        dt_offset: int, 
        ndims_offset: int, 
        shape_type_size: int
    ) -> tuple[list[int], list[int], torch.dtype, int, int]:
        dt = struct.unpack_from("<B", mmap_obj, slot_offset + dt_offset)[0]
        ndims = struct.unpack_from("<B", mmap_obj, slot_offset + ndims_offset)[0]
        
        torch_dt = DT_MAP.get(dt)
        if torch_dt is None:
            raise exc.MalformedMessageError(f"Unknown dtype in header: {dt}")
        
        item_size = torch_dt.itemsize
        shape_offset = slot_offset + header_size
        strides_offset = shape_offset + (shape_type_size * ndims)
        data_offset = strides_offset + (shape_type_size * ndims)
        
        fmt_char = UNSIGNED_FORMATS.get(shape_type_size, 'I')
        
        shape = list(struct.unpack_from(f"<{ndims}{fmt_char}", mmap_obj, shape_offset))
        strides = list(struct.unpack_from(f"<{ndims}{fmt_char}", mmap_obj, strides_offset))
    
        
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        data_size = num_elements * item_size
        
        return shape, strides, torch_dt, data_offset, data_size
```

# zero-tensor-py/tests/test_consumer.py (389 lines)
```python
import struct
import socket
import threading
import mmap
import os
import time
import pytest
import torch
from zero_tensor_py.protocol import DT_F32, DT_I32
from zero_tensor_py.consumer import ZeroTensorConsumer, VERSION, _PROTO_BEGIN_STR
import zero_tensor_py.exceptions as exc


CB_HEAD_OFFSET = 0
CB_HEAD_SIZE = 64 
CB_TAIL_OFFSET = 64
CB_TAIL_SIZE = 64
CB_NSLOTS_OFFSET = 128
CB_NSLOTS_SIZE = 4
CB_SLOT_SIZE_OFFSET = 132
CB_SLOT_SIZE_SIZE = 4
CB_IS_RUNNING_OFFSET = 136
CB_IS_RUNNING_SIZE = 8
CB_SIZE = 152
HEADER_SIZE = 8
HEADER_DT_OFFSET = 0
HEADER_DT_SIZE = 1
HEADER_NDIMS_OFFSET = 1
HEADER_NDIMS_SIZE = 1
HEADER_IS_READY_OFFSET = 2
HEADER_IS_READY_SIZE = 1
SHAPE_TYPE_SIZE = 4


class MockAsyncProducer:
    def __init__(self, 
                 socket_path: str, 
                 shm_path: str, 
                 nslots: int, 
                 slot_size: int, 
                 wrong_vers = False, 
                 wrong_proto = False,
                 missing_proto = False,
                 invalid_proto_val = False):
        self.socket_path = socket_path
        self.shm_path = shm_path
        self.nslots = nslots
        self.slot_size = slot_size
        self.ver = VERSION if not wrong_vers else "0"
        self.missing_proto = missing_proto
        self.invalid_proto_val = invalid_proto_val
        
        self.server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if os.path.exists(socket_path):
            os.remove(socket_path)
        self.server_sock.bind(socket_path)
        self.server_sock.listen(1)
        self.thread = None
        self.exc = None
        self.conn = None
        self.proto = _PROTO_BEGIN_STR if not wrong_proto else "PROTOXD"

    def _build_handshake(self) -> str:
        cb_key = "cb_size" if not self.missing_proto else ""
        cb_key+= (f"={CB_SIZE}" if not self.invalid_proto_val and not self.missing_proto else "")
        if self.invalid_proto_val and not self.missing_proto:
            cb_key+="=f"
        return (
            f"{self.proto} {self.ver} "
            f"{cb_key} "
            f"head_offset={CB_HEAD_OFFSET} head_size={CB_HEAD_SIZE} "
            f"tail_offset={CB_TAIL_OFFSET} tail_size={CB_TAIL_SIZE} "
            f"nslots_offset={CB_NSLOTS_OFFSET} nslots_size={CB_NSLOTS_SIZE} "
            f"slot_size_offset={CB_SLOT_SIZE_OFFSET} slot_size_size={CB_SLOT_SIZE_SIZE} "
            f"is_running_offset={CB_IS_RUNNING_OFFSET} is_running_size={CB_IS_RUNNING_SIZE} "
            f"header_size={HEADER_SIZE} "
            f"dt_offset={HEADER_DT_OFFSET} dt_size={HEADER_DT_SIZE} "
            f"ndims_offset={HEADER_NDIMS_OFFSET} ndims_size={HEADER_NDIMS_SIZE} "
            f"is_ready_offset={HEADER_IS_READY_OFFSET} is_ready_size={HEADER_IS_READY_SIZE} "
            f"shape_type_size={SHAPE_TYPE_SIZE}\n"
        )

    def _init_control_block(self, mm: mmap.mmap):
        mm[CB_NSLOTS_OFFSET:CB_NSLOTS_OFFSET + 4] = struct.pack("<I", self.nslots)
        mm[CB_SLOT_SIZE_OFFSET:CB_SLOT_SIZE_OFFSET + 4] = struct.pack("<I", self.slot_size)
        mm[CB_IS_RUNNING_OFFSET:CB_IS_RUNNING_OFFSET + 8] = struct.pack("<Q", 1)
        mm[CB_HEAD_OFFSET:CB_HEAD_OFFSET + 8] = struct.pack("<Q", 0)
        mm[CB_TAIL_OFFSET:CB_TAIL_OFFSET + 8] = struct.pack("<Q", 0)

    def start(self, batches: list[dict]):
        self.exc = None
        def loop():
            self.server_sock.settimeout(5.0)
            try:
                conn, _ = self.server_sock.accept()
                self.conn = conn
            except socket.timeout:
                return
            try:

                with conn:
                    cmd = b""
                    while b"\n" not in cmd:
                        chunk = conn.recv(16)
                        if not chunk:
                            return
                        cmd += chunk

                    if b"START" not in cmd:
                        return

                    total_size = CB_SIZE + self.nslots * self.slot_size
                    with open(self.shm_path, "r+b") as f:
                        f.truncate(total_size)
                        mm = mmap.mmap(f.fileno(), total_size)
                        self._init_control_block(mm)
                        mm.close()

                    conn.sendall(self._build_handshake().encode('utf-8'))

                    head = 0
                    for batch in batches:
                        slot_idx = head % self.nslots
                        slot_offset = CB_SIZE + slot_idx * self.slot_size

                        while True:
                            with open(self.shm_path, "r+b") as f:
                                mm = mmap.mmap(f.fileno(), total_size)
                                current_tail = struct.unpack_from("<Q", mm, CB_TAIL_OFFSET)[0]
                                mm.close()
                            if head - current_tail < self.nslots:
                                break
                            time.sleep(0.0001)

                        with open(self.shm_path, "r+b") as f:
                            mm = mmap.mmap(f.fileno(), total_size)
                            
                            ndims = len(batch["shape"])
                            mm[slot_offset + HEADER_DT_OFFSET] = batch["dt"]
                            mm[slot_offset + HEADER_NDIMS_OFFSET] = ndims
                            mm[slot_offset + HEADER_IS_READY_OFFSET] = 0
                            
                            shape_fmt = f"<{ndims}I"
                            struct.pack_into(shape_fmt, mm, slot_offset + HEADER_SIZE, *batch["shape"])
                            
                            struct.pack_into(
                                shape_fmt, 
                                mm, 
                                slot_offset + HEADER_SIZE + ndims * SHAPE_TYPE_SIZE,
                                *batch["strides"]
                            )
                            
                            data_offset = slot_offset + HEADER_SIZE + 2 * ndims * SHAPE_TYPE_SIZE
                            mm[data_offset:data_offset + len(batch["data"])] = batch["data"]
                            
                            mm[slot_offset + HEADER_IS_READY_OFFSET] = 1
                            
                            mm[CB_HEAD_OFFSET:CB_HEAD_OFFSET + 8] = struct.pack("<Q", head + 1)
                            
                            mm.close()
                        
                        head += 1
                    
                    timeout = 3.0
                    start_time = time.time()
                    while time.time() - start_time < timeout:
                        with open(self.shm_path, "r+b") as f:
                            mm = mmap.mmap(f.fileno(), total_size)
                            current_tail = struct.unpack_from("<Q", mm, CB_TAIL_OFFSET)[0]
                            mm.close()
                        if current_tail >= len(batches):
                            break
                        time.sleep(0.001)
                    
                    with open(self.shm_path, "r+b") as f:
                        mm = mmap.mmap(f.fileno(), total_size)
                        mm[CB_IS_RUNNING_OFFSET:CB_IS_RUNNING_OFFSET + 8] = struct.pack("<Q", 0)
                        mm.close()
            except Exception as e:
                self.exc = e

        self.thread = threading.Thread(target=loop)
        self.thread.start()

    def stop(self):
        if self.thread:
            self.thread.join(timeout=3.0)
            if self.thread.is_alive():
                raise RuntimeError(
                    f"MockAsyncProducer thread did not terminate within timeout "
                    f"(socket={self.socket_path})")
        self.server_sock.close()
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        if self.exc is not None:
            raise self.exc


@pytest.fixture
def temp_ipc_env(tmp_path):
    socket_path = str(tmp_path / "test_zero_tensor.sock")
    shm_name = f"zt_pytest_shm_{tmp_path.name}"
    shm_path = f"/dev/shm/{shm_name}"
    
    nslots = 2
    slot_size = 1024
    total_size = CB_SIZE + nslots * slot_size
    with open(shm_path, "wb") as f:
        f.write(b"\x00" * total_size)
        
    yield socket_path, shm_name, shm_path
    
    if os.path.exists(shm_path):
        os.remove(shm_path)


def _make_batch(shape: list[int], values: list[float], dt: int = DT_F32) -> dict:
    strides = [1] * len(shape)

    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]

    if dt == DT_F32:
        data = struct.pack(f"<{len(values)}f", *values)
    elif dt == DT_I32:
        data = struct.pack(f"<{len(values)}i", *values)
    else:
        data = struct.pack(f"<{len(values)}i", *([8] * len(values)))

    return {
        "shape": shape,
        "strides": strides,
        "dt": dt,
        "data": data,
    }


def test_async_consumer_end_to_end(temp_ipc_env):
    socket_path, shm_name, shm_path = temp_ipc_env
    nslots = 2
    slot_size = 1024
    
    batches = [
        _make_batch([2, 2], [1.0, 2.0, 3.0, 4.0]),
        _make_batch([2, 2], [5.0, 6.0, 7.0, 8.0]),
        _make_batch([2, 2], [9.0, 10.0, 11.0, 12.0]),
    ]
    
    server = MockAsyncProducer(socket_path, shm_path, nslots, slot_size)
    server.start(batches)

    try:
        collected = []
        with ZeroTensorConsumer(socket_path, shm_name) as consumer:
            for batch in consumer:
                collected.append(batch.clone())
                if len(collected) >= len(batches):
                    break
        
        assert len(collected) == 3
        
        assert collected[0].shape == (2, 2)
        assert collected[0].dtype == torch.float32
        assert torch.allclose(collected[0], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        
        assert torch.allclose(collected[1], torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        assert torch.allclose(collected[2], torch.tensor([[9.0, 10.0], [11.0, 12.0]]))
    finally:
        server.stop()


def test_async_consumer_empty_stream(temp_ipc_env):
    socket_path, shm_name, shm_path = temp_ipc_env
    nslots = 2
    slot_size = 1024
    
    server = MockAsyncProducer(socket_path, shm_path, nslots, slot_size)
    server.start([])
    
    try:
        collected = []
        with ZeroTensorConsumer(socket_path, shm_name) as consumer:
            for batch in consumer:
                collected.append(batch)
                if len(collected) > 10:
                    break
        
        assert len(collected) == 0
    finally:
        server.stop()


def test_async_consumer_3d_tensor(temp_ipc_env):
    socket_path, shm_name, shm_path = temp_ipc_env
    nslots = 2
    slot_size = 4096
    
    values = [float(i) for i in range(24)]
    batches = [_make_batch([2, 3, 4], values)]
    
    server = MockAsyncProducer(socket_path, shm_path, nslots, slot_size)
    server.start(batches)
    
    try:
        collected = []
        with ZeroTensorConsumer(socket_path, shm_name) as consumer:
            for batch in consumer:
                collected.append(batch.clone())
                break
        
        assert len(collected) == 1
        assert collected[0].shape == (2, 3, 4)
        expected = torch.tensor(values).reshape(2, 3, 4)
        assert torch.allclose(collected[0], expected)
    finally:
        server.stop()


def test_consumer_ring_buffer_wrap_around(temp_ipc_env):
    socket_path, shm_name, shm_path = temp_ipc_env
    nslots = 2
    slot_size = 1024
    
    batches = [
        _make_batch([2, 2], [float(i) for i in range(4 * j, 4 * (j + 1))])
        for j in range(5)
    ]
    
    server = MockAsyncProducer(socket_path, shm_path, nslots, slot_size)
    server.start(batches)
    
    try:
        collected = []
        with ZeroTensorConsumer(socket_path, shm_name) as consumer:
            for batch in consumer:
                collected.append(batch.clone())
                if len(collected) >= len(batches):
                    break
        
        assert len(collected) == 5
        
        for j, batch in enumerate(collected):
            expected = torch.tensor([float(i) for i in range(4 * j, 4 * (j + 1))]).reshape(2, 2)
            assert torch.allclose(batch, expected), f"Mismatch at batch {j}"
    finally:
        server.stop()


def test_consumer_detects_producer_death(temp_ipc_env):
    socket_path, shm_name, shm_path = temp_ipc_env
    server = MockAsyncProducer(socket_path, shm_path, nslots=2, slot_size=1024)
    server.start([_make_batch([2, 2], [1.0, 2.0, 3.0, 4.0])] * 5)

    with ZeroTensorConsumer(socket_path, shm_name) as consumer:
        it = iter(consumer)
        next(it)
        while server.conn is None:
            time.sleep(0.001)
        server.conn.close()
        with pytest.raises(exc.ZTConnectionError):
            for _ in it:
                pass

@pytest.mark.parametrize(
        "fail_kwarg,r_exc",
        (
            ["wrong_vers", exc.ProtocolError],
            ["wrong_proto", exc.ProtocolError],
            ["missing_proto", exc.ProtocolError],
            ["invalid_proto_val", exc.MalformedMessageError]
        )
)
def test_invalid_handshake(temp_ipc_env, fail_kwarg, r_exc):
    sock_path, shm_name, shm_path = temp_ipc_env
    server = MockAsyncProducer(sock_path, shm_path, nslots = 2, slot_size = 1024, **{fail_kwarg: True})
    server.start([])

    with pytest.raises(r_exc):
        with ZeroTensorConsumer(sock_path, shm_name) as _:
            assert False, "Should fail"

def test_wrong_dt(temp_ipc_env):
    sock_path, shm_name, shm_path = temp_ipc_env
    server = MockAsyncProducer(sock_path, shm_path, nslots = 2, slot_size = 1024)
    server.start([_make_batch([2, 2], [1.0, 2.0, 3.0, 4.0], dt = 32)] * 3)
    with ZeroTensorConsumer(sock_path, shm_name) as cons:
        it = iter(cons)
        with pytest.raises(exc.MalformedMessageError):
            next(it)
```

# zero-tensor-rs/Cargo.toml (21 lines)
```toml
[package]
name = "zero-tensor-lib"
version = "0.5.0"
edition = "2024"

[dependencies]
bytemuck = { version = "1.25.2", features = ["derive", "extern_crate_alloc"] }
crossbeam-utils = "0.8.22"
ctrlc = "3.5.2"
fastrand = "2.5.0"
half = { version = "2.7.1", features = ["bytemuck"] }
libc = "0.2.186"
ndarray = "0.17.2"
rand = "0.10.2"
rayon = "1.12.0"
smallvec = "1.15.2"
thiserror = "2.0.18"

[dev-dependencies]
tempfile = "3.10"
rstest = "0.26.1"
```

# zero-tensor-rs/src/bin/throughput_bench.rs (100 lines)
```rust
use std::path::Path;
use zero_tensor_lib::core::{
    dataset::{
        ZeroTensorDataset,
        item::{ShapeType, TensorBatchLayout, TensorDT},
    },
    producer::ZeroTensorProducerBuilder,
};

const BATCH_SIZE: usize = 12;
const CHANNELS: ShapeType = 3;
const HEIGHT: ShapeType = 1024;
const WIDTH: ShapeType = 1024;
const STEPS: u64 = 200;
const NSLOTS: u64 = 10;

struct BenchDataset {
    raw_item_size: usize,
    meta: TensorBatchLayout,
    source_buffer: Vec<u8>,
}

impl BenchDataset {
    fn new(raw_item_size: usize) -> Self {
        let shape = vec![CHANNELS, HEIGHT, WIDTH];
        let strides = vec![HEIGHT * WIDTH, WIDTH, 1];
        let meta = TensorBatchLayout::new(shape.into(), strides.into(), TensorDT::F32);
        let mut source = vec![0u8; raw_item_size];
        fastrand::Rng::new().fill(&mut source);

        Self {
            raw_item_size,
            meta,
            source_buffer: source,
        }
    }
}

impl ZeroTensorDataset for BenchDataset {
    type Error = std::io::Error;

    fn len(&self) -> usize {
        BATCH_SIZE * STEPS as usize
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get_batch_layout(&self, _idxs: &[usize]) -> Result<TensorBatchLayout, Self::Error> {
        Ok(self.meta.clone())
    }

    fn write_item_into(&self, idx: usize, buf: &mut [u8]) -> Result<(), Self::Error> {
        let target = &mut buf[..self.raw_item_size];
        target.copy_from_slice(&self.source_buffer[..self.raw_item_size]);

        for byte in target.iter_mut() {
            *byte = byte.wrapping_add(idx as u8);
        }

        Ok(())
    }
}

fn main() {
    let socket_path = Path::new("/tmp/zt_bench.sock");
    let shm_name = "zt_bench";

    if socket_path.exists() {
        let _ = std::fs::remove_file(socket_path);
    }

    let item_elements = CHANNELS * HEIGHT * WIDTH;
    let raw_item_size = item_elements as u64 * 4;

    let slot_size = (raw_item_size * BATCH_SIZE as u64) + 4096;

    println!("[Rust Bench] Initializing ZeroTensorProducer...");
    println!(" -> SHM Name: {}", shm_name);
    println!(
        " -> Slot Size: {:.2} MB",
        slot_size as f64 / 1024.0 / 1024.0
    );

    let mut producer = ZeroTensorProducerBuilder::new(slot_size, shm_name, socket_path)
        .num_slots(NSLOTS)
        .build()
        .expect("Failed to create producer");

    let dataset = BenchDataset::new(raw_item_size as usize);

    println!("[Rust Bench] Ready! Waiting for Python consumer to connect...");

    producer
        .start_streaming(&dataset, BATCH_SIZE)
        .expect("Streaming failed");

    println!("[Rust Bench] Finished streaming");
}
```

# zero-tensor-rs/src/lib.rs (42 lines)
```rust
pub mod core;
pub mod pipeline;
pub mod transform;

#[cfg(test)]
mod tests {
    #[test]
    fn test_shuffle_determinism_with_seed() {
        let seed = Some(1337);
        let len = 1000;

        let mut indices1: Vec<usize> = (0..len).collect();
        let mut rng1 = fastrand::Rng::with_seed(seed.unwrap());
        rng1.shuffle(&mut indices1);

        let mut indices2: Vec<usize> = (0..len).collect();
        let mut rng2 = fastrand::Rng::with_seed(seed.unwrap());
        rng2.shuffle(&mut indices2);

        assert_eq!(
            indices1, indices2,
            "Shuffled indices must be identical with the same seed"
        );
    }

    #[test]
    fn test_shuffle_differs_across_epochs() {
        let base_seed = 42u64;
        let len = 100;

        let mut epoch0: Vec<usize> = (0..len).collect();
        fastrand::Rng::with_seed(base_seed).shuffle(&mut epoch0);

        let mut epoch1: Vec<usize> = (0..len).collect();
        fastrand::Rng::with_seed(base_seed + 1).shuffle(&mut epoch1);

        assert_ne!(
            epoch0, epoch1,
            "Epochs must have different shuffle patterns"
        );
    }
}
```

# zero-tensor-rs/src/pipeline/error.rs (30 lines)
```rust
use crate::transform::TransformError;
use std::{error::Error, fmt};

#[derive(Debug)]
pub struct PipelineError {
    pub step: usize,
    pub error: TransformError,
}

impl PipelineError {
    pub fn new(step: usize, error: TransformError) -> Self {
        Self { step, error }
    }
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Pipeline error during step {}: {}",
            self.step, self.error
        )
    }
}

impl Error for PipelineError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.error)
    }
}
```

# zero-tensor-rs/src/pipeline/mod.rs (39 lines)
```rust
pub mod error;
use std::sync::Arc;

pub use error::PipelineError;

use crate::{core::dataset::item::TensorViewMut, transform::Transform};

#[derive(Clone)]
pub struct Pipeline {
    steps: Vec<Arc<dyn Transform>>,
}

impl Pipeline {
    pub fn new() -> Self {
        let steps = Vec::new();
        Self { steps }
    }

    pub fn then<T: Transform + 'static>(mut self, step: T) -> Self {
        self.steps.push(Arc::new(step));
        self
    }

    pub fn exec(&self, tensor: &mut TensorViewMut) -> Result<(), PipelineError> {
        self.steps
            .iter()
            .enumerate()
            .try_for_each(|(i, step)| step.apply(tensor).map_err(|e| PipelineError::new(i + 1, e)))
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
```

# zero-tensor-rs/src/pipeline/tests.rs (148 lines)
```rust
use super::*;
use crate::{
    core::dataset::item::{TensorBatchLayout, TensorDT},
    transform::{Scale, TransformError},
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct MockTransform {
    calls: Arc<AtomicUsize>,
}

impl Transform for MockTransform {
    fn apply(&self, _: &mut TensorViewMut) -> Result<(), TransformError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct FailingTransform;

impl Transform for FailingTransform {
    fn apply(&self, _: &mut TensorViewMut) -> Result<(), TransformError> {
        Err(TransformError::InvalidValue)
    }
}

fn make_tensor() -> (Vec<u8>, TensorBatchLayout) {
    let data = vec![1.0f32, 2.0, 3.0];
    let raw_bytes = bytemuck::pod_collect_to_vec(&data);

    let layout = TensorBatchLayout::new(vec![data.len()].into(), vec![1].into(), TensorDT::F32);

    (raw_bytes, layout)
}

#[test]
fn empty_pipeline_succeeds() {
    let (mut raw_bytes, layout) = make_tensor();
    let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

    let pipeline = Pipeline::new();

    assert!(pipeline.exec(&mut tensor).is_ok());
}

#[test]
fn executes_all_steps() {
    let calls = Arc::new(AtomicUsize::new(0));

    let pipeline = Pipeline::new()
        .then(MockTransform {
            calls: Arc::clone(&calls),
        })
        .then(MockTransform {
            calls: Arc::clone(&calls),
        })
        .then(MockTransform {
            calls: Arc::clone(&calls),
        });

    let (mut raw_bytes, layout) = make_tensor();
    let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

    pipeline.exec(&mut tensor).unwrap();

    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[test]
fn stops_after_error() {
    let calls = Arc::new(AtomicUsize::new(0));

    let pipeline = Pipeline::new()
        .then(MockTransform {
            calls: Arc::clone(&calls),
        })
        .then(FailingTransform)
        .then(MockTransform {
            calls: Arc::clone(&calls),
        });

    let (mut raw_bytes, layout) = make_tensor();
    let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

    let result = pipeline.exec(&mut tensor);

    assert!(result.is_err());

    // Выполнился только первый transform.
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn returns_correct_step_number() {
    let calls = Arc::new(AtomicUsize::new(0));

    let pipeline = Pipeline::new()
        .then(MockTransform {
            calls: Arc::clone(&calls),
        })
        .then(MockTransform {
            calls: Arc::clone(&calls),
        })
        .then(FailingTransform);

    let (mut raw_bytes, layout) = make_tensor();
    let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

    let error = pipeline.exec(&mut tensor).unwrap_err();

    assert_eq!(error.step, 3);
    assert!(matches!(error.error, TransformError::InvalidValue));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn first_step_error_has_step_one() {
    let pipeline = Pipeline::new().then(FailingTransform);

    let (mut raw_bytes, layout) = make_tensor();
    let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

    let error = pipeline.exec(&mut tensor).unwrap_err();

    assert_eq!(error.step, 1);
    assert!(matches!(error.error, TransformError::InvalidValue));
}

#[test]
fn applies_real_transforms_in_order() {
    let data = vec![1.0f32, 2.0, 3.0];
    let mut raw_bytes = bytemuck::pod_collect_to_vec(&data);

    let layout = TensorBatchLayout::new(vec![3].into(), vec![1].into(), TensorDT::F32);

    let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

    let pipeline = Pipeline::new().then(Scale::new(2.0)).then(Scale::new(3.0));

    pipeline.exec(&mut tensor).unwrap();

    let result: Vec<f32> = bytemuck::pod_collect_to_vec(&raw_bytes);

    assert_eq!(result, vec![6.0, 12.0, 18.0]);
}
```

# zero-tensor-rs/src/transform/add.rs (529 lines)
```rust
use super::{Scalar, Transform, TransformError};
use crate::{core::dataset::item::TensorViewMut, transform::ScalarConversionError};

pub enum OverflowMode {
    Error,
    Wrapping,
}

pub struct Add {
    value: Scalar,
    overflow: OverflowMode,
}

impl Add {
    pub fn new<T: Into<Scalar>>(value: T) -> Self {
        Self {
            value: value.into(),
            overflow: OverflowMode::Error,
        }
    }

    pub fn arith_overflow(self, overflow: OverflowMode) -> Self {
        Self { overflow, ..self }
    }
}

impl Transform for Add {
    fn apply(&self, tensor: &mut TensorViewMut) -> Result<(), TransformError> {
        if let Ok(u) = <Scalar as TryInto<u8>>::try_into(self.value)
            && u == 0
        {
            return Ok(());
        }

        macro_rules! add_int {
            ($ty:ty, $t:expr) => {{
                let h: $ty = self.value.try_into()?;

                if matches!(self.overflow, OverflowMode::Error) {
                    for x in $t.iter() {
                        x.checked_add(h).ok_or(TransformError::Overflow)?;
                    }
                    $t.map_inplace(|x| *x = unsafe { x.unchecked_add(h) });
                } else {
                    $t.map_inplace(|x| *x = x.wrapping_add(h));
                }
            }};
        }

        macro_rules! add {
            ($ty:ty, $t:expr) => {{
                let h: $ty = self.value.try_into()?;
                $t.map_inplace(|x| *x += h);
            }};
        }

        match tensor {
            TensorViewMut::BF16(t) => add!(half::bf16, t),
            TensorViewMut::F16(t) => add!(half::f16, t),
            TensorViewMut::U8(t) => {
                let val: i32 = self.value.try_into()?;
                if val < -(u8::MAX as i32) || val > (u8::MAX as i32) {
                    return Err(ScalarConversionError::Overflow.into());
                }
                if val < 0 {
                    let v = (-val) as u8;
                    match self.overflow {
                        OverflowMode::Error => {
                            for x in t.iter() {
                                x.checked_sub(v).ok_or(TransformError::Overflow)?;
                            }
                            t.map_inplace(|x| *x = unsafe { x.unchecked_sub(v) });
                        }
                        OverflowMode::Wrapping => {
                            t.map_inplace(|x| *x = x.wrapping_sub(v));
                        }
                    }
                } else {
                    let v = val as u8;
                    match self.overflow {
                        OverflowMode::Error => {
                            for x in t.iter() {
                                x.checked_add(v).ok_or(TransformError::Overflow)?;
                            }
                            t.map_inplace(|x| *x = unsafe { x.unchecked_add(v) });
                        }
                        OverflowMode::Wrapping => {
                            t.map_inplace(|x| *x = x.wrapping_add(v));
                        }
                    }
                }
            }
            TensorViewMut::I8(t) => add_int!(i8, t),
            TensorViewMut::I32(t) => add_int!(i32, t),
            TensorViewMut::I64(t) => add_int!(i64, t),
            TensorViewMut::F32(t) => add!(f32, t),
            TensorViewMut::F64(t) => add!(f64, t),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;
    use crate::core::dataset::item::{TensorBatchLayout, TensorDT, TensorViewMut};
    use rstest::rstest;

    fn make_tensor_f32(data: &mut [f32]) -> TensorViewMut<'_> {
        let l = data.len();
        let raw_bytes = bytemuck::cast_slice_mut(data);

        let layout = TensorBatchLayout::new(vec![l].into(), vec![1].into(), TensorDT::F32);

        layout.try_view_mut(raw_bytes).unwrap()
    }

    fn make_tensor_i8(data: &mut [i8]) -> TensorViewMut<'_> {
        let l = data.len();
        let raw_bytes = bytemuck::cast_slice_mut(data);

        let layout = TensorBatchLayout::new(vec![l].into(), vec![1].into(), TensorDT::I8);

        layout.try_view_mut(raw_bytes).unwrap()
    }

    fn make_tensor_i32(data: &mut [i32]) -> TensorViewMut<'_> {
        let l = data.len();
        let raw_bytes = bytemuck::cast_slice_mut(data);

        let layout = TensorBatchLayout::new(vec![l].into(), vec![1].into(), TensorDT::I32);

        layout.try_view_mut(raw_bytes).unwrap()
    }

    fn make_tensor_u8(data: &mut [u8]) -> TensorViewMut<'_> {
        let l = data.len();
        let raw_bytes = bytemuck::cast_slice_mut(data);

        let layout = TensorBatchLayout::new(vec![l].into(), vec![1].into(), TensorDT::U8);

        layout.try_view_mut(raw_bytes).unwrap()
    }

    // ------------------------------------------------------------
    // Basic
    // ------------------------------------------------------------

    #[test]
    fn add_f32() {
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0];

        {
            let mut tensor = make_tensor_f32(&mut data);

            Add::new(2.0).apply(&mut tensor).unwrap();
        }

        assert_eq!(data, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn add_f32_negative() {
        let mut data = vec![10.0f32, 20.0, 30.0];

        {
            let mut tensor = make_tensor_f32(&mut data);

            Add::new(-5.0).apply(&mut tensor).unwrap();
        }

        assert_eq!(data, vec![5.0, 15.0, 25.0]);
    }

    #[test]
    fn add_zero_does_not_modify_tensor() {
        let mut data = vec![1.0f32, 2.0, 3.0];
        let original = data.clone();

        {
            let mut tensor = make_tensor_f32(&mut data);

            Add::new(0.0).apply(&mut tensor).unwrap();
        }

        assert_eq!(data, original);
    }

    // ------------------------------------------------------------
    // Floating point dtypes
    // ------------------------------------------------------------

    #[test]
    fn add_f64() {
        let mut data = vec![1.0f64, 2.0, 3.0];

        let raw_bytes = bytemuck::cast_slice_mut(&mut data);

        let layout = TensorBatchLayout::new(vec![3].into(), vec![1].into(), TensorDT::F64);

        let mut tensor = layout.try_view_mut(raw_bytes).unwrap();

        Add::new(0.5).apply(&mut tensor).unwrap();

        assert_eq!(data, vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn add_f16() {
        let mut data = vec![
            half::f16::from_f32(1.0),
            half::f16::from_f32(2.0),
            half::f16::from_f32(3.0),
        ];

        let raw_bytes = bytemuck::cast_slice_mut(&mut data);

        let layout = TensorBatchLayout::new(vec![3].into(), vec![1].into(), TensorDT::F16);

        let mut tensor = layout.try_view_mut(raw_bytes).unwrap();

        Add::new(0.5).apply(&mut tensor).unwrap();

        let actual: Vec<f32> = data.iter().map(|x| x.to_f32()).collect();

        assert_eq!(actual, vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn add_bf16() {
        let mut data = vec![
            half::bf16::from_f32(1.0),
            half::bf16::from_f32(2.0),
            half::bf16::from_f32(3.0),
        ];

        let raw_bytes = bytemuck::cast_slice_mut(&mut data);

        let layout = TensorBatchLayout::new(vec![3].into(), vec![1].into(), TensorDT::BF16);

        let mut tensor = layout.try_view_mut(raw_bytes).unwrap();

        Add::new(0.5).apply(&mut tensor).unwrap();

        let actual: Vec<f32> = data.iter().map(|x| x.to_f32()).collect();

        assert_eq!(actual, vec![1.5, 2.5, 3.5]);
    }

    // ------------------------------------------------------------
    // Integer dtypes
    // ------------------------------------------------------------

    #[test]
    fn add_i8() {
        let mut data = vec![-10i8, 0, 10];

        {
            let mut tensor = make_tensor_i8(&mut data);

            Add::new(5.0).apply(&mut tensor).unwrap();
        }

        assert_eq!(data, vec![-5, 5, 15]);
    }

    #[test]
    fn add_i32() {
        let mut data = vec![-100i32, 0, 100];

        {
            let mut tensor = make_tensor_i32(&mut data);

            Add::new(50.0).apply(&mut tensor).unwrap();
        }

        assert_eq!(data, vec![-50, 50, 150]);
    }

    #[test]
    fn add_u8() {
        let mut data = vec![0u8, 10, 100];

        {
            let mut tensor = make_tensor_u8(&mut data);

            Add::new(20.0).apply(&mut tensor).unwrap();
        }

        assert_eq!(data, vec![20, 30, 120]);
    }

    // ------------------------------------------------------------
    // Invalid values
    // ------------------------------------------------------------

    #[rstest]
    #[case(f64::NAN)]
    #[case(f64::INFINITY)]
    #[case(f64::NEG_INFINITY)]
    fn integer_add_rejects_non_finite(#[case] value: f64) {
        let mut data = vec![1i32, 2, 3];

        {
            let mut tensor = make_tensor_i32(&mut data);

            let result = Add::new(value).apply(&mut tensor);

            assert!(matches!(
                result,
                Err(TransformError::ScalarConversion(
                    ScalarConversionError::InvalidValue
                ))
            ));
        }

        assert_eq!(data, vec![1, 2, 3]);
    }

    #[test]
    fn integer_add_rejects_fractional_value() {
        let mut data = vec![1i32, 2, 3];
        let original = data.clone();

        {
            let mut tensor = make_tensor_i32(&mut data);

            let result = Add::new(1.5).apply(&mut tensor);

            assert!(matches!(
                result,
                Err(TransformError::ScalarConversion(
                    ScalarConversionError::FractionalValue
                ))
            ));
        }

        assert_eq!(data, original);
    }

    // ------------------------------------------------------------
    // Value outside dtype range
    // ------------------------------------------------------------

    #[test]
    fn add_value_outside_dtype_range() {
        let mut data = vec![1u8, 2, 3];
        let original = data.clone();

        {
            let mut tensor = make_tensor_u8(&mut data);

            let result = Add::new(256).apply(&mut tensor);

            assert!(matches!(
                result,
                Err(TransformError::ScalarConversion(
                    ScalarConversionError::Overflow
                ))
            ));
        }

        assert_eq!(data, original);
    }

    #[test]
    fn add_negative_value_u8() {
        let mut data = vec![1u8, 2, 3];
        let exp: Vec<u8> = data.iter().map(|&x| x - 1).collect();
        {
            let mut tensor = make_tensor_u8(&mut data);

            let result = Add::new(-1).apply(&mut tensor);
            assert!(result.is_ok());
        }

        assert_eq!(data, exp);
    }

    // ------------------------------------------------------------
    // Arithmetic overflow
    // ------------------------------------------------------------

    #[test]
    fn add_positive_arithmetic_overflow() {
        let mut data = vec![100i8, 120, 127];
        let original = data.clone();

        {
            let mut tensor = make_tensor_i8(&mut data);

            let result = Add::new(10.0).apply(&mut tensor);

            assert!(matches!(result, Err(TransformError::Overflow)));
        }

        // Important: no partial modification.
        assert_eq!(data, original);
    }

    #[test]
    fn add_negative_arithmetic_overflow() {
        let mut data = vec![-100i8, -120, -128];
        let original = data.clone();

        {
            let mut tensor = make_tensor_i8(&mut data);

            let result = Add::new(-10.0).apply(&mut tensor);

            assert!(matches!(result, Err(TransformError::Overflow)));
        }

        assert_eq!(data, original);
    }

    // ------------------------------------------------------------
    // Wrapping
    // ------------------------------------------------------------

    #[test]
    fn add_wrapping_positive_overflow() {
        let mut data = vec![127i8, 126, 100];

        {
            let mut tensor = make_tensor_i8(&mut data);

            Add::new(1.0)
                .arith_overflow(OverflowMode::Wrapping)
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, vec![-128, 127, 101]);
    }

    #[test]
    fn add_wrapping_negative_overflow() {
        let mut data = vec![-128i8, -127, -100];

        {
            let mut tensor = make_tensor_i8(&mut data);

            Add::new(-1.0)
                .arith_overflow(OverflowMode::Wrapping)
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, vec![127, -128, -101]);
    }

    // ------------------------------------------------------------
    // All-or-nothing behavior
    // ------------------------------------------------------------

    #[test]
    fn overflow_does_not_partially_modify_tensor() {
        let mut data = vec![1i8, 2, 127, 4, 5];
        let original = data.clone();

        {
            let mut tensor = make_tensor_i8(&mut data);

            let result = Add::new(1.0).apply(&mut tensor);

            assert!(matches!(result, Err(TransformError::Overflow)));
        }

        assert_eq!(data, original);
    }

    // ------------------------------------------------------------
    // Zero-copy
    // ------------------------------------------------------------

    #[test]
    fn add_modifies_original_buffer() {
        let mut data = vec![10i32, 20, 30];

        {
            let mut tensor = make_tensor_i32(&mut data);

            Add::new(5.0).apply(&mut tensor).unwrap();
        }

        assert_eq!(data, vec![15, 25, 35]);
    }

    #[test]
    fn add_exact_positive_boundary() {
        let mut data = vec![120i8];

        {
            let mut tensor = make_tensor_i8(&mut data);

            Add::new(7.0).apply(&mut tensor).unwrap();
        }

        assert_eq!(data[0], 127);
    }

    #[test]
    fn add_exact_negative_boundary() {
        let mut data = vec![-120i8];

        {
            let mut tensor = make_tensor_i8(&mut data);

            Add::new(-7.0).apply(&mut tensor).unwrap();
        }

        assert_eq!(data, vec![-127]);
    }

    #[test]
    fn test_atomic_overflow() {
        let mut data = vec![1i8, 2, 127, 4];
        let original = data.clone();

        let mut tensor = make_tensor_i8(&mut data);

        let result = Add::new(1i8).apply(&mut tensor);

        assert!(matches!(result, Err(TransformError::Overflow)));
        assert_eq!(data, original);
    }
}
```

# zero-tensor-rs/src/transform/clamp.rs (752 lines)
```rust
use crate::transform::{IntoScalarOption, Scalar, ScalarConversionError};

use super::{TensorViewMut, Transform, TransformError};

pub enum IntRoundingMode {
    Error,
    Round,
    Floor,
    Ceil,
}

pub enum OverflowMode {
    Error,
    Clamp,
}

pub struct Clamp {
    min: Option<Scalar>,
    max: Option<Scalar>,
    int_rounding_mode: IntRoundingMode,
    overflow_mode: OverflowMode,
}

impl Clamp {
    pub fn new<T: IntoScalarOption>(min: T, max: T) -> Result<Self, TransformError> {
        let min = min.into_scalar_option();
        let max = max.into_scalar_option();

        if min.is_some_and(|x| x.is_nan()) || max.is_some_and(|x| x.is_nan()) {
            return Err(TransformError::InvalidValue);
        }
        if let (Some(mx), Some(mn)) = (max, min)
            && mx < mn
        {
            return Err(TransformError::InvalidValue);
        }
        Ok(Self {
            min,
            max,
            int_rounding_mode: IntRoundingMode::Error,
            overflow_mode: OverflowMode::Error,
        })
    }

    pub fn int_rounding_mode(self, int_rounding_mode: IntRoundingMode) -> Self {
        Self {
            int_rounding_mode,
            ..self
        }
    }

    pub fn overflow_mode(self, overflow_mode: OverflowMode) -> Self {
        Self {
            overflow_mode,
            ..self
        }
    }

    fn scalar_to_f64(value: Scalar) -> Result<f64, TransformError> {
        value.try_into().map_err(Into::into)
    }

    fn rounded_scalar(&self, value: Scalar) -> Result<Scalar, TransformError> {
        let value = Self::scalar_to_f64(value)?;

        let value = match self.int_rounding_mode {
            IntRoundingMode::Error => {
                return Err(ScalarConversionError::FractionalValue.into());
            }
            IntRoundingMode::Round => value.round(),
            IntRoundingMode::Floor => value.floor(),
            IntRoundingMode::Ceil => value.ceil(),
        };

        Ok(Scalar::F64(value))
    }

    fn resolve_int<T>(&self, value: Scalar, min: T, max: T) -> Result<T, TransformError>
    where
        T: TryFrom<Scalar, Error = ScalarConversionError> + Copy + Into<Scalar>,
    {
        match value.try_into() {
            Ok(value) => Ok(value),

            Err(ScalarConversionError::FractionalValue) => {
                let rounded = self.rounded_scalar(value)?;

                match rounded.try_into() {
                    Ok(value) => Ok(value),

                    Err(ScalarConversionError::Overflow) => {
                        let rounded_f64 = Self::scalar_to_f64(rounded)?;

                        match self.overflow_mode {
                            OverflowMode::Error => Err(ScalarConversionError::Overflow.into()),
                            OverflowMode::Clamp => {
                                let min_scalar: Scalar = min.into();

                                if Scalar::F64(rounded_f64) < min_scalar {
                                    Ok(min)
                                } else {
                                    Ok(max)
                                }
                            }
                        }
                    }

                    Err(e) => Err(e.into()),
                }
            }

            Err(ScalarConversionError::Overflow) => match self.overflow_mode {
                OverflowMode::Error => Err(ScalarConversionError::Overflow.into()),

                OverflowMode::Clamp => {
                    let source = Self::scalar_to_f64(value)?;

                    let min_scalar: Scalar = min.into();
                    let max_scalar: Scalar = max.into();

                    let min_value = Self::scalar_to_f64(min_scalar)?;
                    let max_value = Self::scalar_to_f64(max_scalar)?;

                    if source < min_value {
                        Ok(min)
                    } else if source > max_value {
                        Ok(max)
                    } else {
                        unreachable!("conversion overflow without crossing target bounds");
                    }
                }
            },

            Err(e) => Err(e.into()),
        }
    }

    fn resolve_f32(&self, value: Scalar) -> Result<f32, TransformError> {
        match value.try_into() {
            Ok(value) => Ok(value),

            Err(ScalarConversionError::Overflow) => match self.overflow_mode {
                OverflowMode::Error => Err(ScalarConversionError::Overflow.into()),
                OverflowMode::Clamp => {
                    let value = Self::scalar_to_f64(value)?;

                    if value < f32::MIN as f64 {
                        Ok(f32::MIN)
                    } else {
                        Ok(f32::MAX)
                    }
                }
            },

            Err(e) => Err(e.into()),
        }
    }

    fn resolve_f64(&self, value: Scalar) -> Result<f64, TransformError> {
        Ok(value.try_into()?)
    }

    fn resolve_f16(&self, value: Scalar) -> Result<half::f16, TransformError> {
        match value.try_into() {
            Ok(value) => Ok(value),

            Err(ScalarConversionError::Overflow) => match self.overflow_mode {
                OverflowMode::Error => Err(ScalarConversionError::Overflow.into()),
                OverflowMode::Clamp => {
                    let value = Self::scalar_to_f64(value)?;

                    if value < half::f16::MIN.to_f64() {
                        Ok(half::f16::MIN)
                    } else {
                        Ok(half::f16::MAX)
                    }
                }
            },

            Err(e) => Err(e.into()),
        }
    }

    fn resolve_bf16(&self, value: Scalar) -> Result<half::bf16, TransformError> {
        match value.try_into() {
            Ok(value) => Ok(value),

            Err(ScalarConversionError::Overflow) => match self.overflow_mode {
                OverflowMode::Error => Err(ScalarConversionError::Overflow.into()),
                OverflowMode::Clamp => {
                    let value = Self::scalar_to_f64(value)?;

                    if value < half::bf16::MIN.to_f64() {
                        Ok(half::bf16::MIN)
                    } else {
                        Ok(half::bf16::MAX)
                    }
                }
            },

            Err(e) => Err(e.into()),
        }
    }
}

impl Transform for Clamp {
    fn apply(&self, tensor: &mut TensorViewMut) -> Result<(), TransformError> {
        macro_rules! match_max_min {
            ($max:expr, $min:expr, $t:expr) => {
                match ($min, $max) {
                    (Some(min), Some(max)) => {
                        $t.map_inplace(|x| {
                            *x = (*x).clamp(min, max);
                        });
                    }

                    (Some(min), None) => {
                        $t.map_inplace(|x| {
                            *x = (*x).max(min);
                        });
                    }

                    (None, Some(max)) => {
                        $t.map_inplace(|x| {
                            *x = (*x).min(max);
                        });
                    }

                    (None, None) => {}
                }
            };
        }

        match tensor {
            TensorViewMut::U8(t) => {
                let min = self
                    .min
                    .map(|x| self.resolve_int(x, u8::MIN, u8::MAX))
                    .transpose()?;

                let max = self
                    .max
                    .map(|x| self.resolve_int(x, u8::MIN, u8::MAX))
                    .transpose()?;

                match_max_min!(max, min, t);
            }

            TensorViewMut::I8(t) => {
                let min = self
                    .min
                    .map(|x| self.resolve_int(x, i8::MIN, i8::MAX))
                    .transpose()?;

                let max = self
                    .max
                    .map(|x| self.resolve_int(x, i8::MIN, i8::MAX))
                    .transpose()?;

                match_max_min!(max, min, t);
            }

            TensorViewMut::I32(t) => {
                let min = self
                    .min
                    .map(|x| self.resolve_int(x, i32::MIN, i32::MAX))
                    .transpose()?;

                let max = self
                    .max
                    .map(|x| self.resolve_int(x, i32::MIN, i32::MAX))
                    .transpose()?;

                match_max_min!(max, min, t);
            }

            TensorViewMut::I64(t) => {
                let min = self
                    .min
                    .map(|x| self.resolve_int(x, i64::MIN, i64::MAX))
                    .transpose()?;

                let max = self
                    .max
                    .map(|x| self.resolve_int(x, i64::MIN, i64::MAX))
                    .transpose()?;

                match_max_min!(max, min, t);
            }

            TensorViewMut::F32(t) => {
                let min = self.min.map(|x| self.resolve_f32(x)).transpose()?;

                let max = self.max.map(|x| self.resolve_f32(x)).transpose()?;

                match_max_min!(max, min, t);
            }

            TensorViewMut::F64(t) => {
                let min = self.min.map(|x| self.resolve_f64(x)).transpose()?;

                let max = self.max.map(|x| self.resolve_f64(x)).transpose()?;

                match_max_min!(max, min, t);
            }

            TensorViewMut::F16(t) => {
                let min = self.min.map(|x| self.resolve_f16(x)).transpose()?;

                let max = self.max.map(|x| self.resolve_f16(x)).transpose()?;

                match_max_min!(max, min, t);
            }

            TensorViewMut::BF16(t) => {
                let min = self.min.map(|x| self.resolve_bf16(x)).transpose()?;

                let max = self.max.map(|x| self.resolve_bf16(x)).transpose()?;

                match_max_min!(max, min, t);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dataset::item::{TensorBatchLayout, TensorDT};
    use rstest::rstest;

    fn make_tensor_f32(data: &mut [f32]) -> TensorViewMut<'_> {
        let len = data.len();
        let raw_bytes = bytemuck::cast_slice_mut(data);

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), TensorDT::F32);

        layout.try_view_mut(raw_bytes).unwrap()
    }

    fn make_tensor_f64(data: &mut [f64]) -> TensorViewMut<'_> {
        let len = data.len();
        let raw_bytes = bytemuck::cast_slice_mut(data);

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), TensorDT::F64);

        layout.try_view_mut(raw_bytes).unwrap()
    }

    fn make_tensor_i8(data: &mut [i8]) -> TensorViewMut<'_> {
        let len = data.len();
        let raw_bytes = bytemuck::cast_slice_mut(data);

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), TensorDT::I8);

        layout.try_view_mut(raw_bytes).unwrap()
    }

    fn make_tensor_u8(data: &mut [u8]) -> TensorViewMut<'_> {
        let len = data.len();
        let raw_bytes = bytemuck::cast_slice_mut(data);

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), TensorDT::U8);

        layout.try_view_mut(raw_bytes).unwrap()
    }

    // ---------------------------------------------------------
    // Constructor
    // ---------------------------------------------------------

    #[test]
    fn constructor_accepts_valid_bounds() {
        assert!(Clamp::new(0.0, 10.0).is_ok());
        assert!(Clamp::new(Some(10.0), Some(10.0)).is_ok());
        assert!(Clamp::new(None::<f64>, Some(10.0)).is_ok());
        assert!(Clamp::new(Some(0.0), None::<f64>).is_ok());
        assert!(Clamp::new(None::<f64>, None::<f64>).is_ok());
    }

    #[test]
    fn constructor_rejects_min_greater_than_max() {
        let result = Clamp::new(Some(10.0), Some(0.0));

        assert!(matches!(result, Err(TransformError::InvalidValue)));
    }

    #[test]
    fn constructor_rejects_nan_min() {
        let result = Clamp::new(Some(f64::NAN), Some(10.0));

        assert!(matches!(result, Err(TransformError::InvalidValue)));
    }

    #[test]
    fn constructor_rejects_nan_max() {
        let result = Clamp::new(Some(0.0), Some(f64::NAN));

        assert!(matches!(result, Err(TransformError::InvalidValue)));
    }

    // ---------------------------------------------------------
    // F32
    // ---------------------------------------------------------

    #[test]
    fn clamp_f32_both_bounds() {
        let mut data = vec![-10.0, 0.0, 5.0, 10.0, 20.0];

        {
            let mut tensor = make_tensor_f32(&mut data);

            Clamp::new(Some(0.0), Some(10.0))
                .unwrap()
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, vec![0.0, 0.0, 5.0, 10.0, 10.0]);
    }

    #[test]
    fn clamp_f32_min_only() {
        let mut data = vec![-10.0, 0.0, 5.0];

        {
            let mut tensor = make_tensor_f32(&mut data);

            Clamp::new(Some(0.0), None::<f64>)
                .unwrap()
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, vec![0.0, 0.0, 5.0]);
    }

    #[test]
    fn clamp_f32_max_only() {
        let mut data = vec![0.0, 5.0, 10.0, 20.0];

        {
            let mut tensor = make_tensor_f32(&mut data);

            Clamp::new(None::<f64>, Some(10.0))
                .unwrap()
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, vec![0.0, 5.0, 10.0, 10.0]);
    }

    #[test]
    fn clamp_f32_values_inside_bounds_are_unchanged() {
        let mut data = vec![1.0, 2.5, 5.0, 9.99];

        let original = data.clone();

        {
            let mut tensor = make_tensor_f32(&mut data);

            Clamp::new(Some(0.0), Some(10.0))
                .unwrap()
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, original);
    }

    #[test]
    fn clamp_f32_equal_bounds() {
        let mut data = vec![-10.0, 0.0, 5.0, 10.0];

        {
            let mut tensor = make_tensor_f32(&mut data);

            Clamp::new(Some(5.0), Some(5.0))
                .unwrap()
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, vec![5.0, 5.0, 5.0, 5.0]);
    }

    // ---------------------------------------------------------
    // F64
    // ---------------------------------------------------------

    #[test]
    fn clamp_f64() {
        let mut data = vec![-100.0, -1.5, 0.0, 1.5, 100.0];

        {
            let mut tensor = make_tensor_f64(&mut data);

            Clamp::new(Some(-1.0), Some(1.0))
                .unwrap()
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, vec![-1.0, -1.0, 0.0, 1.0, 1.0]);
    }

    // ---------------------------------------------------------
    // Integer: Error rounding mode
    // ---------------------------------------------------------

    #[rstest]
    #[case(0.5, 10.0)]
    #[case(0.0, 10.5)]
    #[case(-0.5, 10.0)]
    #[case(-10.5, 0.0)]
    fn integer_fractional_bounds_rejected(#[case] min: f64, #[case] max: f64) {
        let mut data = vec![0i8, 5, 10];

        let mut tensor = make_tensor_i8(&mut data);

        let result = Clamp::new(Some(min), Some(max)).unwrap().apply(&mut tensor);

        assert!(matches!(
            result,
            Err(TransformError::ScalarConversion(
                ScalarConversionError::FractionalValue
            ))
        ));
    }

    // ---------------------------------------------------------
    // Integer: exact bounds
    // ---------------------------------------------------------

    #[test]
    fn clamp_i8() {
        let mut data = vec![-128, -100, 0, 100, 127];

        {
            let mut tensor = make_tensor_i8(&mut data);

            Clamp::new(Some(-50.0), Some(50.0))
                .unwrap()
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, vec![-50, -50, 0, 50, 50]);
    }

    #[test]
    fn clamp_u8() {
        let mut data = vec![0, 10, 100, 200, 255];

        {
            let mut tensor = make_tensor_u8(&mut data);

            Clamp::new(Some(50.0), Some(200.0))
                .unwrap()
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, vec![50, 50, 100, 200, 200]);
    }

    // ---------------------------------------------------------
    // Integer rounding
    // ---------------------------------------------------------

    #[rstest]
    #[case(IntRoundingMode::Floor, vec![0, 1, 5, 5])]
    #[case(IntRoundingMode::Ceil,  vec![1, 1, 5, 6])]
    #[case(IntRoundingMode::Round, vec![1, 1, 5, 6])]
    fn integer_rounding_modes(#[case] mode: IntRoundingMode, #[case] expected: Vec<i8>) {
        let mut data = vec![0i8, 1, 5, 10];

        {
            let mut tensor = make_tensor_i8(&mut data);

            Clamp::new(Some(0.5), Some(5.5))
                .unwrap()
                .int_rounding_mode(mode)
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, expected);
    }

    // ---------------------------------------------------------
    // Integer overflow
    // ---------------------------------------------------------

    #[test]
    fn integer_bound_overflow_returns_error() {
        let mut data = vec![0i8, 10, 100];

        {
            let mut tensor = make_tensor_i8(&mut data);

            let result = Clamp::new(Some(-200.0), Some(200.0))
                .unwrap()
                .apply(&mut tensor);

            assert!(matches!(
                result,
                Err(TransformError::ScalarConversion(
                    ScalarConversionError::Overflow
                ))
            ));
        }

        assert_eq!(data, vec![0, 10, 100]);
    }

    #[test]
    fn integer_bound_overflow_can_be_clamped() {
        let mut data = vec![-128, -10, 0, 10, 127];

        {
            let mut tensor = make_tensor_i8(&mut data);

            Clamp::new(Some(-200.0), Some(200.0))
                .unwrap()
                .overflow_mode(OverflowMode::Clamp)
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, vec![-128, -10, 0, 10, 127]);
    }

    #[test]
    fn integer_overflow_clamps_only_outside_side() {
        let mut data = vec![-128, -100, 0, 100, 127];

        {
            let mut tensor = make_tensor_i8(&mut data);

            Clamp::new(Some(-200.0), Some(100.0))
                .unwrap()
                .overflow_mode(OverflowMode::Clamp)
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, vec![-128, -100, 0, 100, 100]);
    }

    // ---------------------------------------------------------
    // Modification / error atomicity
    // ---------------------------------------------------------

    #[test]
    fn clamp_modifies_only_values_outside_bounds() {
        let mut data = vec![-10.0, 1.0, 5.0, 9.0, 20.0];

        {
            let mut tensor = make_tensor_f32(&mut data);

            Clamp::new(Some(0.0), Some(10.0))
                .unwrap()
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, vec![0.0, 1.0, 5.0, 9.0, 10.0]);
    }

    #[test]
    fn failed_clamp_does_not_modify_tensor() {
        let mut data = vec![-10i8, 0, 10, 100];

        let original = data.clone();

        {
            let mut tensor = make_tensor_i8(&mut data);

            let result = Clamp::new(Some(-200.0), Some(200.0))
                .unwrap()
                .apply(&mut tensor);

            assert!(matches!(
                result,
                Err(TransformError::ScalarConversion(
                    ScalarConversionError::Overflow
                ))
            ));
        }

        assert_eq!(data, original);
    }

    // ---------------------------------------------------------
    // No-op
    // ---------------------------------------------------------

    #[test]
    fn clamp_without_bounds_is_noop() {
        let mut data = vec![-100.0f32, 0.0, 100.0];

        let original = data.clone();

        {
            let mut tensor = make_tensor_f32(&mut data);

            Clamp::new(None::<f64>, None::<f64>)
                .unwrap()
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(data, original);
    }

    #[test]
    fn clamp_f16_bound_overflow() {
        let mut data = vec![
            half::f16::from_f32(-1.0),
            half::f16::from_f32(0.0),
            half::f16::from_f32(1.0),
        ];

        {
            let l = data.len();
            let raw_bytes = bytemuck::cast_slice_mut(&mut data);

            let layout = TensorBatchLayout::new(vec![l].into(), vec![1].into(), TensorDT::F16);

            let mut tensor = layout.try_view_mut(raw_bytes).unwrap();

            Clamp::new(Some(-1e10), Some(1e10))
                .unwrap()
                .overflow_mode(OverflowMode::Clamp)
                .apply(&mut tensor)
                .unwrap();
        }

        assert_eq!(
            data,
            vec![
                half::f16::from_f32(-1.0),
                half::f16::from_f32(0.0),
                half::f16::from_f32(1.0),
            ]
        );
    }
}
```

# zero-tensor-rs/src/transform/error.rs (34 lines)
```rust
use std::{error::Error, sync::Arc};

#[derive(Debug, thiserror::Error, Clone)]
pub enum TransformError {
    #[error("UnsupportedDtype")]
    UnsupportedDtype,

    #[error("Overflow")]
    Overflow,

    #[error("Invalid value")]
    InvalidValue,

    #[error("Scalar conversion error: {0}")]
    ScalarConversion(#[from] ScalarConversionError),

    #[error("Custom error: {0}")]
    Custom(Arc<dyn Error + Send + Sync>),
}

#[derive(Debug, thiserror::Error, Clone, Copy)]
pub enum ScalarConversionError {
    #[error("Overflow")]
    Overflow,

    #[error("Unsupported dtype")]
    UnsupportedDtype,

    #[error("Invalid value")]
    InvalidValue,

    #[error("FractionalValue")]
    FractionalValue,
}
```

# zero-tensor-rs/src/transform/helpers.rs (3 lines)
```rust
pub fn is_float_int(val: f64) -> bool {
    val.fract() == 0.0
}
```

# zero-tensor-rs/src/transform/mod.rs (19 lines)
```rust
pub mod add;
pub mod clamp;
pub mod error;
mod helpers;
pub mod scalar;
pub mod scale;
pub mod standardize;

use crate::core::dataset::item::TensorViewMut;

pub use add::Add;
pub use clamp::Clamp;
pub use error::{ScalarConversionError, TransformError};
pub use scalar::{IntoScalarOption, Scalar};
pub use scale::Scale;

pub trait Transform: Send + Sync {
    fn apply(&self, tensor: &mut TensorViewMut) -> Result<(), TransformError>;
}
```

# zero-tensor-rs/src/transform/scalar/cmp.rs (214 lines)
```rust
use std::cmp::Ordering;

use super::Scalar;

macro_rules! promote_partial_cmp {
    ($x:expr, $y:expr) => {
        match ($x, $y) {
            (Scalar::U8(a), Scalar::U8(b)) => a.partial_cmp(b),
            (Scalar::I8(a), Scalar::I8(b)) => a.partial_cmp(b),
            (Scalar::I32(a), Scalar::I32(b)) => a.partial_cmp(b),
            (Scalar::I64(a), Scalar::I64(b)) => a.partial_cmp(b),

            (Scalar::F32(a), Scalar::F32(b)) => a.partial_cmp(b),
            (Scalar::F64(a), Scalar::F64(b)) => a.partial_cmp(b),

            (a, b) => a.to_f64_lossy().partial_cmp(&b.to_f64_lossy()),
        }
    };
}

impl PartialOrd for Scalar {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        promote_partial_cmp!(self, other)
    }
}

impl PartialEq for Scalar {
    fn eq(&self, other: &Self) -> bool {
        self.partial_cmp(other) == Some(Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use super::Scalar;

    #[test]
    fn same_integer_types() {
        assert_eq!(
            Scalar::U8(1).partial_cmp(&Scalar::U8(2)),
            Some(Ordering::Less)
        );
        assert_eq!(
            Scalar::I8(2).partial_cmp(&Scalar::I8(1)),
            Some(Ordering::Greater)
        );
        assert_eq!(Scalar::I32(42), Scalar::I32(42));
        assert_eq!(
            Scalar::I64(-10).partial_cmp(&Scalar::I64(-5)),
            Some(Ordering::Less)
        );
    }

    #[test]
    fn same_float_types() {
        assert_eq!(Scalar::F32(1.5), Scalar::F32(1.5));
        assert!(Scalar::F32(1.0) < Scalar::F32(2.0));

        assert_eq!(Scalar::F64(-1.0), Scalar::F64(-1.0));
        assert!(Scalar::F64(10.0) > Scalar::F64(5.0));
    }

    #[test]
    fn same_half_types() {
        let a = Scalar::F16(half::f16::from_f32(1.5));
        let b = Scalar::F16(half::f16::from_f32(2.5));

        assert!(a < b);

        let a = Scalar::BF16(half::bf16::from_f32(1.5));
        let b = Scalar::BF16(half::bf16::from_f32(1.5));

        assert_eq!(a, b);
    }

    #[test]
    fn mixed_integer_types() {
        assert_eq!(Scalar::U8(42), Scalar::I8(42));
        assert_eq!(
            Scalar::I8(-1).partial_cmp(&Scalar::U8(1)),
            Some(Ordering::Less)
        );

        assert_eq!(Scalar::U8(42), Scalar::I32(42));
        assert_eq!(
            Scalar::I32(-10).partial_cmp(&Scalar::U8(0)),
            Some(Ordering::Less)
        );

        assert_eq!(Scalar::I32(100), Scalar::I64(100));
        assert!(Scalar::I64(101) > Scalar::I32(100));
    }

    #[test]
    fn integer_and_float_comparison() {
        assert_eq!(Scalar::I32(42), Scalar::F32(42.0));
        assert!(Scalar::I32(41) < Scalar::F32(42.0));

        assert_eq!(Scalar::U8(255), Scalar::F32(255.0));
        assert!(Scalar::F32(255.0) > Scalar::I32(100));

        assert_eq!(Scalar::I64(42), Scalar::F64(42.0));
        assert!(Scalar::I64(-1) < Scalar::F64(0.0));
    }

    #[test]
    fn half_and_f32_comparison() {
        let f16 = Scalar::F16(half::f16::from_f32(1.5));
        let f32 = Scalar::F32(1.5);

        assert_eq!(f16, f32);

        let bf16 = Scalar::BF16(half::bf16::from_f32(2.0));
        let f32 = Scalar::F32(3.0);

        assert!(bf16 < f32);
    }

    #[test]
    fn half_and_integer_comparison() {
        let f16 = Scalar::F16(half::f16::from_f32(42.0));
        assert_eq!(f16, Scalar::I32(42));

        let bf16 = Scalar::BF16(half::bf16::from_f32(-10.0));
        assert!(bf16 < Scalar::I64(-5));
    }

    #[test]
    fn f64_promotes_all_other_types() {
        assert_eq!(Scalar::F64(42.0), Scalar::U8(42));
        assert_eq!(Scalar::F64(-42.0), Scalar::I8(-42));
        assert_eq!(Scalar::F64(42.0), Scalar::I32(42));
        assert_eq!(Scalar::F64(42.0), Scalar::I64(42));

        assert_eq!(Scalar::F64(1.5), Scalar::F16(half::f16::from_f32(1.5)));

        assert_eq!(Scalar::F64(1.5), Scalar::BF16(half::bf16::from_f32(1.5)));
    }

    #[test]
    fn comparison_is_symmetric() {
        let pairs = [
            (Scalar::U8(42), Scalar::I8(42)),
            (Scalar::I32(-10), Scalar::I64(-10)),
            (Scalar::I32(100), Scalar::F32(100.0)),
            (Scalar::I64(100), Scalar::F64(100.0)),
            (
                Scalar::F16(half::f16::from_f32(2.0)),
                Scalar::BF16(half::bf16::from_f32(2.0)),
            ),
        ];

        for (a, b) in pairs {
            assert_eq!(a == b, b == a);
            assert_eq!(a.partial_cmp(&b), b.partial_cmp(&a).map(Ordering::reverse));
        }
    }

    #[test]
    fn nan_is_not_equal_to_anything() {
        let nan32 = Scalar::F32(f32::NAN);

        assert_ne!(nan32, Scalar::F32(f32::NAN));
        assert_ne!(nan32, Scalar::F32(1.0));
        assert_ne!(Scalar::F32(1.0), nan32);
    }

    #[test]
    fn nan_has_no_ordering() {
        let nan32 = Scalar::F32(f32::NAN);
        let value32 = Scalar::F32(1.0);

        assert_eq!(nan32.partial_cmp(&value32), None);
        assert_eq!(value32.partial_cmp(&nan32), None);

        let nan64 = Scalar::F64(f64::NAN);

        assert_eq!(nan64.partial_cmp(&Scalar::I64(1)), None);
        assert_eq!(Scalar::I64(1).partial_cmp(&nan64), None);
    }

    #[test]
    fn infinity_comparison() {
        assert!(Scalar::F32(f32::INFINITY) > Scalar::F32(1.0));
        assert!(Scalar::F64(f64::NEG_INFINITY) < Scalar::I32(0));

        assert_eq!(Scalar::F32(f32::INFINITY), Scalar::F64(f64::INFINITY));

        assert!(Scalar::F64(f64::NEG_INFINITY) < Scalar::F16(half::f16::from_f32(-100.0)));
    }

    #[test]
    fn transitivity_for_ordered_values() {
        let a = Scalar::I8(-1);
        let b = Scalar::F32(0.0);
        let c = Scalar::F64(1.0);

        assert!(a < b);
        assert!(b < c);
        assert!(a < c);
    }

    #[test]
    fn precision_edge_i64_and_f32() {
        let a = Scalar::I64(16_777_217);
        let b = Scalar::F32(16_777_216.0);

        assert!(a > b);
        assert!(b < a);
        assert_ne!(a, b);
    }
}
```

# zero-tensor-rs/src/transform/scalar/is_zero.rs (32 lines)
```rust
pub trait IsZero {
    fn eq_zero(self) -> bool;
}

macro_rules! impl_is_zero {
    ($($ty:ty),* $(,)?) => {
        $(
            impl IsZero for $ty {
                #[inline]
                fn eq_zero(self) -> bool {
                    self == 0 as $ty
                }
            }
        )*
    };
}

impl_is_zero!(u8, i8, i32, i64, f32, f64);

impl IsZero for half::f16 {
    #[inline]
    fn eq_zero(self) -> bool {
        self == half::f16::ZERO
    }
}

impl IsZero for half::bf16 {
    #[inline]
    fn eq_zero(self) -> bool {
        self == half::bf16::ZERO
    }
}
```

# zero-tensor-rs/src/transform/scalar/mod.rs (220 lines)
```rust
use super::{ScalarConversionError, helpers::is_float_int};
pub mod cmp;
pub mod is_zero;
pub mod ops;

#[derive(Clone, Copy, Debug)]
pub enum Scalar {
    U8(u8),
    I8(i8),
    I32(i32),
    I64(i64),
    BF16(half::bf16),
    F16(half::f16),
    F32(f32),
    F64(f64),
}

pub trait IntoScalarOption {
    fn into_scalar_option(self) -> Option<Scalar>;
}

macro_rules! int_to_int {
    ($value:expr, $target:ty) => {{
        if $value as i128 > <$target>::MAX as i128 || ($value as i128) < <$target>::MIN as i128 {
            Err(ScalarConversionError::Overflow)
        } else {
            Ok($value as $target)
        }
    }};
}

macro_rules! float_to_int {
    ($f:expr, $tyfrom:ty) => {{
        let f_64 = $f as f64;
        if !f_64.is_finite() {
            Err(ScalarConversionError::InvalidValue)
        } else if !is_float_int(f_64) {
            Err(ScalarConversionError::FractionalValue)
        } else if f_64 > <$tyfrom>::MAX as f64 || f_64 < <$tyfrom>::MIN as f64 {
            Err(ScalarConversionError::Overflow)
        } else {
            Ok(f_64 as $tyfrom)
        }
    }};
}

macro_rules! impl_int_try_from {
    ($ty:ty) => {
        impl TryFrom<Scalar> for $ty {
            type Error = ScalarConversionError;

            fn try_from(value: Scalar) -> Result<$ty, Self::Error> {
                match value {
                    Scalar::U8(i) => int_to_int!(i, $ty),
                    Scalar::I8(i) => int_to_int!(i, $ty),
                    Scalar::I32(i) => int_to_int!(i, $ty),
                    Scalar::I64(i) => int_to_int!(i, $ty),
                    Scalar::BF16(h) => float_to_int!(h.to_f64(), $ty),
                    Scalar::F16(h) => float_to_int!(h.to_f64(), $ty),
                    Scalar::F32(f) => float_to_int!(f, $ty),
                    Scalar::F64(f) => float_to_int!(f, $ty),
                }
            }
        }
    };
}

macro_rules! impl_from {
    ($ty:ty, $variant:ident) => {
        impl From<$ty> for Scalar {
            fn from(value: $ty) -> Self {
                Self::$variant(value)
            }
        }

        impl IntoScalarOption for $ty {
            fn into_scalar_option(self) -> Option<Scalar> {
                Some(Scalar::$variant(self))
            }
        }

        impl IntoScalarOption for Option<$ty> {
            fn into_scalar_option(self) -> Option<Scalar> {
                self.map(Scalar::$variant)
            }
        }
    };
}

macro_rules! float_to_half {
    ($f:expr, $tyfrom:ty) => {{
        let i = $f as f64;
        if i.is_nan() || i.is_infinite() {
            Ok(<$tyfrom>::from_f64(i))
        } else if i < <$tyfrom>::MIN.to_f64() || i > <$tyfrom>::MAX.to_f64() {
            Err(ScalarConversionError::Overflow)
        } else {
            Ok(<$tyfrom>::from_f64(i))
        }
    }};
}

macro_rules! impl_half_try_from {
    ($ty:ty) => {
        impl TryFrom<Scalar> for $ty {
            type Error = ScalarConversionError;

            fn try_from(value: Scalar) -> Result<$ty, Self::Error> {
                match value {
                    Scalar::U8(i) => float_to_half!(i, $ty),
                    Scalar::I8(i) => float_to_half!(i, $ty),
                    Scalar::I32(i) => float_to_half!(i, $ty),
                    Scalar::I64(i) => float_to_half!(i, $ty),
                    Scalar::BF16(h) => Ok(<$ty>::from_f32(h.to_f32())),
                    Scalar::F16(h) => Ok(<$ty>::from_f32(h.to_f32())),
                    Scalar::F32(f) => float_to_half!(f, $ty),
                    Scalar::F64(f) => float_to_half!(f, $ty),
                }
            }
        }
    };
}

macro_rules! int_to_float {
    ($i:expr, $tyfrom:ty) => {
        if $i as i128 > <$tyfrom>::MAX as i128 || ($i as i128) < <$tyfrom>::MIN as i128 {
            Err(ScalarConversionError::Overflow)
        } else {
            Ok($i as $tyfrom)
        }
    };
}

macro_rules! impl_float_try_from {
    ($ty:ty) => {
        impl TryFrom<Scalar> for $ty {
            type Error = ScalarConversionError;

            fn try_from(value: Scalar) -> Result<$ty, Self::Error> {
                match value {
                    Scalar::U8(i) => int_to_float!(i, $ty),
                    Scalar::I8(i) => int_to_float!(i, $ty),
                    Scalar::I32(i) => int_to_float!(i, $ty),
                    Scalar::I64(i) => int_to_float!(i, $ty),
                    Scalar::BF16(h) => Ok(h.to_f32() as $ty),
                    Scalar::F16(h) => Ok(h.to_f32() as $ty),
                    Scalar::F32(f) => Ok(f as $ty),
                    Scalar::F64(f) => Ok(f as $ty),
                }
            }
        }
    };
}

impl_from! {u8, U8}
impl_from! {i8, I8}
impl_from! {i32, I32}
impl_from! {i64, I64}
impl_from! {half::bf16, BF16}
impl_from! {half::f16, F16}
impl_from! {f32, F32}
impl_from! {f64, F64}

impl_int_try_from! {u8}
impl_int_try_from! {i8}
impl_int_try_from! {i32}
impl_int_try_from! {i64}
impl_half_try_from! {half::bf16}
impl_half_try_from! {half::f16}
impl_float_try_from! {f32}
impl_float_try_from! {f64}

macro_rules! float_fn {
    ($fn_name:ident, $ret:ty, $fb:expr) => {
        pub fn $fn_name(self) -> $ret {
            match self {
                Scalar::BF16(f) => f.$fn_name(),
                Scalar::F16(f) => f.$fn_name(),
                Scalar::F32(f) => f.$fn_name(),
                Scalar::F64(f) => f.$fn_name(),
                _ => $fb,
            }
        }
    };
}

impl Scalar {
    pub fn to_f32_lossy(self) -> f32 {
        match self {
            Scalar::U8(v) => v as f32,
            Scalar::I8(v) => v as f32,
            Scalar::I32(v) => v as f32,
            Scalar::I64(v) => v as f32,
            Scalar::F32(v) => v,
            Scalar::F64(v) => v as f32,
            Scalar::BF16(v) => v.to_f32(),
            Scalar::F16(v) => v.to_f32(),
        }
    }

    pub fn to_f64_lossy(self) -> f64 {
        match self {
            Scalar::U8(v) => v as f64,
            Scalar::I8(v) => v as f64,
            Scalar::I32(v) => v as f64,
            Scalar::I64(v) => v as f64,
            Scalar::F32(v) => v as f64,
            Scalar::F64(v) => v,
            Scalar::BF16(v) => v.to_f64(),
            Scalar::F16(v) => v.to_f64(),
        }
    }

    float_fn! {is_nan, bool, false}
    float_fn! {is_finite, bool, true}
    float_fn! {is_infinite, bool, false}
}

#[cfg(test)]
mod tests;
```

# zero-tensor-rs/src/transform/scalar/ops.rs (530 lines)
```rust
use std::ops::{Neg, Not};

use super::Scalar;

macro_rules! promote_types {
    ($x:expr, $y:expr, $op:tt) => {
        match ($x, $y) {
            (Scalar::U8(a), Scalar::U8(b))     => Scalar::U8(a $op b),
            (Scalar::I8(a), Scalar::I8(b))     => Scalar::I8(a $op b),
            (Scalar::I32(a), Scalar::I32(b))   => Scalar::I32(a $op b),
            (Scalar::I64(a), Scalar::I64(b))   => Scalar::I64(a $op b),
            (Scalar::F32(a), Scalar::F32(b))   => Scalar::F32(a $op b),
            (Scalar::F64(a), Scalar::F64(b))   => Scalar::F64(a $op b),
            (Scalar::BF16(a), Scalar::BF16(b)) => Scalar::BF16(half::bf16::from_f32(a.to_f32() $op b.to_f32())),
            (Scalar::F16(a), Scalar::F16(b))   => Scalar::F16(half::f16::from_f32(a.to_f32() $op b.to_f32())),

            (Scalar::U8(a), Scalar::I8(b))     => Scalar::I32((a as i32) $op (b as i32)),
            (Scalar::I8(a), Scalar::U8(b))     => Scalar::I32((a as i32) $op (b as i32)),

            (Scalar::U8(a), Scalar::I32(b))     => Scalar::I32((a as i32) $op b),
            (Scalar::I32(a), Scalar::U8(b))     => Scalar::I32(a $op (b as i32)),

            (Scalar::U8(a), Scalar::I64(b))     => Scalar::I64((a as i64) $op b),
            (Scalar::I64(a), Scalar::U8(b))     => Scalar::I64(a $op (b as i64)),

            (Scalar::I8(a), Scalar::I32(b))     => Scalar::I32((a as i32) $op b),
            (Scalar::I32(a), Scalar::I8(b))     => Scalar::I32(a $op (b as i32)),

            (Scalar::I8(a), Scalar::I64(b))     => Scalar::I64((a as i64) $op b),
            (Scalar::I64(a), Scalar::I8(b))     => Scalar::I64(a $op (b as i64)),

            (Scalar::I32(a), Scalar::I64(b))    => Scalar::I64((a as i64) $op b),
            (Scalar::I64(a), Scalar::I32(b))    => Scalar::I64(a $op (b as i64)),

            (Scalar::I32(a), Scalar::F32(b))    => Scalar::F32((a as f32) $op b),
            (Scalar::F32(a), Scalar::I32(b))    => Scalar::F32(a $op (b as f32)),

            (Scalar::I64(a), Scalar::F32(b))    => Scalar::F64((a as f64) $op (b as f64)),
            (Scalar::F32(a), Scalar::I64(b))    => Scalar::F64((a as f64) $op (b as f64)),

            (Scalar::U8(a), Scalar::F32(b))     => Scalar::F32((a as f32) $op b),
            (Scalar::F32(a), Scalar::U8(b))     => Scalar::F32(a $op (b as f32)),

            (Scalar::F16(a), Scalar::F32(b))    => Scalar::F32(a.to_f32() $op b),
            (Scalar::F32(a), Scalar::F16(b))    => Scalar::F32(a $op b.to_f32()),

            (Scalar::BF16(a), Scalar::F32(b))   => Scalar::F32(a.to_f32() $op b),
            (Scalar::F32(a), Scalar::BF16(b))   => Scalar::F32(a $op b.to_f32()),

            (Scalar::F16(a), Scalar::BF16(b))   => Scalar::F32(a.to_f32() $op b.to_f32()),
            (Scalar::BF16(a), Scalar::F16(b))   => Scalar::F32(a.to_f32() $op b.to_f32()),

            (Scalar::F64(a), any)               => Scalar::F64(a $op any.to_f64_lossy()),
            (any, Scalar::F64(b))               => Scalar::F64(any.to_f64_lossy() $op b),

            (a, b)                              => Scalar::F32(a.to_f32_lossy() $op b.to_f32_lossy()),
        }
    };
}

impl Neg for Scalar {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Scalar::U8(val) => Scalar::I32(-(val as i32)),

            Scalar::I8(val) => Scalar::I8(-val),
            Scalar::I32(val) => Scalar::I32(-val),
            Scalar::I64(val) => Scalar::I64(-val),
            Scalar::F32(val) => Scalar::F32(-val),
            Scalar::F64(val) => Scalar::F64(-val),

            Scalar::BF16(val) => Scalar::BF16(half::bf16::from_f32(-val.to_f32())),
            Scalar::F16(val) => Scalar::F16(half::f16::from_f32(-val.to_f32())),
        }
    }
}

impl Not for Scalar {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            Scalar::U8(val) => Scalar::U8(!val),
            Scalar::I8(val) => Scalar::I8(!val),
            Scalar::I32(val) => Scalar::I32(!val),
            Scalar::I64(val) => Scalar::I64(!val),

            float_scalar => float_scalar,
        }
    }
}

macro_rules! generate_math_ops {
    ($trait_name:ident, $method_name:ident, $op:tt) => {
        impl std::ops::$trait_name for Scalar {
            type Output = Self;

            #[inline]
            fn $method_name(self, rhs: Self) -> Self::Output {
                promote_types!(self, rhs, $op)
            }
        }
    };
}

generate_math_ops!(Add, add, +);
generate_math_ops!(Sub, sub, -);
generate_math_ops!(Mul, mul, *);
generate_math_ops!(Div, div, /);

#[cfg(test)]
mod tests {
    use super::*;

    fn f16(v: f32) -> half::f16 {
        half::f16::from_f32(v)
    }

    fn bf16(v: f32) -> half::bf16 {
        half::bf16::from_f32(v)
    }

    // ------------------------------------------------------------
    // Same type
    // ------------------------------------------------------------

    #[test]
    fn same_type_add() {
        assert!(matches!(Scalar::U8(2) + Scalar::U8(3), Scalar::U8(5)));

        assert!(matches!(Scalar::I8(2) + Scalar::I8(3), Scalar::I8(5)));

        assert!(matches!(Scalar::I32(2) + Scalar::I32(3), Scalar::I32(5)));

        assert!(matches!(Scalar::I64(2) + Scalar::I64(3), Scalar::I64(5)));

        assert!(matches!(
            Scalar::F32(2.0) + Scalar::F32(3.0),
            Scalar::F32(x) if x == 5.0
        ));

        assert!(matches!(
            Scalar::F64(2.0) + Scalar::F64(3.0),
            Scalar::F64(x) if x == 5.0
        ));

        assert!(matches!(
            Scalar::F16(f16(2.0)) + Scalar::F16(f16(3.0)),
            Scalar::F16(x) if x == f16(5.0)
        ));

        assert!(matches!(
            Scalar::BF16(bf16(2.0)) + Scalar::BF16(bf16(3.0)),
            Scalar::BF16(x) if x == bf16(5.0)
        ));
    }

    #[test]
    fn same_type_sub() {
        assert!(matches!(Scalar::I32(10) - Scalar::I32(3), Scalar::I32(7)));

        assert!(matches!(
            Scalar::F32(10.0) - Scalar::F32(3.0),
            Scalar::F32(x) if x == 7.0
        ));
    }

    #[test]
    fn same_type_mul() {
        assert!(matches!(Scalar::I64(6) * Scalar::I64(7), Scalar::I64(42)));

        assert!(matches!(
            Scalar::F64(1.5) * Scalar::F64(2.0),
            Scalar::F64(x) if x == 3.0
        ));
    }

    #[test]
    fn same_type_div() {
        assert!(matches!(Scalar::I32(12) / Scalar::I32(3), Scalar::I32(4)));

        assert!(matches!(
            Scalar::F64(12.0) / Scalar::F64(4.0),
            Scalar::F64(x) if x == 3.0
        ));
    }

    // ------------------------------------------------------------
    // Integer promotion
    // ------------------------------------------------------------

    #[test]
    fn u8_i8_promotes_to_i32() {
        assert!(matches!(Scalar::U8(10) + Scalar::I8(-3), Scalar::I32(7)));

        assert!(matches!(Scalar::I8(-3) + Scalar::U8(10), Scalar::I32(7)));
    }

    #[test]
    fn u8_i32_promotes_to_i32() {
        assert!(matches!(Scalar::U8(10) + Scalar::I32(20), Scalar::I32(30)));

        assert!(matches!(Scalar::I32(20) + Scalar::U8(10), Scalar::I32(30)));
    }

    #[test]
    fn u8_i64_promotes_to_i64() {
        assert!(matches!(Scalar::U8(10) + Scalar::I64(20), Scalar::I64(30)));

        assert!(matches!(Scalar::I64(20) + Scalar::U8(10), Scalar::I64(30)));
    }

    #[test]
    fn i8_i32_promotes_to_i32() {
        assert!(matches!(Scalar::I8(-10) + Scalar::I32(20), Scalar::I32(10)));

        assert!(matches!(Scalar::I32(20) + Scalar::I8(-10), Scalar::I32(10)));
    }

    #[test]
    fn i8_i64_promotes_to_i64() {
        assert!(matches!(Scalar::I8(-10) + Scalar::I64(20), Scalar::I64(10)));

        assert!(matches!(Scalar::I64(20) + Scalar::I8(-10), Scalar::I64(10)));
    }

    #[test]
    fn i32_i64_promotes_to_i64() {
        assert!(matches!(Scalar::I32(10) + Scalar::I64(20), Scalar::I64(30)));

        assert!(matches!(Scalar::I64(20) + Scalar::I32(10), Scalar::I64(30)));
    }

    // ------------------------------------------------------------
    // F32 special promotion
    // ------------------------------------------------------------

    #[test]
    fn i32_f32_promotes_to_f32() {
        assert!(matches!(
            Scalar::I32(2) + Scalar::F32(0.5),
            Scalar::F32(x) if x == 2.5
        ));

        assert!(matches!(
            Scalar::F32(0.5) + Scalar::I32(2),
            Scalar::F32(x) if x == 2.5
        ));
    }

    #[test]
    fn i64_f32_promotes_to_f64() {
        assert!(matches!(
            Scalar::I64(2) + Scalar::F32(0.5),
            Scalar::F64(x) if x == 2.5
        ));

        assert!(matches!(
            Scalar::F32(0.5) + Scalar::I64(2),
            Scalar::F64(x) if x == 2.5
        ));
    }

    #[test]
    fn u8_f32_promotes_to_f32() {
        assert!(matches!(
            Scalar::U8(2) + Scalar::F32(0.5),
            Scalar::F32(x) if x == 2.5
        ));

        assert!(matches!(
            Scalar::F32(0.5) + Scalar::U8(2),
            Scalar::F32(x) if x == 2.5
        ));
    }

    // ------------------------------------------------------------
    // Half precision promotion
    // ------------------------------------------------------------

    #[test]
    fn f16_f32_promotes_to_f32() {
        assert!(matches!(
            Scalar::F16(f16(2.0)) + Scalar::F32(0.5),
            Scalar::F32(x) if x == 2.5
        ));

        assert!(matches!(
            Scalar::F32(0.5) + Scalar::F16(f16(2.0)),
            Scalar::F32(x) if x == 2.5
        ));
    }

    #[test]
    fn bf16_f32_promotes_to_f32() {
        assert!(matches!(
            Scalar::BF16(bf16(2.0)) + Scalar::F32(0.5),
            Scalar::F32(x) if x == 2.5
        ));

        assert!(matches!(
            Scalar::F32(0.5) + Scalar::BF16(bf16(2.0)),
            Scalar::F32(x) if x == 2.5
        ));
    }

    #[test]
    fn f16_bf16_promotes_to_f32() {
        assert!(matches!(
            Scalar::F16(f16(2.0)) + Scalar::BF16(bf16(0.5)),
            Scalar::F32(x) if x == 2.5
        ));

        assert!(matches!(
            Scalar::BF16(bf16(0.5)) + Scalar::F16(f16(2.0)),
            Scalar::F32(x) if x == 2.5
        ));
    }

    // ------------------------------------------------------------
    // F64 always wins
    // ------------------------------------------------------------

    #[test]
    fn f64_wins_over_integer() {
        assert!(matches!(
            Scalar::F64(0.5) + Scalar::I32(2),
            Scalar::F64(x) if x == 2.5
        ));

        assert!(matches!(
            Scalar::I32(2) + Scalar::F64(0.5),
            Scalar::F64(x) if x == 2.5
        ));
    }

    #[test]
    fn f64_wins_over_f32() {
        assert!(matches!(
            Scalar::F64(0.5) + Scalar::F32(2.0),
            Scalar::F64(x) if x == 2.5
        ));

        assert!(matches!(
            Scalar::F32(2.0) + Scalar::F64(0.5),
            Scalar::F64(x) if x == 2.5
        ));
    }

    #[test]
    fn f64_wins_over_f16() {
        assert!(matches!(
            Scalar::F16(f16(2.0)) + Scalar::F64(0.5),
            Scalar::F64(x) if x == 2.5
        ));
    }

    #[test]
    fn f64_wins_over_bf16() {
        assert!(matches!(
            Scalar::BF16(bf16(2.0)) + Scalar::F64(0.5),
            Scalar::F64(x) if x == 2.5
        ));
    }

    // ------------------------------------------------------------
    // Fallback -> F32
    // ------------------------------------------------------------

    #[test]
    fn fallback_integer_f16_promotes_to_f32() {
        assert!(matches!(
            Scalar::I8(2) + Scalar::F16(f16(0.5)),
            Scalar::F32(x) if x == 2.5
        ));
    }

    #[test]
    fn fallback_integer_bf16_promotes_to_f32() {
        assert!(matches!(
            Scalar::I32(2) + Scalar::BF16(bf16(0.5)),
            Scalar::F32(x) if x == 2.5
        ));
    }

    // ------------------------------------------------------------
    // Operand order for non-commutative operators
    // ------------------------------------------------------------

    #[test]
    fn subtraction_preserves_operand_order() {
        assert!(matches!(Scalar::I32(10) - Scalar::U8(3), Scalar::I32(7)));

        assert!(matches!(Scalar::U8(3) - Scalar::I32(10), Scalar::I32(-7)));
    }

    #[test]
    fn division_preserves_operand_order() {
        assert!(matches!(Scalar::I32(20) / Scalar::U8(4), Scalar::I32(5)));

        assert!(matches!(
            Scalar::F32(10.0) / Scalar::I64(2),
            Scalar::F64(x) if x == 5.0
        ));
    }

    #[test]
    fn mixed_multiplication_works() {
        assert!(matches!(
            Scalar::I32(3) * Scalar::F32(2.5),
            Scalar::F32(x) if x == 7.5
        ));
    }

    #[test]
    fn neg_u8_promotes_to_i32() {
        assert_eq!(-Scalar::U8(42), Scalar::I32(-42));

        assert_eq!(-Scalar::U8(0), Scalar::I32(0));

        assert_eq!(-Scalar::U8(u8::MAX), Scalar::I32(-(u8::MAX as i32)));
    }

    #[test]
    fn neg_signed_integers() {
        assert_eq!(-Scalar::I8(42), Scalar::I8(-42));

        assert_eq!(-Scalar::I32(-42), Scalar::I32(42));

        assert_eq!(-Scalar::I64(1_000_000), Scalar::I64(-1_000_000));
    }

    #[test]
    fn neg_floats() {
        assert_eq!(-Scalar::F32(1.5), Scalar::F32(-1.5));

        assert_eq!(-Scalar::F64(-42.25), Scalar::F64(42.25));
    }

    #[test]
    fn neg_half_floats() {
        let bf16 = Scalar::BF16(half::bf16::from_f32(1.5));
        let f16 = Scalar::F16(half::f16::from_f32(-2.25));

        assert_eq!(-bf16, Scalar::BF16(half::bf16::from_f32(-1.5)));

        assert_eq!(-f16, Scalar::F16(half::f16::from_f32(2.25)));
    }

    #[test]
    fn neg_negative_zero() {
        let result = -Scalar::F32(0.0);

        match result {
            Scalar::F32(value) => {
                assert_eq!(value, 0.0);
                assert!(value.is_sign_negative());
            }
            _ => panic!("expected Scalar::F32"),
        }
    }

    #[test]
    fn not_u8() {
        assert_eq!(!Scalar::U8(0b0000_1111), Scalar::U8(0b1111_0000));

        assert_eq!(!Scalar::U8(0), Scalar::U8(u8::MAX));

        assert_eq!(!Scalar::U8(u8::MAX), Scalar::U8(0));
    }

    #[test]
    fn not_i8() {
        assert_eq!(!Scalar::I8(0), Scalar::I8(!0i8));

        assert_eq!(!Scalar::I8(42), Scalar::I8(!42i8));

        assert_eq!(!Scalar::I8(-1), Scalar::I8(!-1i8));
    }

    #[test]
    fn not_i32() {
        assert_eq!(!Scalar::I32(0), Scalar::I32(!0i32));

        assert_eq!(!Scalar::I32(0x0F0F_0F0F), Scalar::I32(!0x0F0F_0F0Fi32));
    }

    #[test]
    fn not_i64() {
        assert_eq!(!Scalar::I64(0), Scalar::I64(!0i64));

        assert_eq!(!Scalar::I64(42), Scalar::I64(!42i64));

        assert_eq!(!Scalar::I64(-1), Scalar::I64(!-1i64));
    }

    #[test]
    fn not_floats_are_identity() {
        assert_eq!(!Scalar::F32(1.5), Scalar::F32(1.5));

        assert_eq!(!Scalar::F64(-42.25), Scalar::F64(-42.25));
    }

    #[test]
    fn not_half_floats_are_identity() {
        let bf16 = half::bf16::from_f32(1.5);
        let f16 = half::f16::from_f32(-2.25);

        assert_eq!(!Scalar::BF16(bf16), Scalar::BF16(bf16));

        assert_eq!(!Scalar::F16(f16), Scalar::F16(f16));
    }

    #[test]
    fn not_integers_is_involution() {
        let u8_value = Scalar::U8(42);
        assert_eq!(!(!u8_value), Scalar::U8(42));

        let i8_value = Scalar::I8(-42);
        assert_eq!(!(!i8_value), Scalar::I8(-42));

        let i32_value = Scalar::I32(123_456);
        assert_eq!(!(!i32_value), Scalar::I32(123_456));

        let i64_value = Scalar::I64(-987_654_321);
        assert_eq!(!(!i64_value), Scalar::I64(-987_654_321));
    }
}
```

# zero-tensor-rs/src/transform/scalar/tests.rs (510 lines)
```rust
use std::assert_matches;

use super::*;
use rstest::rstest;

// ============================================================
// From<T> for Scalar
// ============================================================

#[test]
fn from_integer_types() {
    assert!(matches!(Scalar::from(42u8), Scalar::U8(42)));
    assert!(matches!(Scalar::from(-42i8), Scalar::I8(-42)));
    assert!(matches!(Scalar::from(-42i32), Scalar::I32(-42)));
    assert!(matches!(Scalar::from(-42i64), Scalar::I64(-42)));
}

#[test]
fn from_float_types() {
    assert!(matches!(Scalar::from(1.5f32), Scalar::F32(v) if v == 1.5));
    assert!(matches!(Scalar::from(1.5f64), Scalar::F64(v) if v == 1.5));
}

#[test]
fn from_half_types() {
    let bf16 = half::bf16::from_f32(1.5);
    let f16 = half::f16::from_f32(1.5);

    assert!(matches!(Scalar::from(bf16), Scalar::BF16(v) if v == bf16));
    assert!(matches!(Scalar::from(f16), Scalar::F16(v) if v == f16));
}

// ============================================================
// Integer -> Integer
// ============================================================

#[rstest]
#[case(0i64, 0i32)]
#[case(127i64, 127i32)]
#[case(-128i64, -128i32)]
#[case(0i64, 0i32)]
#[case(i32::MAX as i64, i32::MAX)]
#[case(i32::MIN as i64, i32::MIN)]
fn integer_to_integer_success(#[case] input: i64, #[case] expected: i32) {
    let result = i32::try_from(Scalar::I64(input)).unwrap();
    assert_eq!(result, expected);
}

#[test]
fn i8_boundaries() {
    assert_eq!(i8::try_from(Scalar::I32(i8::MIN as i32)).unwrap(), i8::MIN);

    assert_eq!(i8::try_from(Scalar::I32(i8::MAX as i32)).unwrap(), i8::MAX);
}

#[test]
fn i8_overflow_positive() {
    let result = i8::try_from(Scalar::I32(i8::MAX as i32 + 1));

    assert_matches!(result, Err(ScalarConversionError::Overflow));
}

#[test]
fn i8_overflow_negative() {
    let result = i8::try_from(Scalar::I32(i8::MIN as i32 - 1));

    assert_matches!(result, Err(ScalarConversionError::Overflow));
}

#[test]
fn u8_overflow_negative() {
    let result = u8::try_from(Scalar::I8(-1));

    assert_matches!(result, Err(ScalarConversionError::Overflow));
}

#[test]
fn u8_overflow_positive() {
    let result = u8::try_from(Scalar::I32(256));

    assert_matches!(result, Err(ScalarConversionError::Overflow));
}

#[test]
fn u8_boundaries() {
    assert_eq!(u8::try_from(Scalar::I32(0)).unwrap(), 0);
    assert_eq!(u8::try_from(Scalar::I32(255)).unwrap(), 255);
}

// ============================================================
// Float -> Integer
// ============================================================

#[rstest]
#[case(0.0)]
#[case(1.0)]
#[case(-1.0)]
#[case(127.0)]
#[case(-128.0)]
fn float_to_i8_success(#[case] value: f64) {
    let result = i8::try_from(Scalar::F64(value)).unwrap();
    assert_eq!(result as f64, value);
}

#[test]
fn float_to_integer_rejects_fractional_value() {
    assert_matches!(
        i32::try_from(Scalar::F64(1.5)),
        Err(ScalarConversionError::FractionalValue)
    );

    assert_matches!(
        i32::try_from(Scalar::F32(-1.5)),
        Err(ScalarConversionError::FractionalValue)
    );
}

#[test]
fn float_to_integer_exact_boundary() {
    assert_eq!(i8::try_from(Scalar::F64(i8::MAX as f64)).unwrap(), i8::MAX);

    assert_eq!(i8::try_from(Scalar::F64(i8::MIN as f64)).unwrap(), i8::MIN);
}

#[test]
fn float_to_integer_overflow_positive() {
    assert_matches!(
        i8::try_from(Scalar::F64(128.0)),
        Err(ScalarConversionError::Overflow)
    );
}

#[test]
fn float_to_integer_overflow_negative() {
    assert_matches!(
        i8::try_from(Scalar::F64(-129.0)),
        Err(ScalarConversionError::Overflow)
    );
}

#[test]
fn float_to_integer_nan() {
    assert_matches!(
        i32::try_from(Scalar::F64(f64::NAN)),
        Err(ScalarConversionError::InvalidValue)
    );
}

#[test]
fn float_to_integer_positive_infinity() {
    assert_matches!(
        i32::try_from(Scalar::F64(f64::INFINITY)),
        Err(ScalarConversionError::InvalidValue)
    );
}

#[test]
fn float_to_integer_negative_infinity() {
    assert_matches!(
        i32::try_from(Scalar::F64(f64::NEG_INFINITY)),
        Err(ScalarConversionError::InvalidValue)
    );
}

// ============================================================
// Half -> Integer
// ============================================================

#[test]
fn f16_to_integer() {
    assert_eq!(
        i32::try_from(Scalar::F16(half::f16::from_f32(42.0))).unwrap(),
        42
    );
}

#[test]
fn bf16_to_integer() {
    assert_eq!(
        i32::try_from(Scalar::BF16(half::bf16::from_f32(-42.0))).unwrap(),
        -42
    );
}

#[test]
fn f16_to_integer_fractional() {
    assert_matches!(
        i32::try_from(Scalar::F16(half::f16::from_f32(1.5))),
        Err(ScalarConversionError::FractionalValue)
    );
}

#[test]
fn bf16_to_integer_fractional() {
    assert_matches!(
        i32::try_from(Scalar::BF16(half::bf16::from_f32(1.5))),
        Err(ScalarConversionError::FractionalValue)
    );
}

// ============================================================
// Integer -> Float
// ============================================================

#[test]
fn integer_to_f32() {
    assert_eq!(f32::try_from(Scalar::I32(42)).unwrap(), 42.0);

    assert_eq!(f32::try_from(Scalar::I8(-42)).unwrap(), -42.0);

    assert_eq!(f32::try_from(Scalar::U8(255)).unwrap(), 255.0);
}

#[test]
fn integer_to_f64() {
    assert_eq!(
        f64::try_from(Scalar::I64(-123456789)).unwrap(),
        -123456789.0
    );

    assert_eq!(f64::try_from(Scalar::U8(255)).unwrap(), 255.0);
}

#[test]
fn i64_to_f32_does_not_overflow() {
    let result = f32::try_from(Scalar::I64(i64::MAX));

    assert!(result.is_ok());
    assert!(result.unwrap().is_finite());
}

// ============================================================
// Float -> Float
// ============================================================

#[test]
fn f32_to_f64() {
    let value = 123.5f32;

    let result = f64::try_from(Scalar::F32(value)).unwrap();

    assert_eq!(result, value as f64);
}

#[test]
fn f64_to_f32() {
    let value = 123.5f64;

    let result = f32::try_from(Scalar::F64(value)).unwrap();

    assert_eq!(result, value as f32);
}

#[test]
fn f64_to_f32_special_values() {
    assert!(f32::try_from(Scalar::F64(f64::NAN)).unwrap().is_nan());

    assert_eq!(
        f32::try_from(Scalar::F64(f64::INFINITY)).unwrap(),
        f32::INFINITY
    );

    assert_eq!(
        f32::try_from(Scalar::F64(f64::NEG_INFINITY)).unwrap(),
        f32::NEG_INFINITY
    );
}

// ============================================================
// Float -> Half
// ============================================================

#[test]
fn f32_to_f16() {
    let value = 1.5f32;

    let result = half::f16::try_from(Scalar::F32(value)).unwrap();

    assert_eq!(result.to_f32(), value);
}

#[test]
fn f32_to_bf16() {
    let value = 1.5f32;

    let result = half::bf16::try_from(Scalar::F32(value)).unwrap();

    assert_eq!(result.to_f32(), value);
}

#[test]
fn f64_to_f16() {
    let value = 1.5f64;

    let result = half::f16::try_from(Scalar::F64(value)).unwrap();

    assert_eq!(result.to_f64(), value);
}

// ============================================================
// Integer -> Half
// ============================================================

#[test]
fn integer_to_f16() {
    let result = half::f16::try_from(Scalar::I32(42)).unwrap();

    assert_eq!(result.to_f32(), 42.0);
}

#[test]
fn integer_to_bf16() {
    let result = half::bf16::try_from(Scalar::I32(-42)).unwrap();

    assert_eq!(result.to_f32(), -42.0);
}

// ============================================================
// Half -> Float
// ============================================================

#[test]
fn f16_to_f32() {
    let value = half::f16::from_f32(1.5);

    let result = f32::try_from(Scalar::F16(value)).unwrap();

    assert_eq!(result, value.to_f32());
}

#[test]
fn bf16_to_f64() {
    let value = half::bf16::from_f32(1.5);

    let result = f64::try_from(Scalar::BF16(value)).unwrap();

    assert_eq!(result, value.to_f64());
}

// ============================================================
// Half -> Half
// ============================================================

#[test]
fn f16_to_f16() {
    let value = half::f16::from_f32(1.5);

    let result = half::f16::try_from(Scalar::F16(value)).unwrap();

    assert_eq!(result, value);
}

#[test]
fn bf16_to_bf16() {
    let value = half::bf16::from_f32(1.5);

    let result = half::bf16::try_from(Scalar::BF16(value)).unwrap();

    assert_eq!(result, value);
}

#[test]
fn f16_to_bf16() {
    let value = half::f16::from_f32(1.5);

    let result = half::bf16::try_from(Scalar::F16(value)).unwrap();

    assert_eq!(result.to_f32(), value.to_f32());
}

#[test]
fn bf16_to_f16() {
    let value = half::bf16::from_f32(1.5);

    let result = half::f16::try_from(Scalar::BF16(value)).unwrap();

    assert_eq!(result.to_f32(), value.to_f32());
}

// ============================================================
// Half overflow
// ============================================================

#[test]
fn f32_to_f16_overflow() {
    let value = half::f16::MAX.to_f64() * 2.0;

    assert_matches!(
        half::f16::try_from(Scalar::F64(value)),
        Err(ScalarConversionError::Overflow)
    );
}

#[test]
fn f32_to_bf16_overflow() {
    let value = half::bf16::MAX.to_f64() * 2.0;

    assert_matches!(
        half::bf16::try_from(Scalar::F64(value)),
        Err(ScalarConversionError::Overflow)
    );
}

// ============================================================
// Special float values -> half
// ============================================================

#[test]
fn nan_to_f16() {
    let result = half::f16::try_from(Scalar::F64(f64::NAN));

    assert!(result.is_ok());
    assert!(result.unwrap().is_nan());
}

#[test]
fn infinity_to_f16() {
    let result = half::f16::try_from(Scalar::F64(f64::INFINITY));
    println!("{result:?}");
    assert!(result.is_ok());
    assert!(result.unwrap().is_infinite());
}

#[test]
fn into_scalar_option_for_values() {
    assert_eq!(42u8.into_scalar_option(), Some(Scalar::U8(42)));

    assert_eq!((-42i8).into_scalar_option(), Some(Scalar::I8(-42)));

    assert_eq!(12345i32.into_scalar_option(), Some(Scalar::I32(12345)));

    assert_eq!(
        (-999_999i64).into_scalar_option(),
        Some(Scalar::I64(-999_999))
    );

    assert_eq!(
        half::bf16::from_f32(1.5).into_scalar_option(),
        Some(Scalar::BF16(half::bf16::from_f32(1.5)))
    );

    assert_eq!(
        half::f16::from_f32(-2.25).into_scalar_option(),
        Some(Scalar::F16(half::f16::from_f32(-2.25)))
    );

    assert_eq!(3.5f32.into_scalar_option(), Some(Scalar::F32(3.5)));

    assert_eq!((-7.25f64).into_scalar_option(), Some(Scalar::F64(-7.25)));
}

#[test]
fn into_scalar_option_for_some_values() {
    let value: Option<u8> = Some(42);
    assert_eq!(value.into_scalar_option(), Some(Scalar::U8(42)));

    let value: Option<i8> = Some(-42);
    assert_eq!(value.into_scalar_option(), Some(Scalar::I8(-42)));

    let value: Option<i32> = Some(12345);
    assert_eq!(value.into_scalar_option(), Some(Scalar::I32(12345)));

    let value: Option<i64> = Some(-999_999);
    assert_eq!(value.into_scalar_option(), Some(Scalar::I64(-999_999)));

    let value: Option<half::bf16> = Some(half::bf16::from_f32(1.5));
    assert_eq!(
        value.into_scalar_option(),
        Some(Scalar::BF16(half::bf16::from_f32(1.5)))
    );

    let value: Option<half::f16> = Some(half::f16::from_f32(-2.25));
    assert_eq!(
        value.into_scalar_option(),
        Some(Scalar::F16(half::f16::from_f32(-2.25)))
    );

    let value: Option<f32> = Some(3.5);
    assert_eq!(value.into_scalar_option(), Some(Scalar::F32(3.5)));

    let value: Option<f64> = Some(-7.25);
    assert_eq!(value.into_scalar_option(), Some(Scalar::F64(-7.25)));
}

#[test]
fn into_scalar_option_for_none() {
    let value: Option<u8> = None;
    assert_eq!(value.into_scalar_option(), None);

    let value: Option<i8> = None;
    assert_eq!(value.into_scalar_option(), None);

    let value: Option<i32> = None;
    assert_eq!(value.into_scalar_option(), None);

    let value: Option<i64> = None;
    assert_eq!(value.into_scalar_option(), None);

    let value: Option<half::bf16> = None;
    assert_eq!(value.into_scalar_option(), None);

    let value: Option<half::f16> = None;
    assert_eq!(value.into_scalar_option(), None);

    let value: Option<f32> = None;
    assert_eq!(value.into_scalar_option(), None);

    let value: Option<f64> = None;
    assert_eq!(value.into_scalar_option(), None);
}
```

# zero-tensor-rs/src/transform/scale.rs (719 lines)
```rust
use super::{Transform, error::TransformError};
use crate::{core::dataset::item::TensorViewMut, transform::Scalar};

pub struct Scale {
    factor: Scalar,
}

impl Scale {
    pub fn new<T: Into<Scalar>>(factor: T) -> Self {
        Scale {
            factor: factor.into(),
        }
    }
}

impl Transform for Scale {
    fn apply(&self, tensor: &mut TensorViewMut) -> Result<(), TransformError> {
        match tensor {
            TensorViewMut::BF16(t) => {
                let factor: half::bf16 = self.factor.try_into()?;
                t.map_inplace(|x| *x *= factor);
            }
            TensorViewMut::F32(t) => {
                let factor: f32 = self.factor.try_into()?;
                t.map_inplace(|x| *x *= factor);
            }
            TensorViewMut::F64(t) => {
                let factor: f64 = self.factor.try_into()?;

                t.map_inplace(|x| *x *= factor);
            }
            TensorViewMut::U8(t) => {
                let factor = self.factor.try_into()?;

                for x in t.iter() {
                    x.checked_mul(factor).ok_or(TransformError::Overflow)?;
                }

                t.map_inplace(|x| *x = unsafe { x.unchecked_mul(factor) });
            }
            TensorViewMut::F16(t) => {
                let factor: half::f16 = self.factor.try_into()?;

                t.map_inplace(|x| *x *= factor);
            }
            TensorViewMut::I8(t) => {
                let factor = self.factor.try_into()?;

                for x in t.iter() {
                    x.checked_mul(factor).ok_or(TransformError::Overflow)?;
                }

                t.map_inplace(|x| *x = unsafe { x.unchecked_mul(factor) });
            }
            TensorViewMut::I32(t) => {
                let factor = self.factor.try_into()?;

                for x in t.iter() {
                    x.checked_mul(factor).ok_or(TransformError::Overflow)?;
                }

                t.map_inplace(|x| *x = unsafe { x.unchecked_mul(factor) });
            }
            TensorViewMut::I64(t) => {
                let factor = self.factor.try_into()?;

                for x in t.iter() {
                    x.checked_mul(factor).ok_or(TransformError::Overflow)?;
                }

                t.map_inplace(|x| *x = unsafe { x.unchecked_mul(factor) });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::dataset::item::{TensorBatchLayout, TensorDT},
        transform::ScalarConversionError,
    };
    use rstest::rstest;

    macro_rules! bytes {
        ($data:expr) => {{
            let data = $data;
            let len = data.len();
            let raw_bytes: Vec<u8> = bytemuck::pod_collect_to_vec(&data);
            (raw_bytes, len)
        }};
    }

    macro_rules! assert_result {
        ($raw_bytes:expr, $expected:expr, $ty:ty) => {{
            let result: Vec<$ty> = bytemuck::pod_collect_to_vec(&$raw_bytes);

            assert_eq!(result, $expected);
        }};
    }

    // ============================================================
    // FLOATS
    // ============================================================

    #[rstest]
    #[case(TensorDT::BF16)]
    #[case(TensorDT::F16)]
    #[case(TensorDT::F32)]
    #[case(TensorDT::F64)]
    fn scale_float_positive_factor(#[case] dt: TensorDT) {
        let factor = 2.0;

        let (mut raw_bytes, len) = match dt {
            TensorDT::BF16 => {
                let data = vec![
                    half::bf16::from_f64(-2.0),
                    half::bf16::from_f64(0.0),
                    half::bf16::from_f64(3.0),
                ];

                bytes!(data)
            }

            TensorDT::F16 => {
                let data = vec![
                    half::f16::from_f64(-2.0),
                    half::f16::from_f64(0.0),
                    half::f16::from_f64(3.0),
                ];

                bytes!(data)
            }

            TensorDT::F32 => {
                let data = vec![-2.0f32, 0.0, 3.0];

                bytes!(data)
            }

            TensorDT::F64 => {
                let data = vec![-2.0f64, 0.0, 3.0];

                bytes!(data)
            }

            _ => unreachable!(),
        };

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), dt);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        Scale::new(factor).apply(&mut tensor).unwrap();

        match dt {
            TensorDT::BF16 => {
                assert_result!(
                    raw_bytes,
                    vec![
                        half::bf16::from_f64(-4.0),
                        half::bf16::from_f64(0.0),
                        half::bf16::from_f64(6.0),
                    ],
                    half::bf16
                );
            }

            TensorDT::F16 => {
                assert_result!(
                    raw_bytes,
                    vec![
                        half::f16::from_f64(-4.0),
                        half::f16::from_f64(0.0),
                        half::f16::from_f64(6.0),
                    ],
                    half::f16
                );
            }

            TensorDT::F32 => {
                assert_result!(raw_bytes, vec![-4.0f32, 0.0, 6.0], f32);
            }

            TensorDT::F64 => {
                assert_result!(raw_bytes, vec![-4.0f64, 0.0, 6.0], f64);
            }

            _ => unreachable!(),
        }
    }

    #[rstest]
    #[case(TensorDT::BF16)]
    #[case(TensorDT::F16)]
    #[case(TensorDT::F32)]
    #[case(TensorDT::F64)]
    fn scale_float_negative_factor(#[case] dt: TensorDT) {
        let factor = -2.0;

        let (mut raw_bytes, len) = match dt {
            TensorDT::BF16 => {
                let data = vec![
                    half::bf16::from_f64(-2.0),
                    half::bf16::from_f64(0.0),
                    half::bf16::from_f64(3.0),
                ];

                bytes!(data)
            }

            TensorDT::F16 => {
                let data = vec![
                    half::f16::from_f64(-2.0),
                    half::f16::from_f64(0.0),
                    half::f16::from_f64(3.0),
                ];

                bytes!(data)
            }

            TensorDT::F32 => {
                let data = vec![-2.0f32, 0.0, 3.0];

                bytes!(data)
            }

            TensorDT::F64 => {
                let data = vec![-2.0f64, 0.0, 3.0];

                bytes!(data)
            }

            _ => unreachable!(),
        };

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), dt);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        Scale::new(factor).apply(&mut tensor).unwrap();

        match dt {
            TensorDT::BF16 => {
                assert_result!(
                    raw_bytes,
                    vec![
                        half::bf16::from_f64(4.0),
                        half::bf16::from_f64(0.0),
                        half::bf16::from_f64(-6.0),
                    ],
                    half::bf16
                );
            }

            TensorDT::F16 => {
                assert_result!(
                    raw_bytes,
                    vec![
                        half::f16::from_f64(4.0),
                        half::f16::from_f64(0.0),
                        half::f16::from_f64(-6.0),
                    ],
                    half::f16
                );
            }

            TensorDT::F32 => {
                assert_result!(raw_bytes, vec![4.0f32, 0.0, -6.0], f32);
            }

            TensorDT::F64 => {
                assert_result!(raw_bytes, vec![4.0f64, 0.0, -6.0], f64);
            }

            _ => unreachable!(),
        }
    }

    #[rstest]
    #[case(TensorDT::BF16)]
    #[case(TensorDT::F16)]
    #[case(TensorDT::F32)]
    #[case(TensorDT::F64)]
    fn scale_float_zero_factor(#[case] dt: TensorDT) {
        let (mut raw_bytes, len) = match dt {
            TensorDT::BF16 => {
                let data = vec![half::bf16::from_f64(-10.0), half::bf16::from_f64(5.0)];

                bytes!(data)
            }

            TensorDT::F16 => {
                let data = vec![half::f16::from_f64(-10.0), half::f16::from_f64(5.0)];

                bytes!(data)
            }

            TensorDT::F32 => {
                let data = vec![-10.0f32, 5.0];

                bytes!(data)
            }

            TensorDT::F64 => {
                let data = vec![-10.0f64, 5.0];

                bytes!(data)
            }

            _ => unreachable!(),
        };

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), dt);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        Scale::new(0.0).apply(&mut tensor).unwrap();

        match dt {
            TensorDT::BF16 => {
                assert_result!(
                    raw_bytes,
                    vec![half::bf16::from_f64(0.0), half::bf16::from_f64(0.0),],
                    half::bf16
                );
            }

            TensorDT::F16 => {
                assert_result!(
                    raw_bytes,
                    vec![half::f16::from_f64(0.0), half::f16::from_f64(0.0),],
                    half::f16
                );
            }

            TensorDT::F32 => {
                assert_result!(raw_bytes, vec![0.0f32, 0.0], f32);
            }

            TensorDT::F64 => {
                assert_result!(raw_bytes, vec![0.0f64, 0.0], f64);
            }

            _ => unreachable!(),
        }
    }

    // ============================================================
    // INTEGERS
    // ============================================================

    #[rstest]
    #[case(TensorDT::U8)]
    #[case(TensorDT::I8)]
    #[case(TensorDT::I32)]
    #[case(TensorDT::I64)]
    fn scale_integer_positive_factor(#[case] dt: TensorDT) {
        let (mut raw_bytes, len) = match dt {
            TensorDT::U8 => {
                let data = vec![0u8, 1, 10, 100];
                bytes!(data)
            }

            TensorDT::I8 => {
                let data = vec![-10i8, 0, 5, 50];
                bytes!(data)
            }

            TensorDT::I32 => {
                let data = vec![-10i32, 0, 5, 100];
                bytes!(data)
            }

            TensorDT::I64 => {
                let data = vec![-10i64, 0, 5, 100];
                bytes!(data)
            }

            _ => unreachable!(),
        };

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), dt);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        Scale::new(2.0).apply(&mut tensor).unwrap();

        match dt {
            TensorDT::U8 => {
                assert_result!(raw_bytes, vec![0u8, 2, 20, 200], u8);
            }

            TensorDT::I8 => {
                assert_result!(raw_bytes, vec![-20i8, 0, 10, 100], i8);
            }

            TensorDT::I32 => {
                assert_result!(raw_bytes, vec![-20i32, 0, 10, 200], i32);
            }

            TensorDT::I64 => {
                assert_result!(raw_bytes, vec![-20i64, 0, 10, 200], i64);
            }

            _ => unreachable!(),
        }
    }

    #[rstest]
    #[case(TensorDT::I8)]
    #[case(TensorDT::I32)]
    #[case(TensorDT::I64)]
    fn scale_integer_negative_factor(#[case] dt: TensorDT) {
        let (mut raw_bytes, len) = match dt {
            TensorDT::I8 => {
                let data = vec![-10i8, 0, 5];
                bytes!(data)
            }

            TensorDT::I32 => {
                let data = vec![-10i32, 0, 5];
                bytes!(data)
            }

            TensorDT::I64 => {
                let data = vec![-10i64, 0, 5];
                bytes!(data)
            }

            _ => unreachable!(),
        };

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), dt);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        Scale::new(-2.0).apply(&mut tensor).unwrap();

        match dt {
            TensorDT::I8 => {
                assert_result!(raw_bytes, vec![20i8, 0, -10], i8);
            }

            TensorDT::I32 => {
                assert_result!(raw_bytes, vec![20i32, 0, -10], i32);
            }

            TensorDT::I64 => {
                assert_result!(raw_bytes, vec![20i64, 0, -10], i64);
            }

            _ => unreachable!(),
        }
    }

    #[rstest]
    #[case(TensorDT::U8)]
    #[case(TensorDT::I8)]
    #[case(TensorDT::I32)]
    #[case(TensorDT::I64)]
    fn scale_integer_zero_factor(#[case] dt: TensorDT) {
        let (mut raw_bytes, len) = match dt {
            TensorDT::U8 => {
                let data = vec![1u8, 10, 100];
                bytes!(data)
            }

            TensorDT::I8 => {
                let data = vec![-10i8, 0, 10];
                bytes!(data)
            }

            TensorDT::I32 => {
                let data = vec![-100i32, 0, 100];
                bytes!(data)
            }

            TensorDT::I64 => {
                let data = vec![-100i64, 0, 100];
                bytes!(data)
            }

            _ => unreachable!(),
        };

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), dt);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        Scale::new(0.0).apply(&mut tensor).unwrap();

        match dt {
            TensorDT::U8 => {
                assert_result!(raw_bytes, vec![0u8, 0, 0], u8);
            }

            TensorDT::I8 => {
                assert_result!(raw_bytes, vec![0i8, 0, 0], i8);
            }

            TensorDT::I32 => {
                assert_result!(raw_bytes, vec![0i32, 0, 0], i32);
            }

            TensorDT::I64 => {
                assert_result!(raw_bytes, vec![0i64, 0, 0], i64);
            }

            _ => unreachable!(),
        }
    }

    // ============================================================
    // FRACTIONAL FACTOR FOR INTEGER TENSORS
    // ============================================================

    #[rstest]
    #[case(TensorDT::U8)]
    #[case(TensorDT::I8)]
    #[case(TensorDT::I32)]
    #[case(TensorDT::I64)]
    fn integer_scale_rejects_fractional_factor(#[case] dt: TensorDT) {
        let (mut raw_bytes, len) = match dt {
            TensorDT::U8 => {
                let data = vec![1u8, 2, 3];
                bytes!(data)
            }

            TensorDT::I8 => {
                let data = vec![1i8, 2, 3];
                bytes!(data)
            }

            TensorDT::I32 => {
                let data = vec![1i32, 2, 3];
                bytes!(data)
            }

            TensorDT::I64 => {
                let data = vec![1i64, 2, 3];
                bytes!(data)
            }

            _ => unreachable!(),
        };

        let original = raw_bytes.clone();

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), dt);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        let result = Scale::new(2.5).apply(&mut tensor);

        assert!(matches!(
            result,
            Err(TransformError::ScalarConversion(
                ScalarConversionError::FractionalValue
            ))
        ));

        assert_eq!(raw_bytes, original);
    }

    // ============================================================
    // OVERFLOW
    // ============================================================

    #[test]
    fn scale_u8_overflow_does_not_modify_tensor() {
        let data = vec![10u8, 127, 200];
        let expected = data.clone();

        let (mut raw_bytes, len) = bytes!(data);

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), TensorDT::U8);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        let result = Scale::new(2u8).apply(&mut tensor);

        assert!(matches!(result, Err(TransformError::Overflow)));

        assert_result!(raw_bytes, expected, u8);
    }

    #[test]
    fn scale_i8_positive_overflow_does_not_modify_tensor() {
        let data = vec![10i8, 64, 100];
        let expected = data.clone();

        let (mut raw_bytes, len) = bytes!(data);

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), TensorDT::I8);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        let result = Scale::new(2i8).apply(&mut tensor);

        assert!(matches!(result, Err(TransformError::Overflow)));

        assert_result!(raw_bytes, expected, i8);
    }

    #[test]
    fn scale_i8_negative_overflow_does_not_modify_tensor() {
        let data = vec![-100i8, -64, 10];
        let expected = data.clone();

        let (mut raw_bytes, len) = bytes!(data);

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), TensorDT::I8);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        let result = Scale::new(2i8).apply(&mut tensor);

        assert!(matches!(result, Err(TransformError::Overflow)));

        assert_result!(raw_bytes, expected, i8);
    }

    #[test]
    fn scale_i32_overflow_does_not_modify_tensor() {
        let data = vec![i32::MAX, 10];
        let expected = data.clone();

        let (mut raw_bytes, len) = bytes!(data);

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), TensorDT::I32);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        let result = Scale::new(2i32).apply(&mut tensor);

        assert!(matches!(result, Err(TransformError::Overflow)));

        assert_result!(raw_bytes, expected, i32);
    }

    #[test]
    fn scale_i64_overflow_does_not_modify_tensor() {
        let data = vec![i64::MAX, 10];
        let expected = data.clone();

        let (mut raw_bytes, len) = bytes!(data);

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), TensorDT::I64);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        let result = Scale::new(2i64).apply(&mut tensor);

        assert!(matches!(result, Err(TransformError::Overflow)));

        assert_result!(raw_bytes, expected, i64);
    }

    // ============================================================
    // FACTOR CONVERSION OVERFLOW
    // ============================================================

    #[test]
    fn scale_u8_rejects_out_of_range_factor() {
        let data = vec![1u8, 2, 3];
        let expected = data.clone();

        let (mut raw_bytes, len) = bytes!(data);

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), TensorDT::U8);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        let result = Scale::new(300i32).apply(&mut tensor);

        assert!(result.is_err());

        assert_result!(raw_bytes, expected, u8);
    }

    #[test]
    fn scale_i8_rejects_out_of_range_factor() {
        let data = vec![1i8, 2, 3];
        let expected = data.clone();

        let (mut raw_bytes, len) = bytes!(data);

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), TensorDT::I8);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        let result = Scale::new(200i32).apply(&mut tensor);

        assert!(result.is_err());

        assert_result!(raw_bytes, expected, i8);
    }

    #[test]
    fn scale_i32_rejects_out_of_range_factor() {
        let data = vec![1i32, 2, 3];
        let expected = data.clone();

        let (mut raw_bytes, len) = bytes!(data);

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), TensorDT::I32);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        let result = Scale::new(i64::MAX).apply(&mut tensor);

        assert!(result.is_err());

        assert_result!(raw_bytes, expected, i32);
    }
}
```

# zero-tensor-rs/src/transform/standardize.rs (297 lines)
```rust
use super::{Scalar, TensorViewMut, Transform, TransformError, scalar::is_zero::IsZero};

pub struct Standardize {
    mean: Scalar,
    std: Scalar,
}

impl Standardize {
    pub fn new<T: Into<Scalar>, M: Into<Scalar>>(mean: T, std: M) -> Result<Self, TransformError> {
        let mean = mean.into();
        let std = std.into();
        if !mean.is_finite() || !std.is_finite() || std == 0.into() {
            return Err(TransformError::InvalidValue);
        }

        Ok(Self { mean, std })
    }
}

impl Transform for Standardize {
    fn apply(&self, tensor: &mut TensorViewMut) -> Result<(), TransformError> {
        macro_rules! standardize {
            ($ty:ty, $t:expr) => {{
                let mean: $ty = self.mean.try_into()?;
                let std: $ty = self.std.try_into()?;
                if std.eq_zero() {
                    return Err(TransformError::InvalidValue);
                }
                $t.map_inplace(|x| *x = (*x - mean) / std);
            }};
        }
        match tensor {
            TensorViewMut::BF16(t) => standardize!(half::bf16, t),
            TensorViewMut::F16(t) => standardize!(half::f16, t),
            TensorViewMut::F32(t) => standardize!(f32, t),
            TensorViewMut::F64(t) => standardize!(f64, t),
            _ => {
                return Err(TransformError::UnsupportedDtype);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dataset::item::{TensorBatchLayout, TensorDT};
    use rstest::rstest;

    macro_rules! assert_invalid_value {
        ($expr:expr) => {
            assert!(matches!($expr, Err(TransformError::InvalidValue)));
        };
    }

    #[rstest]
    #[case(TensorDT::F16)]
    #[case(TensorDT::BF16)]
    #[case(TensorDT::F32)]
    #[case(TensorDT::F64)]
    fn standardize_float_tensors(#[case] dt: TensorDT) {
        let (mut raw_bytes, len) = match dt {
            TensorDT::F16 => {
                let data = vec![
                    half::f16::from_f32(1.0),
                    half::f16::from_f32(3.0),
                    half::f16::from_f32(5.0),
                ];
                let len = data.len();
                (bytemuck::pod_collect_to_vec(&data), len)
            }

            TensorDT::BF16 => {
                let data = vec![
                    half::bf16::from_f32(1.0),
                    half::bf16::from_f32(3.0),
                    half::bf16::from_f32(5.0),
                ];
                let len = data.len();
                (bytemuck::pod_collect_to_vec(&data), len)
            }

            TensorDT::F32 => {
                let data = vec![1.0f32, 3.0, 5.0];
                let len = data.len();
                (bytemuck::pod_collect_to_vec(&data), len)
            }

            TensorDT::F64 => {
                let data = vec![1.0f64, 3.0, 5.0];
                let len = data.len();
                (bytemuck::pod_collect_to_vec(&data), len)
            }

            _ => unreachable!(),
        };

        let layout = TensorBatchLayout::new(vec![len].into(), vec![1].into(), dt);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        let transform = Standardize::new(Scalar::from(3.0f64), Scalar::from(2.0f64)).unwrap();

        transform.apply(&mut tensor).unwrap();

        match dt {
            TensorDT::F16 => {
                let result: Vec<half::f16> = bytemuck::pod_collect_to_vec(&raw_bytes);

                let expected = vec![
                    half::f16::from_f32(-1.0),
                    half::f16::from_f32(0.0),
                    half::f16::from_f32(1.0),
                ];

                assert_eq!(result, expected);
            }

            TensorDT::BF16 => {
                let result: Vec<half::bf16> = bytemuck::pod_collect_to_vec(&raw_bytes);

                let expected = vec![
                    half::bf16::from_f32(-1.0),
                    half::bf16::from_f32(0.0),
                    half::bf16::from_f32(1.0),
                ];

                assert_eq!(result, expected);
            }

            TensorDT::F32 => {
                let result: Vec<f32> = bytemuck::pod_collect_to_vec(&raw_bytes);

                assert_eq!(result, vec![-1.0, 0.0, 1.0]);
            }

            TensorDT::F64 => {
                let result: Vec<f64> = bytemuck::pod_collect_to_vec(&raw_bytes);

                assert_eq!(result, vec![-1.0, 0.0, 1.0]);
            }

            _ => unreachable!(),
        }
    }

    #[test]
    fn standardize_accepts_mixed_scalar_types() {
        let data = vec![1.0f32, 3.0, 5.0];
        let mut raw_bytes = bytemuck::pod_collect_to_vec(&data);

        let layout = TensorBatchLayout::new(vec![data.len()].into(), vec![1].into(), TensorDT::F32);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        let transform = Standardize::new(Scalar::from(3i32), Scalar::from(2i8)).unwrap();

        transform.apply(&mut tensor).unwrap();

        let result: Vec<f32> = bytemuck::pod_collect_to_vec(&raw_bytes);

        assert_eq!(result, vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn standardize_negative_std() {
        let data = vec![1.0f32, 3.0, 5.0];
        let mut raw_bytes = bytemuck::pod_collect_to_vec(&data);

        let layout = TensorBatchLayout::new(vec![data.len()].into(), vec![1].into(), TensorDT::F32);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        let transform = Standardize::new(Scalar::from(3.0), Scalar::from(-2.0)).unwrap();

        transform.apply(&mut tensor).unwrap();

        let result: Vec<f32> = bytemuck::pod_collect_to_vec(&raw_bytes);

        assert_eq!(result, vec![1.0, -0.0, -1.0]);
    }

    #[rstest]
    #[case(Scalar::from(0u8))]
    #[case(Scalar::from(0i8))]
    #[case(Scalar::from(0i32))]
    #[case(Scalar::from(0i64))]
    #[case(Scalar::from(0.0f32))]
    #[case(Scalar::from(0.0f64))]
    #[case(Scalar::from(half::f16::from_f32(0.0)))]
    #[case(Scalar::from(half::bf16::from_f32(0.0)))]
    fn standardize_rejects_zero_std(#[case] std: Scalar) {
        assert_invalid_value!(Standardize::new(Scalar::from(0.0), std));
    }

    #[rstest]
    #[case(Scalar::from(f32::NAN))]
    #[case(Scalar::from(f64::NAN))]
    #[case(Scalar::from(half::f16::NAN))]
    #[case(Scalar::from(half::bf16::NAN))]
    fn standardize_rejects_nan_mean(#[case] mean: Scalar) {
        assert_invalid_value!(Standardize::new(mean, Scalar::from(1.0),));
    }

    #[rstest]
    #[case(Scalar::from(f32::NAN))]
    #[case(Scalar::from(f64::NAN))]
    #[case(Scalar::from(half::f16::NAN))]
    #[case(Scalar::from(half::bf16::NAN))]
    fn standardize_rejects_nan_std(#[case] std: Scalar) {
        assert_invalid_value!(Standardize::new(Scalar::from(0.0), std,));
    }

    #[rstest]
    #[case(Scalar::from(f32::INFINITY))]
    #[case(Scalar::from(f32::NEG_INFINITY))]
    #[case(Scalar::from(f64::INFINITY))]
    #[case(Scalar::from(f64::NEG_INFINITY))]
    #[case(Scalar::from(half::f16::INFINITY))]
    #[case(Scalar::from(half::f16::NEG_INFINITY))]
    #[case(Scalar::from(half::bf16::INFINITY))]
    #[case(Scalar::from(half::bf16::NEG_INFINITY))]
    fn standardize_rejects_infinite_mean(#[case] mean: Scalar) {
        assert_invalid_value!(Standardize::new(mean, Scalar::from(1.0),));
    }

    #[rstest]
    #[case(Scalar::from(f32::INFINITY))]
    #[case(Scalar::from(f32::NEG_INFINITY))]
    #[case(Scalar::from(f64::INFINITY))]
    #[case(Scalar::from(f64::NEG_INFINITY))]
    #[case(Scalar::from(half::f16::INFINITY))]
    #[case(Scalar::from(half::f16::NEG_INFINITY))]
    #[case(Scalar::from(half::bf16::INFINITY))]
    #[case(Scalar::from(half::bf16::NEG_INFINITY))]
    fn standardize_rejects_infinite_std(#[case] std: Scalar) {
        assert_invalid_value!(Standardize::new(Scalar::from(0.0), std,));
    }

    #[rstest]
    #[case(TensorDT::U8)]
    #[case(TensorDT::I8)]
    #[case(TensorDT::I32)]
    #[case(TensorDT::I64)]
    fn standardize_rejects_integer_tensors(#[case] dt: TensorDT) {
        macro_rules! make_data {
            ($ty:ty) => {{
                let data = vec![1 as $ty, 2 as $ty, 3 as $ty];
                bytemuck::pod_collect_to_vec(&data)
            }};
        }

        let mut raw_bytes = match dt {
            TensorDT::U8 => make_data!(u8),
            TensorDT::I8 => make_data!(i8),
            TensorDT::I32 => make_data!(i32),
            TensorDT::I64 => make_data!(i64),
            _ => unreachable!(),
        };

        let original = raw_bytes.clone();

        let layout = TensorBatchLayout::new(vec![3].into(), vec![1].into(), dt);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        let transform = Standardize::new(Scalar::from(2.0), Scalar::from(1.0)).unwrap();

        let result = transform.apply(&mut tensor);

        assert!(matches!(result, Err(TransformError::UnsupportedDtype)));

        assert_eq!(raw_bytes, original);
    }

    #[test]
    fn standardize_fails_when_std_becomes_zero_after_conversion() {
        let transform = Standardize::new(Scalar::from(0.0f64), Scalar::from(1e-100f64)).unwrap();

        let data = vec![1.0f32, 2.0f32];
        let mut raw_bytes = bytemuck::pod_collect_to_vec(&data);
        let original = raw_bytes.clone();

        let layout = TensorBatchLayout::new(vec![data.len()].into(), vec![1].into(), TensorDT::F32);

        let mut tensor = layout.try_view_mut(&mut raw_bytes).unwrap();

        let result = transform.apply(&mut tensor);

        assert!(matches!(result, Err(TransformError::InvalidValue)));

        drop(tensor);
        assert_eq!(raw_bytes, original);
    }
}
```

# zero-tensor-rs/tests/integration_consumer.py (49 lines)
```python
import sys
import torch
from zero_tensor_py import ZeroTensorConsumer

def main():
    socket_path = sys.argv[1]
    shm_name = sys.argv[2]
    batch_size = int(sys.argv[4])

    print(f"[PyConsumer] Connecting to {socket_path}...")
    with ZeroTensorConsumer(socket_path, shm_name) as consumer:
        for epoch in range(3):
            step = 0
            for batch in consumer:
                b, h, w = batch.shape
                assert b == batch_size, f"Step {step}: Expected B={batch_size}, got {b}"
                
                for i in range(b):
                    item = batch[i]
                    
                    non_zero_mask = (item != 0)
                    
                    if not non_zero_mask.any():
                        continue
                        
                    rows_with_data = non_zero_mask.any(dim=1)
                    cols_with_data = non_zero_mask.any(dim=0)
                    
                    last_row = rows_with_data.nonzero(as_tuple=True)[0][-1].item()
                    last_col = cols_with_data.nonzero(as_tuple=True)[0][-1].item()
                    
                    if last_row < h - 1:
                        assert torch.all(item[last_row + 1:, :] == 0), \
                            f"Step {step}, Item {i}: Bottom padding is not zero!"
                            
                    if last_col < w - 1:
                        assert torch.all(item[:, last_col + 1:] == 0), \
                            f"Step {step}, Item {i}: Right padding is not zero!"
                
                print(f"Step {step}: Verified batch shape [{b}, {h}, {w}] with correct padding.")
                step += 1
                
            print(f"Epoch {epoch} done")
                
    print("Dynamic batching integration test PASSED.")
    sys.exit(0)

if __name__ == "__main__":
    main()
```

# zero-tensor-rs/tests/integration_producer.rs (149 lines)
```rust
use std::process::Command;
use std::thread;
use std::time::Duration;
use tempfile::tempdir;
use zero_tensor_lib::core::{
    buffer::get_dt_size,
    dataset::{
        ZTDatasetError, ZeroTensorDataset,
        item::{ShapeType, ShapeVec, StrideVec, TensorBatchLayout, TensorDT},
    },
    producer::ZeroTensorProducerBuilder,
};

#[derive(Debug)]
struct TestError(String);
impl std::fmt::Display for TestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for TestError {}
impl ZTDatasetError for TestError {
    fn index(&self) -> Option<usize> {
        None
    }
}

struct DynamicDataset {
    shapes: Vec<(ShapeType, ShapeType)>, // (H, W)
}

impl DynamicDataset {
    fn new(num_items: usize) -> Self {
        let mut rng = fastrand::Rng::new();
        let shapes = (0..num_items)
            .map(|_| (rng.usize(2..6), rng.usize(2..6)))
            .collect();
        Self { shapes }
    }
}

impl ZeroTensorDataset for DynamicDataset {
    type Error = TestError;

    fn len(&self) -> usize {
        self.shapes.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get_batch_layout(&self, indices: &[usize]) -> Result<TensorBatchLayout, Self::Error> {
        if indices.is_empty() {
            return Err(TestError("Empty batch".into()));
        }

        let (max_h, max_w) = indices
            .iter()
            .map(|&i| self.shapes[i])
            .fold((0, 0), |(mh, mw), (h, w)| (mh.max(h), mw.max(w)));

        let mut shape = ShapeVec::new();
        shape.push(max_h);
        shape.push(max_w);

        let mut strides = StrideVec::new();
        strides.push(max_w);
        strides.push(1);

        Ok(TensorBatchLayout::new(shape, strides, TensorDT::F32))
    }

    fn write_item_into(&self, idx: usize, buf: &mut [u8]) -> Result<(), Self::Error> {
        let (h, w) = self.shapes[idx];
        let total_els = (h * w) as usize;

        let f32_buf =
            unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut f32, total_els) };

        for r in 0..h {
            for c in 0..w {
                f32_buf[(r * w + c) as usize] = (r * 10 + c + idx * 100) as f32;
            }
        }
        Ok(())
    }
}

#[test]
fn test_dynamic_batching_e2e() {
    let dir = tempdir().unwrap();
    let socket_path = dir.path().join("dyn_test.sock");
    let shm_name = "zt_dyn_integration";

    let batch_size = 4;
    let steps = 3;
    let dataset = DynamicDataset::new(batch_size * steps);

    let max_item_bytes = (5 * 5 * get_dt_size(TensorDT::F32)) as usize;
    let slot_size = (max_item_bytes * batch_size) + 4096;

    let mut producer = ZeroTensorProducerBuilder::new(slot_size as u64, shm_name, &socket_path)
        .num_slots(3)
        .build()
        .expect("Failed to init producer");

    let consumer_socket = socket_path.clone();
    let consumer_shm = shm_name.to_string();

    let python_handle = thread::spawn(move || {
        thread::sleep(Duration::from_millis(200));

        let root_dir = std::env::current_dir()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let python_project_dir = root_dir.join("zero-tensor-py");
        let consumer_script = root_dir.join("zero-tensor-rs/tests/integration_consumer.py");
        let python_path = python_project_dir.join("src");

        let status = Command::new("uv")
            .arg("--directory")
            .arg(&python_project_dir)
            .arg("run")
            .arg("python3")
            .arg(&consumer_script)
            .arg(&consumer_socket)
            .arg(&consumer_shm)
            .arg(slot_size.to_string())
            .arg(batch_size.to_string())
            .arg(steps.to_string())
            .env("PYTHONPATH", python_path)
            .status()
            .expect("Failed to execute python consumer");

        assert!(
            status.success(),
            "Python consumer failed with status: {:?}",
            status
        );
    });

    producer
        .start_streaming(&dataset, batch_size)
        .expect("Streaming failed");
    python_handle.join().expect("Consumer thread panicked");
}
```

# zero-tensor-rs/tests/integration_signals.rs (48 lines)
```rust
use std::{
    path::Path,
    process::{Command, Stdio},
    thread,
    time::Duration,
};

use tempfile::tempdir;

#[test]
fn test_cleanup_on_intercept() {
    let dir = tempdir().unwrap();
    let sock_path = dir.path().join("integration_test.sock");
    let shm_name = "zt_integration_test_shm";

    let bin_path = env!("CARGO_BIN_EXE_throughput_bench");

    let mut child = Command::new(bin_path)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("Failed to run executable");

    thread::sleep(Duration::from_millis(500));

    let pid = child.id();
    unsafe {
        libc::kill(pid as i32, libc::SIGINT);
    }
    let status = child.wait().expect("Failed to wait on child process");
    assert!(
        !status.success(),
        "Process should exit with non-zero code on SIGINT"
    );
    assert!(
        !sock_path.exists(),
        "Socket file must be cleaned up on SIGINT!"
    );

    #[cfg(target_os = "linux")]
    {
        let shm_path = format!("/dev/shm/{}", shm_name);
        assert!(
            !Path::new(&shm_path).exists(),
            "SHM segment must be unlinked on SIGINT!"
        );
    }
}
```

