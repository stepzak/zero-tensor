import mmap
import os
import socket
from typing import Generator, Optional
from .protocol import TensorHeaderParser
import gc

import torch


class ZeroTensorConsumer:
    def __init__(self, socket_path: str, shm_name: str, slot_size: int, nslots: int = 2):
        self.socket_path = socket_path
        self.shm_name = os.path.join("/dev/shm", shm_name)
        self.slot_size = slot_size
        self.total_size = slot_size * nslots
        self.nslots = nslots

        self.sock: Optional[socket.socket] = None
        self.shm_file = None
        self.mem: Optional[mmap.mmap] = None

    def close(self):
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
        if self.sock is not None:
            self.sock.close()
            self.sock = None
        if self.shm_file is not None:
            self.shm_file.close()
            self.shm_file = None

    def __enter__(self) -> "ZeroTensorConsumer":
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.sock.connect(self.socket_path)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self.socket_path}: {e}")
        try:
            self.shm_file = open(self.shm_name, "r+b")
            self.mem = mmap.mmap(self.shm_file.fileno(), self.total_size)
        except Exception as e:
            self.close()
            raise OSError(f"Failed to open {self.shm_name}: {e}")
        
    def __iter__(self) -> Generator[torch.Tensor, None, None]:
        if self.sock is None or self.shm_file is None:
            raise RuntimeError("Consumer is not connected. Use 'with' or 'connect'")

        buf = bytearray()
        while True:
            c = self.sock.recv(1)
            if not c:
                break

            if c == b'\n':
                msg = buf.decode("utf-8").strip()
                buf.clear()
                if msg.startswith("READY"):
                    try:
                        offset = int(msg.split()[1])
                    except (IndexError, ValueError):
                        raise RuntimeError(f"Malformed message from receiver: {msg}. Expected: READY <offset>")
                
                    shape, strides, dt, data_offset, data_size = TensorHeaderParser.parse_meta(self.mem, offset)

                    raw_view = memoryview(self.mem)[data_offset:data_offset+data_size]
                    flat_tensor = torch.frombuffer(raw_view, dtype = dt)
                    batch_tensor = torch.as_strided(flat_tensor, shape, strides)
                    yield batch_tensor

                    self.sock.sendall(b"RELEASE\n")
            else:
                buf.extend(c)