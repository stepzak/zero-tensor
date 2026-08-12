import gc
import mmap
import os
import select
import socket
import struct
import time
from typing import Generator, Optional

import torch
from zero_tensor_py.protocol import TensorHeaderParser

VERSION = "0.5.0"
_CONTROL_START_MSG = b"START\n"
_CONTROL_STOP_MSG = b"STOP\n"
_CONTROL_NEXT_EPOCH_MSG = b"EPOCH_DONE\n"
_SOCK_WAIT_POLL_TIMEOUT = 0.00001

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

    def _parse_handshake(self, handshake_str: str):
        parts = handshake_str.strip().split()
        if not parts or parts[0] != "ZT":
            raise ValueError(f"Invalid handshake protocol: {handshake_str}")
        if parts[1] != VERSION:
            raise RuntimeError(f"Invalid protocol version, consumer is {VERSION}, producer is {parts[1]}")
        for part in parts[2:]:
            if "=" in part:
                key, val = part.split("=", 1)
                self.handshake_dict[key] = int(val)
        
        self.cb_size = self.handshake_dict["cb_size"]
        self.head_offset = self.handshake_dict["head_offset"]
        self.tail_offset = self.handshake_dict["tail_offset"]
        self.is_running_offset = self.handshake_dict["is_running_offset"]
        
        self.header_size = self.handshake_dict["header_size"]
        self.dt_offset = self.handshake_dict["dt_offset"]
        self.ndims_offset = self.handshake_dict["ndims_offset"]
        self.is_ready_offset = self.handshake_dict["is_ready_offset"]
        self.shape_type_size = self.handshake_dict["shape_type_size"]

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.sock.connect(self.socket_path)
            self.sock.sendall(_CONTROL_START_MSG)
            
            handshake_bytes = b""
            while b"\n" not in handshake_bytes:
                chunk = self.sock.recv(4096)
                if not chunk:
                    raise ConnectionError("Producer closed connection during handshake")
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
            
            self.sock.setblocking(False)
            
        except Exception as e:
            self.close()
            raise ConnectionError(f"Failed to connect and initialize: {e}")

    def close(self):
        if self.mem is not None and self.is_running_offset > 0:
            try:
                self.mem[self.is_running_offset:self.is_running_offset + 8] = struct.pack("<Q", 0)
            except Exception:
                pass

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
        
    def __iter__(self) -> Generator[torch.Tensor, None, None]:
        if self.sock is None or self.mem is None:
            raise RuntimeError("Consumer is not connected. Use 'with' or 'connect'")
        return self._iter_epoch()
    
    def _iter_epoch(self) -> Generator[torch.Tensor, None, None]:
        if self.mem is None:
            raise RuntimeError("Memory not mapped")
            
        tail = struct.unpack_from("<Q", self.mem, self.tail_offset)[0]
        
        while True:
            is_running = struct.unpack_from("<Q", self.mem, self.is_running_offset)[0]
            if is_running == 0:
                break
            head = struct.unpack_from("<Q", self.mem, self.head_offset)[0]
            while head <= tail:
                try:
                    readable, _, _ = select.select([self.sock], [], [], _SOCK_WAIT_POLL_TIMEOUT)
                    if readable:
                        is_running = struct.unpack_from("<Q", self.mem, self.is_running_offset)[0]
                        if is_running == 0:
                            return
                        chunk = self.sock.recv(1024)
                        if chunk == b"":
                            raise ConnectionAbortedError("Producer disonnected")
                        if _CONTROL_NEXT_EPOCH_MSG in chunk:
                            return
                    head = struct.unpack_from("<Q", self.mem, self.head_offset)[0]
                except BlockingIOError:
                    pass
                continue
                
            slot_idx = tail % self.nslots
            slot_offset = self.cb_size + (slot_idx * self.slot_size)
            
            if not TensorHeaderParser.is_slot_ready(self.mem, slot_offset, self.is_ready_offset):
                time.sleep(_SOCK_WAIT_POLL_TIMEOUT)
                continue
                
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
                struct.Struct("<Q").pack_into(self.mem, self.tail_offset, tail)