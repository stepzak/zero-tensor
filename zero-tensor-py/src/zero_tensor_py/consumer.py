import collections
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
        self._slot_events = None
        self._release_tail = 0
        self._current_slot_idx = 0
        self._pending_releases = None
        self._keys = None

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
            if torch.cuda.is_available():
                self._slot_events = [torch.cuda.Event(enable_timing=False) for _ in range(self.nslots)]
            else:
                self._slot_events = None
            self._pending_releases: collections.deque[tuple[int, torch.cuda.Event]] = collections.deque()
            self._release_tail = self._load_tail()
            self._current_slot_idx = None
            self._keys = self.handshake_dict["keys"]
            
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
        if self._pending_releases is not None:
            self._pending_releases.clear()
        self._slot_events = None
        self._current_slot_idx = None

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
            
        read_tail = self._load_tail()
        try:
            while True:
                is_running = self._load_is_running()
                if is_running == 0:
                    break

                slot_idx = read_tail % self.nslots
                if read_tail - self._release_tail >= self.nslots:
                    self._drain_releases(force_up_to=read_tail - self.nslots + 1)

                head = self._load_head()
                while head <= read_tail:
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

                self._current_slot_idx = slot_idx
                self._pending_releases.append((slot_idx, None))

                try:
                    yield batch_tensor
                finally:
                    self._current_slot_idx = None
                    read_tail += 1
                    self._drain_releases(self._release_tail)
        finally:
            if self._pending_releases:
                self._drain_releases(force_up_to=self._release_tail + len(self._pending_releases))
            self._current_slot_idx = None

    def to_device(self, tensor: torch.Tensor,
                    device: torch.device | None = None,
                    dtype: torch.dtype | None = None,
                    non_blocking: bool = False,
                    copy: bool = False,
                    *,
                    memory_format: torch.memory_format | None = None,
                    stream: torch.cuda.Stream | None = None,
                   ) -> torch.Tensor:
        if self._current_slot_idx is None:
            raise RuntimeError("to_device() called outside active iteration")
        
        gpu_tensor = tensor.to(device = device, 
                                dtype = dtype,
                                non_blocking = non_blocking,
                                copy = copy, 
                                memory_format=memory_format)
        
        is_view = not copy and self._slot_events is not None and tensor.data_ptr() == gpu_tensor.data_ptr()

        if is_view and non_blocking and self._slot_events is not None:
            if not stream:
                stream = torch.cuda.current_stream()
            idx = self._current_slot_idx
            event = self._slot_events[idx]
            event.record(stream)
            if self._pending_releases and self._pending_releases[-1][0] == idx:
                self._pending_releases[-1] = (idx, event)
                
        return gpu_tensor
        
    def _drain_releases(self, force_up_to: int) -> None:
        while self._pending_releases:
            _, event = self._pending_releases[0]
            must_wait = self._release_tail < force_up_to
            if event is not None:
                if not event.query():
                    if not must_wait:
                        break
                    event.synchronize()

            self._pending_releases.popleft()
            self._release_tail += 1
            self._store_tail(self._release_tail)

