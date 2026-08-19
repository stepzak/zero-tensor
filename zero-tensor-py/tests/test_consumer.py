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
