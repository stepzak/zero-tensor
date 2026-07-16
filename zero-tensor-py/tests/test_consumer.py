import struct
import socket
import threading
import mmap
import os
import pytest
import torch
from zero_tensor_py.protocol import TensorHeaderParser, DT_F32
from zero_tensor_py.consumer import ZeroTensorConsumer

class MockRustProducerServer:
    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self.server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if os.path.exists(socket_path):
            os.remove(socket_path)
        self.server_sock.bind(socket_path)
        self.server_sock.listen(1)
        self.thread = None
        self.should_stop = False

    def start(self, offsets_to_send):
        def loop():
            self.server_sock.settimeout(1.0)
            try:
                conn, _ = self.server_sock.accept()
            except socket.timeout:
                return

            with conn:
                for offset in offsets_to_send:
                    conn.sendall(f"READY {offset}\n".encode('utf-8'))
                    
                    resp = b""
                    while b"\n" not in resp:
                        chunk = conn.recv(1)
                        if not chunk:
                            break
                        resp += chunk
                    
                    if not resp.startswith(b"RELEASE"):
                        print(f"Protocol violation in test: expected RELEASE, got {resp}")
                        break

        self.thread = threading.Thread(target=loop)
        self.thread.start()

    def stop(self):
        self.should_stop = True
        if self.thread:
            self.thread.join()
        self.server_sock.close()
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)



@pytest.fixture
def temp_ipc_env(tmp_path):
    socket_path = str(tmp_path / "test_zero_tensor.sock")
    shm_name = "zt_pytest_shm"
    shm_path = f"/dev/shm/{shm_name}"
    
    yield socket_path, shm_name, shm_path
    
    if os.path.exists(shm_path):
        os.remove(shm_path)


def test_tensor_header_parser_compat():
    buffer = bytearray(256)
    offset = 0

    dt = DT_F32
    ndims = 3 
    reserved = b"\x00" * 6
    
    shape = [2, 2, 3]
    strides_bytes = [24, 12, 4]

    header_bytes = struct.pack("<BB6s", dt, ndims, reserved)
    buffer[offset:offset+8] = header_bytes

    struct.pack_into("<3I", buffer, offset + 8, *shape)
    struct.pack_into("<3I", buffer, offset + 20, *strides_bytes)

    p_shape, p_strides, p_dtype, data_offset, data_size = TensorHeaderParser.parse_meta(buffer, offset)

    assert p_shape == [2, 2, 3]
    assert p_strides == [6, 3, 1]
    assert p_dtype == torch.float32
    assert data_offset == offset + 8 + 12 + 12
    assert data_size == 2 * 2 * 3 * 4


def test_consumer_end_to_end_lifecycle(temp_ipc_env):
    socket_path, shm_name, shm_path = temp_ipc_env
    slot_size = 1024
    nslots = 2
    total_size = slot_size * nslots

    with open(shm_path, "wb") as f:
        f.write(b"\x00" * total_size)

    with open(shm_path, "r+b") as f:
        mem = mmap.mmap(f.fileno(), total_size)
        
        offset_0 = 0
        struct.pack_into("<BB6s", mem, offset_0, DT_F32, 2, b"\x00" * 6)
        struct.pack_into("<2I", mem, offset_0 + 8, 2, 2)
        struct.pack_into("<2I", mem, offset_0 + 16, 8, 4)
        struct.pack_into("<4f", mem, offset_0 + 24, 1.0, 2.0, 3.0, 4.0)

        offset_1 = slot_size
        struct.pack_into("<BB6s", mem, offset_1, DT_F32, 2, b"\x00" * 6)
        struct.pack_into("<2I", mem, offset_1 + 8, 2, 2)
        struct.pack_into("<2I", mem, offset_1 + 16, 8, 4)
        struct.pack_into("<4f", mem, offset_1 + 24, 5.0, 6.0, 7.0, 8.0)
        mem.close()

    server = MockRustProducerServer(socket_path)
    server.start(offsets_to_send=[offset_0, offset_1])

    try:
        batches_collected = []
        with ZeroTensorConsumer(socket_path, shm_name, slot_size, nslots=nslots) as consumer:
            for batch in consumer:
                batches_collected.append(batch.clone())

        assert len(batches_collected) == 2
        
        assert batches_collected[0].shape == (2, 2)
        assert batches_collected[0].dtype == torch.float32
        assert torch.allclose(batches_collected[0], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

        assert torch.allclose(batches_collected[1], torch.tensor([[5.0, 6.0], [7.0, 8.0]]))

    finally:
        server.stop()