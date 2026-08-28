import time
import pytest
from zero_tensor_py.consumer import ZeroTensorConsumer
import zero_tensor_py.exceptions as exc
from mocks.producer_mock import MockAsyncProducer
from helpers import _make_batch


def test_async_consumer_empty_stream(temp_ipc_env):
    socket_path, shm_name, shm_path = temp_ipc_env
    nslots = 2
    slot_size = 4096
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

def test_consumer_ring_buffer_wrap_around(temp_ipc_env):
    socket_path, shm_name, shm_path = temp_ipc_env
    nslots = 2
    slot_size = 4096
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
                assert "data" in batch
                collected.append(batch["data"].clone())
                if len(collected) >= len(batches):
                    break
        assert len(collected) == 5
       
    finally:
        server.stop()

def test_consumer_detects_producer_death(temp_ipc_env):
    socket_path, shm_name, shm_path = temp_ipc_env
    server = MockAsyncProducer(socket_path, shm_path, nslots=2, slot_size=4096)
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
    server = MockAsyncProducer(sock_path, shm_path, nslots=2, slot_size=4096, **{fail_kwarg: True})
    server.start([])
    with pytest.raises(r_exc):
        with ZeroTensorConsumer(sock_path, shm_name) as _:
            assert False, "Should fail"

def test_wrong_dt(temp_ipc_env):
    sock_path, shm_name, shm_path = temp_ipc_env
    server = MockAsyncProducer(sock_path, shm_path, nslots=2, slot_size=4096)
    server.start([_make_batch([2, 2], [1.0, 2.0, 3.0, 4.0], dt=32)] * 3)
    with ZeroTensorConsumer(sock_path, shm_name) as cons:
        it = iter(cons)
        with pytest.raises(exc.MalformedMessageError):
            next(it)