import pytest
import torch
from unittest.mock import patch
from zero_tensor_py import ZeroTensorConsumer
from mocks.producer_mock import MockAsyncProducer
from mocks.cuda_mock import MockedGPU, MockEvent
from helpers import _make_batch

NSLOTS = 2
SLOT_SIZE = 4096

class TestSlotEventsInitialization:
    def test_slot_events_initialized_when_cuda_available(self, temp_ipc_env):
        socket_path, shm_name, shm_path = temp_ipc_env
        batches = [_make_batch([2, 2], [1.0, 2.0, 3.0, 4.0])]
        producer = MockAsyncProducer(socket_path, shm_path, NSLOTS, SLOT_SIZE)
        producer.start(batches)
        try:
            with MockedGPU():
                with ZeroTensorConsumer(socket_path, shm_name) as consumer:
                    assert consumer._slot_events is not None
                    assert len(consumer._slot_events) == NSLOTS
                    assert all(isinstance(e, MockEvent) for e in consumer._slot_events)
                    for _ in consumer:
                        pass
        finally:
            producer.stop()

    def test_slot_events_none_when_cuda_unavailable(self, temp_ipc_env):
        socket_path, shm_name, shm_path = temp_ipc_env
        batches = [_make_batch([2, 2], [1.0, 2.0, 3.0, 4.0])]
        producer = MockAsyncProducer(socket_path, shm_path, NSLOTS, SLOT_SIZE)
        producer.start(batches)
        try:
            with patch('torch.cuda.is_available', return_value=False):
                with ZeroTensorConsumer(socket_path, shm_name) as consumer:
                    assert consumer._slot_events is None
                    for _ in consumer:
                        pass
        finally:
            producer.stop()


class TestToDeviceEventRecording:
    def test_non_blocking_records_event(self, temp_ipc_env):
        socket_path, shm_name, shm_path = temp_ipc_env
        batches = [_make_batch([2, 2], [1.0, 2.0, 3.0, 4.0])]
        producer = MockAsyncProducer(socket_path, shm_path, NSLOTS, SLOT_SIZE)
        producer.start(batches)
        try:
            with MockedGPU():
                with ZeroTensorConsumer(socket_path, shm_name) as consumer:
                    for batch in consumer:
                        consumer.to_device(batch, device='cuda', non_blocking=True)
                        assert len(consumer._pending_releases) > 0
                        slot_idx, event = consumer._pending_releases[-1]
                        assert event is not None
                        assert isinstance(event, MockEvent)
                        assert event._recorded
                        break
        finally:
            producer.stop()

    def test_blocking_does_not_record_event(self, temp_ipc_env):
        socket_path, shm_name, shm_path = temp_ipc_env
        batches = [_make_batch([2, 2], [1.0, 2.0, 3.0, 4.0])]
        producer = MockAsyncProducer(socket_path, shm_path, NSLOTS, SLOT_SIZE)
        producer.start(batches)
        try:
            with MockedGPU():
                with ZeroTensorConsumer(socket_path, shm_name) as consumer:
                    for batch in consumer:
                        consumer.to_device(batch, device='cuda', non_blocking=False)
                        slot_idx, event = consumer._pending_releases[-1]
                        assert event is None
                        break
        finally:
            producer.stop()

    def test_copy_does_not_record_event(self, temp_ipc_env):
        socket_path, shm_name, shm_path = temp_ipc_env
        batches = [_make_batch([2, 2], [1.0, 2.0, 3.0, 4.0])]
        producer = MockAsyncProducer(socket_path, shm_path, NSLOTS, SLOT_SIZE)
        producer.start(batches)
        try:
            with MockedGPU():
                with ZeroTensorConsumer(socket_path, shm_name) as consumer:
                    for batch in consumer:
                        consumer.to_device(batch, device='cuda', non_blocking=True, copy=True)
                        slot_idx, event = consumer._pending_releases[-1]
                        assert event is None
                        break
        finally:
            producer.stop()

    def test_dtype_change_does_not_record_event(self, temp_ipc_env):
        socket_path, shm_name, shm_path = temp_ipc_env
        batches = [_make_batch([2, 2], [1.0, 2.0, 3.0, 4.0])]
        producer = MockAsyncProducer(socket_path, shm_path, NSLOTS, SLOT_SIZE)
        producer.start(batches)
        try:
            with MockedGPU():
                with ZeroTensorConsumer(socket_path, shm_name) as consumer:
                    for batch in consumer:
                        result = consumer.to_device(
                            batch, device='cuda', non_blocking=True, dtype=torch.float16
                        )
                        assert result["data"].data_ptr() != batch["data"].data_ptr()
                        slot_idx, event = consumer._pending_releases[-1]
                        assert event is None
                        break
        finally:
            producer.stop()


class TestToDeviceOutsideIteration:
    def test_to_device_outside_iteration_raises(self, temp_ipc_env):
        socket_path, shm_name, shm_path = temp_ipc_env
        batches = [_make_batch([2, 2], [1.0, 2.0, 3.0, 4.0])]
        producer = MockAsyncProducer(socket_path, shm_path, NSLOTS, SLOT_SIZE)
        producer.start(batches)
        try:
            with MockedGPU():
                with ZeroTensorConsumer(socket_path, shm_name) as consumer:
                    for _ in consumer:
                        pass
                    dummy_tensor = torch.zeros(2, 2)
                    with pytest.raises(RuntimeError, match="outside active iteration"):
                        consumer.to_device(dummy_tensor, device='cuda', non_blocking=True)
        finally:
            producer.stop()


class TestDrainReleases:
    def test_pending_releases_accumulate(self, temp_ipc_env):
        socket_path, shm_name, shm_path = temp_ipc_env
        batches = [
            _make_batch([2, 2], [1.0, 2.0, 3.0, 4.0]),
            _make_batch([2, 2], [5.0, 6.0, 7.0, 8.0]),
        ]
        producer = MockAsyncProducer(socket_path, shm_path, NSLOTS, SLOT_SIZE)
        producer.start(batches)
        try:
            with MockedGPU():
                with ZeroTensorConsumer(socket_path, shm_name) as consumer:
                    initial_release_tail = consumer._release_tail
                    count = 0
                    for batch in consumer:
                        consumer.to_device(batch, device='cuda', non_blocking=True)
                        count += 1
                        if count >= 2:
                            break
                    assert consumer._release_tail > initial_release_tail
        finally:
            producer.stop()

    def test_release_tail_advances_after_iteration(self, temp_ipc_env):
        socket_path, shm_name, shm_path = temp_ipc_env
        batches = [_make_batch([2, 2], [1.0, 2.0, 3.0, 4.0])]
        producer = MockAsyncProducer(socket_path, shm_path, NSLOTS, SLOT_SIZE)
        producer.start(batches)
        try:
            with MockedGPU():
                with ZeroTensorConsumer(socket_path, shm_name) as consumer:
                    initial_release_tail = consumer._release_tail
                    for batch in consumer:
                        consumer.to_device(batch, device='cuda', non_blocking=True)
                        break
                    assert consumer._release_tail > initial_release_tail
        finally:
            producer.stop()


class TestDataRacePrevention:
    def test_previous_batch_remains_valid(self, temp_ipc_env):
        socket_path, shm_name, shm_path = temp_ipc_env
        batches = [
            _make_batch([2, 2], [1.0, 2.0, 3.0, 4.0]),
            _make_batch([2, 2], [5.0, 6.0, 7.0, 8.0]),
        ]
        producer = MockAsyncProducer(socket_path, shm_path, NSLOTS, SLOT_SIZE)
        producer.start(batches)
        try:
            with MockedGPU():
                with ZeroTensorConsumer(socket_path, shm_name) as consumer:
                    prev_batch = None
                    count = 0
                    for batch in consumer:
                        gpu_batch = consumer.to_device(batch, device='cuda', non_blocking=True)
                        if prev_batch is not None:
                            _ = prev_batch["data"].sum().item()
                        prev_batch = gpu_batch
                        count += 1
                        if count >= 2:
                            break
                    assert prev_batch is not None
                    _ = prev_batch["data"].sum().item()
        finally:
            producer.stop()


class TestCpuFallback:
    def test_to_device_without_gpu_returns_tensor(self, temp_ipc_env):
        socket_path, shm_name, shm_path = temp_ipc_env
        batches = [_make_batch([2, 2], [1.0, 2.0, 3.0, 4.0])]
        producer = MockAsyncProducer(socket_path, shm_path, NSLOTS, SLOT_SIZE)
        producer.start(batches)
        try:
            with patch('torch.cuda.is_available', return_value=False):
                with ZeroTensorConsumer(socket_path, shm_name) as consumer:
                    for batch in consumer:
                        result = consumer.to_device(batch, device='cpu')
                        assert result is not None
                        assert torch.allclose(result["data"], batch["data"])
                        slot_idx, event = consumer._pending_releases[-1]
                        assert event is None
                        break
        finally:
            producer.stop()