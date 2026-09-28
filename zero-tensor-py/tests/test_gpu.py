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

    def test_copy_records_event(self, temp_ipc_env):
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
                        assert event is not None
                        break
        finally:
            producer.stop()

    def test_dtype_change_records_event(self, temp_ipc_env):
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
                        assert event is not None
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
@pytest.mark.parametrize("as_dict", [False, True])
@pytest.mark.parametrize("copy,dtype", [(False, None), (True, None), (False, torch.float16)])
def test_async_transfer_holds_slot_until_copy_completes(as_dict, copy, dtype):
    import collections
    from unittest.mock import Mock

    consumer = ZeroTensorConsumer("unused", "unused")
    consumer._current_slot_idx = 0
    consumer._slot_events = [None]
    consumer._pending_releases = collections.deque([(0, None)])
    consumer._store_tail = Mock()
    source = torch.arange(4, dtype=torch.float32)
    with MockedGPU():
        result = consumer.to_device(
            {"image": source} if as_dict else source,
            device="cuda", non_blocking=True, copy=copy, dtype=dtype,
        )
        destination = result["image"] if as_dict else result
        assert destination.data_ptr() != source.data_ptr()
        event = consumer._pending_releases[0][1]
        assert event is not None
        event._completed = False
        consumer._drain_releases(0)
        consumer._store_tail.assert_not_called()
        assert consumer._release_tail == 0
        event._completed = True
        consumer._drain_releases(0)
        consumer._store_tail.assert_called_once_with(1)


def test_explicit_stream_runs_copy_and_waits_for_previous_transfer():
    import collections
    from contextlib import contextmanager
    from unittest.mock import Mock
    from mocks.cuda_mock import MockStream, mock_tensor_to

    consumer = ZeroTensorConsumer("unused", "unused")
    consumer._current_slot_idx = 0
    consumer._slot_events = [None]
    consumer._pending_releases = collections.deque([(0, None)])
    source = torch.ones(4)
    first_stream, second_stream = MockStream(), MockStream()
    second_stream.wait_event = Mock()
    active = []

    @contextmanager
    def stream_context(stream):
        active.append(stream)
        try:
            yield
        finally:
            active.pop()

    copies = []

    def checked_copy(tensor, **kwargs):
        copies.append(active[-1])
        return mock_tensor_to(tensor, **kwargs)

    with MockedGPU(), patch("torch.cuda.stream", stream_context), patch.object(torch.Tensor, "to", checked_copy):
        consumer.to_device(source, device="cuda", non_blocking=True, stream=first_stream)
        first_event = consumer._pending_releases[0][1]
        consumer.to_device(source, device="cuda", non_blocking=True, stream=second_stream)
        second_event = consumer._pending_releases[0][1]
        assert copies == [first_stream, second_stream]
        second_stream.wait_event.assert_called_once_with(first_event)
        assert first_event.stream is first_stream
        assert second_event.stream is second_stream
        assert second_event is not first_event


def test_forced_release_waits_for_incomplete_event():
    import collections
    from unittest.mock import Mock

    consumer = ZeroTensorConsumer("unused", "unused")
    event = MockEvent()
    event._completed = False
    consumer._pending_releases = collections.deque([(0, event)])
    consumer._store_tail = Mock(side_effect=lambda _: pytest.fail("Released before completion") if not event.query() else None)
    consumer._drain_releases(1)
    assert event.query()
    consumer._store_tail.assert_called_once_with(1)


def test_close_waits_for_active_transfer_before_unmapping():
    import collections
    from unittest.mock import Mock

    consumer = ZeroTensorConsumer("unused", "unused")
    event = MockEvent()
    event._completed = False
    consumer._pending_releases = collections.deque([(0, event)])
    consumer._store_tail = Mock()
    consumer.mem = Mock()
    consumer.mem.close.side_effect = lambda: pytest.fail("Unmapped during transfer") if not event.query() else None
    consumer.close()
    assert event.query()
    consumer._store_tail.assert_called_once_with(1)
