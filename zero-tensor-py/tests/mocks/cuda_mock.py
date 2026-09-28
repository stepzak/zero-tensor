import torch
from contextlib import nullcontext
from unittest.mock import patch


class MockEvent:
    def __init__(self, enable_timing=False):
        self.enable_timing = enable_timing
        self._recorded = False
        self._completed = True

    def record(self, stream=None):
        self._recorded = True
        self.stream = stream

    def synchronize(self):
        self._completed = True

    def query(self):
        return self._completed

    def elapsed_time(self, end_event):
        return 0.0


class MockStream:
    def __init__(self):
        pass

    def synchronize(self):
        pass

    def wait_event(self, event):
        pass


_original_tensor_to = torch.Tensor.to


def mock_tensor_to(self, device=None, dtype=None, non_blocking=False,
                   copy=False, memory_format=None):
    # A CPU -> CUDA transfer has distinct storage, even without copy=True.
    cuda_copy = device is not None and torch.device(device).type == "cuda"
    return _original_tensor_to(self, dtype=dtype, copy=copy or cuda_copy)


def mock_cuda_available():
    return True


def mock_current_stream(device=None):
    return MockStream()


def mock_cuda_event(enable_timing=False):
    return MockEvent(enable_timing)


def mock_cuda_stream():
    return MockStream()


class MockedGPU:
    def __enter__(self):
        self.patches = [
            patch('torch.cuda.is_available', mock_cuda_available),
            patch('torch.cuda.Event', mock_cuda_event),
            patch('torch.cuda.Stream', mock_cuda_stream),
            patch('torch.cuda.stream', lambda stream: nullcontext()),
            patch('torch.cuda.current_stream', mock_current_stream),
            patch.object(torch.Tensor, 'to', mock_tensor_to),
        ]
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for p in self.patches:
            p.stop()