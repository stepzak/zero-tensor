import struct
from zero_tensor_py.protocol import DT_F32, DT_I32

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