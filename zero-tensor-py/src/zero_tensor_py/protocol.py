import struct
import torch
#from _typeshed import ReadableBuffer

DT_F16: int = 0
DT_F32: int = 1
DT_F64: int = 2
DT_BF16: int = 3
DT_I8: int = 4
DT_I32: int = 5
DT_I64: int = 6
DT_U8: int = 7

DT_MAP: dict[int, torch.dtype] = {
    DT_U8: torch.uint8,
    DT_BF16: torch.bfloat16,
    DT_F16: torch.float16,
    DT_F32: torch.float32,
    DT_F64: torch.float64,
    DT_I32: torch.int32,
    DT_I64: torch.int64,
    DT_I8: torch.int8,
}

class TensorHeaderParser:
    """
    Parser TensorHeader from shared memory
    """

    HEADER_SIZE = 8
    SHAPE_TYPE_SIZE = 4
    STRIDES_TYPE_SIZE = 4

    @staticmethod
    def parse_meta(mmap_obj, offset: int) -> tuple[int, int, int, int, int]:
        """
        Parses metadata
        
        :param mmap_obj: pointer to a mmap buffer
        :type mmap_obj: ReadableBuffer
        :param offset: Offset to the beginning of the tensor
        :type offset: int
        :return: (shape, strides, torch_dt, data_offset, data_size)
        :rtype: tuple[int, int, int, int, int]
        """
        dt, ndims = struct.unpack_from("<BB", mmap_obj, offset)

        torch_dt = DT_MAP.get(dt)
        if not torch_dt:
            raise ValueError("Unknown dtype in header")
        
        item_size = torch_dt.itemsize
        shape_offset = offset + TensorHeaderParser.HEADER_SIZE
        strides_offset = shape_offset + (TensorHeaderParser.SHAPE_TYPE_SIZE * ndims)
        data_offset = strides_offset + (TensorHeaderParser.STRIDES_TYPE_SIZE * ndims)

        shape = list(struct.unpack_from(f"<{ndims}I", mmap_obj, shape_offset))
        rust_strides = list(struct.unpack_from(f"<{ndims}I", mmap_obj, strides_offset))
        strides = [s // item_size for s in rust_strides]

        num_elements = 1
        for dim in shape:
            num_elements *= dim
        data_size = num_elements * item_size
        
        return shape, strides, torch_dt, data_offset, data_size
