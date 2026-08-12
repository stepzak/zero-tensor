import struct
import torch

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

UNSIGNED_FORMATS = {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}


class TensorHeaderParser:
    """
    Parser TensorHeader from shared memory
    """
    @staticmethod
    def is_slot_ready(mmap_obj, slot_offset: int, is_ready_offset_in_header: int) -> bool:
        val = struct.unpack_from("<B", mmap_obj, slot_offset + is_ready_offset_in_header)[0]
        return val == 1

    @staticmethod
    def parse_meta(
        mmap_obj, 
        slot_offset: int, 
        header_size: int, 
        dt_offset: int, 
        ndims_offset: int, 
        shape_type_size: int
    ) -> tuple[list[int], list[int], torch.dtype, int, int]:
        dt = struct.unpack_from("<B", mmap_obj, slot_offset + dt_offset)[0]
        ndims = struct.unpack_from("<B", mmap_obj, slot_offset + ndims_offset)[0]
        
        torch_dt = DT_MAP.get(dt)
        if torch_dt is None:
            raise ValueError(f"Unknown dtype in header: {dt}")
        
        item_size = torch_dt.itemsize
        shape_offset = slot_offset + header_size
        strides_offset = shape_offset + (shape_type_size * ndims)
        data_offset = strides_offset + (shape_type_size * ndims)
        
        fmt_char = UNSIGNED_FORMATS.get(shape_type_size, 'I')
        
        shape = list(struct.unpack_from(f"<{ndims}{fmt_char}", mmap_obj, shape_offset))
        rust_strides = list(struct.unpack_from(f"<{ndims}{fmt_char}", mmap_obj, strides_offset))
        
        strides = [s // item_size for s in rust_strides]
        
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        data_size = num_elements * item_size
        
        return shape, strides, torch_dt, data_offset, data_size