import mmap
import os
import socket
import struct
import threading
import time

from zero_tensor_py.consumer import _PROTO_BEGIN_STR, VERSION


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