import pytest
import os
from mocks.producer_mock import CB_SIZE


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