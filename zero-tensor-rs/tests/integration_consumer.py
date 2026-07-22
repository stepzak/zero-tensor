import sys
import torch
from zero_tensor_py import ZeroTensorConsumer

def main():
    socket_path = sys.argv[1]
    shm_name = sys.argv[2]
    slot_size = int(sys.argv[3])
    max_steps = int(sys.argv[4])
    batch_size = 2

    print(f"[Python] Connecting to {socket_path} (shm: {shm_name})...")

    with ZeroTensorConsumer(socket_path, shm_name, slot_size, batch_size) as consumer:
        steps = 0
        for batch in consumer:
            assert batch.shape == (batch_size, 2, 3), f"Wrong shape. Expected: ({batch_size}, 2, 3), got: {batch.shape}"
            assert batch.stride() == (6, 3, 1),  f"Wrong stride. Expected: (6, 3, 1), got: {batch.stride()}"
            assert batch.dtype == torch.float32, f"Wrong dtype. Expected: {torch.float32}, got {batch.dtype}"

            idx_0 = steps * batch_size
            idx_1 = steps * batch_size + 1

            assert torch.allclose(batch[0, 0, 0], torch.tensor(idx_0, dtype=torch.float32))
            assert torch.allclose(batch[0, 1, 0], torch.tensor(idx_0 + 0.5, dtype=torch.float32))
            
            assert torch.allclose(batch[1, 0, 0], torch.tensor(idx_1, dtype=torch.float32))
            assert torch.allclose(batch[1, 1, 0], torch.tensor(idx_1 + 0.5, dtype=torch.float32))

            print(f"[Python] Step {steps} verified successfully.")
            steps += 1
            if steps >= max_steps:
                break
    
    print("[Python] Integration consumer finished successfully.")
    sys.exit(0)

if __name__ == "__main__":
    main()
