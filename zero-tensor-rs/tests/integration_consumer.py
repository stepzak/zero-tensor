import sys
import torch
from zero_tensor_py import ZeroTensorConsumer

def main():
    socket_path = sys.argv[1]
    shm_name = sys.argv[2]
    slot_size = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    max_steps = int(sys.argv[5])

    print(f"[PyConsumer] Connecting to {socket_path}...")
    
    with ZeroTensorConsumer(socket_path, shm_name, slot_size, nslots=3) as consumer:
        step = 0
        for batch in consumer:
            b, h, w = batch.shape
            assert b == batch_size, f"Step {step}: Expected B={batch_size}, got {b}"
            
            for i in range(b):
                item = batch[i]
                
                non_zero_mask = (item != 0)
                
                if not non_zero_mask.any():
                    continue
                    
                rows_with_data = non_zero_mask.any(dim=1)
                cols_with_data = non_zero_mask.any(dim=0)
                
                last_row = rows_with_data.nonzero(as_tuple=True)[0][-1].item()
                last_col = cols_with_data.nonzero(as_tuple=True)[0][-1].item()
                
                if last_row < h - 1:
                    assert torch.all(item[last_row + 1:, :] == 0), \
                        f"Step {step}, Item {i}: Bottom padding is not zero!"
                        
                if last_col < w - 1:
                    assert torch.all(item[:, last_col + 1:] == 0), \
                        f"Step {step}, Item {i}: Right padding is not zero!"
            
            print(f"Step {step}: Verified batch shape [{b}, {h}, {w}] with correct padding.")
            step += 1
            
            if step >= max_steps:
                break
                
    print("Dynamic batching integration test PASSED.")
    sys.exit(0)

if __name__ == "__main__":
    main()