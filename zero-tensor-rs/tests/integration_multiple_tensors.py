import sys
import torch
from zero_tensor_py import ZeroTensorConsumer

def main():
    if len(sys.argv) < 6:
        print("Usage: python integration_consumer.py <socket> <shm> <slot_size> <batch_size> <steps>")
        sys.exit(1)

    socket_path = sys.argv[1]
    shm_name = sys.argv[2]
    batch_size = int(sys.argv[4])
    steps = int(sys.argv[5])

    print(f"[PyConsumer] Connecting to {socket_path}...")
    
    try:
        with ZeroTensorConsumer(socket_path, shm_name) as consumer:
            for _ in range(2):
                step = 0
                for batch in consumer:
                    assert "img" in batch, "Missing 'img' key"
                    assert "lbl" in batch, "Missing 'lbl' key"
                    img = batch["img"]
                    lbl = batch["lbl"]
                    
                    b, h, w = img.shape
                    assert b == batch_size, f"Step {step}: Expected B={batch_size}, got {b}"
                    assert h == 2 and w == 2, f"Step {step}: Expected H=2, W=2, got {h}x{w}"
                    
                    for i in range(b):
                        global_idx = step * batch_size + i
                        
                        expected_img_vals = [(global_idx * 10 + j) for j in range(4)]
                        actual_img_vals = img[i].flatten().tolist()
                        assert torch.allclose(img[i], torch.tensor(expected_img_vals, dtype=torch.float32).reshape(2, 2)), \
                            f"Step {step}, Item {i}: Img mismatch. Expected {expected_img_vals}, got {actual_img_vals}"
                        
                        expected_lbl_val = global_idx * 100
                        actual_lbl_val = lbl[i].item()
                        assert actual_lbl_val == expected_lbl_val, \
                            f"Step {step}, Item {i}: Lbl mismatch. Expected {expected_lbl_val}, got {actual_lbl_val}"

                    step += 1
                    if step >= steps:
                        break
                
                print(f"[PyConsumer] Verified {step} batches successfully.")
                
    except Exception as e:
        print(f"[PyConsumer] ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("[PyConsumer] Dynamic batching integration test PASSED.")
    sys.exit(0)

if __name__ == "__main__":
    main()