import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

def verify_net_ha_ml():
    print("--- Net-HA-ML Hardware Verification ---")
    
    # 1. Check CUDA and Device
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    
    print(f"Target GPU: {gpu_name}")
    print(f"Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
    
    # 2. Verify FP8 Support (Required for Net-HA-ML)
    # RTX 4070 (Ada Lovelace) is 8.9, which is perfect.
    fp8_supported = compute_cap >= (8, 9)
    print(f"Native FP8 Support: {'READY' if fp8_supported else 'UNSUPPORTED'}")

    # 3. Functional Test: FP8 Linear Layer
    # We use a dummy input and an FP8-enabled Linear layer
    model = te.Linear(128, 64).to(device)
    dummy_input = torch.randn(16, 128).to(device)
    
    # Define an FP8 recipe
    fp8_recipe = recipe.Format.E4M3 # 4-bit exponent, 3-bit mantissa
    
    try:
        # Use the transformer-engine autocast to trigger 8-bit math
        with te.fp8_autocast(enabled=True):
            output = model(dummy_input)
        
        print(f"FP8 Forward Pass: SUCCESS")
        print(f"Output Shape: {output.shape}")
        print("\nNET-HA-ML ENGINE STATUS: ALL SYSTEMS NOMINAL")
        
    except Exception as e:
        print(f"\nNET-HA-ML ENGINE STATUS: FAILED")
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_net_ha_ml()
