import torch
import os
import numpy as np
from model import NetHAMLModel
from data_loader import NetHAMLDataset
from collections import Counter

def test_offenders_standalone():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = 'data/processed'
    model_path = "net_ha_ml_best.pth"

    print("Loading dataset mapping...")
    temp_ds = NetHAMLDataset(data_dir)
    target_names = temp_ds.label_encoder.classes_
    
    print("\nLoading model...")
    # Your model currently uses 3 metadata features
    model = NetHAMLModel(num_classes=len(target_names), temporal_dim=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    print("\nTesting files containing 'FT' in their label...")
    print(f"{'Filename':<35} | {'True Label':<10} | {'Predictions (Count)'}")
    print("-" * 90)

    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            for f in sorted(os.listdir(data_dir)):
                if not f.endswith('.pt'): continue

                data = torch.load(os.path.join(data_dir, f), map_location='cpu')
                true_label = data['label']

                # We only want to interrogate the File Transfers
                if "FT" not in true_label.upper(): 
                    continue

                num_flows = data['temporal'].shape[0]
                all_preds = []

                # Process in small VRAM-safe batches
                batch_size = 32
                for i in range(0, num_flows, batch_size):
                    end = min(i + batch_size, num_flows)

                    t_batch = data['temporal'][i:end].to(device)
                    s_batch = data['spatial'][i:end].to(device)
                    m_batch = data['metadata'][i:end].to(device)

                    out, _ = model(t_batch, s_batch, m_batch)
                    preds = torch.argmax(out, dim=1).cpu().numpy()
                    all_preds.extend(preds)

                # Map indices back to string names (e.g., [1, 1, 4] -> ["FT", "FT", "VOIP"])
                pred_names = [target_names[idx] for idx in all_preds]
                counts = Counter(pred_names)

                # Format the output to show exactly what the model guessed
                pred_str = " | ".join([f"{k}: {v}" for k, v in counts.items()])
                print(f"{f:<35} | {true_label:<10} | {pred_str}")
                
                torch.cuda.empty_cache()

if __name__ == "__main__":
    test_offenders_standalone()
