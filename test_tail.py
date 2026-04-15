import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast
import transformer_engine.pytorch as te
import os
import glob
from sklearn.preprocessing import LabelEncoder

# Import your custom modules
from model import NetHAMLModel

class TailDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir='data/processed_all'):
        pt_files = glob.glob(os.path.join(data_dir, '*.pt'))
        all_temporal, all_spatial, all_metadata, all_labels = [], [], [], []
        
        print(f"Loading {len(pt_files)} processed PCAP tensor files for TAIL analysis...")
        for f in pt_files:
            data = torch.load(f, weights_only=True)
            # Filter for flows that have at least 256 packets
            # temporal shape: (flows, 1024, 5)
            # We want to check how many packets are non-zero in the first column (size)
            temp = data['temporal']
            # Find indices where the 128th packet is non-zero
            valid_mask = temp[:, 255, 0] != 0 
            
            if not valid_mask.any():
                continue
                
            num_flows = valid_mask.sum().item()
            
            label = data['label']
            fname = os.path.basename(f).lower()
            if 'skype' in fname and ('audio' in fname or 'video' in fname or 'file' in fname or 'voip' in fname):
                if 'vpn' in fname:
                    label = 'VPN-VOIP'
                else:
                    label = 'VOIP'
            
            # Slice packets 128 to 256
            all_temporal.append(temp[valid_mask, 128:256, :])
            all_spatial.append(data['spatial'][valid_mask])
            all_metadata.append(data['metadata'][valid_mask])
            all_labels.extend([label] * num_flows)

        if not all_temporal:
            raise ValueError("No flows found with at least 256 packets.")

        self.temporal = torch.cat(all_temporal, dim=0).cuda()
        self.spatial = torch.cat(all_spatial, dim=0).cuda()
        
        # Metadata logic: We'll use the metadata already in the file 
        # (which is based on the full flow now)
        raw_metadata = torch.log1p(torch.cat(all_metadata, dim=0)).cuda()
        
        # Explicit feature extraction for the TAIL segment (128:256)
        print("Extracting explicitly engineered top-3 features for the TAIL segment...")
        sizes = torch.abs(self.temporal[:, :, 0])
        directions = self.temporal[:, :, 2]
        
        max_size = torch.max(sizes, dim=1)[0].unsqueeze(1)
        max_size_log = torch.log1p(max_size)
        flips = torch.sum(directions[:, :-1] != directions[:, 1:], dim=1, keepdim=True).float() / 128.0
        fwd_pkts = torch.sum(directions == 1, dim=1, keepdim=True).float()
        bwd_pkts = torch.sum(directions == -1, dim=1, keepdim=True).float()
        dir_ratio = fwd_pkts / (bwd_pkts + 1e-5)
        dir_ratio_log = torch.log1p(dir_ratio)
        
        extracted_feats = torch.cat([max_size_log, flips, dir_ratio_log], dim=1)
        self.metadata = torch.cat([raw_metadata, extracted_feats], dim=1)
        
        self.label_encoder = LabelEncoder()
        # Ensure classes match the training set order
        # Hardcoded from data_loader.py output to be safe
        classes = ['CHAT', 'FT', 'MAIL', 'STREAMING', 'VOIP', 'VPN-CHAT', 'VPN-FT', 'VPN-MAIL', 'VPN-P2P', 'VPN-STREAMING', 'VPN-VOIP']
        self.label_encoder.classes_ = np.array(classes)
        self.labels = torch.tensor(self.label_encoder.transform(all_labels), dtype=torch.long).cuda()
        
        print(f"Loaded {len(self.labels)} tail-segment flows (packets 128-256) into VRAM.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.temporal[idx], self.spatial[idx], self.metadata[idx], self.labels[idx]

def evaluate_tail():
    device = torch.device("cuda")
    weights_path = "net_ha_ml_best.pth" 
    
    try:
        dataset = TailDataset()
    except Exception as e:
        print(f"Error loading tail dataset: {e}")
        return
        
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    target_names = dataset.label_encoder.classes_
    
    model = NetHAMLModel(
        num_classes=len(target_names), 
        temporal_dim=5,
        metadata_dim=7,     
        d_model=256,        
        num_layers=4        
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"Successfully loaded weights from {weights_path}")
    except FileNotFoundError:
        print(f"Error: {weights_path} not found.")
        return

    model.eval() 
    all_preds, all_labels = [], []
    
    print(f"Running Tail Inference on {len(dataset)} flows (Packets 128-256)...")
    
    with torch.no_grad():
        with autocast('cuda', dtype=torch.bfloat16):
            with te.fp8_autocast(enabled=True):
                for temporal, spatial, metadata, labels in loader:
                    pad_len = (8 - temporal.size(0) % 8) % 8
                    if pad_len > 0:
                        temporal = torch.cat([temporal, temporal[:pad_len]], dim=0)
                        spatial = torch.cat([spatial, spatial[:pad_len]], dim=0)
                        metadata = torch.cat([metadata, metadata[:pad_len]], dim=0)
                    
                    class_out, _ = model(temporal, spatial, metadata)
                    if pad_len > 0: class_out = class_out[:-pad_len]
                    _, predicted = torch.max(class_out, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
    print("\n=== TAIL EVALUATION COMPLETE (Packets 128-256) ===")
    print("\nClassification Report (4 Decimals):")
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4, zero_division=0))
    
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)
    
    print("\n=== TAIL OVERALL METRICS (4 DECIMALS) ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

if __name__ == "__main__":
    evaluate_tail()
