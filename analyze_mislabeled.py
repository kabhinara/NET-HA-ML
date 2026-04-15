import torch
import numpy as np
import os
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch.amp import autocast
import transformer_engine.pytorch as te

from model import NetHAMLModel
from data_loader import NetHAMLDataset

def extract_comprehensive_features(temp_tensor, spat_tensor):
    features = []
    for i in range(temp_tensor.shape[0]):
        temp = temp_tensor[i].numpy()
        spat = spat_tensor[i].numpy()
        
        sizes = np.abs(temp[:, 0])
        iats = temp[:, 1]
        directions = temp[:, 2]
        
        max_size = np.max(sizes)
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)
        
        large_pkt_500 = np.sum(sizes > 500) / 1024.0
        large_pkt_1200 = np.sum(sizes > 1200) / 1024.0
        small_pkt_100 = np.sum(sizes < 100) / 1024.0
        
        valid_iats = iats[iats > 0]
        jitter = np.std(valid_iats) if len(valid_iats) > 0 else 0
        mean_iat = np.mean(valid_iats) if len(valid_iats) > 0 else 0
        max_iat = np.max(valid_iats) if len(valid_iats) > 0 else 0
        total_time = np.sum(iats)
        
        flips = np.sum(directions[:-1] != directions[1:]) / 1024.0
        fwd_pkts = np.sum(directions == 1)
        bwd_pkts = np.sum(directions == -1)
        dir_ratio = fwd_pkts / (bwd_pkts + 1e-5)
        
        sparsity = np.sum(spat == 0) / 4096.0
        spatial_mean = np.mean(spat)
        spatial_std = np.std(spat)
        
        features.append([
            max_size, mean_size, std_size, 
            large_pkt_500, large_pkt_1200, small_pkt_100,
            jitter, mean_iat, max_iat, total_time,
            flips, dir_ratio,
            sparsity, spatial_mean, spatial_std
        ])
    return np.array(features)

def analyze():
    device = torch.device("cuda")
    data_dir = 'data/processed'
    weights_path = "net_ha_ml_best.pth" 
    
    print("Loading dataset for analysis...")
    dataset = NetHAMLDataset(data_dir)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    target_names = dataset.label_encoder.classes_
    num_classes = len(target_names)
    
    model = NetHAMLModel(
        num_classes=num_classes, 
        temporal_dim=5, 
        metadata_dim=7,
        d_model=256, 
        num_layers=4
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    
    all_preds, all_labels, all_filenames = [], [], []
    
    print("Running full inference to find misclassifications...")
    with torch.no_grad():
        with autocast('cuda', dtype=torch.bfloat16):
            with te.fp8_autocast(enabled=True):
                for temporal, spatial, metadata, labels, _, fnames in loader:
                    pad_len = (8 - temporal.size(0) % 8) % 8
                    if pad_len > 0:
                        temporal = torch.cat([temporal, temporal[:pad_len]], dim=0)
                        spatial = torch.cat([spatial, spatial[:pad_len]], dim=0)
                        metadata = torch.cat([metadata, metadata[:pad_len]], dim=0)
                    
                    class_out, _ = model(temporal, spatial, metadata)
                    if pad_len > 0:
                        class_out = class_out[:-pad_len]
                        
                    _, predicted = torch.max(class_out, 1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_filenames.extend(fnames)
                    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_filenames = np.array(all_filenames)
    
    voip_idx = list(target_names).index("VOIP")
    ft_idx = list(target_names).index("FT")
    stream_idx = list(target_names).index("STREAMING")
    
    # Collect specific misclassifications
    ft_pred_voip_mask = (all_labels == ft_idx) & (all_preds == voip_idx)
    voip_pred_ft_mask = (all_labels == voip_idx) & (all_preds == ft_idx)
    voip_pred_stream_mask = (all_labels == voip_idx) & (all_preds == stream_idx)
    
    ft_pred_voip_fnames = np.unique(all_filenames[ft_pred_voip_mask])
    voip_pred_ft_fnames = np.unique(all_filenames[voip_pred_ft_mask])
    voip_pred_stream_fnames = np.unique(all_filenames[voip_pred_stream_mask])
    
    print(f"\nFound {len(ft_pred_voip_fnames)} files with FT->VOIP errors")
    print(f"Found {len(voip_pred_ft_fnames)} files with VOIP->FT errors")
    print(f"Found {len(voip_pred_stream_fnames)} files with VOIP->STREAMING errors")
    
    # We will test separability using the 15 features 
    # Grab all raw tensors needed for testing
    print("\nExtracting comprehensive features for statistical tests...")
    
    all_ft_feats = []
    all_voip_feats = []
    all_streaming_feats = []
    
    bad_ft_voip_feats = []
    bad_voip_ft_feats = []
    bad_voip_stream_feats = []
    
    # We load them from file directly to get pure lists
    for f in os.listdir(data_dir):
        if not f.endswith('.pt'): continue
        data = torch.load(os.path.join(data_dir, f), map_location='cpu')
        fname = os.path.basename(f)
        
        feats = extract_comprehensive_features(data['temporal'], data['spatial'])
        
        # Where does this file belong logically
        label_raw = data['label'].upper()
        
        is_skype = 'skype' in fname.lower()
        is_ft = 'FT' in label_raw or 'FILE' in fname.lower()
        is_voip = 'VOIP' in label_raw or 'AUDIO' in fname.lower() or 'VIDEO' in fname.lower()
        is_stream = 'STREAM' in label_raw
        
        is_pure_ft = is_ft and not is_skype
        is_merged_voip = is_voip or (is_skype and is_ft)
        
        if is_pure_ft: all_ft_feats.append(feats)
        if is_merged_voip: all_voip_feats.append(feats)
        if is_stream: all_streaming_feats.append(feats)
        
        if fname in ft_pred_voip_fnames: bad_ft_voip_feats.append(feats)
        if fname in voip_pred_ft_fnames: bad_voip_ft_feats.append(feats)
        if fname in voip_pred_stream_fnames: bad_voip_stream_feats.append(feats)
        
    all_ft_feats = np.vstack(all_ft_feats) if all_ft_feats else np.array([])
    all_voip_feats = np.vstack(all_voip_feats) if all_voip_feats else np.array([])
    all_streaming_feats = np.vstack(all_streaming_feats) if all_streaming_feats else np.array([])
    
    bad_ft_voip_feats = np.vstack(bad_ft_voip_feats) if bad_ft_voip_feats else np.array([])
    bad_voip_ft_feats = np.vstack(bad_voip_ft_feats) if bad_voip_ft_feats else np.array([])
    bad_voip_stream_feats = np.vstack(bad_voip_stream_feats) if bad_voip_stream_feats else np.array([])
    
    def test_separability(name, X_target, X_base, bad_fnames):
        if len(X_target) == 0 or len(X_base) == 0: return []
        
        min_samples = min(len(X_target), len(X_base))
        from sklearn.utils import resample
        X_target_b = resample(X_target, replace=False, n_samples=min_samples, random_state=42)
        X_base_b = resample(X_base, replace=False, n_samples=min_samples, random_state=42)
        
        X = np.vstack([X_target_b, X_base_b])
        y = np.array([1]*min_samples + [0]*min_samples)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test)) * 100
        
        print(f"\nSeparability for {name}: {acc:.2f}% (Using Random Forest)")
        
        if acc < 85.0:
            print(f"--> WARNING: Separability is < 85%. These files physically belong in the other class.")
            return list(bad_fnames)
        else:
            print(f"--> These flows ARE statistically distinct. The NN just needs more training.")
            return []
            
    relabel_as_voip = []
    relabel_as_ft = []
    relabel_as_stream = []

    print("\n--- Testing Bad FTs vs ALL VOIP ---")
    bad_ft = test_separability("Bad FT -> VOIP", bad_ft_voip_feats, all_voip_feats, ft_pred_voip_fnames)
    relabel_as_voip.extend(bad_ft)

    print("\n--- Testing Bad VOIPs vs ALL FT ---")
    bad_voip_to_ft = test_separability("Bad VOIP -> FT", bad_voip_ft_feats, all_ft_feats, voip_pred_ft_fnames)
    relabel_as_ft.extend(bad_voip_to_ft)
    
    print("\n--- Testing Bad VOIPs vs ALL STREAMING ---")
    bad_voip_to_stream = test_separability("Bad VOIP -> STREAMING", bad_voip_stream_feats, all_streaming_feats, voip_pred_stream_fnames)
    relabel_as_stream.extend(bad_voip_to_stream)
    
    print("\n=== RECOMMENDED RELABELS ===")
    print(f"Change to VOIP: {relabel_as_voip}")
    print(f"Change to FT: {relabel_as_ft}")
    print(f"Change to STREAMING: {relabel_as_stream}")

if __name__ == "__main__":
    analyze()
