import torch
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import resample

def extract_comprehensive_features(temp_tensor, spat_tensor):
    features = []
    for i in range(temp_tensor.shape[0]):
        temp = temp_tensor[i].numpy()
        spat = spat_tensor[i].numpy()
        
        sizes = np.abs(temp[:, 0])
        iats = temp[:, 1]
        directions = temp[:, 2]
        
        # --- Size Features ---
        max_size = np.max(sizes)
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)
        
        large_pkt_500 = np.sum(sizes > 500) / 1024.0
        large_pkt_1200 = np.sum(sizes > 1200) / 1024.0
        small_pkt_100 = np.sum(sizes < 100) / 1024.0
        
        # --- Timing Features ---
        valid_iats = iats[iats > 0]
        jitter = np.std(valid_iats) if len(valid_iats) > 0 else 0
        mean_iat = np.mean(valid_iats) if len(valid_iats) > 0 else 0
        max_iat = np.max(valid_iats) if len(valid_iats) > 0 else 0
        total_time = np.sum(iats)
        
        # --- Direction / Flow Features ---
        flips = np.sum(directions[:-1] != directions[1:]) / 1024.0
        fwd_pkts = np.sum(directions == 1)
        bwd_pkts = np.sum(directions == -1)
        dir_ratio = fwd_pkts / (bwd_pkts + 1e-5)
        
        # --- Payload (Spatial) Features ---
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

def test_scp1_vs_voip():
    data_dir = 'data/processed'
    X_scp, X_voip = [], []
    
    print("Extracting features for scp1.pcapng.pt vs ALL VOIP...")
    
    for f in os.listdir(data_dir):
        if not f.endswith('.pt'): continue
        
        data = torch.load(os.path.join(data_dir, f), map_location='cpu')
        true_label = data['label'].upper()
        filename = f.lower()
        
        # Determine if it's our target scp1 file
        is_scp1 = 'scp1.pcapng.pt' in filename
        
        # With our new labeling, VOIP includes original VOIP and Skype VOIP/FT
        # but let's just grab anything that's labeled VOIP (ignoring VPN for now, or including it)
        # Actually let's catch all that ends up in 'VOIP' class based on our new data_loader logic
        is_voip = False
        if 'skype' in filename and ('audio' in filename or 'video' in filename or 'file' in filename or 'voip' in filename):
            if 'vpn' not in filename:
                is_voip = True
        elif 'VOIP' in true_label and 'VPN' not in true_label:
            is_voip = True
            
        if not is_scp1 and not is_voip:
            continue
            
        feats = extract_comprehensive_features(data['temporal'], data['spatial'])
        
        if is_scp1:
            X_scp.append(feats)
        elif is_voip:
            X_voip.append(feats)

    if len(X_scp) == 0:
        print("Could not find scp1.pcapng.pt or it has 0 flows.")
        return

    X_scp = np.vstack(X_scp)
    X_voip = np.vstack(X_voip)
    
    print(f"\nOriginal counts -> SCP1: {len(X_scp)}, ALL VOIP: {len(X_voip)}")
    
    # Balance dataset (Undersample VOIP to match SCP1)
    min_samples = min(len(X_scp), len(X_voip))
    X_scp_b = resample(X_scp, replace=False, n_samples=min_samples, random_state=42)
    X_voip_b = resample(X_voip, replace=False, n_samples=min_samples, random_state=42)
    
    print(f"Balanced counts -> SCP1: {len(X_scp_b)}, ALL VOIP: {len(X_voip_b)}")
    
    X = np.vstack([X_scp_b, X_voip_b])
    # Label 1 = SCP1, Label 0 = VOIP
    y = np.array([1] * len(X_scp_b) + [0] * len(X_voip_b))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining Random Forest on all 15 features...")
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    print(f"Binary Classification Accuracy: {acc:.2f}%")
    
    print("\n=== CONFUSION MATRIX ===")
    cm = confusion_matrix(y_test, preds)
    print(f"                 Predicted VOIP          Predicted SCP1")
    print(f"Actual VOIP            {cm[0, 0]:<20}    {cm[0, 1]:<18}")
    print(f"Actual SCP1            {cm[1, 0]:<20}    {cm[1, 1]:<18}")
    
    feature_names = [
        "Max Pkt Size", "Mean Pkt Size", "Std Pkt Size", 
        "Pkt Ratio > 500B", "Pkt Ratio > 1200B", "Pkt Ratio < 100B",
        "Jitter (Std IAT)", "Mean IAT", "Max IAT", "Total Active Time",
        "Direction Flips", "Direction Ratio (Fwd/Bwd)",
        "Payload Sparsity (Zeros)", "Spatial Payload Mean", "Spatial Payload Std"
    ]
    
    importances = clf.feature_importances_
    print("\n=== TOP Differentiating Features ===")
    for fname, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        if imp > 0.01: # Only show features with > 1% importance
            print(f"  {fname:<25}: {imp*100:.1f}%")

if __name__ == "__main__":
    test_scp1_vs_voip()
