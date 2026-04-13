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

def test_skype_voip_vs_pure_voip():
    data_dir = 'data/processed'
    X_skype_voip, X_pure_voip = [], []
    
    print("Extracting 15 comprehensive features for Skype VOIP vs Pure VOIP...")
    
    for f in os.listdir(data_dir):
        if not f.endswith('.pt'): continue
        
        data = torch.load(os.path.join(data_dir, f), map_location='cpu')
        true_label = data['label'].upper()
        filename = f.lower()

        # We only care about VOIP traffic
        is_voip = 'VOIP' in true_label or 'AUDIO' in filename or 'VIDEO' in filename
        is_ft = 'FT' in true_label or 'FILE' in filename
        
        # Skip FT, we only want VOIP
        if is_ft or not is_voip:
            continue
            
        is_skype = 'skype' in filename
            
        feats = extract_comprehensive_features(data['temporal'], data['spatial'])
        
        if is_skype:
            X_skype_voip.append(feats)
        else:
            X_pure_voip.append(feats)

    X_skype_voip = np.vstack(X_skype_voip)
    X_pure_voip = np.vstack(X_pure_voip)
    
    print(f"\nOriginal counts -> Skype VOIP: {len(X_skype_voip)}, Pure VOIP: {len(X_pure_voip)}")
    
    # Balance the dataset (Undersample Pure VOIP to match Skype VOIP)
    min_samples = min(len(X_skype_voip), len(X_pure_voip))
    X_skype_voip_b = resample(X_skype_voip, replace=False, n_samples=min_samples, random_state=42)
    X_pure_voip_b = resample(X_pure_voip, replace=False, n_samples=min_samples, random_state=42)
    
    print(f"Balanced counts -> Skype VOIP: {len(X_skype_voip_b)}, Pure VOIP: {len(X_pure_voip_b)}")
    
    X = np.vstack([X_skype_voip_b, X_pure_voip_b])
    # Label 1 = Skype VOIP, Label 0 = Pure VOIP
    y = np.array([1] * len(X_skype_voip_b) + [0] * len(X_pure_voip_b))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining Random Forest on all 15 features...")
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    print(f"Binary Classification Accuracy: {acc:.2f}%")
    
    print("\n=== CONFUSION MATRIX ===")
    cm = confusion_matrix(y_test, preds)
    print(f"                 Predicted Pure VOIP     Predicted Skype VOIP")
    print(f"Actual Pure VOIP       {cm[0, 0]:<20}    {cm[0, 1]:<18}")
    print(f"Actual Skype VOIP      {cm[1, 0]:<20}    {cm[1, 1]:<18}")
    
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
        if imp > 0.02: # Only show features with > 2% importance
            print(f"  {fname:<25}: {imp*100:.1f}%")

if __name__ == "__main__":
    test_skype_voip_vs_pure_voip()
