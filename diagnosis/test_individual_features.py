import torch
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

def extract_features(temp_tensor, spat_tensor):
    features = []
    for i in range(temp_tensor.shape[0]):
        temp = temp_tensor[i].numpy()
        spat = spat_tensor[i].numpy()
        
        sizes = np.abs(temp[:, 0])
        iats = temp[:, 1]
        directions = temp[:, 2]
        
        valid_iats = iats[iats > 0]
        
        # 1. Total Active Time
        total_time = np.sum(iats)
        
        # 2. Direction Ratio (Fwd/Bwd)
        fwd_pkts = np.sum(directions == 1)
        bwd_pkts = np.sum(directions == -1)
        dir_ratio = fwd_pkts / (bwd_pkts + 1e-5)
        
        # 3. Spatial Payload Mean
        spatial_mean = np.mean(spat)
        
        # 4. Spatial Payload Std
        spatial_std = np.std(spat)
        
        # 5. Max Pkt Size
        max_size = np.max(sizes)
        
        # 6. Mean IAT
        mean_iat = np.mean(valid_iats) if len(valid_iats) > 0 else 0
        
        # 7. Jitter (Std IAT)
        jitter = np.std(valid_iats) if len(valid_iats) > 0 else 0
        
        # 8. Max IAT
        max_iat = np.max(valid_iats) if len(valid_iats) > 0 else 0
        
        # 9. Mean Pkt Size
        mean_size = np.mean(sizes)

        features.append([
            total_time, dir_ratio, spatial_mean, spatial_std,
            max_size, mean_iat, jitter, max_iat, mean_size
        ])
    return np.array(features)

def test_individual_features():
    data_dir = 'data/processed'
    X_ft, X_voip = [], []
    
    print("Extracting individual features for balanced Skype FT vs Skype VOIP...")
    
    for f in os.listdir(data_dir):
        if not f.endswith('.pt'): continue
        
        data = torch.load(os.path.join(data_dir, f), map_location='cpu')
        true_label = data['label'].upper()
        filename = f.lower()

        is_ft = 'FT' in true_label or 'FILE' in filename
        is_voip = 'VOIP' in true_label or 'AUDIO' in filename or 'VIDEO' in filename
        is_skype = 'skype' in filename
        
        if not is_skype:
            continue
            
        feats = extract_features(data['temporal'], data['spatial'])
        
        if is_ft:
            X_ft.append(feats)
        elif is_voip:
            X_voip.append(feats)

    X_ft = np.vstack(X_ft)
    X_voip = np.vstack(X_voip)
    
    # Balance dataset
    X_voip_balanced = resample(X_voip, replace=False, n_samples=len(X_ft), random_state=42)
    
    X = np.vstack([X_ft, X_voip_balanced])
    y = np.array([1] * len(X_ft) + [0] * len(X_voip_balanced))
    
    feature_names = [
        "Total Active Time",
        "Direction Ratio (Fwd/Bwd)",
        "Spatial Payload Mean",
        "Spatial Payload Std",
        "Max Pkt Size",
        "Mean IAT",
        "Jitter (Std IAT)",
        "Max IAT",
        "Mean Pkt Size"
    ]
    
    print(f"Dataset Balanced: {len(X_ft)} Skype FT vs {len(X_voip_balanced)} Skype VOIP\n")
    print("Testing each feature individually (Accuracy on 20% test set):\n")
    
    results = []
    
    # Test each feature individually
    for i, name in enumerate(feature_names):
        # Extract just the single column for this feature
        X_single = X[:, [i]]
        
        X_train, X_test, y_train, y_test = train_test_split(X_single, y, test_size=0.2, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds) * 100
        
        results.append((name, acc))
        
    # Sort and print results
    results.sort(key=lambda x: x[1], reverse=True)
    
    for name, acc in results:
        print(f"{name:<25}: {acc:.2f}%")

if __name__ == "__main__":
    test_individual_features()
