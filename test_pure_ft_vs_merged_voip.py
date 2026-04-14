import torch
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import resample

def extract_features(temp_tensor, spat_tensor):
    features = []
    for i in range(temp_tensor.shape[0]):
        temp = temp_tensor[i].numpy()
        spat = spat_tensor[i].numpy()
        
        sizes = np.abs(temp[:, 0])
        directions = temp[:, 2]
        
        flips = np.sum(directions[:-1] != directions[1:]) / 1024.0
        spatial_std = np.std(spat)
        max_size = np.max(sizes)
        
        fwd_pkts = np.sum(directions == 1)
        bwd_pkts = np.sum(directions == -1)
        dir_ratio = fwd_pkts / (bwd_pkts + 1e-5)
        
        spatial_mean = np.mean(spat)
        
        features.append([flips, spatial_std, max_size, dir_ratio, spatial_mean])
    return np.array(features)

def main():
    data_dir = 'data/processed'
    X_ft, X_voip = [], []
    
    print("Extracting features for Pure FT vs (ALL VOIP + Skype FT)...")
    
    for f in os.listdir(data_dir):
        if not f.endswith('.pt'): continue
        
        data = torch.load(os.path.join(data_dir, f), map_location='cpu')
        true_label = data['label'].upper()
        filename = f.lower()

        is_ft = 'FT' in true_label or 'FILE' in filename
        is_voip = 'VOIP' in true_label or 'AUDIO' in filename or 'VIDEO' in filename
        is_skype = 'skype' in filename
        
        # New grouping logic
        # Class 1: Pure FT (FT but NOT Skype)
        is_pure_ft = is_ft and not is_skype
        
        # Class 0: Merged VOIP (All VOIP + Skype FT)
        is_merged_voip = is_voip or (is_skype and is_ft)
        
        if not is_pure_ft and not is_merged_voip:
            continue
            
        feats = extract_features(data['temporal'], data['spatial'])
        
        if is_pure_ft:
            X_ft.append(feats)
        elif is_merged_voip:
            X_voip.append(feats)

    X_ft = np.vstack(X_ft)
    X_voip = np.vstack(X_voip)
    
    print(f"\nOriginal counts -> Pure FT: {len(X_ft)}, Merged VOIP: {len(X_voip)}")
    
    # Balance dataset
    min_samples = min(len(X_ft), len(X_voip))
    X_ft_b = resample(X_ft, replace=False, n_samples=min_samples, random_state=42)
    X_voip_b = resample(X_voip, replace=False, n_samples=min_samples, random_state=42)
    
    print(f"Balanced counts -> Pure FT: {len(X_ft_b)}, Merged VOIP: {len(X_voip_b)}")
    
    X = np.vstack([X_ft_b, X_voip_b])
    # Label 1 = Pure FT, Label 0 = Merged VOIP
    y = np.array([1] * len(X_ft_b) + [0] * len(X_voip_b))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- Test A: Direction Flips Alone ---
    print("\n--- Test A: Direction Flips Alone ---")
    clf_a = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf_a.fit(X_train[:, [0]], y_train)
    preds_a = clf_a.predict(X_test[:, [0]])
    acc_a = accuracy_score(y_test, preds_a) * 100
    print(f"Accuracy: {acc_a:.2f}%")
    
    cm_a = confusion_matrix(y_test, preds_a)
    print(f"                 Predicted M-VOIP        Predicted P-FT")
    print(f"Actual M-VOIP          {cm_a[0, 0]:<20}    {cm_a[0, 1]:<18}")
    print(f"Actual P-FT            {cm_a[1, 0]:<20}    {cm_a[1, 1]:<18}")
    
    # --- Test B: Top 5 Differentiators ---
    print("\n--- Test B: All 5 Top Differentiators ---")
    feature_names = ["Direction Flips", "Spatial Payload Std", "Max Pkt Size", "Direction Ratio (Fwd/Bwd)", "Spatial Payload Mean"]
    
    clf_b = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf_b.fit(X_train, y_train)
    preds_b = clf_b.predict(X_test)
    acc_b = accuracy_score(y_test, preds_b) * 100
    print(f"Accuracy: {acc_b:.2f}%")
    
    cm_b = confusion_matrix(y_test, preds_b)
    print(f"                 Predicted M-VOIP        Predicted P-FT")
    print(f"Actual M-VOIP          {cm_b[0, 0]:<20}    {cm_b[0, 1]:<18}")
    print(f"Actual P-FT            {cm_b[1, 0]:<20}    {cm_b[1, 1]:<18}")
    
    print("\nFeature Importances (Test B):")
    importances = clf_b.feature_importances_
    for fname, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"  {fname:<25}: {imp*100:.1f}%")

if __name__ == "__main__":
    main()
