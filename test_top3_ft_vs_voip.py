import torch
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import resample

def extract_top3_features(temp_tensor):
    features = []
    for i in range(temp_tensor.shape[0]):
        temp = temp_tensor[i].numpy()
        
        sizes = np.abs(temp[:, 0])
        directions = temp[:, 2]
        
        # 1. Direction Flips
        flips = np.sum(directions[:-1] != directions[1:]) / 1024.0
        
        # 2. Max Packet Size
        max_size = np.max(sizes)
        
        # 3. Direction Ratio (Fwd/Bwd)
        fwd_pkts = np.sum(directions == 1)
        bwd_pkts = np.sum(directions == -1)
        dir_ratio = fwd_pkts / (bwd_pkts + 1e-5)
        
        features.append([flips, max_size, dir_ratio])
    return np.array(features)

def test_top3_differentiators():
    data_dir = 'data/processed'
    X_ft, X_voip = [], []
    
    print("Extracting TOP 3 features for Pure FT vs (ALL VOIP + Skype FT)...")
    
    for f in os.listdir(data_dir):
        if not f.endswith('.pt'): continue
        
        data = torch.load(os.path.join(data_dir, f), map_location='cpu')
        true_label = data['label'].upper()
        filename = f.lower()

        is_ft = 'FT' in true_label or 'FILE' in filename
        is_voip = 'VOIP' in true_label or 'AUDIO' in filename or 'VIDEO' in filename
        is_skype = 'skype' in filename
        
        # Class 1: Pure FT (FT but NOT Skype)
        is_pure_ft = is_ft and not is_skype
        
        # Class 0: Merged VOIP (All VOIP + Skype FT)
        is_merged_voip = is_voip or (is_skype and is_ft)
        
        if not is_pure_ft and not is_merged_voip:
            continue
            
        feats = extract_top3_features(data['temporal'])
        
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
    y = np.array([1] * len(X_ft_b) + [0] * len(X_voip_b))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n--- Test C: Top 3 Differentiators Combined ---")
    feature_names = ["Direction Flips", "Max Pkt Size", "Direction Ratio (Fwd/Bwd)"]
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    print(f"Accuracy: {acc:.2f}%\n")
    
    cm = confusion_matrix(y_test, preds)
    print(f"                 Predicted M-VOIP        Predicted P-FT")
    print(f"Actual M-VOIP          {cm[0, 0]:<20}    {cm[0, 1]:<18}")
    print(f"Actual P-FT            {cm[1, 0]:<20}    {cm[1, 1]:<18}\n")
    
    print("Feature Importances:")
    importances = clf.feature_importances_
    for fname, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"  {fname:<25}: {imp*100:.1f}%")

if __name__ == "__main__":
    test_top3_differentiators()
