import torch
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import resample

def extract_flips_feature(temp_tensor):
    features = []
    for i in range(temp_tensor.shape[0]):
        temp = temp_tensor[i].numpy()
        directions = temp[:, 2]
        
        # --- Direction Flips Feature ---
        flips = np.sum(directions[:-1] != directions[1:]) / 1024.0
        
        features.append([flips])
    return np.array(features)

def test_scp1_vs_voip_flips():
    data_dir = 'data/processed'
    X_scp, X_voip = [], []
    
    print("Extracting ONLY Direction Flips feature for scp1.pcapng.pt vs ALL VOIP...")
    
    for f in os.listdir(data_dir):
        if not f.endswith('.pt'): continue
        
        data = torch.load(os.path.join(data_dir, f), map_location='cpu')
        true_label = data['label'].upper()
        filename = f.lower()
        
        is_scp1 = 'scp1.pcapng.pt' in filename
        
        is_voip = False
        if 'skype' in filename and ('audio' in filename or 'video' in filename or 'file' in filename or 'voip' in filename):
            if 'vpn' not in filename:
                is_voip = True
        elif 'VOIP' in true_label and 'VPN' not in true_label:
            is_voip = True
            
        if not is_scp1 and not is_voip:
            continue
            
        feats = extract_flips_feature(data['temporal'])
        
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
    
    # Balance dataset
    min_samples = min(len(X_scp), len(X_voip))
    X_scp_b = resample(X_scp, replace=False, n_samples=min_samples, random_state=42)
    X_voip_b = resample(X_voip, replace=False, n_samples=min_samples, random_state=42)
    
    print(f"Balanced counts -> SCP1: {len(X_scp_b)}, ALL VOIP: {len(X_voip_b)}")
    
    X = np.vstack([X_scp_b, X_voip_b])
    y = np.array([1] * len(X_scp_b) + [0] * len(X_voip_b))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining Random Forest on ONLY Direction Flips feature...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    print(f"Binary Classification Accuracy: {acc:.2f}%")
    
    print("\n=== CONFUSION MATRIX ===")
    cm = confusion_matrix(y_test, preds)
    print(f"                 Predicted VOIP          Predicted SCP1")
    print(f"Actual VOIP            {cm[0, 0]:<20}    {cm[0, 1]:<18}")
    print(f"Actual SCP1            {cm[1, 0]:<20}    {cm[1, 1]:<18}")
    
if __name__ == "__main__":
    test_scp1_vs_voip_flips()
