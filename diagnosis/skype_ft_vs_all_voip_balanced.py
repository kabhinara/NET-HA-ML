import torch
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import resample

def extract_features(temp_tensor):
    features = []
    for i in range(temp_tensor.shape[0]):
        temp = temp_tensor[i].numpy()
        sizes = np.abs(temp[:, 0])
        directions = temp[:, 2]
        
        max_size = np.max(sizes)
        mean_size = np.mean(sizes)
        flips = np.sum(directions[:-1] != directions[1:]) / 1024.0
        
        fwd_pkts = np.sum(directions == 1)
        bwd_pkts = np.sum(directions == -1)
        dir_ratio = fwd_pkts / (bwd_pkts + 1e-5)
        
        features.append([max_size, mean_size, flips, dir_ratio])
    return np.array(features)

def test_balanced():
    data_dir = 'data/processed'
    X_ft, X_voip = [], []
    
    print("Extracting features for BALANCED Skype FT vs ALL VOIP...")
    
    for f in os.listdir(data_dir):
        if not f.endswith('.pt'): continue
        
        data = torch.load(os.path.join(data_dir, f), map_location='cpu')
        true_label = data['label'].upper()
        filename = f.lower()

        is_ft = 'FT' in true_label or 'FILE' in filename
        is_voip = 'VOIP' in true_label or 'AUDIO' in filename or 'VIDEO' in filename
        is_skype = 'skype' in filename
        
        feats = extract_features(data['temporal'])
        
        if is_skype and is_ft:
            X_ft.append(feats)
        elif is_voip:
            X_voip.append(feats)

    X_ft = np.vstack(X_ft)
    X_voip = np.vstack(X_voip)
    
    print(f"Original counts -> Skype FT: {len(X_ft)}, ALL VOIP: {len(X_voip)}")
    
    # Undersample VOIP to match Skype FT
    X_voip_balanced = resample(X_voip, replace=False, n_samples=len(X_ft), random_state=42)
    
    print(f"Balanced counts -> Skype FT: {len(X_ft)}, ALL VOIP: {len(X_voip_balanced)}")
    
    # Combine and create labels
    X = np.vstack([X_ft, X_voip_balanced])
    y = np.array([1] * len(X_ft) + [0] * len(X_voip_balanced))
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    def evaluate_model(name, X_tr, X_te, feature_names):
        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        clf.fit(X_tr, y_train)
        preds = clf.predict(X_te)
        
        acc = accuracy_score(y_test, preds) * 100
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.2f}%")
        
        cm = confusion_matrix(y_test, preds)
        print("Confusion Matrix:")
        print(f"                 Predicted VOIP    Predicted Skype FT")
        print(f"Actual VOIP      {cm[0, 0]:<14}    {cm[0, 1]:<18}")
        print(f"Actual Skype FT  {cm[1, 0]:<14}    {cm[1, 1]:<18}")
        
        print("\nFeature Importances:")
        importances = clf.feature_importances_
        for fname, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
            print(f"  {fname:<25}: {imp*100:.1f}%")

    # Test 1: Max Size Only
    evaluate_model(
        "Model 1: Max Packet Size ONLY", 
        X_train[:, [0]], X_test[:, [0]], 
        ["Max Packet Size"]
    )
    
    # Test 2: All 4 targeted features
    feature_names = ["Max Packet Size", "Mean Packet Size", "Direction Flip Rate", "Direction Ratio (Fwd/Bwd)"]
    evaluate_model(
        "Model 2: Size & Asymmetry Features", 
        X_train, X_test, 
        feature_names
    )

if __name__ == "__main__":
    test_balanced()
