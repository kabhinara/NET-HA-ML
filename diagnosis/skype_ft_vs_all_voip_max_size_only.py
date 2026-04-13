import torch
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def extract_max_size_feature(temp_tensor):
    features = []
    # Using only temporal tensor as we only need sizes
    for i in range(temp_tensor.shape[0]):
        temp = temp_tensor[i].numpy()
        
        sizes = np.abs(temp[:, 0])
        
        # 1. Packet Size Max
        max_size = np.max(sizes)

        features.append([
            max_size
        ])
        
    return np.array(features)

def test_skype_ft_vs_all_voip_max_size_only():
    data_dir = 'data/processed'
    X, y = [], []
    
    print("Extracting ONLY Max Size feature for Skype FT vs ALL VOIP...")
    
    skype_ft_count = 0
    all_voip_count = 0
    
    for f in os.listdir(data_dir):
        if not f.endswith('.pt'): continue
        
        data = torch.load(os.path.join(data_dir, f), map_location='cpu')
        true_label = data['label'].upper()
        filename = f.lower()

        is_ft = 'FT' in true_label or 'FILE' in filename
        is_voip = 'VOIP' in true_label or 'AUDIO' in filename or 'VIDEO' in filename
        is_skype = 'skype' in filename
        
        # We want: Class 1 = Skype FT, Class 0 = ALL VOIP
        if is_skype and is_ft:
            label = 1
            skype_ft_count += data['temporal'].shape[0]
        elif is_voip:
            label = 0
            all_voip_count += data['temporal'].shape[0]
        else:
            continue
            
        feats = extract_max_size_feature(data['temporal'])
        X.append(feats)
        y.extend([label] * feats.shape[0])

    X = np.vstack(X)
    y = np.array(y)
    
    print(f"Extracted {len(y)} flows (Skype FT: {skype_ft_count}, ALL VOIP: {all_voip_count})")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining Random Forest Classifier on ONLY Max Size feature...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    print(f"Binary Classification Accuracy: {accuracy_score(y_test, preds) * 100:.2f}%\n")
    
    feature_names = [
        "Max Packet Size"
    ]
    
    importances = clf.feature_importances_
    
    print("=== FEATURE IMPORTANCES ===")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"{name:<25}: {imp*100:.1f}% importance")

if __name__ == "__main__":
    test_skype_ft_vs_all_voip_max_size_only()
