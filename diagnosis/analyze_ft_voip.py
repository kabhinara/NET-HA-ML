import torch
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def extract_micro_features(temp_tensor, spat_tensor):
    features = []
    for i in range(temp_tensor.shape[0]):
        temp = temp_tensor[i].numpy()
        spat = spat_tensor[i].numpy()
        
        sizes = np.abs(temp[:, 0])
        iats = temp[:, 1]
        directions = temp[:, 2]
        
        jitter = np.std(iats[iats > 0]) if len(iats[iats > 0]) > 0 else 0
        large_pkt_ratio_500 = np.sum(sizes > 500) / 1024.0
        large_pkt_ratio_1200 = np.sum(sizes > 1200) / 1024.0
        flips = np.sum(directions[:-1] != directions[1:]) / 1024.0
        sparsity = np.sum(spat == 0) / 4096.0
        max_size = np.max(sizes)
        mean_size = np.mean(sizes)
        
        # New feature: Flow directionality
        # Ratio of forward to backward packets
        fwd_pkts = np.sum(directions == 1)
        bwd_pkts = np.sum(directions == -1)
        dir_ratio = fwd_pkts / (bwd_pkts + 1e-5)

        features.append([
            jitter, 
            large_pkt_ratio_500, 
            large_pkt_ratio_1200, 
            flips, 
            sparsity, 
            max_size, 
            mean_size,
            dir_ratio
        ])
        
    return np.array(features)

def analyze():
    data_dir = 'data/processed'
    
    # Store data by category
    categories = {
        'skype_ft': [],
        'skype_voip': [],
        'pure_ft': [],
        'pure_voip': []
    }
    
    print("Extracting features from all FT and VOIP flows...")
    for f in os.listdir(data_dir):
        if not f.endswith('.pt'): continue
        
        data = torch.load(os.path.join(data_dir, f), map_location='cpu')
        true_label = data['label'].upper()
        filename = f.lower()

        is_ft = 'FT' in true_label or 'FILE' in filename
        is_voip = 'VOIP' in true_label or 'AUDIO' in filename or 'VIDEO' in filename
        is_skype = 'skype' in filename
        
        if is_skype and 'file' in filename:
            cat = 'skype_ft'
        elif is_skype and ('audio' in filename or 'video' in filename):
            cat = 'skype_voip'
        elif is_ft and not is_skype:
            cat = 'pure_ft'
        elif is_voip and not is_skype:
            cat = 'pure_voip'
        else:
            continue
            
        feats = extract_micro_features(data['temporal'], data['spatial'])
        categories[cat].append(feats)

    # Convert to numpy arrays
    for k in categories:
        if len(categories[k]) > 0:
            categories[k] = np.vstack(categories[k])
        else:
            categories[k] = np.array([])
            
    print(f"Flow counts:")
    print(f"Skype FT: {categories['skype_ft'].shape[0]}")
    print(f"Skype VOIP: {categories['skype_voip'].shape[0]}")
    print(f"Pure FT: {categories['pure_ft'].shape[0]}")
    print(f"Pure VOIP: {categories['pure_voip'].shape[0]}")
    
    feature_names = [
        "Jitter (IAT StdDev)", 
        "Large Pkt Ratio (>500B)", 
        "Large Pkt Ratio (>1200B)", 
        "Direction Flip Rate", 
        "Payload Sparsity", 
        "Max Packet Size",
        "Mean Packet Size",
        "Direction Ratio (Fwd/Bwd)"
    ]
    
    def train_and_eval(name, X_pos, X_neg):
        if len(X_pos) == 0 or len(X_neg) == 0:
            print(f"Skipping {name} due to missing data")
            return
            
        X = np.vstack([X_pos, X_neg])
        y = np.array([1]*len(X_pos) + [0]*len(X_neg))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds) * 100
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.2f}%")
        
        importances = clf.feature_importances_
        for fname, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
            if imp > 0.01:
                print(f"{fname:<25}: {imp*100:.1f}%")

    train_and_eval("Skype FT vs Skype VOIP", categories['skype_ft'], categories['skype_voip'])
    train_and_eval("Pure FT vs Pure VOIP", categories['pure_ft'], categories['pure_voip'])
    
    # All FT vs All VOIP
    all_ft = np.vstack([categories['skype_ft'], categories['pure_ft']])
    all_voip = np.vstack([categories['skype_voip'], categories['pure_voip']])
    train_and_eval("ALL FT vs ALL VOIP", all_ft, all_voip)

if __name__ == "__main__":
    analyze()
