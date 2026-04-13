import torch
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def extract_max_size_feature(temp_tensor):
    features = []
    for i in range(temp_tensor.shape[0]):
        temp = temp_tensor[i].numpy()
        sizes = np.abs(temp[:, 0])
        max_size = np.max(sizes)
        features.append([max_size])
    return np.array(features)

def analyze_misclassifications():
    data_dir = 'data/processed'
    X, y, source_info = [], [], []
    
    print("Extracting ONLY Max Size feature for Skype FT vs ALL VOIP...")
    
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
            source_type = "Skype FT"
        elif is_voip:
            label = 0
            source_type = "Skype VOIP" if is_skype else "Pure VOIP"
        else:
            continue
            
        feats = extract_max_size_feature(data['temporal'])
        X.append(feats)
        y.extend([label] * feats.shape[0])
        source_info.extend([source_type] * feats.shape[0])

    X = np.vstack(X)
    y = np.array(y)
    source_info = np.array(source_info)
    
    # Split data while keeping track of indices so we can look up the source
    indices = np.arange(len(y))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set: {len(y_train)} samples")
    print(f"Testing set: {len(y_test)} samples (Skype FT: {np.sum(y_test==1)}, ALL VOIP: {np.sum(y_test==0)})")
    
    print("\nTraining Random Forest Classifier on ONLY Max Size feature...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    
    acc = accuracy_score(y_test, preds) * 100
    print(f"\nBinary Classification Accuracy: {acc:.2f}%")
    
    # Let's break down the predictions
    print("\n=== MISCLASSIFICATION BREAKDOWN ===")
    
    # y_test == 1 (True Skype FT), preds == 0 (Predicted VOIP)
    fn_mask = (y_test == 1) & (preds == 0)
    # y_test == 0 (True ALL VOIP), preds == 1 (Predicted Skype FT)
    fp_mask = (y_test == 0) & (preds == 1)
    
    # True Positives & True Negatives
    tp_mask = (y_test == 1) & (preds == 1)
    tn_mask = (y_test == 0) & (preds == 0)
    
    print(f"Total True Skype FT correctly predicted as Skype FT: {np.sum(tp_mask)}")
    print(f"Total True VOIP correctly predicted as VOIP: {np.sum(tn_mask)}")
    print("-" * 50)
    print(f"Total Skype FT misclassified as VOIP (False Negatives): {np.sum(fn_mask)}")
    print(f"Total VOIP misclassified as Skype FT (False Positives): {np.sum(fp_mask)}")
    print("-" * 50)
    
    print("\nWhere did the False Positives (VOIP predicted as Skype FT) come from?")
    fp_sources = source_info[idx_test[fp_mask]]
    from collections import Counter
    fp_counts = Counter(fp_sources)
    for source, count in fp_counts.items():
        print(f"  - {source}: {count} flows")
        
    print("\nWhere did the True Negatives (VOIP correctly predicted as VOIP) come from?")
    tn_sources = source_info[idx_test[tn_mask]]
    tn_counts = Counter(tn_sources)
    for source, count in tn_counts.items():
        print(f"  - {source}: {count} flows")

    print("\n=== CONFUSION MATRIX ===")
    cm = confusion_matrix(y_test, preds)
    print(f"                 Predicted VOIP    Predicted Skype FT")
    print(f"Actual VOIP      {cm[0, 0]:<14}    {cm[0, 1]:<18}")
    print(f"Actual Skype FT  {cm[1, 0]:<14}    {cm[1, 1]:<18}")

if __name__ == "__main__":
    analyze_misclassifications()
