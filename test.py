import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from torch.amp import autocast # NEW: For Flash Attention compatibility

# Import your custom modules
from model import NetHAMLModel
from data_loader import NetHAMLDataset

def evaluate_net_ha_ml():
    # 1. Configuration
    device = torch.device("cuda")
    data_dir = 'data/processed'
    weights_path = "net_ha_ml_best.pth" 
    
    # 2. Dataset & Loader
    test_dataset = NetHAMLDataset(data_dir)
    # Using batch_size=64 to keep VRAM usage stable on your 8GB 4070
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    target_names = test_dataset.label_encoder.classes_
    
    # 3. Initialize Architecture
    print(f"Loading Net-HA-ML (4-layer, 256-dim, 1024-sequence)...")
    model = NetHAMLModel(
        num_classes=11, 
        temporal_dim=5,     
        d_model=256,        
        num_layers=4        
    ).to(device)
    
    # Load weights
    try:
        # weights_only=True is best practice for security and speed
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"Successfully loaded weights from {weights_path}")
    except FileNotFoundError:
        print(f"Error: {weights_path} not found.")
        return

    model.eval() 
    
    all_preds = []
    all_labels = []
    
    print(f"Running Inference on {len(test_dataset)} flows (using Flash Attention)...")
    
    # torch.no_grad() reduces memory consumption during inference
    with torch.no_grad():
        # NEW: autocast is REQUIRED here to enable the Flash Attention kernel in model.py
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            for temporal, spatial, metadata, labels, _ in test_loader:
                
                # Move batch to GPU
                temporal = temporal.to(device)
                spatial = spatial.to(device)
                metadata = metadata.to(device)
                
                # Forward pass
                class_out, _ = model(temporal, spatial, metadata)
                
                # Get predicted class
                _, predicted = torch.max(class_out, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
    print("\n=== EVALUATION COMPLETE ===")
    
    # 4. Generate Classification Report
    print("\nClassification Report:")
    # target_names provides the actual labels (VOIP, FT, etc.)
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
    
    # 5. Render High-Res Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(14, 11))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    
    plt.title('Net-HA-ML: Traffic Classification Performance (1024 Packets)', fontsize=16)
    plt.ylabel('True Traffic Class', fontsize=14)
    plt.xlabel('Predicted Traffic Class', fontsize=14)
    
    plot_path = 'confusion_matrix_final.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"\nSaved high-resolution matrix to: {plot_path}")

if __name__ == "__main__":
    evaluate_net_ha_ml()
