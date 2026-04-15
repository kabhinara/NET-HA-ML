import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score
from torch.utils.data import DataLoader
from torch.amp import autocast
import transformer_engine.pytorch as te

# Import your custom modules
from model import NetHAMLModel
from data_loader import NetHAMLDataset

def evaluate_net_ha_ml():
    device = torch.device("cuda")
    data_dir = 'data/processed'
    weights_path = "net_ha_ml_best.pth" 
    
    # 2. Dataset & Loader
    test_dataset = NetHAMLDataset(data_dir)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    target_names = test_dataset.label_encoder.classes_
    num_classes = len(target_names)
    
    print(f"Loading Net-HA-ML (4-layer, 256-dim)...")
    model = NetHAMLModel(
        num_classes=num_classes, 
        temporal_dim=5,
        metadata_dim=7,     
        d_model=256,        
        num_layers=4        
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"Successfully loaded weights from {weights_path}")
    except FileNotFoundError:
        print(f"Error: {weights_path} not found.")
        return

    model.eval() 
    
    all_preds = []
    all_labels = []
    
    print(f"Running Inference on ALL {len(test_dataset)} flows (using Flash Attention & FP8)...")
    
    with torch.no_grad():
        with autocast('cuda', dtype=torch.bfloat16):
            with te.fp8_autocast(enabled=True):
                for temporal, spatial, metadata, labels, pkt_counts, filenames in test_loader:
                    # Pad batch size to multiple of 8 if necessary for FP8 execution
                    pad_len = (8 - temporal.size(0) % 8) % 8
                    if pad_len > 0:
                        temporal = torch.cat([temporal, temporal[:pad_len]], dim=0)
                        spatial = torch.cat([spatial, spatial[:pad_len]], dim=0)
                        metadata = torch.cat([metadata, metadata[:pad_len]], dim=0)
                    
                    class_out, _ = model(temporal, spatial, metadata)
                    
                    if pad_len > 0:
                        class_out = class_out[:-pad_len]
                        
                    _, predicted = torch.max(class_out, 1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
    print("\n=== EVALUATION COMPLETE ===")
    
    print("\nClassification Report (4 Decimals):")
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4, zero_division=0))
    
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)
    
    print("\n=== OVERALL METRICS (4 DECIMALS) ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(14, 11))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    
    plt.title('Net-HA-ML: Traffic Classification Performance', fontsize=16)
    plt.ylabel('True Traffic Class', fontsize=14)
    plt.xlabel('Predicted Traffic Class', fontsize=14)
    
    plot_path = 'confusion_matrix_final.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"\nSaved high-resolution matrix to: {plot_path}")

if __name__ == "__main__":
    evaluate_net_ha_ml()
