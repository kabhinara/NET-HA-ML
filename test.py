import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Import your custom modules
from model import NetHAMLModel
from data_loader import NetHAMLDataset
from torch.utils.data import DataLoader

def evaluate_net_ha_ml():
    # 1. Configuration
    device = torch.device("cuda")
    
    # IMPORTANT: Point this to your TEST data. 
    # If you haven't split your data, use the same CSV for now just to test the code.
    test_csv_path = "data/iscx_vpn2016.csv" 
    
    # 2. Safe Inference Loader
    test_dataset = NetHAMLDataset('data/processed')
    
    # We keep drop_last=True because the RTX 4070 still requires 
    # batch alignment (mult of 8) for the FP8 inference kernels!
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)
    
    # 3. Load the Architecture and Weights
    print("Loading Net-HA-ML Architecture...")
    model = NetHAMLModel(num_classes=11, temporal_dim=16).to(device)
    
    # Load the weights from the training run
    model.load_state_dict(torch.load("net_ha_ml_best.pth", weights_only=True))
    model.eval() # Disable dropout and batch norm
    
    # Tracking
    all_preds = []
    all_labels = []
    
    print(f"Running Inference on {device}...")
    with torch.no_grad():
        for temporal, spatial, metadata, labels in test_loader:
            temporal, spatial, metadata, labels = temporal.to(device), spatial.to(device), metadata.to(device), labels.to(device)
    
            # You MUST also pass the metadata to the model forward pass
            outputs = model(temporal, spatial, metadata)	    
            temporal, spatial = temporal.to(device), spatial.to(device)
            
            # Forward pass
            outputs = model(temporal, spatial, metadata)
            _, predicted = torch.max(outputs.data, 1)
            
            # Move back to CPU for Scikit-Learn
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print("\n=== INFERENCE COMPLETE ===")
    
    # 4. Generate Classification Report (Precision, Recall, F1)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))
    
    # 5. Render High-Res Confusion Matrix for the Report
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                cbar_kws={'label': 'Number of Packets'})
    
    plt.title('Net-HA-ML: FP8 Confusion Matrix (ISCX-VPN2016)', fontsize=16)
    plt.ylabel('True Traffic Class', fontsize=14)
    plt.xlabel('Predicted Traffic Class', fontsize=14)
    
    # Save the plot directly to your folder
    plot_path = 'confusion_matrix_v1.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300) # 300 DPI is standard for academic papers
    print(f"\nSaved high-resolution matrix to: {plot_path}")

if __name__ == "__main__":
    evaluate_net_ha_ml()
