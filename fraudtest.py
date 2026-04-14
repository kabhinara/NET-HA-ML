import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

# Import your existing pipeline
from model import NetHAMLModel
from data_loader import NetHAMLDataset

def test_and_track_errors():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = 'data/processed'
    model_path = "net_ha_ml_best.pth"

    print("Loading Dataset & Normalization Scalers...")
    test_dataset = NetHAMLDataset(data_dir)
    target_names = test_dataset.label_encoder.classes_
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("Loading Model...")
    model = NetHAMLModel(num_classes=len(target_names), temporal_dim=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []

    print("\n" + "="*80)
    print(f"{'SOURCE FILE':<35} | {'TRUE LABEL':<15} | {'PREDICTED LABEL':<15}")
    print("="*80)

    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            for batch in test_loader:
                # Safely unpack the batch (checks if your DataLoader returns 5 or 6 items)
                if len(batch) >= 5:
                    temp = batch[0].to(device)
                    spat = batch[1].to(device)
                    meta = batch[2].to(device)
                    labels = batch[3].to(device)
                    filenames = batch[-1] # Assumes filename is the very last item returned
                else:
                    raise ValueError("Your NetHAMLDataset is not returning filenames! See note below.")

                out, _ = model(temp, spat, meta)
                preds = torch.argmax(out, dim=1)

                preds_cpu = preds.cpu().numpy()
                labels_cpu = labels.cpu().numpy()

                all_preds.extend(preds_cpu)
                all_labels.extend(labels_cpu)

                # Real-time error tracking
                for i in range(len(preds_cpu)):
                    if preds_cpu[i] != labels_cpu[i]:
                        true_class = target_names[labels_cpu[i]]
                        pred_class = target_names[preds_cpu[i]]
                        fname = filenames[i]
                        
                        # Prints immediately to the terminal when the model gets it wrong
                        print(f"{fname:<35} | {true_class:<15} | {pred_class:<15}")

    print("="*80)
    print("\nEvaluation complete. Generating metrics...")

    # 1. Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # 2. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Net-HA-ML: Real-Time Error Tracking Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    
    matrix_name = 'confusion_matrix_tracked.png'
    plt.savefig(matrix_name, dpi=300)
    print(f"Saved high-resolution matrix to: {matrix_name}\n")

if __name__ == "__main__":
    test_and_track_errors()
