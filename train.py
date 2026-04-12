import torch
import torch.nn as nn
import torch.optim as optim
import transformer_engine.pytorch as te
from model import NetHAMLModel
from data_loader import get_net_ha_ml_loaders

def train_net_ha_ml():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    data_dir = "data/processed"
    epochs = 300
    batch_size = 64
    
    # Class weights for Weighted CrossEntropy
    # [CHAT, FT, MAIL, STREAMING, VOIP, VPN-CHAT, VPN-FT, VPN-MAIL, VPN-P2P, VPN-STREAMING, VPN-VOIP]
    # Based on rough frequency estimates:
    weights = torch.tensor([1.5, 0.6, 2.0, 1.2, 1.0, 4.0, 3.5, 8.0, 4.5, 2.5, 1.5]).to(device)
    
    model = NetHAMLModel(num_classes=11, temporal_dim=16).to(device)
    train_loader = get_net_ha_ml_loaders(data_dir, batch_size=batch_size)
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    
    print(f"Starting Training on {torch.cuda.get_device_name(0)}...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for temporal, spatial, metadata, labels in train_loader:
            temporal, spatial, metadata, labels = temporal.to(device), spatial.to(device), metadata.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with te.fp8_autocast(enabled=True):
                outputs = model(temporal, spatial, metadata)
                loss = criterion(outputs, labels)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, Best: {best_acc:.2f}%, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), "net_ha_ml_best.pth")

    print("Training Complete.")

if __name__ == "__main__":
    train_net_ha_ml()
