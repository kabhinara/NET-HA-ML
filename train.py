import torch
import torch.nn as nn
import torch.optim as optim
import transformer_engine.pytorch as te
import os
from model import NetHAMLModel
from data_loader import get_net_ha_ml_loaders

def train_net_ha_ml():
    # Performance Optimization: Fused kernels
    torch.backends.cudnn.benchmark = True
    
    device = torch.device("cuda")
    data_dir = "data/processed"
    epochs = 300
    batch_size = 128
    accumulation_steps = 2 # Effective batch size = 256
    
    model = NetHAMLModel(num_classes=11, temporal_dim=16).to(device)
    train_loader = get_net_ha_ml_loaders(data_dir, batch_size=batch_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    stagnation_counter = 0
    regression_counter = 0
    
    print(f"Starting Training on {torch.cuda.get_device_name(0)}...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        optimizer.zero_grad()
        
        for i, (temporal, spatial, labels) in enumerate(train_loader):
            temporal, spatial, labels = temporal.to(device), spatial.to(device), labels.to(device)
            
            with te.fp8_autocast(enabled=True):
                outputs = model(temporal, spatial)
                loss = criterion(outputs, labels) / accumulation_steps
                
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, Best: {best_acc:.2f}%, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            regression_counter = 0
            stagnation_counter = 0
            torch.save(model.state_dict(), "net_ha_ml_best.pth")
        else:
            if epoch_acc < best_acc - 0.5:
                regression_counter += 1
            if regression_counter >= 4:
                print(f"Regression detected (4 epochs down)! Aborting at Epoch {epoch+1}.")
                break
            
            stagnation_counter += 1
            if stagnation_counter >= 15:
                print(f"Model stuck (no improvement for 15 epochs)! Aborting at Epoch {epoch+1}.")
                break

    torch.save(model.state_dict(), "net_ha_ml_v1.pth")
    print("Training Complete. Model saved.")
    
    # Final check
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for temporal, spatial, labels in train_loader:
            outputs = model(temporal.to(device), spatial.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
    print(f'Accuracy of the network on the training sets: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    train_net_ha_ml()
