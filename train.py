import torch
import torch.nn as nn
import torch.optim as optim
import transformer_engine.pytorch as te
from model import NetHAMLModel
from data_loader import get_net_ha_ml_loaders

def train_net_ha_ml():
    device = torch.device("cuda")
    csv_path = "data/iscx_vpn2016.csv"
    epochs = 150
    batch_size = 128
    
    model = NetHAMLModel(num_classes=16).to(device)
    train_loader = get_net_ha_ml_loaders(csv_path, batch_size=batch_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4, steps_per_epoch=len(train_loader), epochs=epochs)
    
    print(f"Starting Training on {torch.cuda.get_device_name(0)}...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for temporal, spatial, labels in train_loader:
            temporal, spatial, labels = temporal.to(device), spatial.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with te.fp8_autocast(enabled=True):
                outputs = model(temporal, spatial)
                loss = criterion(outputs, labels)
                
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
            
        # Early exit if we hit >99.0% on the training set
        if epoch_acc >= 99.0:
            print(f"Goal Reached! 99% accuracy achieved at Epoch {epoch+1}!")
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
