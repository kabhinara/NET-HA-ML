import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, confusion_matrix

def test_temporal_cnn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = 'data/processed'
    X_ft, X_voip = [], []
    
    print("Extracting raw temporal sequences (first 256 packets) for Skype FT vs Skype VOIP...")
    
    SEQ_LEN = 256
    
    for f in os.listdir(data_dir):
        if not f.endswith('.pt'): continue
        
        data = torch.load(os.path.join(data_dir, f), map_location='cpu')
        true_label = data['label'].upper()
        filename = f.lower()

        is_ft = 'FT' in true_label or 'FILE' in filename
        is_voip = 'VOIP' in true_label or 'AUDIO' in filename or 'VIDEO' in filename
        is_skype = 'skype' in filename
        
        if not is_skype:
            continue
            
        # temp tensor shape: (flows, 1024, 5)
        # We'll just take the first SEQ_LEN packets to represent the sequence
        temp_data = data['temporal'][:, :SEQ_LEN, :].numpy()
        
        if is_ft:
            X_ft.append(temp_data)
        elif is_voip:
            X_voip.append(temp_data)

    X_ft = np.vstack(X_ft)
    X_voip = np.vstack(X_voip)
    
    # Balance dataset
    X_voip_balanced = resample(X_voip, replace=False, n_samples=len(X_ft), random_state=42)
    
    X = np.vstack([X_ft, X_voip_balanced])
    y = np.array([1] * len(X_ft) + [0] * len(X_voip_balanced))
    
    print(f"Dataset Balanced: {len(X_ft)} Skype FT vs {len(X_voip_balanced)} Skype VOIP")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize features slightly to help CNN learn
    # Features: [Size, IAT, Direction, Flag1, Flag2...]
    # We will just pass them raw but converted to float
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    
    class SimpleTempCNN(nn.Module):
        def __init__(self):
            super(SimpleTempCNN, self).__init__()
            # input shape: (batch, channels=5, seq_len=256)
            self.conv1 = nn.Conv1d(5, 32, kernel_size=5, padding=2)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool1d(2)
            
            self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool1d(2)
            
            self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.relu3 = nn.ReLU()
            self.pool3 = nn.MaxPool1d(2)
            
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(128 * 32, 128) # 256 / 8 = 32
            self.relu4 = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, 2)
            
        def forward(self, x):
            x = x.transpose(1, 2) # (batch, seq_len, channels) -> (batch, channels, seq_len)
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.pool3(self.relu3(self.conv3(x)))
            x = self.flatten(x)
            x = self.dropout(self.relu4(self.fc1(x)))
            x = self.fc2(x)
            return x
            
    model = SimpleTempCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    print(f"Training 1D CNN on exact packet sequences for 20 epochs on {device}...")
    
    for epoch in range(1, 21):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(out, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            
        train_acc = correct / total * 100
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
            
    model.eval()
    with torch.no_grad():
        out = model(X_test_t)
        preds = torch.argmax(out, dim=1).cpu().numpy()
        
    acc = accuracy_score(y_test, preds) * 100
    print(f"\nFinal Sequence CNN Test Accuracy: {acc:.2f}%")
    
    print("\n=== CONFUSION MATRIX ===")
    cm = confusion_matrix(y_test, preds)
    print(f"                 Predicted Skype VOIP    Predicted Skype FT")
    print(f"Actual Skype VOIP      {cm[0, 0]:<20}    {cm[0, 1]:<18}")
    print(f"Actual Skype FT        {cm[1, 0]:<20}    {cm[1, 1]:<18}")

if __name__ == "__main__":
    test_temporal_cnn()
