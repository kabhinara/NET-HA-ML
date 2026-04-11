import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformer_engine.pytorch as te
from data_loader import NetHAMLDataset

class SimpleMLP(nn.Module):
    def __init__(self, num_classes=16):
        super().__init__()
        self.net = nn.Sequential(
            te.Linear(32, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            te.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            te.Linear(128, num_classes)
        )
    def forward(self, t, s):
        t = t.view(t.size(0), -1)
        return self.net(t)

if __name__ == '__main__':
    ds = NetHAMLDataset('data/iscx_vpn2016.csv')
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)
    device = 'cuda'
    model = SimpleMLP().to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(10):
        model.train()
        total_acc = 0
        total = 0
        for t, s, l in loader:
            t, s, l = t.to(device), s.to(device), l.to(device)
            opt.zero_grad()
            with te.fp8_autocast(enabled=True):
                out = model(t, s)
                loss = crit(out, l)
            loss.backward()
            opt.step()
            
            _, p = out.max(1)
            total_acc += (p == l).sum().item()
            total += l.size(0)
        print(f"Epoch {epoch}, Acc: {total_acc/total*100:.2f}%")
