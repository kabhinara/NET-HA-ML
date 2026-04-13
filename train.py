import torch
import torch.nn as nn
import torch.optim as optim
import transformer_engine.pytorch as te
from torch import autocast                  # NEW: Import autocast
from torch.cuda.amp import GradScaler       # NEW: Import GradScaler
from model import NetHAMLModel
from data_loader import NetHAMLDataset

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        raw_ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-raw_ce_loss)
        focal_term = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_weights = self.alpha[targets]
            focal_loss = alpha_weights * focal_term * raw_ce_loss
        else:
            focal_loss = focal_term * raw_ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

torch.set_float32_matmul_precision('high')	

def train_net_ha_ml():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    data_dir = "data/processed"
    epochs = 300
    
    batch_size = 64            
    accumulation_steps = 1   
    lambda_reg = 0.1

    dataset = NetHAMLDataset(data_dir)
    num_samples = len(dataset.labels)
    
    weights = torch.tensor([1.5, 3.5, 4.0, 1.2, 1.0, 4.0, 3.5, 12.0, 4.5, 2.5, 1.5]).to(device)
    model = NetHAMLModel(
        num_classes=11, 
        temporal_dim=5, 
        d_model=256, 
        num_layers=4
    ).to(device)
    
    criterion_class = FocalLoss(alpha=weights, gamma=2.0)
    criterion_reg = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01, fused=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # NEW: Initialize the Gradient Scaler
    scaler = torch.amp.GradScaler('cuda')

    print(f"Starting Training: Physical Batch {batch_size}, Effective Batch {batch_size * accumulation_steps}")
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        correct = torch.tensor(0, device=device)
        
        indices = torch.randperm(num_samples, device=device)
        optimizer.zero_grad() 

        for i in range(0, num_samples - batch_size, batch_size):
            batch_idx = indices[i:i+batch_size]
            
            temporal = dataset.temporal[batch_idx]
            spatial = dataset.spatial[batch_idx]
            metadata = dataset.metadata[batch_idx]
            labels = dataset.labels[batch_idx]
            pkt_counts = dataset.pkt_counts[batch_idx]

            # NEW: Force bfloat16 for Flash Attention compatibility, safely nesting the TE FP8 cast
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                with te.fp8_autocast(enabled=True):
                    class_preds, pkt_preds = model(temporal, spatial, metadata)
                    loss_class = criterion_class(class_preds, labels)
                    
                    log_true_counts = torch.log1p(pkt_counts)
                    loss_reg = criterion_reg(pkt_preds.squeeze(), log_true_counts.squeeze())
                    
                    loss = (loss_class + (lambda_reg * loss_reg)) / accumulation_steps

            # NEW: Use scaler for backward pass
            scaler.scale(loss).backward()

            if (i // batch_size + 1) % accumulation_steps == 0:
                # NEW: Use scaler to step the optimizer and update
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.detach() * accumulation_steps
            _, predicted = torch.max(class_preds, 1)
            correct += (predicted == labels).sum()

        epoch_loss = total_loss.item() / (num_samples // batch_size)
        epoch_acc = 100 * correct.item() / num_samples
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")	
       
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), "net_ha_ml_best.pth")
            print(f"  --> New Best Accuracy! Saved: {best_acc:.2f}%")
        
        torch.cuda.empty_cache()

    print("Training Complete.")
    torch.save(model.state_dict(), "net_ha_ml_final.pth")

if __name__ == "__main__":
    train_net_ha_ml()
