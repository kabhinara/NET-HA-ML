import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
import math
from torch.utils.checkpoint import checkpoint

class TETransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = te.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = te.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.activation = nn.GELU()

    def forward(self, src):
        src_norm = self.norm1(src)
        src2 = F.scaled_dot_product_attention(src_norm, src_norm, src_norm, is_causal=False)
        src = src + self.dropout1(src2)
        src_norm2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_norm2))))
        src = src + self.dropout2(src2)
        return src

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class NetHAMLModel(nn.Module):
    def __init__(self, num_classes=11, temporal_dim=16, seq_len=1024, d_model=512, nhead=8, num_layers=8):
        super().__init__()
        self.num_classes = num_classes
        
        # --- 1. SPATIAL PATH (CNN) ---
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            te.Linear(256 * 8 * 8, 1024),
            nn.ReLU()
        )
        
        # --- 2. TEMPORAL PATH (1D-CNN Micro-cluster Extraction) ---
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(temporal_dim, 128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, d_model, kernel_size=3, padding='same'),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len+1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.transformer_layers = nn.ModuleList([
            TETransformerLayer(d_model, nhead, (d_model * 4) + (8 - (d_model*4)%8)%8, dropout=0.0)
            for _ in range(num_layers)
        ])
        
        self.temporal_out = te.Linear(d_model, 1024)
        
        # --- 3. FUSION HEAD ---
        self.classifier = nn.Sequential(
            te.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            te.Linear(1024, 16)
        )

    def forward(self, temporal_x, spatial_x):
        B = temporal_x.size(0)
        
        spatial_feat = self.cnn(spatial_x) # (B, 1024)
        
        # Temporal Path (Expects B, F, T)
        x = temporal_x.transpose(1, 2)
        x = self.temporal_encoder(x) # (B, d_model, 1024)
        x = x.transpose(1, 2) # (B, 1024, d_model)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_seq = torch.cat((cls_tokens, x), dim=1) # (B, 1025, d_model)
        x_seq = self.pos_encoder(x_seq)
        
        for i, layer in enumerate(self.transformer_layers):
            if i < 4:
                x_seq = checkpoint(layer, x_seq, use_reentrant=True)
            else:
                x_seq = layer(x_seq)
            
        temporal_feat = self.temporal_out(x_seq[:, 0, :]) # (B, 1024)
        
        fused = torch.cat((spatial_feat, temporal_feat), dim=1) # (B, 2048)
        # Return only the valid classes
        return self.classifier(fused)[:, :self.num_classes]
