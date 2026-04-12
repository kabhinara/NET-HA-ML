import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
import math

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
    def __init__(self, d_model, max_len=129):
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
    def __init__(self, num_classes=11, temporal_dim=4, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.num_classes = num_classes
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            te.Linear(256 * 8 * 8, 512),
            nn.ReLU()
        )
        
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(temporal_dim, 128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, d_model, kernel_size=3, padding='same'),
            nn.BatchNorm1d(d_model), nn.ReLU()
        )
        
        self.pos_encoder = PositionalEncoding(d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.transformer_layers = nn.ModuleList([
            TETransformerLayer(d_model, nhead, d_model * 4, dropout=0.0)
            for _ in range(num_layers)
        ])
        
        self.temporal_out = te.Linear(d_model, 512)
        
        self.metadata_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        
        # Classifier with output padding for FP8
        self.classifier = nn.Sequential(
            te.Linear(512 + 512 + 16, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            te.Linear(1024, 16)
        )

    def forward(self, temporal_x, spatial_x, metadata_x):
        B = temporal_x.size(0)
        spatial_feat = self.cnn(spatial_x)
        
        x = temporal_x.transpose(1, 2)
        x = self.temporal_encoder(x)
        x = x.transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_seq = torch.cat((cls_tokens, x), dim=1)
        x_seq = self.pos_encoder(x_seq)
        
        # Direct layers (no checkpointing)
        for layer in self.transformer_layers:
            x_seq = layer(x_seq)
            
        temporal_feat = self.temporal_out(x_seq[:, 0, :])
        meta_feat = self.metadata_mlp(metadata_x)
        
        fused = torch.cat((spatial_feat, temporal_feat, meta_feat), dim=1)
        return self.classifier(fused)[:, :self.num_classes]
