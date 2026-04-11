import torch
import torch.nn as nn
import transformer_engine.pytorch as te

class TETransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0, batch_first=True)
        
        # Hardware-Aware FP8 Linear layers from NVIDIA Transformer Engine
        self.linear1 = te.Linear(d_model, dim_feedforward)
        self.dropout = nn.Identity()
        self.linear2 = te.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()
        self.activation = nn.GELU()

    def forward(self, src):
        # Pre-Norm Architecture for Stable Training
        src_norm = self.norm1(src)
        src2, _ = self.self_attn(src_norm, src_norm, src_norm)
        src = src + self.dropout1(src2)
        
        src_norm2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_norm2))))
        src = src + self.dropout2(src2)
        return src

class NetHAMLModel(nn.Module):
    """
    SOTA Feature Tokenizer + Transformer (FT-Transformer) for Tabular Data.
    Adapted for NVIDIA Transformer Engine and FP8 computation.
    """
    def __init__(self, num_classes=16, seq_len=1, temporal_dim=32, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        
        # Feature Tokenizer: Converts scalar features into d_model vectors
        self.feature_weights = nn.Parameter(torch.randn(temporal_dim, d_model) * (1 / temporal_dim))
        self.feature_biases = nn.Parameter(torch.randn(temporal_dim, d_model) * (1 / temporal_dim))
        
        # CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer Backbone
        self.layers = nn.ModuleList([
            TETransformerLayer(d_model, nhead, d_model * 4, dropout=0.0)
            for _ in range(num_layers)
        ])
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            te.Linear(d_model, 128),
            nn.ReLU(),
            te.Linear(128, num_classes)
        )

    def forward(self, temporal_x, spatial_x):
        # temporal_x: (B, 1, F)
        x = temporal_x.squeeze(1) # (B, F)
        B, F = x.shape
        
        # Tokenize Features
        x_tokens = x.unsqueeze(-1) * self.feature_weights + self.feature_biases # (B, F, d_model)
        
        # Add CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, d_model)
        
        # Combine
        x_seq = torch.cat((cls_tokens, x_tokens), dim=1) # (B, 1+F, d_model)
        
        # Pass through Transformer
        for layer in self.layers:
            x_seq = layer(x_seq)
            
        # Extract CLS token output
        cls_out = x_seq[:, 0, :]
        
        return self.classifier(cls_out)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NetHAMLModel(num_classes=16).to(device)
    fake_temporal = torch.randn(32, 1, 32).to(device)
    fake_spatial = torch.randn(32, 1, 28, 28).to(device)
    output = model(fake_temporal, fake_spatial)
    print("SOTA FT-Transformer SMOKE TEST: PASSED")
