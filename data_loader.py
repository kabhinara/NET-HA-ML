import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder

class NetHAMLDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        
        # 1. Labels
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.df['traffic_type'])
        
        # 2. Features (All 23 statistical columns)
        feature_cols = [
            'duration', 'total_fiat', 'total_biat', 'min_fiat', 'min_biat', 
            'max_fiat', 'max_biat', 'mean_fiat', 'mean_biat', 'flowPktsPerSecond', 
            'flowBytesPerSecond', 'min_flowiat', 'max_flowiat', 'mean_flowiat', 
            'std_flowiat', 'min_active', 'mean_active', 'max_active', 
            'std_active', 'min_idle', 'mean_idle', 'max_idle', 'std_idle'
        ]
        
        # Robustly transform the data because network flow features have extreme outliers
        from sklearn.preprocessing import QuantileTransformer
        self.scaler = QuantileTransformer(output_distribution='normal')
        scaled_data = self.scaler.fit_transform(self.df[feature_cols].values)
        
        # 3. Hardware Alignment: Pad 23 features to 32
        # (32 is divisible by 16, as required by the RTX 4070 FP8 kernels)
        padding_size = 32 - len(feature_cols)
        padding = np.zeros((scaled_data.shape[0], padding_size))
        self.features = np.hstack((scaled_data, padding))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Shape: [1, 32] - Satisfies FP8 kernel requirements
        temporal_x = torch.tensor(self.features[idx], dtype=torch.float32).unsqueeze(0)
        
        # Dummy spatial data for the CNN head
        spatial_x = torch.zeros((1, 28, 28)) 
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return temporal_x, spatial_x, label

def get_net_ha_ml_loaders(csv_path, batch_size=32):
    dataset = NetHAMLDataset(csv_path)
    # drop_last=True is CRITICAL for FP8 batch alignment (Batch must be mult of 8)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
