import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class NetHAMLDataset(Dataset):
    def __init__(self, data_dir='data/processed'):
        pt_files = glob.glob(os.path.join(data_dir, '*.pt'))
        
        all_temporal, all_spatial, all_metadata, all_labels = [], [], [], []
        
        print(f"Loading {len(pt_files)} processed PCAP tensor files...")
        for f in pt_files:
            data = torch.load(f, weights_only=True)
            num_flows = data['temporal'].shape[0]
            all_temporal.append(data['temporal'])
            all_spatial.append(data['spatial'])
            all_metadata.append(data['metadata'])
            all_labels.extend([data['label']] * num_flows)
            
        self.temporal = torch.cat(all_temporal, dim=0)
        self.spatial = torch.cat(all_spatial, dim=0)
        self.metadata = torch.cat(all_metadata, dim=0)
        
        self.label_encoder = LabelEncoder()
        self.labels = torch.tensor(self.label_encoder.fit_transform(all_labels), dtype=torch.long)
        
        print(f"Loaded {len(self.labels)} total network flows.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        temporal = self.temporal[idx][:128, :]
        spatial = self.spatial[idx]
        metadata = torch.log1p(self.metadata[idx])
        return temporal, spatial, metadata, self.labels[idx]

def get_net_ha_ml_loaders(data_dir='data/processed', batch_size=64):
    dataset = NetHAMLDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
