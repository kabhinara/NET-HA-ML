import torch
import os

def expose_the_ft_fraud():
    data_dir = 'data/processed'
    for f in os.listdir(data_dir):
        if f.endswith('.pt'):
            d = torch.load(os.path.join(data_dir, f))
            if d['label'] == "FT":
                # Grab the first flow in the batch, and its first 30 packet sizes
                sizes = d['temporal'][0, :30, 0].numpy()
                print(f"Raw packet sizes for an 'FT' flow:\n{sizes}")
                return

expose_the_ft_fraud()
