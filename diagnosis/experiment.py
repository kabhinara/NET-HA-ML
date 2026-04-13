import torch
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

def diagnostic_analysis(data_dir):
    streak_voip, streak_ft = [], []
    ratio_voip, ratio_ft = [], []
    
    print("Extracting Size Heuristics from processed tensors...")

    for f in os.listdir(data_dir):
        if f.endswith('.pt'):
            d = torch.load(os.path.join(data_dir, f))
            if d['label'] in ["VOIP", "FT", "VPN-VOIP", "VPN-FT"]:
                # temp_batch is (Num_Flows, 1024, 5)
                temp_batch = d['temporal'].numpy()
                
                # Loop through the individual flows in the batch
                for i in range(temp_batch.shape[0]):
                    temp = temp_batch[i] # Now it is correctly (1024, 5)
                    
                    # Extract absolute packet sizes
                    sizes = np.abs(temp[:, 0])

                    # --- Heuristic 1: The MTU Streak ---
                    # Find the longest unbroken chain of packets larger than 1200 bytes
                    is_large = sizes > 1200
                    streaks = [sum(g) for k, g in itertools.groupby(is_large) if k]
                    max_streak = max(streaks) if streaks else 0
                    
                    # --- Heuristic 2: The Small-to-Large Ratio ---
                    # Ratio of control/voice packets (<300B) to data packets (>1200B)
                    num_small = np.sum(sizes < 300)
                    num_large = np.sum(sizes > 1200)
                    size_ratio = num_large / (num_small + 1e-6)
                    
                    if "VOIP" in d['label']:
                        streak_voip.append(max_streak)
                        ratio_voip.append(size_ratio)
                    else:
                        streak_ft.append(max_streak)
                        ratio_ft.append(size_ratio)

    print("Generating KDE plots...")
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.kdeplot(streak_voip, label="VOIP", fill=True)
    sns.kdeplot(streak_ft, label="FT", fill=True)
    plt.title("MTU Streak Separation (Max Consecutive > 1200B)")
    plt.xlabel("Number of Consecutive Large Packets")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    sns.kdeplot(ratio_voip, label="VOIP", fill=True)
    sns.kdeplot(ratio_ft, label="FT", fill=True)
    plt.title("Small-to-Large Packet Ratio")
    plt.xlabel("Ratio (Large Packets / Small Packets)")
    
    # We clip the x-axis for the ratio plot because FT might have extreme outliers
    plt.xlim(-1, max(np.percentile(ratio_voip, 95), np.percentile(ratio_ft, 95)) * 1.5)
    plt.legend()
    
    plt.tight_layout()
    plot_path = "specialist_diagnostic_size.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Diagnostic complete. Check {plot_path}")

if __name__ == "__main__":
    diagnostic_analysis('data/processed')
