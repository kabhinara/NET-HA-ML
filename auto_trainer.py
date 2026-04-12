import os
import glob
import time
import subprocess
import torch

def main():
    print("Timer started: Waiting 25 minutes (1500 seconds)...", flush=True)
    time.sleep(1500)
    
    downloads_dirs = [os.path.expanduser("~/Downloads"), os.path.expanduser("~/Download")]
    data_dir = os.path.abspath("data")
    
    print("\n--- PHASE 1: EXTRACTION ---", flush=True)
    for ddir in downloads_dirs:
        if not os.path.exists(ddir):
            continue
            
        zip_files = glob.glob(os.path.join(ddir, "*.zip"))
        for zf in zip_files:
            print(f"Extracting {zf} into {data_dir}...", flush=True)
            # -n: never overwrite, -j: junk paths (flatten directory structure)
            subprocess.run(f"unzip -n -j '{zf}' -d '{data_dir}'", shell=True)
            
        tar_files = glob.glob(os.path.join(ddir, "*.tar.gz")) + glob.glob(os.path.join(ddir, "*.tgz"))
        for tf in tar_files:
            print(f"Extracting {tf} into {data_dir}...", flush=True)
            subprocess.run(f"tar -xzf '{tf}' -C '{data_dir}'", shell=True)
            
        # Also copy loose pcap files just in case they weren't archived
        pcap_files = glob.glob(os.path.join(ddir, "*.pcap")) + glob.glob(os.path.join(ddir, "*.pcapng"))
        for pf in pcap_files:
            print(f"Copying {pf} into {data_dir}...", flush=True)
            subprocess.run(f"cp -n '{pf}' '{data_dir}'", shell=True)

    print("\n--- PHASE 2: PREPROCESSING ---", flush=True)
    subprocess.run("conda run --no-capture-output -n net_ha_ml python preprocess_pcaps.py", shell=True)
    
    print("\n--- PHASE 3: CLASS VALIDATION ---", flush=True)
    pt_files = glob.glob(os.path.join(data_dir, "processed", "*.pt"))
    labels = set()
    for f in pt_files:
        try:
            data = torch.load(f, weights_only=True)
            labels.add(data['label'])
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    print(f"Found {len(labels)} classes: {labels}", flush=True)
    
    print("\n--- PHASE 4: EXECUTION ---", flush=True)
    if len(labels) >= 14:
        print("Target of 14 classes reached. Initiating final model training...", flush=True)
        subprocess.run("conda run --no-capture-output -n net_ha_ml python train.py", shell=True)
        print("Training complete.", flush=True)
    else:
        print(f"Class count ({len(labels)}) is less than 14. Aborting training.", flush=True)
        print("Powering off the PC as requested...", flush=True)
        # Try graceful poweroff
        subprocess.run("systemctl poweroff || sudo poweroff || poweroff", shell=True)

if __name__ == "__main__":
    main()
