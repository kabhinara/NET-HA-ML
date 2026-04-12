import os
import glob
import torch
import numpy as np
import time
import struct
from scapy.utils import RawPcapReader, RawPcapNgReader

def get_label(filename):
    fname = os.path.basename(filename).lower()
    is_vpn = 'vpn' in fname
    
    if any(k in fname for k in ['chat', 'aim', 'icq']): base = "CHAT"
    elif 'email' in fname: base = "MAIL"
    elif any(k in fname for k in ['ftps', 'scp', 'sftp', 'skype_file', 'skype_files']): base = "FT"
    elif any(k in fname for k in ['audio', 'video', 'voip', 'skype']): base = "VOIP"
    elif any(k in fname for k in ['netflix', 'spotify', 'vimeo', 'youtube']): base = "STREAMING"
    elif 'bittorrent' in fname: base = "P2P"
    else: base = "BROWSING"
    
    return f"VPN-{base}" if is_vpn else base

def parse_packet(pkt_data, linktype):
    offset = 0
    if linktype == 1: # Ethernet
        if len(pkt_data) < 14: return None
        ethertype = struct.unpack("!H", pkt_data[12:14])[0]
        offset = 14
        if ethertype == 0x8100: # VLAN
            if len(pkt_data) < 18: return None
            ethertype = struct.unpack("!H", pkt_data[16:18])[0]
            offset = 18
        if ethertype != 0x0800: return None
    elif linktype in [101, 12, 14, 228, 229]: # Raw IP (VPN tun0 captures)
        offset = 0
        if len(pkt_data) < 20 or (pkt_data[0] >> 4) != 4: return None
    elif linktype == 113: # Linux SLL
        if len(pkt_data) < 16: return None
        ethertype = struct.unpack("!H", pkt_data[14:16])[0]
        offset = 16
        if ethertype != 0x0800: return None
    elif linktype == 0: # Loopback Null
        offset = 4
        if len(pkt_data) < 24 or (pkt_data[offset] >> 4) != 4: return None
    else:
        return None

    ip_header = pkt_data[offset:]
    if len(ip_header) < 20: return None
    
    ihl = (ip_header[0] & 0x0F) * 4
    proto = ip_header[9]
    src_ip = ip_header[12:16]
    dst_ip = ip_header[16:20]
    
    if proto not in (6, 17): # Not TCP or UDP
        return None
        
    transport_offset = offset + ihl
    if len(pkt_data) < transport_offset + 4: return None
    
    sport, dport = struct.unpack("!HH", pkt_data[transport_offset:transport_offset+4])
    
    # We use raw bytes for the flow tuple to make it fast
    flow_id = tuple(sorted([src_ip + struct.pack("!H", sport), dst_ip + struct.pack("!H", dport)]) + [bytes([proto])])
    
    # TCP header length is in byte 12 of TCP header
    if proto == 6:
        if len(pkt_data) < transport_offset + 13:
            payload_offset = transport_offset + 20
        else:
            tcp_hl = ((pkt_data[transport_offset + 12] >> 4) & 0x0F) * 4
            payload_offset = transport_offset + tcp_hl
    else:
        payload_offset = transport_offset + 8
        
    payload = pkt_data[payload_offset:]
    return flow_id, payload

def preprocess_pcap(pcap_path, output_dir):
    out_path = os.path.join(output_dir, os.path.basename(pcap_path) + '.pt')
    if os.path.exists(out_path):
        print(f"Skipping {os.path.basename(pcap_path)}, already processed.", flush=True)
        return
        
    label_str = get_label(pcap_path)
    flows = {}
    
    print(f"Processing {os.path.basename(pcap_path)} -> Label: {label_str}", flush=True)
    
    # Determine reader based on extension
    ReaderClass = RawPcapNgReader if pcap_path.lower().endswith('.pcapng') else RawPcapReader
    
    try:
        with ReaderClass(pcap_path) as pcap_reader:
            global_linktype = getattr(pcap_reader, 'linktype', 1) # Default to Ethernet if missing
            for i, pcap_data in enumerate(pcap_reader):
                # We can now increase the limit drastically to 500k packets per file because we use raw parsing!
                if i >= 500000:
                    break
                    
                pkt_data = pcap_data[0] if isinstance(pcap_data, tuple) else pcap_data
                if not isinstance(pkt_data, bytes):
                    continue
                
                ts = 0.0
                wirelen = len(pkt_data)
                linktype = global_linktype
                
                if isinstance(pcap_data, tuple) and len(pcap_data) > 1:
                    meta = pcap_data[1]
                    if hasattr(meta, 'sec') and hasattr(meta, 'usec'):
                        ts = meta.sec + (meta.usec / 1e6)
                    if hasattr(meta, 'wirelen'):
                        wirelen = meta.wirelen
                    if hasattr(meta, 'linktype'):
                        linktype = meta.linktype
                
                parsed = parse_packet(pkt_data, linktype)
                if parsed is None: continue
                
                flow_id, payload = parsed
                
                if flow_id not in flows:
                    flows[flow_id] = {'packets': [], 'start_time': ts, 'payload_bytes': bytearray()}
                    
                flow = flows[flow_id]
                
                if len(flow['packets']) < 32:
                    iat = ts - flow['start_time'] if len(flow['packets']) == 0 else ts - flow['packets'][-1]['time']
                    flow['packets'].append({
                        'size': wirelen,
                        'iat': iat,
                        'time': ts
                    })
                    
                if len(flow['payload_bytes']) < 784 and payload:
                    flow['payload_bytes'].extend(payload)
                    
    except Exception as e:
        print(f"  [!] Error reading {pcap_path}: {e}")
        return
        
    X_temporal = []
    X_spatial = []
    
    for flow_id, flow_data in flows.items():
        if len(flow_data['packets']) < 3:
            continue
            
        # Padded to 1024 packets (16 for FP8 Tensor Cores)
        temporal = np.zeros((1024, 16), dtype=np.float32)
        for j, p in enumerate(flow_data['packets']):
            temporal[j, 0] = p['size']
            temporal[j, 1] = p['iat']

        # Spatial: Flatten 4096 bytes into 64x64 image
        spatial = np.zeros(4096, dtype=np.float32)
        payload = flow_data['payload_bytes'][:4096]
        spatial[:len(payload)] = list(payload)
        spatial = spatial.reshape((1, 64, 64)) / 255.0 # Normalize pixel values
        X_temporal.append(temporal)
        X_spatial.append(spatial)
        
    if len(X_temporal) == 0:
        print("  -> No valid flows found.", flush=True)
        return
        
    X_temporal = torch.tensor(np.array(X_temporal))
    X_spatial = torch.tensor(np.array(X_spatial))
    
    torch.save({
        'temporal': X_temporal,
        'spatial': X_spatial,
        'label': label_str
    }, out_path)
    print(f"  -> Saved {len(X_temporal)} flows to {out_path}", flush=True)

if __name__ == '__main__':
    output_directory = 'data/processed'
    os.makedirs(output_directory, exist_ok=True)
    
    pcap_files = glob.glob('data/**/*.pcap', recursive=True) + glob.glob('data/**/*.pcapng', recursive=True)
    print(f"Found {len(pcap_files)} PCAP files. Starting ultra-fast raw preprocessing...", flush=True)
    
    start_time = time.time()
    for p in pcap_files:
        preprocess_pcap(p, output_directory)
        
    print(f"\nRaw Preprocessing Complete! Took {time.time() - start_time:.2f} seconds.", flush=True)
