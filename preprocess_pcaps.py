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
    if linktype == 1:
        if len(pkt_data) < 14: return None
        ethertype = struct.unpack("!H", pkt_data[12:14])[0]
        offset = 14
        if ethertype == 0x8100:
            if len(pkt_data) < 18: return None
            ethertype = struct.unpack("!H", pkt_data[16:18])[0]
            offset = 18
        if ethertype != 0x0800: return None
    elif linktype in [101, 12, 14, 228, 229]:
        offset = 0
        if len(pkt_data) < 20 or (pkt_data[0] >> 4) != 4: return None
    elif linktype == 113:
        if len(pkt_data) < 16: return None
        ethertype = struct.unpack("!H", pkt_data[14:16])[0]
        offset = 16
        if ethertype != 0x0800: return None
    elif linktype == 0:
        offset = 4
        if len(pkt_data) < 24 or (pkt_data[offset] >> 4) != 4: return None
    else: return None

    ip_header = pkt_data[offset:]
    if len(ip_header) < 20: return None
    src_ip = ip_header[12:16]
    dst_ip = ip_header[16:20]
    proto = ip_header[9]
    ihl = (ip_header[0] & 0x0F) * 4
    
    if proto not in (6, 17): return None
    transport_offset = offset + ihl
    if len(pkt_data) < transport_offset + 4: return None
    sport, dport = struct.unpack("!HH", pkt_data[transport_offset:transport_offset+4])
    flow_id = tuple(sorted([src_ip + struct.pack("!H", sport), dst_ip + struct.pack("!H", dport)]) + [bytes([proto])])
    
    if proto == 6:
        if len(pkt_data) < transport_offset + 13: payload_offset = transport_offset + 20
        else: payload_offset = transport_offset + ((pkt_data[transport_offset + 12] >> 4) & 0x0F) * 4
    else: payload_offset = transport_offset + 8
    
    return flow_id, pkt_data[payload_offset:], src_ip

def preprocess_pcap(pcap_path, output_dir):
    out_path = os.path.join(output_dir, os.path.basename(pcap_path) + '.pt')
    if os.path.exists(out_path): return
    label_str = get_label(pcap_path)
    flows = {}
    print(f"Processing {os.path.basename(pcap_path)} -> Label: {label_str}", flush=True)
    ReaderClass = RawPcapNgReader if pcap_path.lower().endswith('.pcapng') else RawPcapReader
    try:
        with ReaderClass(pcap_path) as pcap_reader:
            global_linktype = getattr(pcap_reader, 'linktype', 1)
            for i, pcap_data in enumerate(pcap_reader):
                if i >= 500000: break
                pkt_data = pcap_data[0] if isinstance(pcap_data, tuple) else pcap_data
                if not isinstance(pkt_data, bytes): continue
                
                ts, wirelen = 0.0, len(pkt_data)
                linktype = global_linktype
                if isinstance(pcap_data, tuple) and len(pcap_data) > 1:
                    meta = pcap_data[1]
                    if hasattr(meta, 'sec'): ts = meta.sec + (meta.usec / 1e6)
                    if hasattr(meta, 'wirelen'): wirelen = meta.wirelen
                    if hasattr(meta, 'linktype'): linktype = meta.linktype
                
                parsed = parse_packet(pkt_data, linktype)
                if parsed is None: continue
                flow_id, payload, src_ip = parsed
                
                if flow_id not in flows:
                    flows[flow_id] = {'packets': [], 'start_time': ts, 'payload_bytes': bytearray(), 'initiator': src_ip}
                flow = flows[flow_id]
                
                if len(flow['packets']) < 128:
                    direction = 1 if src_ip == flow['initiator'] else -1
                    iat = ts - flow['start_time'] if len(flow['packets']) == 0 else ts - flow['packets'][-1]['time']
                    flow['packets'].append({'size': wirelen * direction, 'iat': iat, 'time': ts})
                    
                if len(flow['payload_bytes']) < 4096 and payload:
                    flow['payload_bytes'].extend(payload)
                    
    except Exception as e:
        print(f"  [!] Error: {e}")
        return
        
    X_temporal, X_spatial, X_meta = [], [], []
    for flow_id, flow_data in flows.items():
        if len(flow_data['packets']) < 3: continue
        
        # 1. Temporal
        temporal = np.zeros((128, 16), dtype=np.float32)
        for j, p in enumerate(flow_data['packets']):
            temporal[j, 0] = p['size']
            temporal[j, 1] = p['iat']
        
        # 2. Spatial
        spatial = np.zeros(4096, dtype=np.float32)
        payload = flow_data['payload_bytes'][:4096]
        spatial[:len(payload)] = list(payload)
        spatial = spatial.reshape((1, 64, 64)) / 255.0
        
        # 3. Metadata
        total_bytes = sum(abs(p['size']) for p in flow_data['packets'])
        duration = flow_data['packets'][-1]['time'] - flow_data['packets'][0]['time']
        mean_size = total_bytes / len(flow_data['packets'])
        metadata = np.array([total_bytes, duration, mean_size], dtype=np.float32)
        
        X_temporal.append(temporal)
        X_spatial.append(spatial)
        X_meta.append(metadata)
    
    if len(X_temporal) == 0: return
    torch.save({
        'temporal': torch.tensor(np.array(X_temporal)), 
        'spatial': torch.tensor(np.array(X_spatial)), 
        'metadata': torch.tensor(np.array(X_meta)),
        'label': label_str
    }, out_path)
    print(f"  -> Saved {len(X_temporal)} flows", flush=True)

if __name__ == '__main__':
    output_directory = 'data/processed'
    os.makedirs(output_directory, exist_ok=True)
    pcap_files = glob.glob('data/**/*.pcap', recursive=True) + glob.glob('data/**/*.pcapng', recursive=True)
    for p in pcap_files: preprocess_pcap(p, output_directory)
