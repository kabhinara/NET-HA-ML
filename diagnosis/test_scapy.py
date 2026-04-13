import scapy.all as scapy
import os

pcap_file = 'data/aim_chat_3a.pcap'
print(f"Reading {pcap_file}...")

# Read just the first 100 packets to test
packets = scapy.rdpcap(pcap_file, count=100)

flows = {}
for pkt in packets:
    if scapy.IP in pkt:
        src = pkt[scapy.IP].src
        dst = pkt[scapy.IP].dst
        proto = pkt[scapy.IP].proto
        sport = pkt.sport if hasattr(pkt, 'sport') else 0
        dport = pkt.dport if hasattr(pkt, 'dport') else 0
        
        flow_tuple = tuple(sorted([f"{src}:{sport}", f"{dst}:{dport}"]) + [str(proto)])
        
        if flow_tuple not in flows:
            flows[flow_tuple] = []
        flows[flow_tuple].append(pkt)

print(f"Found {len(flows)} flows in the first 100 packets.")
for f, pkts in list(flows.items())[:2]:
    print(f"Flow {f}: {len(pkts)} packets")
    if len(pkts) > 0 and scapy.Raw in pkts[0]:
        print(f"  First packet payload bytes: {len(pkts[0][scapy.Raw].load)}")
