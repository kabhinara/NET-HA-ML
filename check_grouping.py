import pandas as pd
df = pd.read_csv('data/iscx_vpn2016.csv')

# 1. 2-class: VPN vs Non-VPN
df['is_vpn'] = df['traffic_type'].str.startswith('VPN').astype(int)
grouped_2c = df.groupby(list(df.columns[:-2]))['is_vpn'].agg(lambda x: x.value_counts().max())
print(f"Max accuracy for VPN vs Non-VPN: {grouped_2c.sum() / len(df) * 100:.2f}%")

# 2. 7-class: Traffic Category (ignoring VPN)
df['category'] = df['traffic_type'].str.replace('VPN-', '')
grouped_7c = df.groupby(list(df.columns[:-3]))['category'].agg(lambda x: x.value_counts().max())
print(f"Max accuracy for 7 Traffic Categories: {grouped_7c.sum() / len(df) * 100:.2f}%")
