import pandas as pd

df = pd.read_csv('data/iscx_vpn2016.csv')
features = list(df.columns[:-1])

# For each group of identical features, what is the most common label?
# The maximum possible accuracy is if we predict the most common label for each group perfectly.
grouped = df.groupby(features)['traffic_type'].agg(lambda x: x.value_counts().max())
max_correct = grouped.sum()
total_rows = len(df)

print(f"Maximum theoretical training accuracy: {max_correct / total_rows * 100:.2f}%")
print(f"Total rows: {total_rows}, Max correct: {max_correct}")
