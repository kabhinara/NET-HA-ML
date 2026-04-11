import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/iscx_vpn2016.csv')
X = df.drop('traffic_type', axis=1)
y = LabelEncoder().fit_transform(df['traffic_type'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=300, max_depth=30, random_state=42)
rf.fit(X_train, y_train)
print(f"RF Train Accuracy: {rf.score(X_train, y_train)*100:.2f}%")
print(f"RF Test Accuracy: {rf.score(X_test, y_test)*100:.2f}%")
