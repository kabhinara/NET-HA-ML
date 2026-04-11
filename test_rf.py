import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/iscx_vpn2016.csv')

feature_cols = [
    'duration', 'total_fiat', 'total_biat', 'min_fiat', 'min_biat', 
    'max_fiat', 'max_biat', 'mean_fiat', 'mean_biat', 'flowPktsPerSecond', 
    'flowBytesPerSecond', 'min_flowiat', 'max_flowiat', 'mean_flowiat', 
    'std_flowiat', 'min_active', 'mean_active', 'max_active', 
    'std_active', 'min_idle', 'mean_idle', 'max_idle', 'std_idle'
]

X = df[feature_cols]
y = LabelEncoder().fit_transform(df['traffic_type'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)
print(f"RF Accuracy: {accuracy_score(y_test, preds)*100:.2f}%")
