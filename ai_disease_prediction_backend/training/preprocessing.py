import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_clean_csv(file_path: str, dropna: bool = True):
    df = pd.read_csv(file_path)
    if dropna:
        df = df.dropna()
    return df

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
