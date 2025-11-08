# filename: train_heart.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Load & clean
def load_and_clean_csv(path):
    df = pd.read_csv(path)

    # Encode target: Presence = 1, Absence = 0
    df["Heart Disease"] = df["Heart Disease"].map({"Presence": 1, "Absence": 0})

    # Handle categorical-like columns if needed (already numeric here)
    return df

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler



model_dir = "../models"
model_file = "heart_model.pkl"
os.makedirs(model_dir, exist_ok=True)


# Load dataset
df = load_and_clean_csv("../datasets/Heart_Disease_Prediction.csv")

X = df.drop("Heart Disease", axis=1)
y = df["Heart Disease"]

# Scale features
X_scaled, scaler = scale_features(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"✅ Heart Disease Model Accuracy: {accuracy:.4f}")

# Save
joblib.dump((model, scaler), f"{model_dir}/{model_file}")
print("✅ Heart Disease model trained & saved.")
