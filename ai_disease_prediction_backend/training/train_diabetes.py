# training/train_diabetes.py
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler




def load_and_clean_csv(path: str):
    # Load CSV
    df = pd.read_csv(path)
    
    # Replace missing values (NA/NaN) with column mean
    df = df.fillna(df.mean(numeric_only=True))
    
    # Convert all numeric columns to float safely
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # If any leftover NA after coercion, fill again
    df = df.fillna(df.mean(numeric_only=True))
    
    return df



def scale_features(X: pd.DataFrame):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# ---- Training ----
if __name__ == "__main__":

    model_dir = "../models"
    model_file = "diabetes_model.pkl"
    os.makedirs(model_dir, exist_ok=True)

    # Load dataset
    df = load_and_clean_csv("../datasets/diabetes.csv")

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Scale features
    X_scaled, scaler = scale_features(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    # Accuracy
    accuracy = model.score(X_test, y_test)
    print(f"✅ Diabetes Model Accuracy: {accuracy:.4f}")


    # Save model & scaler
    joblib.dump((model, scaler), f"{model_dir}/{model_file}")
    print("✅ Diabetes model trained & saved.")
