import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os


model_dir = "../models"
model_file = "kidney_model.pkl"
os.makedirs(model_dir, exist_ok=True)

def train_kidney():
    # === Load dataset ===
    dataset_path = os.path.join(os.path.dirname(__file__), "../datasets/kidney_disease.csv")
    print(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)

    # Drop ID column if exists
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    # Handle target column
    df["classification"] = df["classification"].str.strip().str.lower()
    df["classification"] = df["classification"].replace({"ckd": 1, "notckd": 0})
    df = df.dropna(subset=["classification"])
    y = df["classification"].astype(int)
    X = df.drop(columns=["classification"])

    # Separate features
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    print("Numeric cols:", numeric_cols)
    print("Categorical cols:", categorical_cols)

    # Preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    # Model
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    # === Cross-validation ===
    print("\n=== Cross-validation ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")

    print(f"CV Accuracies: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # === Train final model on full data ===
    model.fit(X, y)

    # Save model
    save_path = os.path.join(os.path.dirname(__file__), f"{model_dir}/{model_file}")
    joblib.dump(model, save_path)
    print(f"\n✅ Kidney model + preprocessing saved to: {save_path}")


if __name__ == "__main__":

    train_kidney()
