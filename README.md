# AI-Disease-Prediction 🔬🩺

![Build Status](https://img.shields.io/badge/status-active-brightgreen)

## ✨ Developed by .....

A full-stack **AI Disease Prediction** project that demonstrates training ML models (scikit-learn) and serving them via a **FastAPI** backend with a cross-platform **Flutter** frontend. The app predicts **Diabetes**, **Heart Disease**, and **Chronic Kidney Disease (CKD)** from user-entered clinical features and returns a user-friendly result + suggestion.

---

## 🚀 Features

* ✅ Predict Diabetes, Heart Disease and Chronic Kidney Disease using ML models.
* ⚡ FastAPI backend exposing REST endpoints for each prediction.
* 📱 Flutter frontend with responsive forms and result cards.
* 🧪 Uses CSV datasets for training and scikit-learn `Pipeline` for preprocessing + model.
* 🔁 Model training scripts included and models saved as `*.pkl`.
* 💬 Clear JSON responses containing `status`, `prediction` and `suggestion` for the UI.

---

## ⚡ Tech Stack

* **Backend:** Python, FastAPI, Uvicorn
* **ML:** scikit-learn, pandas, numpy, joblib
* **Frontend:** Flutter (Dart)
* **Data:** CSV (datasets folder)
* **Format:** REST (JSON)

---

## ⚡ Quick Run App With Automations

### 1. Install Dependencies & Setup

Run once to prepare everything:

```bash
python runApp.py -i
```

This command will:

* Install all requirements
* Download datasets
* Train models
* Start both the **backend** and **frontend** automatically

👉 If Google Chrome is installed, the frontend opens automatically.
If not, just copy the frontend address into your browser.

---

### 2. Run App Normally

After the first setup, you can start the app simply by running:

```bash
python runApp.py
```

---

## 🛠️ Backend — Run & Develop

### 1. Create environment & install

```bash
cd ai_disease_prediction_backend
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the API server

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Server default base URL: `http://127.0.0.1:8000`

---

## 🧠 Training models

Training scripts are included (example names):

* `train_diabetes.py`
* `train_heart.py`
* `train_kidney.py`

Each training script should:

1. Load the CSV dataset from `datasets/`.
2. Build preprocessing pipeline (imputers, scalers, one-hot encoders).
3. Train a classifier (e.g. `RandomForestClassifier`) wrapped in a `Pipeline` (preprocessor + classifier).
4. Evaluate with cross-validation.
5. Save the trained pipeline with `joblib.dump(model, "../models/<model_name>.pkl")`.

Example run:

```bash
python train_kidney.py
# created ai_disease_prediction_backend/models/kidney_model.pkl
```

**Important implementation notes (common pitfalls & fixes):**

* If you use a `ColumnTransformer` that refers to column names (strings), pass a pandas `DataFrame` into the pipeline `predict()` / `transform()` call. If you pass a NumPy array, the `ColumnTransformer` will raise:
  `ValueError: Specifying the columns using strings is only supported for dataframes.`

  Fixes:

  * Call `model.predict(pd.DataFrame([row], columns=<column_names>))` OR
  * Use integer column indices in the `ColumnTransformer`.

* If you saved a full `Pipeline` (preprocessor + classifier) then load it with `joblib.load(...)` and call `model.predict(...)` directly with the *raw* input (DataFrame or correctly ordered array) — the pipeline will handle preprocessing.

* A `422 Unprocessable Content` from FastAPI usually indicates the JSON sent does not match the Pydantic model. Verify names, types and required fields.

---

## 📱 Frontend (Flutter)

### 1. Install

```bash
cd ai_disease_prediction_frontend
flutter pub get
```

### 2. Configure backend base URL

* In the Flutter app there's a global `AppState` that stores `baseUrl`. Point it to your FastAPI server:

```dart
// example
final baseUrl = 'http://127.0.0.1:8000';
```

### 3. Run the app

```bash
flutter run
```

### UI Overview

* Each disease has a dedicated form page (e.g. `diabetes_form_page.dart`, `heart_form_page.dart`, `kidney_form_page.dart`).
* Inputs validate numeric fields and offer dropdowns for categorical fields.
* On success the backend returns a JSON with `status`, `prediction`, and `suggestion` which are displayed in a result card.

---

## 🧩 Example: Curl test

```bash
curl -X POST "http://127.0.0.1:8000/kidney/predict" \
  -H "Content-Type: application/json" \
  -d '{ "age": 48.0, "bp": 80.0, "sg": 1.02, "al": 1.0, "su": 0.0, "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent", "bgr": 121.0, "bu": 36.0, "sc": 1.2, "sod": 135.0, "pot": 4.5, "hemo": 15.4, "pcv": 44.0, "wc": 7800.0, "rc": 5.2, "htn": "yes", "dm": "yes", "cad": "no", "appet": "good", "pe": "no", "ane": "no" }'
```

---

## 🐟 License

This project is open-source and available under the **MIT License**.

---

🌟 If you find this project useful, please give it a s
