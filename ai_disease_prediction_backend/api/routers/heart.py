from fastapi import APIRouter
from pydantic import BaseModel
import joblib
import numpy as np

router = APIRouter()

# Load trained model & scaler
model, scaler = joblib.load("ai_disease_prediction_backend/models/heart_model.pkl")

class HeartInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@router.post("/predict")
def predict_heart(data: HeartInput):
    # Convert input to numpy array
    input_data = np.array([[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal
    ]])

    # Scale input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Build response
    if prediction == 1:
        result = {
            "status": "positive",
            "prediction": "Heart Disease",
            "suggestion": "⚠️ Please consult a cardiologist immediately for further checkup."
        }
    else:
        result = {
            "status": "negative",
            "prediction": "No Heart Disease",
            "suggestion": "🎉 Great! Keep maintaining a healthy lifestyle and regular checkups."
        }

    return result
