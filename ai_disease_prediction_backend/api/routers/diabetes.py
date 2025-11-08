from fastapi import APIRouter
from pydantic import BaseModel
import joblib
import numpy as np

router = APIRouter()

# Load trained model & scaler
model, scaler = joblib.load("ai_disease_prediction_backend/models/diabetes_model.pkl")

class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@router.post("/predict")
def predict_diabetes(data: DiabetesInput):
    # Convert input to numpy array
    input_data = np.array([[
        data.Pregnancies, data.Glucose, data.BloodPressure,
        data.SkinThickness, data.Insulin, data.BMI,
        data.DiabetesPedigreeFunction, data.Age
    ]])

    # Scale the input
    if scaler:
        input_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data)[0]

    # Build result like heart & kidney
    if prediction == 1:
        result = {
            "status": "positive",
            "prediction": "Diabetic",
            "suggestion": "⚠️ Please consult an endocrinologist or diabetologist for further evaluation and management."
        }
    else:
        result = {
            "status": "negative",
            "prediction": "Not Diabetic",
            "suggestion": "🎉 Great! Maintain a healthy lifestyle, exercise regularly, and keep monitoring your glucose levels."
        }

    return result
