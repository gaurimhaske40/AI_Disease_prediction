from fastapi import APIRouter
from pydantic import BaseModel
import joblib
import numpy as np

router = APIRouter()

# Load trained kidney model
model = joblib.load("ai_disease_prediction_backend/models/kidney_model.pkl")

class KidneyInput(BaseModel):
    # Define all input fields that your dataset requires
    # (Example: Replace with actual columns from kidney_disease.csv)
    age: float
    bp: float
    sg: float
    al: float
    su: float
    rbc: str
    pc: str
    pcc: str
    ba: str
    bgr: float
    bu: float
    sc: float
    sod: float
    pot: float
    hemo: float
    pcv: float
    wc: float
    rc: float
    htn: str
    dm: str
    cad: str
    appet: str
    pe: str
    ane: str

# @router.post("/predict")
# def predict_kidney(data: KidneyInput):
#     # Convert input to numpy (in the same feature order as training dataset)
#     input_data = np.array([[
#         data.age, data.bp, data.sg, data.al, data.su, data.rbc,
#         data.pc, data.pcc, data.ba, data.bgr, data.bu, data.sc,
#         data.sod, data.pot, data.hemo, data.pcv, data.wc, data.rc,
#         data.htn, data.dm, data.cad, data.appet, data.pe, data.ane
#     ]], dtype=object)  # keep dtype=object because of categorical fields

#     # Predict
#     prediction = model.predict(input_data)[0]

#     if prediction == 1:
#         result = {
#             "status": "positive",
#             "prediction": "Chronic Kidney Disease",
#             "suggestion": "⚠️ Please consult a nephrologist immediately for proper diagnosis and treatment."
#         }
#     else:
#         result = {
#             "status": "negative",
#             "prediction": "No Chronic Kidney Disease",
#             "suggestion": "🎉 Great! Keep maintaining a healthy lifestyle and go for regular health checkups."
#         }

#     return result


import pandas as pd

@router.post("/predict")
def predict_kidney(kidney: KidneyInput):
    data_dict = kidney.model_dump()
    input_df = pd.DataFrame([data_dict])  # ✅ create dataframe with column names
    
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        result = {
            "status": "positive",
            "prediction": "Chronic Kidney Disease",
            "suggestion": "⚠️ Please consult a nephrologist immediately for proper diagnosis and treatment."
        }
    else:
        result = {
            "status": "negative",
            "prediction": "No Chronic Kidney Disease",
            "suggestion": "🎉 Great! Keep maintaining a healthy lifestyle and go for regular health checkups."
        }
    return result
