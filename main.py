from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

# Automatically find the correct path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Load all saved files
model          = joblib.load(os.path.join(MODELS_DIR, 'heart_model.pkl'))
scaler         = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
num_imputer    = joblib.load(os.path.join(MODELS_DIR, 'num_imputer.pkl'))
cat_imputer    = joblib.load(os.path.join(MODELS_DIR, 'cat_imputer.pkl'))
label_encoders = joblib.load(os.path.join(MODELS_DIR, 'label_encoders.pkl'))

# ─── Create the app ─────────────────────────────────
app = FastAPI()

# ─── CORS ───────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Input Schema ───────────────────────────────────
class PatientData(BaseModel):
    Age: float
    Sex: str
    ChestPainType: str
    RestingBP: float
    Cholesterol: float
    FastingBS: float
    RestingECG: str
    MaxHR: float
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str

# ─── Column types ───────────────────────────────────
numerical_columns   = ['Age', 'RestingBP', 'Cholesterol',
                       'FastingBS', 'MaxHR', 'Oldpeak']

categorical_columns = ['Sex', 'ChestPainType', 'RestingECG',
                       'ExerciseAngina', 'ST_Slope']

# ─── Home route ─────────────────────────────────────
@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running!"}

# ─── Predict route ──────────────────────────────────
@app.post("/predict")
def predict(data: PatientData):

    # Step 1 — Build DataFrame
    input_dict = {
        'Age':            [data.Age],
        'Sex':            [data.Sex],
        'ChestPainType':  [data.ChestPainType],
        'RestingBP':      [data.RestingBP],
        'Cholesterol':    [data.Cholesterol],
        'FastingBS':      [data.FastingBS],
        'RestingECG':     [data.RestingECG],
        'MaxHR':          [data.MaxHR],
        'ExerciseAngina': [data.ExerciseAngina],
        'Oldpeak':        [data.Oldpeak],
        'ST_Slope':       [data.ST_Slope],
    }
    df = pd.DataFrame(input_dict)

    # Step 2 — Impute numerical
    df[numerical_columns] = num_imputer.transform(df[numerical_columns])

    # Step 3 — Impute categorical
    df[categorical_columns] = cat_imputer.transform(df[categorical_columns])

    # Step 4 — Encode categorical
    for col in categorical_columns:
        le = label_encoders[col]
        df[col] = le.transform(df[col])

    # Step 5 — Scale numerical
    df[numerical_columns] = scaler.transform(df[numerical_columns])

    # Step 6 — Predict
    prediction  = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "result":      "At Risk" if prediction == 1 else "Low Risk",
        "probability": round(float(probability) * 100, 2),
        "message":     "Please consult a doctor for proper medical advice."
    }