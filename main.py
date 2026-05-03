from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Heart Disease Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

model          = joblib.load(os.path.join(MODELS_DIR, 'heart_model.pkl'))
scaler         = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
num_imputer    = joblib.load(os.path.join(MODELS_DIR, 'num_imputer.pkl'))
cat_imputer    = joblib.load(os.path.join(MODELS_DIR, 'cat_imputer.pkl'))
label_encoders = joblib.load(os.path.join(MODELS_DIR, 'label_encoders.pkl'))

NUMERICAL_COLS   = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
CATEGORICAL_COLS = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
ALL_FEATURES     = NUMERICAL_COLS + CATEGORICAL_COLS

class PatientData(BaseModel):
    Age:            float
    Sex:            str
    ChestPainType:  str
    RestingBP:      float
    Cholesterol:    float
    FastingBS:      int
    RestingECG:     str
    MaxHR:          float
    ExerciseAngina: str
    Oldpeak:        float
    ST_Slope:       str

def preprocess(data: PatientData) -> np.ndarray:
    raw = {
        "Age":            data.Age,
        "RestingBP":      data.RestingBP,
        "Cholesterol":    data.Cholesterol,
        "FastingBS":      data.FastingBS,
        "MaxHR":          data.MaxHR,
        "Oldpeak":        data.Oldpeak,
        "Sex":            data.Sex,
        "ChestPainType":  data.ChestPainType,
        "RestingECG":     data.RestingECG,
        "ExerciseAngina": data.ExerciseAngina,
        "ST_Slope":       data.ST_Slope,
    }
    num_arr = np.array([[raw[c] for c in NUMERICAL_COLS]])
    num_arr = num_imputer.transform(num_arr)
    num_arr = scaler.transform(num_arr)
    cat_arr     = np.array([[raw[c] for c in CATEGORICAL_COLS]])
    cat_arr     = cat_imputer.transform(cat_arr)
    cat_encoded = np.zeros_like(cat_arr, dtype=float)
    for i, col in enumerate(CATEGORICAL_COLS):
        le = label_encoders[col]
        cat_encoded[0, i] = le.transform([cat_arr[0, i]])[0]
    return np.hstack([num_arr, cat_encoded])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: PatientData):
    X          = preprocess(data)
    pred       = int(model.predict(X)[0])
    prob       = float(model.predict_proba(X)[0][1])
    risk_pct   = round(prob * 100, 1)
    risk_label = "High Risk" if pred == 1 else "Low Risk"

    # Feature importances
    importances = model.feature_importances_
    feature_scores = [
        {"feature": name, "importance": round(float(imp) * 100, 1)}
        for name, imp in zip(ALL_FEATURES, importances)
    ]
    feature_scores.sort(key=lambda x: x["importance"], reverse=True)
    top_features = feature_scores[:6]

    return {
        "prediction":   pred,
        "probability":  prob,
        "risk_percent": risk_pct,
        "risk_label":   risk_label,
        "top_features": top_features,
    }

STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))