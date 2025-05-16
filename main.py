# app.py
import pickle
import json
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

# ----------------------------------------------------------------------
# 1) Define the request schema
# ----------------------------------------------------------------------
class DepressionInput(BaseModel):
    Degree: str
    City: str
    Gender: str
    Age: int
    Dietary_Habits: str
    Sleep_Duration: str
    Suicidal_Thoughts: bool
    Family_History: bool
    CGPA: float                 # raw on 0–10 scale
    Academic_Pressure: float
    Work_Pressure: float
    Study_Satisfaction: float
    Job_Satisfaction: float
    Work_Study_Hours: float
    Financial_Stress: float

# ----------------------------------------------------------------------
# 2) Load model, scaler, encoders & selected_features at startup
# ----------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, encoders, selected_features

    # Load the trained Keras model
    model = tf.keras.models.load_model("model.h5")

    # Load the StandardScaler
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Load LabelEncoders
    encoders = {}
    for col in [
        'degree',
        'city',
        'gender',
        'dietary_habits',
        'sleep_duration',
        'suicidal_thoughts',
        'family_history'
    ]:
        with open(f"le_{col}.pkl", "rb") as f:
            encoders[col] = pickle.load(f)

    # Load which features to keep, in order
    with open("selected_features.json", "r") as f:
        selected_features = json.load(f)

    yield
    # no teardown needed

app = FastAPI(lifespan=lifespan)

# ----------------------------------------------------------------------
# 3) Preprocessing helper – now only emits selected_features
# ----------------------------------------------------------------------
def preprocess(payload: DepressionInput) -> np.ndarray:
    # 1) Rescale CGPA to 0–4
    cgpa_scaled = (payload.CGPA / 10) * 4
    

    # 2) Build a full dict of all possible features
    features_dict = {}
    # 2a) Encode string categories
    for field in ['Degree', 'City', 'Gender', 'Dietary_Habits', 'Sleep_Duration']:
        key = field.lower()
        val = getattr(payload, field)
        features_dict[key] = encoders[key].transform([val])[0]

    # 2b) Encode boolean fields ("Yes"/"No")
    for field in ['Suicidal_Thoughts', 'Family_History']:
        key = field.lower()
        val = getattr(payload, field)
        val_str = "Yes" if val else "No"
        features_dict[key] = encoders[key].transform([val_str])[0]

    # 2c) Numeric features
    features_dict['cgpa_scaled']      = cgpa_scaled
    features_dict['academic_pressure'] = payload.Academic_Pressure
    features_dict['work_pressure']     = payload.Work_Pressure
    features_dict['study_satisfaction'] = payload.Study_Satisfaction
    features_dict['job_satisfaction']   = payload.Job_Satisfaction
    features_dict['work_study_hours']   = payload.Work_Study_Hours
    features_dict['financial_stress']   = payload.Financial_Stress
    features_dict['age'] = payload.Age

    # 3) Select only the features you saved earlier
    raw = [features_dict[f] for f in selected_features]

    # 4) Scale to the model’s expected range
    arr = np.array(raw, dtype=np.float32).reshape(1, -1)
    return scaler.transform(arr)

# ----------------------------------------------------------------------
# 4) Prediction endpoint
# ----------------------------------------------------------------------
@app.post("/predict")
def predict_depression(payload: DepressionInput):
    x = preprocess(payload)
    prob = model.predict(x)[0][0]
    return {
        "depression_probability": float(prob),
        "depression_prediction": bool(prob >= 0.5)
    }
