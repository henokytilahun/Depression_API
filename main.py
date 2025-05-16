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
    Age: int = None
    Dietary_Habits: str = None
    Suicidal_Thoughts: bool = None
    Academic_Pressure: float = None
    Financial_Stress: float = None
    Study_Satisfaction: float = None

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
    for col in ['dietary_habits', 'suicidal_thoughts']:
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
    # Build a full dict of all possible features
    features_dict = {}
    
    # Encode string categories
    features_dict['dietary_habits'] = encoders['dietary_habits'].transform([payload.Dietary_Habits])[0]
    
    # Encode boolean fields ("Yes"/"No")
    val_str = "Yes" if payload.Suicidal_Thoughts else "No"
    features_dict['suicidal_thoughts'] = encoders['suicidal_thoughts'].transform([val_str])[0]
    
    # Numeric features
    features_dict['age'] = payload.Age
    features_dict['academic_pressure'] = payload.Academic_Pressure
    features_dict['financial_stress'] = payload.Financial_Stress
    features_dict['study_satisfaction'] = payload.Study_Satisfaction
    

    # Select only the features you saved earlier
    raw = [features_dict[f] for f in selected_features]

    # Scale to the model's expected range
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
