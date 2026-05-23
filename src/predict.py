import joblib
import numpy as np
import os

def predict(features):
    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "iris_model.pkl")
    model = joblib.load(os.path.abspath(model_path))
    prediction = model.predict([features])
    return int(prediction[0])