import pytest
import joblib
import os
import sys
sys.path.insert(0, "src")

from train import train

def test_training_runs():
    train()
    assert os.path.exists("model/iris_model.pkl"), "Model file not created"

def test_model_prediction():
    model = joblib.load("model/iris_model.pkl")
    sample = [[5.1, 3.5, 1.4, 0.2]]
    result = model.predict(sample)
    assert result[0] in [0, 1, 2], "Invalid prediction class"

def test_model_accuracy():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    iris = load_iris()
    X, y = iris.data, iris.target
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = joblib.load("model/iris_model.pkl")
    acc = accuracy_score(y_test, model.predict(X_test))
    assert acc > 0.90, f"Accuracy too low: {acc}"