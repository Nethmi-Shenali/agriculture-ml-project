"""
Model evaluation: RMSE, R², and cross-validation scores.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from src.preprocessing import load_data, preprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _data_path():
    for name in ("agriculture-and-rural-development_lka.csv", "sri_lanka_agriculture.csv"):
        p = os.path.join(PROJECT_ROOT, "data", name)
        if os.path.exists(p):
            return p
    return os.path.join(PROJECT_ROOT, "data", "agriculture-and-rural-development_lka.csv")

DATA_PATH = _data_path()
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

df = load_data(DATA_PATH)
X_train, X_test, y_train, y_test, _, _ = preprocess(df)

model = joblib.load(os.path.join(MODELS_DIR, "trained_model.pkl"))

predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

# Cross-validation (neg MSE, then convert to RMSE)
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
cv_rmse = np.sqrt(-cv_scores)

print("--- Test set ---")
print("RMSE:", rmse)
print("R2 Score:", r2)
print("--- 5-fold CV (train) ---")
print("CV RMSE mean:", cv_rmse.mean())
print("CV RMSE std:", cv_rmse.std())
