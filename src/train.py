"""
Model training with Random Forest and optional GridSearchCV.
Saves model, scaler, and feature names for the app and explainability.
"""
import os
import sys

# Allow running as python src/train.py from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from src.preprocessing import load_data, preprocess

# Paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _data_path():
    for name in ("agriculture-and-rural-development_lka.csv", "sri_lanka_agriculture.csv"):
        p = os.path.join(PROJECT_ROOT, "data", name)
        if os.path.exists(p):
            return p
    return os.path.join(PROJECT_ROOT, "data", "agriculture-and-rural-development_lka.csv")

DATA_PATH = _data_path()
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

df = load_data(DATA_PATH)
X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)

# Hyperparameter grid for distinction-level tuning
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [8, 10, 12],
    "min_samples_split": [2, 5],
}

base_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    base_model, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

model = grid_search.best_estimator_
print("Best params:", grid_search.best_params_)

joblib.dump(model, os.path.join(MODELS_DIR, "trained_model.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))

print("Model, scaler, and feature names saved to models/")
