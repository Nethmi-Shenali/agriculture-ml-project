"""
SHAP explainability: summary plot and per-prediction waterfall support.
"""

import os
import sys
import shap
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import load_data, preprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "agriculture-and-rural-development_lka.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

df = load_data(DATA_PATH)
X_train, X_test, y_train, y_test, _, feature_names = preprocess(df)

model = joblib.load(os.path.join(MODELS_DIR, "trained_model.pkl"))

# Modern SHAP API
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Summary plot
shap.plots.beeswarm(shap_values, show=False)