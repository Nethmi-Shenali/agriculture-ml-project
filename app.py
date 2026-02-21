"""
Sri Lanka Cereal Production Prediction — Streamlit Dashboard
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------------------------
# Paths
# -------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

def _data_path():
    for name in (
        "agriculture-and-rural-development_lka.csv",
        "sri_lanka_agriculture.csv"
    ):
        p = os.path.join(PROJECT_ROOT, "data", name)
        if os.path.exists(p):
            return p
    return os.path.join(
        PROJECT_ROOT,
        "data",
        "agriculture-and-rural-development_lka.csv"
    )

DATA_PATH = _data_path()

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Sri Lanka Agriculture ML Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌾 Sri Lanka Cereal Production Prediction")
st.markdown(
    "**Predicting cereal production using agricultural and rural development indicators — Explainable ML**"
)

# -------------------------------------------------
# Load Model Artifacts
# -------------------------------------------------
@st.cache_resource
def load_artifacts():
    model_path = os.path.join(MODELS_DIR, "trained_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    names_path = os.path.join(MODELS_DIR, "feature_names.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None, None

    return (
        joblib.load(model_path),
        joblib.load(scaler_path),
        joblib.load(names_path) if os.path.exists(names_path) else None,
    )

model, scaler, feature_names = load_artifacts()

if model is None or scaler is None:
    st.warning("⚠️ Train the model first using: python src/train.py")
    st.stop()

# -------------------------------------------------
# Load ML Data
# -------------------------------------------------
@st.cache_data
def load_ml_data():
    from src.preprocessing import load_data, preprocess
    df = load_data(DATA_PATH)
    return preprocess(df)

X_train, X_test, y_train, y_test, _, feat_names = load_ml_data()

# -------------------------------------------------
# Load SHAP Explainer
# -------------------------------------------------
@st.cache_resource
def compute_shap(_model, _X_train, _X_test):
    explainer = shap.Explainer(_model, _X_train)
    return explainer(_X_test)

shap_values = compute_shap(model, X_train, X_test)

# -------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------
st.sidebar.header("📥 Input Agricultural Indicators")

inputs = {}
if feature_names:
    default_vals = dict(zip(
        feature_names,
        [200.0, 19.0, 1800.0, 25.0, 950000.0, 11.0, 100.0, 100.0],
    ))

    for col in feature_names:
        val = default_vals.get(col, 100.0)
        inputs[col] = st.sidebar.number_input(
            col,
            value=val,
            format="%.2f"
        )

# -------------------------------------------------
# Tabs
# -------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🎯 Prediction",
     "📊 Model Performance",
     "🔍 Explainability",
     "📈 Data Overview"]
)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
with tab1:
    st.subheader("Cereal Production Prediction")

    X_input = np.array([[inputs[col] for col in feature_names]])
    X_scaled = scaler.transform(X_input)

    prediction = model.predict(X_scaled)[0]

    st.success(f"Predicted Cereal Production: {prediction:,.2f}")

    # Save for explainability
    st.session_state["last_scaled_input"] = X_scaled
# -------------------------------------------------
# Model Performance
# -------------------------------------------------
with tab2:
    st.subheader("Model Performance")

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{rmse:,.2f}")
    c2.metric("R² Score", f"{r2:.4f}")
    c3.metric("Test Samples", len(y_test))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(y_test, preds)
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--"
    )
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

# -------------------------------------------------
# Explainability
# -------------------------------------------------
with tab3:
    st.subheader("Explainability — SHAP")

    # Use user input instead of test sample
    if "last_scaled_input" in st.session_state:

        X_scaled = st.session_state["last_scaled_input"]

        explainer = shap.Explainer(model, X_train)
        shap_value = explainer(X_scaled)

        st.write("### Feature Contribution to This Prediction")

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_value[0], show=False)
        st.pyplot(fig)

    else:
        st.info("Adjust inputs in Prediction tab first.")
# -------------------------------------------------
# Data Overview
# -------------------------------------------------
with tab4:
    st.subheader("Dataset Overview")

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        st.dataframe(df.head(20), use_container_width=True)

        numeric = df.select_dtypes(include=[np.number])
        if len(numeric.columns) > 1:
            corr = numeric.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0)
            st.pyplot(fig)
    else:
        st.warning("Data file not found in data/ folder.")