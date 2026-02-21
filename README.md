# Predicting Cereal Production in Sri Lanka Using Agricultural and Rural Development Indicators with Explainable Machine Learning

Academic, policy-relevant ML project: regression on cereal production with SHAP explainability and a Streamlit dashboard.

## How to see the folder structure

- **In File Explorer:** Open `C:\Users\Shenali\Desktop\L4S1\ML\agriculture-ml-project`. You should see `data/`, `models/`, `src/`, `app.py`, `requirements.txt`, and `README.md`.
- **In VS Code / Cursor:** Open the folder **File → Open Folder** and choose `agriculture-ml-project`. The left sidebar shows the tree.
- **In terminal:** From the project folder run:
  ```bash
  dir /s /b
  ```
  (or `tree /f` if you have `tree`). Or in PowerShell: `Get-ChildItem -Recurse | Select-Object FullName`.

## Project structure

```
agriculture-ml-project/
├── data/
│   └── agriculture-and-rural-development_lka.csv   ← your World Bank CSV (long format OK)
├── models/                                          ← created when you run train.py
│   ├── trained_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── src/
│   ├── __init__.py
│   ├── preprocessing.py   ← loads & converts World Bank long format to wide
│   ├── train.py
│   ├── evaluate.py
│   └── explainability.py
├── app.py
├── requirements.txt
└── README.md
```

## Setup

1. **Create and activate virtual environment** (in the project folder)
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Data**
   - The project expects **`agriculture-and-rural-development_lka.csv`** in the **`data/`** folder.
   - Your file from Downloads has been copied into `data/` for you. If you use a new copy, place it there.
   - The code accepts the World Bank long-format (Indicator Name, Year, Value) and converts it automatically.

## How to run

**From the project root** (`agriculture-ml-project`), with the virtual environment activated:

1. **Train the model** (Random Forest + GridSearchCV)
   ```bash
   python src/train.py
   ```
   This creates the `models/` folder and saves `trained_model.pkl`, `scaler.pkl`, and `feature_names.pkl`.

2. **Evaluate**
   ```bash
   python src/evaluate.py
   ```
   Prints RMSE, R², and 5-fold cross-validation RMSE.

3. **Run SHAP explainability** (optional)
   ```bash
   python src/explainability.py
   ```

4. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```
   - **Prediction:** set inputs in the sidebar.
   - **Model Performance:** RMSE, R², actual vs predicted plot.
   - **Explainability:** SHAP summary and waterfall.
   - **Data Overview:** table and correlation heatmap.

## Report structure (suggested)

1. Introduction  
2. Problem Statement  
3. Dataset Description  
4. Preprocessing  
5. Model Selection  
6. Training & Hyperparameters  
7. Evaluation Metrics  
8. Explainability Results  
9. Critical Discussion  
10. Conclusion  

## Tech stack

- **Regression:** Random Forest (scikit-learn), GridSearchCV.  
- **Explainability:** SHAP (summary plot + waterfall).  
- **Dashboard:** Streamlit (tabs, KPIs, correlation heatmap).
