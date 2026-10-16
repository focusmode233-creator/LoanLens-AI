# LoanLens AI — Educational Loan Approval Classification Prototype

## Overview
This project demonstrates a machine-learning workflow for classifying loan application records as Approved or Rejected using a public educational dataset.

**Important:** This is an educational prototype, not an official HDFC Bank product or a real lending decision system. Model outputs and probabilities are not guarantees of actual approval.

## Project structure
- `app.py` — Streamlit dashboard and prediction interface
- `train_model.py` — trains and evaluates the Logistic Regression model
- `compare_models.py` — compares Logistic Regression and Random Forest
- `explore_data.py` — basic dataset exploration
- `analyze_data.py` — loan-status counts and chart
- `check_setup.py` — checks core Python libraries
- `main.py` — introductory script
- `loan_model.pkl` — saved trained pipeline used by the app
- `model_comparison.csv` — saved model comparison results
- `data/loan_dataset.csv` — dataset
- `requirements.txt` — Python dependencies

## Run locally
Use Python 3.11–3.13 if possible.

### Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks activation, you can run Streamlit without activating the environment after installing dependencies:
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Retrain / compare models
Run from the project root (the folder containing `data`):
```powershell
python compare_models.py
```
This script trains and compares models and overwrites `loan_model.pkl` and `model_comparison.csv`.

## Notes
- The app expects `data/loan_dataset.csv`, `loan_model.pkl`, and `model_comparison.csv` to be present relative to `app.py`.
- Dataset statistics describe only this dataset, not actual bank operations.
- Verify model evaluation methodology and results before making final performance claims.
