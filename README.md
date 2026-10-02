# HealthyReadmit AI — Patient Readmission Prediction

A portfolio-ready machine learning product that predicts 30-day hospital readmission risk and turns model output into an explainable patient prioritization workflow.

## What it demonstrates
- Exploratory/feature-oriented synthetic healthcare dataset generation
- Classification model comparison: Logistic Regression vs Random Forest
- ROC-AUC, precision, recall, F1 and accuracy
- Probability-based risk bands
- Human-readable risk drivers and recommended follow-up actions
- FastAPI prediction and cohort APIs
- Responsive light UI with patient queue, model performance and patient detail drawer

## Important
This project uses synthetic data for demonstration only. It is not validated for clinical use and must not be used to make real medical decisions.

## Backend
```powershell
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000/docs

## Frontend
In a second terminal:
```powershell
cd frontend
npm install
npm run dev
```
Open http://localhost:3000

## Core API
- GET /api/overview
- GET /api/trends
- GET /api/patients
- GET /api/patient/{patient_id}
- POST /api/predict
- POST /api/retrain
