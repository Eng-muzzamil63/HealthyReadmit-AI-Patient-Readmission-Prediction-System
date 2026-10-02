from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_FILE = MODEL_DIR / "readmission_model.joblib"
METRICS_FILE = MODEL_DIR / "metrics.joblib"
DATA_FILE = MODEL_DIR / "patients.csv"

NUMERIC = [
    "age", "length_of_stay", "previous_admissions", "emergency_visits",
    "medication_count", "comorbidity_count", "lab_instability", "followup_days",
    "discharge_risk_score"
]
CATEGORICAL = ["gender", "primary_diagnosis", "insurance_type", "discharge_disposition"]


def make_dataset(n: int = 6500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 91, n)
    length = np.clip(rng.gamma(2.2, 2.2, n).round(), 1, 30).astype(int)
    prev = np.clip(rng.poisson(0.8, n), 0, 7)
    emergency = np.clip(rng.poisson(1.1, n), 0, 8)
    meds = np.clip(rng.poisson(5, n), 0, 18)
    comorb = np.clip(rng.poisson(2.0, n), 0, 8)
    lab = np.clip(rng.beta(2.0, 5.0, n) * 10, 0, 10)
    follow = np.clip(rng.normal(8, 5, n), 1, 30).round().astype(int)
    discharge_risk = np.clip(0.15 + 0.025*(age-40) + 0.10*comorb + 0.07*prev + 0.04*emergency + 0.03*length + rng.normal(0, 0.5, n), 0, 10)

    gender = rng.choice(["Female", "Male", "Other"], n, p=[0.49, 0.49, 0.02])
    diagnosis = rng.choice(["Heart Failure", "COPD", "Diabetes", "Pneumonia", "Kidney Disease", "Other"], n, p=[.18,.14,.20,.18,.10,.20])
    insurance = rng.choice(["Private", "Medicare", "Medicaid", "Self-pay"], n, p=[.45,.28,.20,.07])
    disposition = rng.choice(["Home", "Home Health", "Skilled Nursing", "Rehab"], n, p=[.62,.20,.10,.08])

    diagnosis_risk = pd.Series(diagnosis).map({"Heart Failure": .65, "COPD": .48, "Diabetes": .32, "Pneumonia": .52, "Kidney Disease": .60, "Other": .10}).to_numpy()
    disposition_risk = pd.Series(disposition).map({"Home": .20, "Home Health": .08, "Skilled Nursing": -.05, "Rehab": -.12}).to_numpy()
    logit = -3.0 + 0.018*age + 0.12*length + 0.34*prev + 0.24*emergency + 0.10*meds + 0.32*comorb + 0.27*lab + 0.33*diagnosis_risk + disposition_risk - 0.035*follow + rng.normal(0, 0.35, n)
    prob = 1/(1+np.exp(-logit))
    y = rng.binomial(1, np.clip(prob, .02, .92))

    df = pd.DataFrame({
        "patient_id": [f"PT-{100000+i}" for i in range(n)],
        "age": age,
        "gender": gender,
        "primary_diagnosis": diagnosis,
        "insurance_type": insurance,
        "length_of_stay": length,
        "previous_admissions": prev,
        "emergency_visits": emergency,
        "medication_count": meds,
        "comorbidity_count": comorb,
        "lab_instability": lab.round(2),
        "followup_days": follow,
        "discharge_disposition": disposition,
        "discharge_risk_score": discharge_risk.round(2),
        "readmitted_30d": y,
    })
    return df


def train() -> dict:
    df = make_dataset()
    df.to_csv(DATA_FILE, index=False)
    X = df[NUMERIC + CATEGORICAL]
    y = df["readmitted_30d"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.22, stratify=y, random_state=42)

    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=280, max_depth=12, min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=-1),
    }
    results = {}
    fitted = {}
    for name, estimator in models.items():
        pipe = Pipeline([( "preprocess", pre), ("model", estimator)])
        pipe.fit(X_train, y_train)
        p = pipe.predict_proba(X_test)[:,1]
        pred = (p >= .5).astype(int)
        results[name] = {
            "roc_auc": round(float(roc_auc_score(y_test, p)), 4),
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        }
        fitted[name] = pipe
    champion = max(results, key=lambda k: results[k]["roc_auc"])
    joblib.dump(fitted[champion], MODEL_FILE)
    joblib.dump({"results": results, "champion": champion}, METRICS_FILE)
    return {"results": results, "champion": champion, "positive_rate": round(float(y.mean()),4)}


def load_assets() -> tuple:
    if not MODEL_FILE.exists() or not METRICS_FILE.exists() or not DATA_FILE.exists():
        train()
    return joblib.load(MODEL_FILE), joblib.load(METRICS_FILE), pd.read_csv(DATA_FILE)


def score(row: dict) -> dict:
    model, metrics, _ = load_assets()
    x = pd.DataFrame([{k: row[k] for k in NUMERIC + CATEGORICAL}])
    p = float(model.predict_proba(x)[0,1])
    if p >= .75: band = "Critical"
    elif p >= .55: band = "High"
    elif p >= .30: band = "Moderate"
    else: band = "Low"
    factors = []
    if row["previous_admissions"] >= 2: factors.append(("Previous admissions", row["previous_admissions"], "higher recurrence signal"))
    if row["emergency_visits"] >= 2: factors.append(("Emergency visits", row["emergency_visits"], "frequent acute-care utilization"))
    if row["comorbidity_count"] >= 4: factors.append(("Comorbidity burden", row["comorbidity_count"], "multiple chronic conditions"))
    if row["followup_days"] >= 14: factors.append(("Follow-up delay", row["followup_days"], "longer gap after discharge"))
    if row["lab_instability"] >= 6: factors.append(("Lab instability", row["lab_instability"], "higher physiologic variability"))
    if row["medication_count"] >= 9: factors.append(("Medication complexity", row["medication_count"], "more medications to manage"))
    if row["length_of_stay"] >= 8: factors.append(("Length of stay", row["length_of_stay"], "longer inpatient course"))
    if not factors: factors.append(("Baseline risk profile", round(p,2), "no single dominant driver"))
    factors = [{"name":a,"value":b,"reason":c} for a,b,c in factors[:5]]
    action = "Standard follow-up" if band=="Low" else "Schedule follow-up within 14 days" if band=="Moderate" else "Arrange early follow-up and medication review" if band=="High" else "Escalate to care-management review within 7 days"
    return {"probability": round(p,4), "band": band, "drivers": factors, "recommended_action": action, "model": metrics["champion"]}
