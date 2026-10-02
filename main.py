from __future__ import annotations
from typing import Any
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .model import load_assets, train, score, NUMERIC, CATEGORICAL

app = FastAPI(title="HealthyReadmit AI", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class Patient(BaseModel):
    age: int = Field(55, ge=18, le=100)
    gender: str = "Female"
    primary_diagnosis: str = "Heart Failure"
    insurance_type: str = "Private"
    length_of_stay: int = Field(5, ge=1, le=60)
    previous_admissions: int = Field(1, ge=0, le=20)
    emergency_visits: int = Field(1, ge=0, le=20)
    medication_count: int = Field(5, ge=0, le=40)
    comorbidity_count: int = Field(2, ge=0, le=15)
    lab_instability: float = Field(3.5, ge=0, le=10)
    followup_days: int = Field(7, ge=1, le=60)
    discharge_disposition: str = "Home"
    discharge_risk_score: float = Field(3.0, ge=0, le=10)

@app.get("/")
def root(): return {"name":"HealthyReadmit AI","status":"ok","docs":"/docs"}

@app.get("/api/overview")
def overview():
    _, meta, df = load_assets()
    p = df.readmitted_30d.mean(); high = int((df.readmitted_30d == 1).sum())
    return {"patients":len(df),"readmission_rate":round(p*100,1),"high_risk_candidates":high,"champion_model":meta["champion"],"metrics":meta["results"]}

@app.get("/api/trends")
def trends():
    _, _, df = load_assets()
    bins = pd.qcut(df["age"], 7, duplicates="drop")
    trend = df.assign(age_band=bins.astype(str)).groupby("age_band", observed=False)["readmitted_30d"].mean().reset_index()
    return [{"label":str(r.age_band),"rate":round(float(r.readmitted_30d*100),1)} for _,r in trend.iterrows()]

@app.get("/api/patients")
def patients(search: str = "", band: str = Query("all")):
    _, _, df = load_assets(); rows=[]
    for _, r in df.head(320).iterrows():
        vals=r.to_dict(); s=score({k:vals[k] for k in NUMERIC+CATEGORICAL});
        if band!="all" and s["band"].lower()!=band.lower(): continue
        if search and search.lower() not in str(vals["patient_id"]).lower() and search.lower() not in str(vals["primary_diagnosis"]).lower(): continue
        rows.append({"patient_id":vals["patient_id"],"age":int(vals["age"]),"diagnosis":vals["primary_diagnosis"],"risk":s["probability"],"band":s["band"],"length_of_stay":int(vals["length_of_stay"]),"previous_admissions":int(vals["previous_admissions"]),"followup_days":int(vals["followup_days"])})
    rows.sort(key=lambda x:x["risk"], reverse=True)
    return rows[:100]

@app.get("/api/patient/{patient_id}")
def patient(patient_id: str):
    _, _, df=load_assets(); hit=df[df.patient_id==patient_id]
    if hit.empty: raise HTTPException(404,"Patient not found")
    vals=hit.iloc[0].to_dict(); inp={k:vals[k] for k in NUMERIC+CATEGORICAL}; s=score(inp)
    return {"patient_id":patient_id,**{k:vals[k] for k in ["age","gender","primary_diagnosis","insurance_type","length_of_stay","previous_admissions","emergency_visits","medication_count","comorbidity_count","lab_instability","followup_days","discharge_disposition","discharge_risk_score"]},**s}

@app.post("/api/predict")
def predict(patient: Patient): return score(patient.model_dump())

@app.post("/api/retrain")
def retrain(): return train()
