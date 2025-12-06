from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime

app = FastAPI(title="Cloud Clinical Note API", version="1.0-lite")

# -------- Dummy Pipeline --------
class DummyPipeline:
    def process_patient(self, data):
        name = data["name"]
        age = data["age"]
        gender = data["gender"]
        symptoms = data["symptoms"]
        scan = data.get("scan_result", "No imaging performed")

        note = (
            f"{name}, a {age}-year-old {gender.lower()} patient, presents with {symptoms}. "
            f"Imaging findings: {scan}. "
            "This clinical note was generated using a lightweight demo pipeline "
            "optimized for low-memory cloud deployment."
        )

        return {
            "patient_id": "DEMO-" + datetime.now().strftime("%Y%m%d%H%M%S"),
            "timestamp": datetime.now().isoformat(),
            "patient_data": data,
            "clinical_documentation": {
                "generated_note": note,
                "icd_coding": {
                    "code": "J18.9",
                    "description": "Pneumonia, unspecified organism",
                    "confidence": 0.80,
                    "evidence": {"reason": "Symptoms suggest respiratory infection"}
                },
            },
            "metadata": {"mode": "cloud-lite"}
        }

pipeline = DummyPipeline()

# -------- API Models --------
class Patient(BaseModel):
    name: str
    age: int
    gender: str
    symptoms: str
    scan_result: Optional[str] = "No imaging"
    medical_history: Optional[str] = "None"

# -------- Endpoints --------
@app.get("/")
async def root():
    return {"status": "running", "mode": "cloud-lite"}

@app.post("/process_patient")
async def process_patient(patient: Patient):
    return pipeline.process_patient(patient.dict())
