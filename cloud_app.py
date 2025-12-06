
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime
from tinydb import TinyDB

app = FastAPI(title="Cloud Clinical Note API", version="1.1-lite")

# -------- Database (NEW) --------
db = TinyDB("clinical_records.json")   # simple JSON DB on disk

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

        result = {
            "patient_id": "DEMO-" + datetime.now().strftime("%Y%m%d%H%M%S"),
            "timestamp": datetime.now().isoformat(),
            "patient_data": data,
            "clinical_documentation": {
                "generated_note": note,
                "icd_coding": {
                    "code": "J18.9",
                    "description": "Pneumonia, unspecified organism",
                    "confidence": 0.80,
                    "evidence": {"reason": "Symptoms suggest respiratory infection"},
                },
            },
            "metadata": {"mode": "cloud-lite"},
        }
        return result

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
    result = pipeline.process_patient(patient.dict())

    # 🔥 SAVE to DB
    db.insert({
        "id": result["patient_id"],
        "timestamp": result["timestamp"],
        "patient": result["patient_data"],
        "note": result["clinical_documentation"]["generated_note"],
        "icd": result["clinical_documentation"]["icd_coding"],
    })

    return result

# NEW: list all saved records
@app.get("/records")
async def get_records():
    records = db.all()
    return {"count": len(records), "records": records}
