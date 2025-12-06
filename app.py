import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from tinydb import TinyDB, Query
from datetime import datetime

# ---------------- PATH SETUP ----------------
current_file = Path(__file__).resolve()
project_root = current_file.parent
src_path = project_root / "Src"

for p in (project_root, src_path):
    if str(p) not in sys.path:
        sys.path.append(str(p))

# ---------------- PIPELINE IMPORT ----------------
try:
    from workflow_pipeline import AutomatedWorkflowPipeline
except ImportError as e:
    print(f"Error importing pipeline: {e}")
    AutomatedWorkflowPipeline = None


# ---------------- FASTAPI APP ----------------
app = FastAPI(
    title="Clinical Documentation AI API",
    description="API for generating clinical notes + ICD-10 codes",
    version="1.1.0"
)

# ---------------- DATABASE (NEW) ----------------
db = TinyDB("clinical_records.json")  # stores all generated notes


# ---------------- PIPELINE INIT ----------------
pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    try:
        print("Initializing AutomatedWorkflowPipeline...")
        pipeline = AutomatedWorkflowPipeline()
        print("✅ Pipeline initialized")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")


# ---------------- Pydantic MODELS ----------------
class PatientInput(BaseModel):
    name: str
    age: int
    gender: str
    symptoms: str
    scan_result: Optional[str] = "No imaging performed"
    medical_history: Optional[str] = "None"
    vital_signs: Optional[Dict[str, Any]] = {}


class ICDInfo(BaseModel):
    code: str
    description: str
    confidence: float
    evidence: Dict[str, Any]


class ClinicalDoc(BaseModel):
    generated_note: str
    icd_coding: ICDInfo


class ProcessResponse(BaseModel):
    patient_id: str
    timestamp: str
    patient_data: Dict[str, Any]
    clinical_documentation: ClinicalDoc
    metadata: Dict[str, Any]


# ---------------- BASIC ROUTES ----------------
@app.get("/")
async def root():
    return {"message": "Clinical Documentation API running", "use": "/process_patient"}


@app.get("/health")
async def health_check():
    return {"status": "healthy" if pipeline else "pipeline_not_loaded"}


# ---------------- PROCESS ONE PATIENT ----------------
@app.post("/process_patient", response_model=ProcessResponse)
async def process_patient_endpoint(patient: PatientInput):

    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    patient_data = patient.dict()

    try:
        result = pipeline.process_patient(patient_data)

        if not result:
            raise HTTPException(status_code=500, detail="Pipeline returned no data")

        # ---------------- SAVE TO DATABASE (NEW) ----------------
        db.insert({
            "id": result["patient_id"],
            "timestamp": result["timestamp"],
            "patient": result["patient_data"],
            "note": result["clinical_documentation"]["generated_note"],
            "icd": result["clinical_documentation"]["icd_coding"]
        })

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ---------------- PROCESS MULTIPLE PATIENTS ----------------
@app.post("/process_batch")
async def process_batch_endpoint(patients: List[PatientInput]):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    results = []
    for patient in patients:
        try:
            res = pipeline.process_patient(patient.dict())
            if res:
                db.insert(res)  # optional: save batch results too
                results.append(res)
        except Exception as e:
            print(f"Error processing {patient.name}: {e}")

    return {"processed_count": len(results), "results": results}



# ---------------- VIEW ALL RECORDS (NEW) ----------------
@app.get("/records")
async def get_records():
    """Returns ALL saved clinical notes."""
    records = db.all()
    return {"count": len(records), "records": records}


# ---------------- RUN LOCALLY ----------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
