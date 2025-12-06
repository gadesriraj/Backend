import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# ------- PATH SETUP --------
# app.py and Src/ are in the same folder in the backend repo
current_file = Path(__file__).resolve()
project_root = current_file.parent           # backend folder
src_path = project_root / "Src"             # backend/Src

# Add both project root and Src to sys.path
for p in (project_root, src_path):
    if str(p) not in sys.path:
        sys.path.append(str(p))

# ------- PIPELINE IMPORT --------
try:
    # workflow_pipeline.py is inside Src/
    from workflow_pipeline import AutomatedWorkflowPipeline
    # if this still fails, we can also try:
    # from Src.workflow_pipeline import AutomatedWorkflowPipeline
except ImportError as e:
    print(f"Error importing pipeline: {e}")
    AutomatedWorkflowPipeline = None


app = FastAPI(
    title="Clinical Documentation AI API",
    description="API for generating clinical notes and assigning ICD-10 codes from patient data.",
    version="1.0.0"
)

# Initialize pipeline
pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    try:
        print("Initializing AutomatedWorkflowPipeline...")
        pipeline = AutomatedWorkflowPipeline()
        print("✅ Pipeline initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")

# Pydantic models for request/response
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

@app.get("/")
async def root():
    return {"message": "Clinical Documentation AI API is running. Use /docs for Swagger UI."}

@app.get("/health")
async def health_check():
    if pipeline:
        return {"status": "healthy", "pipeline_loaded": True}
    return {"status": "degraded", "pipeline_loaded": False}

@app.post("/process_patient", response_model=ProcessResponse)
async def process_patient_endpoint(patient: PatientInput):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Convert Pydantic model to dict for the pipeline
    patient_data = patient.dict()
    
    try:
        result = pipeline.process_patient(patient_data)
        if not result:
            raise HTTPException(status_code=500, detail="Pipeline returned no result")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_batch")
async def process_batch_endpoint(patients: List[PatientInput]):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    results = []
    for patient in patients:
        try:
            res = pipeline.process_patient(patient.dict())
            if res:
                results.append(res)
        except Exception as e:
            print(f"Error processing patient {patient.name}: {e}")
            
    return {"processed_count": len(results), "results": results}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)