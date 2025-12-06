from typing import Dict
from datetime import datetime

# ========================================
# STEP 1: PREPARE AND FORMAT INPUT DATA
# ========================================

class DataPreparationPipeline:
    """Prepare and format patient data for model input"""

    def __init__(self):
        self.prepared_data = []

    def prepare_patient_json(self, patient_data: Dict) -> Dict:
        """Convert raw patient data to structured JSON format"""
        return {
            "PatientName": str(patient_data.get('name', 'Unknown')),
            "Age": int(patient_data.get('age', 0)),
            "Gender": str(patient_data.get('gender', 'Not specified')),
            "Symptoms": str(patient_data.get('symptoms', '')),
            "ScanResult": str(patient_data.get('scan_result', 'No imaging')),
            "MedicalHistory": str(patient_data.get('medical_history', '')),
            "VitalSigns": patient_data.get('vital_signs', {}),
            "Timestamp": datetime.now().isoformat()
        }

    def format_for_model(self, patient_json: Dict) -> str:
        """Format patient data as prompt for LLM"""
        symptoms = patient_json.get('Symptoms', 'General checkup')
        age = patient_json.get('Age', 'Unknown')
        gender = patient_json.get('Gender', 'Unknown')
        scan = patient_json.get('ScanResult', 'No imaging')

        prompt = f"""Generate a brief clinical note. Do NOT repeat information.

Patient: {age}yo {gender}
Chief Complaint: {symptoms}
Imaging: {scan}

Write concise clinical note with: Chief complaint. Physical exam findings. Assessment. Plan for follow-up."""

        return prompt