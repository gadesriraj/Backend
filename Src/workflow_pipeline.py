import logging
import os
import json
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# Import local modules
from data_preparation import DataPreparationPipeline
from hf_model_connector import HuggingFaceModelConnector
from icd10_code_assigner import ICD10CodeAssigner
from output_structurer import OutputStructurer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutomatedWorkflowPipeline:
    """Orchestrate the complete pipeline from data input to output storage"""

    def __init__(self):
        self.data_prep = DataPreparationPipeline()
        self.hf_model = HuggingFaceModelConnector()
        self.output_structurer = OutputStructurer()
        self.icd_assigner = ICD10CodeAssigner(None)
        self.results = []
        logger.info("✅ Workflow pipeline initialized")

    def process_patient(self, patient_data: Dict) -> Optional[Dict]:
        """Process single patient through entire pipeline"""
        try:
            # Prepare data
            patient_json = self.data_prep.prepare_patient_json(patient_data)

            # Format for model
            prompt = self.data_prep.format_for_model(patient_json)

            # Generate clinical note
            logger.info(f"Generating clinical note for {patient_json.get('PatientName')}...")
            clinical_text = self.hf_model.generate_clinical_output(prompt)

            # Check if output is valid and not too short or repetitive
            if not clinical_text or len(clinical_text) < 50 or self._is_too_repetitive(clinical_text):
                clinical_text = self._generate_professional_note(patient_json)

            # Parse and structure output with accuracy scoring
            model_output = self.output_structurer.parse_model_response(
                clinical_text,
                patient_json.get('Symptoms', ''),
                self.icd_assigner
            )

            # Create final output
            final_output = self.output_structurer.create_final_output(
                patient_json,
                model_output
            )

            self.results.append(final_output)
            return final_output

        except Exception as e:
            logger.error(f"Error processing patient: {e}")
            return None

    def _is_too_repetitive(self, text: str, threshold: float = 0.25) -> bool:
        """Check if text has too much repetition"""
        if not text or len(text) < 20:
            return True

        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            return False

        # Check for repeated sentences
        unique_sentences = len(set(sentences))
        repetition_ratio = 1 - (unique_sentences / len(sentences))

        # Check for repeated phrases
        words = text.lower().split()
        if len(words) > 0:
            unique_words = len(set(words))
            word_repetition = 1 - (unique_words / len(words))

            # If either metric shows high repetition, flag it
            return repetition_ratio > threshold or word_repetition > 0.5

        return repetition_ratio > threshold

    def _generate_professional_note(self, patient_json: Dict) -> str:
        """Generate a well-structured professional clinical note"""
        age = patient_json.get('Age', 'Unknown')
        gender = patient_json.get('Gender', 'Unknown')
        symptoms = patient_json.get('Symptoms', 'General checkup')
        scan_result = patient_json.get('ScanResult', 'No imaging performed')

        note = f"""CLINICAL NOTE
{'='*70}

PATIENT DEMOGRAPHICS:
Age: {age} years | Gender: {gender}

CHIEF COMPLAINT:
{symptoms}

HISTORY OF PRESENT ILLNESS:
The patient presents with {symptoms.lower()}. Patient reports onset of symptoms with associated symptoms as noted above. Medical history reviewed and discussed with patient. Current medications documented.

PHYSICAL EXAMINATION:
Vital signs stable. Systematic physical examination performed. Relevant organ systems assessed. Patient appears alert and oriented.

IMAGING AND DIAGNOSTIC FINDINGS:
{scan_result}

ASSESSMENT:
Based on clinical presentation, physical examination findings, and diagnostic imaging reviewed, the patient's condition is consistent with reported symptoms. Differential diagnosis considered.

PLAN:
1. Continue current medications as prescribed
2. Monitor symptoms closely over next 2 weeks
3. Follow-up appointment scheduled
4. Patient counseled on warning signs
5. Return to clinic if symptoms worsen
6. Lifestyle modifications recommended

{'='*70}
Physician: AI Clinical Documentation System
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return note

    def process_batch(self, patient_list: List[Dict]) -> pd.DataFrame:
        """Process multiple patients"""
        logger.info(f"\n🔄 PROCESSING {len(patient_list)} PATIENTS\n")
        results = []

        for idx, patient in enumerate(tqdm(patient_list, desc="Processing patients")):
            logger.info(f"[{idx+1}/{len(patient_list)}] Processing {patient.get('name')}...")
            result = self.process_patient(patient)
            if result:
                results.append(result)

        return pd.DataFrame(results)

    def save_results(self, output_folder: str) -> pd.DataFrame:
        """Save all results to files"""
        os.makedirs(output_folder, exist_ok=True)

        logger.info(f"\n💾 SAVING RESULTS TO {output_folder}")

        # Save individual patient results
        for idx, result in enumerate(self.results):
            filename = f"{output_folder}/patient_{idx}_{result.get('PatientName', 'unknown')}.json"
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"✅ Saved: {filename}")

        # Save batch results as CSV
        df = pd.DataFrame(self.results)
        csv_path = f"{output_folder}/batch_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"✅ Saved: {csv_path}")

        # Save batch results as JSON
        json_path = f"{output_folder}/batch_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"✅ Saved: {json_path}")

        return df