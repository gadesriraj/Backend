import logging
import torch
from typing import Dict
from transformers import pipeline

logger = logging.getLogger(__name__)

# ========================================
# STEP 2: CONNECT TO HUGGING FACE MODEL
# ========================================

class HuggingFaceModelConnector:
    """Handle Hugging Face model connections (Free, No API Key)"""

    def __init__(self):
        self.model_type = None
        self.generator = None
        self.initialize_models()

    def initialize_models(self):
        """Initialize Hugging Face models for clinical text generation"""
        try:
            logger.info("🤖 Loading Hugging Face medical models...")
            logger.info("⏳ This may take 2-3 minutes on first run (models are cached)")

            # Use Text2Text generation with better model
            logger.info("Loading model for clinical note generation...")
            self.generator = pipeline(
                "text2text-generation",
                model="google/flan-t5-large",
                device=0 if torch.cuda.is_available() else -1,
                framework="pt"
            )

            self.model_type = "Google FLAN-T5 Large"
            logger.info(f"✅ {self.model_type} loaded successfully!")
            logger.info(f"   Running on: {'GPU' if torch.cuda.is_available() else 'CPU'}")

        except Exception as e:
            logger.warning(f"Could not load FLAN-T5 Large: {e}. Trying base model...")
            try:
                self.generator = pipeline(
                    "text2text-generation",
                    model="google/flan-t5-base",
                    device=0 if torch.cuda.is_available() else -1
                )
                self.model_type = "Google FLAN-T5 Base"
                logger.info(f"✅ {self.model_type} loaded successfully!")
            except Exception as e2:
                logger.warning(f"Could not load FLAN-T5: {e2}")
                logger.info("Using rule-based template generation...")
                self.model_type = "Rule-Based Template"
                self.generator = None

    def generate_clinical_output(self, prompt: str) -> str:
        """Generate clinical note using Hugging Face model"""
        if self.generator:
            try:
                result = self.generator(
                    prompt,
                    max_length=250,
                    min_length=60,
                    do_sample=False,
                    num_beams=5,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    length_penalty=1.5
                )
                generated_text = result[0]['generated_text'].strip()

                # Clean up any remaining repetitions
                cleaned_text = self._aggressive_remove_repetitions(generated_text)
                return cleaned_text
            except Exception as e:
                logger.warning(f"Generation failed: {e}")
                return None

        return None

    def _aggressive_remove_repetitions(self, text: str) -> str:
        """Aggressively remove all types of repetitions"""
        # Split by periods
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        if not sentences:
            return text

        # Remove exact duplicates first
        seen = []
        for sent in sentences:
            if sent not in seen:
                seen.append(sent)

        # Remove sentences that are substrings of other sentences
        filtered = []
        for sentence in seen:
            is_substring = False
            for other in seen:
                if sentence != other and sentence.lower() in other.lower():
                    is_substring = True
                    break
            if not is_substring:
                filtered.append(sentence)

        # Remove highly similar sentences
        final_sentences = []
        for sentence in filtered:
            is_similar = False
            for existing in final_sentences:
                similarity = self._calculate_similarity(sentence, existing)
                if similarity > 0.75:  # 75% similar = remove
                    is_similar = True
                    break
            if not is_similar:
                final_sentences.append(sentence)

        return '. '.join(final_sentences) + '.' if final_sentences else text

    def _calculate_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate similarity between two sentences using Jaccard similarity"""
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _are_sentences_similar(self, sent1: str, sent2: str, threshold: float = 0.8) -> bool:
        """Check if two sentences are similar"""
        return self._calculate_similarity(sent1, sent2) > threshold

    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_type,
            "device": "GPU" if torch.cuda.is_available() else "CPU",
            "framework": "PyTorch",
            "source": "Hugging Face Hub"
        }