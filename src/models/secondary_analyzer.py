
import torch
from transformers import pipeline
from typing import List

class SecondaryAIAnalyzer:
    """A.I.(2) - Secondary analyzer for follow-up questions and context refinement"""
    def __init__(self):
        print("🔄 Loading Secondary AI Analyzer...")
        try:
            # Using a different model for A.I.(2) - FLAN-T5 for structured analysis
            self.analyzer = pipeline("text2text-generation", model="google/flan-t5-small", model_kwargs={"torch_dtype": torch.float16})
            print("✅ Secondary AI Analyzer loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading Secondary AI Analyzer: {e}")
            self.analyzer = None

    def generate_follow_up_questions(self, symptoms: str, initial_diagnosis: str, confidence: float) -> List[str]:
        """Generate targeted follow-up questions based on low confidence diagnosis"""
        # Use fallback questions directly for better reliability
        return self._get_fallback_questions(initial_diagnosis)

    def _get_fallback_questions(self, diagnosis: str) -> List[str]:
        """Fallback questions when AI generation fails"""
        question_bank = {
            "Viral Infection": ["How high is the fever?", "Any sore throat or runny nose?"],
            "Bacterial Infection": ["Is there localized pain or swelling?", "How long have symptoms persisted?"],
            "Pneumonia": ["Any colored phlegm when coughing?", "Difficulty breathing or chest pain?"],
            "Gastroenteritis": ["Any blood in stool or vomit?", "Recent food or travel history?"],
            "Migraine": ["Is pain on one side of head?", "Any visual disturbances?"],
            "Cardiac": ["Any chest pain or pressure?", "Any shortness of breath with activity?"],
            "Musculoskeletal": ["What activities trigger the pain?", "Any recent injuries or physical strain?"],
            "Back Pain": ["Does the pain radiate down your legs?", "Any numbness or tingling?"],
            "Digestive": ["What foods trigger symptoms?", "Any blood in stool or severe cramping?"],
            "Headache": ["Is pain throbbing or constant?", "Any sensitivity to light or sound?"],
            "Skin": ["When did the rash first appear?", "Any known allergies or recent exposures?"],
            "Anxiety": ["What triggers these feelings?", "Any physical symptoms like rapid heartbeat?"],
            "Sleep": ["How many hours of sleep per night?", "Any snoring or breathing issues?"],
            "Neurological": ["Any weakness or coordination problems?", "When do symptoms occur most?"],
            "Urinary": ["Any fever or back pain?", "How often do you urinate per day?"],
            "Fatigue": ["How long have you felt tired?", "Any changes in appetite or weight?"],
            "Default": ["Rate symptom severity 1-10?", "Any recent changes in symptoms?"]
        }
        
        for key in question_bank:
            if key.lower() in diagnosis.lower():
                return question_bank[key]
        return question_bank["Default"]

    def suggest_additional_tests(self, symptoms: str, diagnosis: str) -> List[str]:
        """Suggest additional tests or information that could help diagnosis"""
        if not self.analyzer:
            return ["Temperature measurement", "Blood pressure check"]
        
        # Use fallback tests directly for better reliability
        return self._get_fallback_tests(diagnosis)

    def _get_fallback_tests(self, diagnosis: str) -> List[str]:
        """Fallback test suggestions"""
        test_bank = {
            "fever": ["Temperature measurement", "Hydration status check"],
            "respiratory": ["Breathing rate count", "Oxygen saturation if available"],
            "cardiac": ["Pulse rate check", "Blood pressure measurement", "ECG if available"],
            "gastrointestinal": ["Hydration assessment", "Abdominal tenderness check"],
            "musculoskeletal": ["Range of motion test", "Physical examination", "Posture assessment"],
            "back": ["Straight leg raise test", "Flexibility assessment", "Neurological exam"],
            "pain": ["Pain scale rating", "Movement triggers assessment", "Physical examination"],
            "digestive": ["Food diary tracking", "Hydration assessment", "Abdominal examination"],
            "headache": ["Blood pressure check", "Vision screening", "Neck mobility test"],
            "skin": ["Patch test consideration", "Medication review", "Environmental assessment"],
            "anxiety": ["Stress level assessment", "Sleep pattern review", "Breathing exercises"],
            "sleep": ["Sleep diary", "Caffeine intake review", "Bedroom environment check"],
            "neurological": ["Balance test", "Reflexes check", "Coordination assessment"],
            "urinary": ["Urine analysis", "Fluid intake tracking", "Bladder diary"],
            "fatigue": ["Energy level tracking", "Activity assessment", "Nutrition review"],
            "default": ["Vital signs check", "Symptom duration tracking"]
        }
        
        for key in test_bank:
            if key in diagnosis.lower():
                return test_bank[key]
        return test_bank["default"]
