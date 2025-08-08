
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
        if not self.analyzer:
            return ["Can you provide more details about the symptoms?", "When did the symptoms start?"]
        
        try:
            prompt = f"Generate 2 specific medical follow-up questions for these symptoms: {symptoms}. Initial diagnosis was {initial_diagnosis} with {confidence:.1%} confidence."
            result = self.analyzer(prompt, max_new_tokens=100, temperature=0.3)
            generated_text = result[0]["generated_text"]
            
            # Fallback to predefined questions if generation fails
            if len(generated_text.strip()) < 10:
                return self._get_fallback_questions(initial_diagnosis)
            
            return [q.strip() for q in generated_text.split('?') if q.strip()][:2]
            
        except Exception as e:
            print(f"Error generating follow-up questions: {e}")
            return self._get_fallback_questions(initial_diagnosis)

    def _get_fallback_questions(self, diagnosis: str) -> List[str]:
        """Fallback questions when AI generation fails"""
        question_bank = {
            "Viral Infection": ["How high is the fever?", "Any sore throat or runny nose?"],
            "Bacterial Infection": ["Is there localized pain or swelling?", "How long have symptoms persisted?"],
            "Pneumonia": ["Any colored phlegm when coughing?", "Difficulty breathing or chest pain?"],
            "Gastroenteritis": ["Any blood in stool or vomit?", "Recent food or travel history?"],
            "Migraine": ["Is pain on one side of head?", "Any visual disturbances?"],
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
        
        try:
            prompt = f"Suggest 3 simple tests or observations for symptoms: {symptoms} with suspected {diagnosis}"
            result = self.analyzer(prompt, max_new_tokens=80, temperature=0.3)
            generated_text = result[0]["generated_text"]
            
            if len(generated_text.strip()) > 10:
                tests = [t.strip() for t in generated_text.split(',') if t.strip()]
                return tests[:3]
            
        except Exception as e:
            print(f"Error suggesting tests: {e}")
        
        # Fallback suggestions
        return self._get_fallback_tests(diagnosis)

    def _get_fallback_tests(self, diagnosis: str) -> List[str]:
        """Fallback test suggestions"""
        test_bank = {
            "fever": ["Temperature measurement", "Hydration status check"],
            "respiratory": ["Breathing rate count", "Oxygen saturation if available"],
            "cardiac": ["Pulse rate check", "Blood pressure measurement"],
            "gastrointestinal": ["Hydration assessment", "Abdominal tenderness check"],
            "default": ["Vital signs check", "Symptom duration tracking"]
        }
        
        for key in test_bank:
            if key in diagnosis.lower():
                return test_bank[key]
        return test_bank["default"]
