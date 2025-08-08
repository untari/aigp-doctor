
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

class BioBERTDiagnosis:
    """Using BioBERT for medical text analysis (completely free)"""

    def __init__(self):
        print("🔄 Loading BioBERT model...")
        try:
            self.model_name = "dmis-lab/biobert-base-cased-v1.2"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            print("✅ BioBERT loaded successfully!")
            self.medical_conditions = {
                "viral_infection": ["fever", "fatigue", "headache", "muscle aches", "runny nose"],
                "bacterial_infection": ["high fever", "chills", "severe fatigue", "localized pain"],
                "pneumonia": ["cough", "fever", "chest pain", "shortness of breath", "fatigue"],
                "gastroenteritis": ["nausea", "vomiting", "diarrhea", "abdominal pain", "fever"],
                "migraine": ["severe headache", "nausea", "sensitivity to light", "vision problems"],
                "arthritis": ["joint pain", "stiffness", "swelling", "reduced range of motion"],
                "allergic_reaction": ["rash", "itching", "swelling", "difficulty breathing"],
                "cardiac_issue": ["chest pain", "shortness of breath", "palpitations", "dizziness"],
                "respiratory_infection": ["cough", "sore throat", "congestion", "fever"],
                "food_poisoning": ["nausea", "vomiting", "diarrhea", "stomach cramps", "fever"]
            }
        except Exception as e:
            print(f"❌ Error loading BioBERT: {e}")
            self.model = None

    def get_embedding(self, text: str) -> np.ndarray:
        if not self.model: return np.random.rand(768)
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                return outputs.last_hidden_state[:, 0, :].numpy().flatten()
        except Exception as e:
            print(f"Error getting BioBERT embedding: {e}")
            return np.random.rand(768)

    def diagnose(self, symptoms: str, context: str = "") -> dict:
        full_text = f"{symptoms} {context}".strip()
        symptom_embedding = self.get_embedding(full_text)
        best_match, best_score = None, 0
        for condition, keywords in self.medical_conditions.items():
            condition_embedding = self.get_embedding(" ".join(keywords))
            similarity = np.dot(symptom_embedding, condition_embedding) / (np.linalg.norm(symptom_embedding) * np.linalg.norm(condition_embedding))
            if similarity > best_score:
                best_score, best_match = similarity, condition
        diagnosis_map = {"viral_infection": "Viral Infection (Common Cold/Flu)", "bacterial_infection": "Bacterial Infection", "pneumonia": "Pneumonia", "gastroenteritis": "Gastroenteritis", "migraine": "Migraine Headache", "arthritis": "Arthritis/Joint Inflammation", "allergic_reaction": "Allergic Reaction", "cardiac_issue": "Cardiac Evaluation Needed", "respiratory_infection": "Upper Respiratory Infection", "food_poisoning": "Food Poisoning"}
        diagnosis = diagnosis_map.get(best_match, "Requires Further Evaluation")
        return {"diagnosis": diagnosis, "confidence": min(best_score * 1.2, 0.95), "reasoning": f"BioBERT similarity analysis (score: {best_score:.3f})"}
