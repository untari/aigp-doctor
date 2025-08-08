
from transformers import pipeline

class ClinicalBERTAnalysis:
    """Using Clinical BERT for medical analysis"""
    def __init__(self):
        print("🔄 Loading Clinical BERT...")
        try:
            self.model_name = "emilyalsentzer/Bio_ClinicalBERT"
            self.classifier = pipeline("text-classification", model=self.model_name, tokenizer=self.model_name, return_all_scores=True)
            print("✅ Clinical BERT loaded!")
        except Exception as e:
            print(f"❌ Error loading Clinical BERT: {e}")
            self.classifier = None

    def analyze_severity(self, symptoms: str) -> dict:
        if not self.classifier: return {"severity": "moderate", "confidence": 0.5}
        try:
            results = self.classifier(symptoms)
            # The model will return a list of dictionaries for each label.
            # We need to find the label with the highest score.
            best_result = max(results[0], key=lambda x: x['score'])
            severity = best_result['label']
            confidence = best_result['score']
            return {"severity": severity, "confidence": confidence}
        except Exception as e:
            print(f"Error in severity analysis: {e}")
            return {"severity": "moderate", "confidence": 0.5}
