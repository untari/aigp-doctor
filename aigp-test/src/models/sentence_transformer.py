
from sentence_transformers import SentenceTransformer

class SentenceTransformerDiagnosis:
    """Using Sentence Transformers for semantic similarity"""
    def __init__(self):
        print("🔄 Loading Sentence Transformer...")
        try:
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.knowledge_base = [
                "Fever headache and fatigue usually indicate viral infection", 
                "Chest pain and shortness of breath may suggest cardiac problems", 
                "Nausea vomiting and diarrhea often indicate gastroenteritis", 
                "Severe headache with nausea may be migraine", 
                "Joint pain and stiffness suggest arthritis", 
                "Cough fever and chest pain may indicate pneumonia", 
                "Rash and itching often indicate allergic reaction", 
                "Abdominal pain with fever may be appendicitis",
                "Persistent cough with blood may indicate respiratory infection",
                "Sudden severe headache may indicate medical emergency",
                "Difficulty breathing with chest tightness suggests asthma or cardiac issue",
                "Skin rash with fever may indicate infectious disease"
            ]
            self.kb_embeddings = self.model.encode(self.knowledge_base)
            print("✅ Sentence Transformer loaded!")
        except Exception as e:
            print(f"❌ Error loading Sentence Transformer: {e}")
            self.model = None

    def find_similar_cases(self, symptoms: str) -> dict:
        if not self.model: return {"similar_case": "Unable to analyze", "confidence": 0.3}
        try:
            symptom_embedding = self.model.encode([symptoms])
            similarities = self.model.similarity(symptom_embedding, self.kb_embeddings)[0]
            best_idx = similarities.argmax()
            return {"similar_case": self.knowledge_base[best_idx], "confidence": similarities[best_idx].item()}
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return {"similar_case": "Analysis failed", "confidence": 0.2}
