
"""
BioBERT Medical Text Analysis Module
===================================

This module implements medical diagnosis using BioBERT, a pre-trained BERT model
specifically fine-tuned on biomedical texts. It provides semantic analysis of
medical symptoms and matches them against a knowledge base of medical conditions.

Key Features:
- Pre-trained BioBERT model for medical text understanding
- Symptom-to-condition mapping using semantic similarity
- Confidence scoring based on embedding similarity
- Fallback error handling for model loading failures

Model Source: dmis-lab/biobert-base-cased-v1.2
License: Open source, suitable for research and educational use
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

class BioBERTDiagnosis:
    """
    BioBERT-based medical diagnosis system
    
    This class uses the BioBERT model to analyze medical symptoms and provide
    diagnosis suggestions based on semantic similarity to known medical conditions.
    """

    def __init__(self):
        """
        Initialize BioBERT model and medical knowledge base
        
        Loads the pre-trained BioBERT model and sets up the medical conditions
        database for symptom matching. Includes error handling for model loading failures.
        """
        print("🔄 Loading BioBERT model...")
        try:
            # Load BioBERT model specifically fine-tuned for biomedical text
            self.model_name = "dmis-lab/biobert-base-cased-v1.2"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            print("✅ BioBERT loaded successfully!")
            
            # Medical knowledge base: condition -> typical symptoms mapping
            # This serves as the reference for similarity matching
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
            self.model = None  # Graceful degradation

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate BioBERT embeddings for medical text
        
        Converts medical text into high-dimensional vector representations
        that capture semantic meaning for similarity comparisons.
        
        Args:
            text (str): Medical text to embed
            
        Returns:
            np.ndarray: 768-dimensional embedding vector
        """
        # Fallback to random vector if model failed to load
        # NOTE: This is a robustness issue - should be handled better
        if not self.model: 
            return np.random.rand(768)  # ROBUSTNESS ISSUE: Silent failure
        
        try:
            # Tokenize and encode text for BioBERT processing
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding (first token) as text representation
                return outputs.last_hidden_state[:, 0, :].numpy().flatten()
        except Exception as e:
            print(f"Error getting BioBERT embedding: {e}")
            return np.random.rand(768)  # ROBUSTNESS ISSUE: Silent failure

    def diagnose(self, symptoms: str, context: str = "") -> dict:
        """
        Perform medical diagnosis using BioBERT similarity matching
        
        Analyzes patient symptoms by comparing their semantic embeddings
        to known medical conditions in the knowledge base.
        
        Args:
            symptoms (str): Patient-reported symptoms
            context (str): Additional context (history, test results)
            
        Returns:
            dict: Diagnosis result with condition, confidence, and reasoning
        """
        # Combine symptoms and context for comprehensive analysis
        full_text = f"{symptoms} {context}".strip()
        symptom_embedding = self.get_embedding(full_text)
        
        # Find best matching condition using cosine similarity
        best_match, best_score = None, 0
        for condition, keywords in self.medical_conditions.items():
            condition_embedding = self.get_embedding(" ".join(keywords))
            # Calculate cosine similarity between symptom and condition embeddings
            similarity = np.dot(symptom_embedding, condition_embedding) / (
                np.linalg.norm(symptom_embedding) * np.linalg.norm(condition_embedding)
            )
            if similarity > best_score:
                best_score, best_match = similarity, condition
        
        # Map internal condition names to human-readable diagnoses
        diagnosis_map = {
            "viral_infection": "Viral Infection (Common Cold/Flu)", 
            "bacterial_infection": "Bacterial Infection", 
            "pneumonia": "Pneumonia", 
            "gastroenteritis": "Gastroenteritis", 
            "migraine": "Migraine Headache", 
            "arthritis": "Arthritis/Joint Inflammation", 
            "allergic_reaction": "Allergic Reaction", 
            "cardiac_issue": "Cardiac Evaluation Needed", 
            "respiratory_infection": "Upper Respiratory Infection", 
            "food_poisoning": "Food Poisoning"
        }
        
        diagnosis = diagnosis_map.get(best_match, "Requires Further Evaluation")
        
        # Apply confidence boost (1.2x) but cap at 95% to avoid overconfidence
        return {
            "diagnosis": diagnosis, 
            "confidence": min(best_score * 1.2, 0.95), 
            "reasoning": f"BioBERT similarity analysis (score: {best_score:.3f})"
        }
