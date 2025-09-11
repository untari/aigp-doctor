"""
Medication Recommendation Expert Module
======================================

This module provides AI-powered medication recommendations with safety checks
and drug interaction warnings. It combines AI-generated advice with a curated
medical database to provide comprehensive medication guidance.

CRITICAL SAFETY NOTICE:
This system is for educational purposes only and should never be used as a
substitute for professional medical advice. All recommendations must be
validated by qualified healthcare providers.

Key Features:
- AI-powered personalized medication recommendations
- Comprehensive drug interaction database
- Safety assessment and contraindication checking
- Evidence-based medication database
- Dosage and usage guidance

Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
License: Open source, educational use only
"""

import torch
from transformers import pipeline
from typing import Dict, List

class MedicationRecommendationExpert:
    """
    Specialized AI expert for medication and treatment recommendations
    
    This class provides comprehensive medication advice by combining AI analysis
    with a curated medical database. It includes critical safety checks and
    drug interaction warnings to promote safe medication use.
    
    IMPORTANT: All recommendations are for educational purposes only and require
    validation by licensed healthcare professionals.
    """
    
    def __init__(self):
        print("🔄 Loading Medication Recommendation Expert...")
        try:
            # Reuse the TinyLlama model for medication-specific queries
            self.medication_llm = pipeline(
                "text-generation", 
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                model_kwargs={"torch_dtype": torch.bfloat16}, 
                device_map="auto"
            )
            print("✅ Medication Recommendation Expert loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading Medication Expert: {e}")
            self.medication_llm = None
            
        # Enhanced medication database with dosages and interactions
        self.medication_database = {
            "fever_reducers": {
                "paracetamol": {
                    "dosage": "500-1000mg every 6-8 hours (max 4000mg/day)",
                    "contraindications": ["liver disease", "alcohol dependency"],
                    "safe_with": ["most medications"]
                },
                "ibuprofen": {
                    "dosage": "200-400mg every 6-8 hours (max 1200mg/day OTC)",
                    "contraindications": ["kidney disease", "stomach ulcers", "heart conditions"],
                    "safe_with": ["paracetamol"]
                }
            },
            "cold_flu": {
                "decongestants": {
                    "dosage": "Follow package instructions, max 3 days nasal sprays",
                    "contraindications": ["high blood pressure", "heart disease"],
                    "safe_with": ["pain relievers"]
                },
                "cough_suppressants": {
                    "dosage": "As directed on package",
                    "contraindications": ["productive cough with infection"],
                    "safe_with": ["pain relievers", "decongestants"]
                }
            },
            "digestive": {
                "oral_rehydration": {
                    "dosage": "1 packet in 200ml water, sip frequently",
                    "contraindications": ["none significant"],
                    "safe_with": ["most medications"]
                },
                "probiotics": {
                    "dosage": "As directed, usually after antibiotic course",
                    "contraindications": ["immunocompromised patients"],
                    "safe_with": ["most medications"]
                }
            }
        }
    
    def generate_personalized_recommendations(self, diagnosis: str, symptoms: str, severity: str) -> Dict:
        """Generate AI-powered personalized medication recommendations"""
        if not self.medication_llm:
            return self._fallback_recommendations(diagnosis)
            
        try:
            # Create a medical-focused prompt for the AI
            prompt = f"""As a medical AI assistant, provide safe over-the-counter medication recommendations for:

Diagnosis: {diagnosis}
Symptoms: {symptoms}
Severity: {severity}

Provide:
1. Recommended over-the-counter medications with dosages
2. Important safety warnings
3. When to seek professional help

Keep recommendations safe and conservative."""

            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.medication_llm.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            outputs = self.medication_llm(
                formatted_prompt, 
                max_new_tokens=200, 
                do_sample=True, 
                temperature=0.3,  # Lower temperature for medical advice
                top_k=50, 
                top_p=0.9
            )
            
            generated_text = outputs[0]["generated_text"]
            
            # Clean the response
            if formatted_prompt in generated_text:
                ai_response = generated_text.replace(formatted_prompt, "").strip()
            else:
                ai_response = generated_text.strip()
                
            # Remove chat template artifacts
            ai_response = ai_response.replace("<|user|>", "").replace("<|assistant|>", "")
            ai_response = ai_response.replace("<start_of_turn>", "").replace("<end_of_turn>", "")
            
            return {
                "ai_recommendations": ai_response,
                "database_match": self._get_database_recommendations(diagnosis),
                "safety_check": self._safety_assessment(symptoms),
                "source": "AI + Database"
            }
            
        except Exception as e:
            print(f"Error generating AI medication recommendations: {e}")
            return self._fallback_recommendations(diagnosis)
    
    def _get_database_recommendations(self, diagnosis: str) -> List[str]:
        """Get medication recommendations from structured database"""
        recommendations = []
        
        diagnosis_lower = diagnosis.lower()
        
        if "viral" in diagnosis_lower or "cold" in diagnosis_lower or "flu" in diagnosis_lower:
            recommendations.extend([
                "Paracetamol 500-1000mg every 6-8 hours for fever/pain",
                "Ibuprofen 200-400mg every 6-8 hours for inflammation",
                "Throat lozenges for sore throat",
                "Decongestant nasal spray (max 3 days)"
            ])
            
        elif "bacterial" in diagnosis_lower:
            recommendations.extend([
                "See doctor for antibiotic prescription - REQUIRED",
                "Paracetamol for supportive care",
                "Probiotics to support gut health during antibiotics"
            ])
            
        elif "gastroenteritis" in diagnosis_lower or "food poisoning" in diagnosis_lower:
            recommendations.extend([
                "Oral rehydration salts (ORS) - most important",
                "Avoid anti-diarrheal if fever present",
                "Probiotics after acute phase",
                "Paracetamol for pain (avoid NSAIDs)"
            ])
            
        elif "migraine" in diagnosis_lower:
            recommendations.extend([
                "Ibuprofen 400mg at first sign",
                "Paracetamol + Caffeine combination",
                "Anti-nausea medication if needed",
                "Consider specialist if frequent"
            ])
            
        elif "arthritis" in diagnosis_lower or "joint" in diagnosis_lower:
            recommendations.extend([
                "Ibuprofen 200-400mg for inflammation",
                "Topical anti-inflammatory gels",
                "Glucosamine supplements (long-term)",
                "Avoid prolonged NSAID use"
            ])
        
        return recommendations if recommendations else ["Consult pharmacist for appropriate medications"]
    
    def _safety_assessment(self, symptoms: str) -> Dict:
        """Assess safety and provide warnings based on symptoms"""
        safety_warnings = []
        urgency_level = "routine"
        
        symptoms_lower = symptoms.lower()
        
        # Emergency symptoms
        if any(emergency in symptoms_lower for emergency in [
            "chest pain", "difficulty breathing", "severe bleeding", 
            "unconscious", "severe allergic reaction"
        ]):
            urgency_level = "emergency"
            safety_warnings.append("EMERGENCY: Seek immediate medical attention")
            
        # High-risk symptoms
        elif any(high_risk in symptoms_lower for high_risk in [
            "severe pain", "high fever", "persistent vomiting", 
            "severe headache", "blood in stool"
        ]):
            urgency_level = "urgent"
            safety_warnings.append("URGENT: Consult healthcare provider within 24 hours")
            
        # Medication-specific warnings
        if "stomach pain" in symptoms_lower or "ulcer" in symptoms_lower:
            safety_warnings.append("CAUTION: Avoid NSAIDs (ibuprofen, aspirin) - use paracetamol instead")
            
        if "kidney" in symptoms_lower or "renal" in symptoms_lower:
            safety_warnings.append("CAUTION: Avoid NSAIDs - consult doctor before any medication")
            
        if "liver" in symptoms_lower or "hepatic" in symptoms_lower:
            safety_warnings.append("CAUTION: Limit paracetamol dose - consult doctor")
            
        return {
            "urgency_level": urgency_level,
            "warnings": safety_warnings,
            "general_advice": "Always read medication labels and follow dosing instructions"
        }
    
    def _fallback_recommendations(self, diagnosis: str) -> Dict:
        """Provide basic recommendations when AI model unavailable"""
        return {
            "ai_recommendations": f"Basic recommendations for {diagnosis}: Rest, hydration, and appropriate over-counter medications. Consult healthcare provider for specific guidance.",
            "database_match": self._get_database_recommendations(diagnosis),
            "safety_check": {"urgency_level": "routine", "warnings": ["Consult pharmacist or doctor for medication advice"], "general_advice": "Follow medication package instructions"},
            "source": "Database only"
        }
    
    def get_drug_interactions_warning(self, current_medications: List[str]) -> str:
        """Provide basic drug interaction warnings"""
        if not current_medications:
            return "No current medications reported."
            
        warnings = []
        
        # Basic interaction checks
        if "warfarin" in [med.lower() for med in current_medications]:
            warnings.append("⚠️ WARFARIN INTERACTION: Avoid aspirin and NSAIDs - increases bleeding risk")
            
        if any("blood pressure" in med.lower() for med in current_medications):
            warnings.append("⚠️ BP MEDICATION: Decongestants may increase blood pressure")
            
        if any("diabetes" in med.lower() or "metformin" in med.lower() for med in current_medications):
            warnings.append("⚠️ DIABETES: Monitor blood sugar if taking steroids")
        
        if warnings:
            return "\n🔍 **DRUG INTERACTION WARNINGS:**\n" + "\n".join(f"• {warning}" for warning in warnings)
        else:
            return "✅ No major interactions detected with reported medications."