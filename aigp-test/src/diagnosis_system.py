
from datetime import datetime
from typing import Dict

from src.models.biobert import BioBERTDiagnosis
from src.models.clinical_bert import ClinicalBERTAnalysis
from src.models.sentence_transformer import SentenceTransformerDiagnosis
from src.models.question_generator import DynamicQuestionGenerator
from src.models.image_expert import ImageAnalysisExpert
from src.models.general_expert import GeneralAIExpert
from src.models.secondary_analyzer import SecondaryAIAnalyzer
from src.models.feedback_generator import ActionableFeedbackGenerator
from src.models.medication_expert import MedicationRecommendationExpert

class DiagnosisSystem:
    """Complete diagnosis system using only free, open-source models"""
    def __init__(self):
        print("🚀 Initializing Free Open-Source AI Diagnosis System...")
        self.biobert = BioBERTDiagnosis()
        self.clinical_bert = ClinicalBERTAnalysis()
        self.sentence_transformer = SentenceTransformerDiagnosis()
        self.question_generator = DynamicQuestionGenerator()
        self.image_expert = ImageAnalysisExpert()
        self.general_expert = GeneralAIExpert()
        self.secondary_ai = SecondaryAIAnalyzer()  # A.I.(2)
        self.feedback_generator = ActionableFeedbackGenerator()
        self.medication_expert = MedicationRecommendationExpert()  # New AI expert
        print("✅ System ready!")

    def comprehensive_diagnosis(self, symptoms: str, context: str = "") -> Dict:
        """A.I.(1) - Primary diagnosis using ensemble of medical models"""
        self.log_step("🔍 A.I.(1) Starting comprehensive analysis", f"Symptoms: {symptoms}, Context: {context}")
        
        # Check for emergency symptoms first
        emergency_alert = self.feedback_generator.generate_emergency_recommendations(symptoms)
        
        biobert_result = self.biobert.diagnose(symptoms, context)
        severity_result = self.clinical_bert.analyze_severity(symptoms)
        similarity_result = self.sentence_transformer.find_similar_cases(symptoms)
        
        # Enhanced confidence calculation with context weighting
        base_confidence = (biobert_result['confidence'] * 0.5 + severity_result['confidence'] * 0.2 + similarity_result['confidence'] * 0.3)
        
        # Context bonus - more context increases confidence
        context_bonus = min(len(context.split()) * 0.01, 0.1) if context else 0
        
        final_confidence = base_confidence + context_bonus
        if severity_result['severity'] == 'severe': final_confidence = min(final_confidence + 0.1, 0.95)
        elif severity_result['severity'] == 'mild': final_confidence = max(final_confidence - 0.1, 0.3)
        
        result = {
            "diagnosis": biobert_result['diagnosis'],
            "confidence": final_confidence,
            "severity": severity_result['severity'],
            "similar_case": similarity_result['similar_case'],
            "reasoning": f"A.I.(1) Multi-model analysis: BioBERT ({biobert_result['confidence']:.2f}), ClinicalBERT ({severity_result['confidence']:.2f}), SentenceTransformer ({similarity_result['confidence']:.2f}), Context bonus: {context_bonus:.2f}",
            "ai_stage": "primary"
        }
        
        # Add emergency alert if detected
        if emergency_alert:
            result["emergency_alert"] = emergency_alert
            result["severity"] = "emergency"
        
        return result

    def enhanced_diagnosis_with_context(self, symptoms: str, all_context: str, previous_diagnosis: Dict) -> Dict:
        """Re-run A.I.(1) with enhanced context after A.I.(2) gathers more info"""
        self.log_step("🔄 A.I.(1) Re-analysis with enhanced context", f"Previous confidence: {previous_diagnosis['confidence']:.2f}")
        
        # Run primary diagnosis again with all accumulated context
        new_result = self.comprehensive_diagnosis(symptoms, all_context)
        
        # Compare with previous result to show improvement
        confidence_improvement = new_result['confidence'] - previous_diagnosis['confidence']
        new_result['improvement'] = confidence_improvement
        new_result['reasoning'] += f" | Confidence improved by {confidence_improvement:.2f}"
        
        return new_result

    def secondary_analysis(self, symptoms: str, initial_diagnosis: Dict) -> Dict:
        """A.I.(2) - Secondary analysis for follow-up questions and test suggestions"""
        self.log_step("🔍 A.I.(2) Secondary analysis", f"Initial confidence: {initial_diagnosis['confidence']:.2f}")
        
        follow_up_questions = self.secondary_ai.generate_follow_up_questions(
            symptoms, initial_diagnosis['diagnosis'], initial_diagnosis['confidence']
        )
        
        suggested_tests = self.secondary_ai.suggest_additional_tests(
            symptoms, initial_diagnosis['diagnosis']
        )
        
        return {
            "follow_up_questions": follow_up_questions,
            "suggested_tests": suggested_tests,
            "ai_stage": "secondary",
            "reasoning": "A.I.(2) analysis for context enhancement"
        }

    def analyze_image_in_context(self, image, current_context: str) -> str:
        """Integrate image analysis into the diagnostic flow"""
        image_description = self.image_expert.analyze_image(image)
        
        # Use A.I.(2) to relate image to current symptoms
        if self.secondary_ai.analyzer:
            try:
                prompt = f"How does this image relate to these symptoms: {current_context}. Image shows: {image_description}"
                result = self.secondary_ai.analyzer(prompt, max_new_tokens=100, temperature=0.3)
                contextualized_analysis = result[0]["generated_text"]
                return f"Image analysis: {image_description}. Medical relevance: {contextualized_analysis}"
            except:
                return f"Image analysis: {image_description}"
        
        return f"Image analysis: {image_description}"

    def get_enhanced_medication_recommendations(self, diagnosis: str, symptoms: str, severity: str, current_medications: list = None) -> Dict:
        """Get AI-powered medication recommendations with safety checks"""
        self.log_step("💊 Getting enhanced medication recommendations", f"Diagnosis: {diagnosis}")
        
        # Get personalized recommendations from AI expert
        recommendations = self.medication_expert.generate_personalized_recommendations(
            diagnosis, symptoms, severity
        )
        
        # Add drug interaction warnings if current medications provided
        if current_medications:
            interaction_warning = self.medication_expert.get_drug_interactions_warning(current_medications)
            recommendations["drug_interactions"] = interaction_warning
        
        return recommendations

    def log_step(self, step: str, details: str = ""):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {step}"
        if details: log_entry += f": {details}"
        print(log_entry)
