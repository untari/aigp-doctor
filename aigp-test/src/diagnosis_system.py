
"""
Diagnosis System Core Module
============================

This module implements the central diagnosis system that coordinates multiple AI models
to provide comprehensive medical diagnosis assistance. It follows a two-stage AI approach:
- A.I.(1): Primary analysis using ensemble of specialized medical models
- A.I.(2): Secondary analysis for follow-up questions and context refinement

The system integrates multiple open-source medical AI models including BioBERT,
Clinical BERT, Sentence Transformers, and specialized image analysis models.

Security Note: All models run locally without external API calls for privacy.
"""

from datetime import datetime
from typing import Dict

# Import all specialized AI model components
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
    """
    Complete diagnosis system using only free, open-source models
    
    This class orchestrates multiple AI models to provide comprehensive medical
    diagnosis assistance. It implements a two-stage analysis approach:
    
    Stage 1 (A.I.1): Primary diagnosis using ensemble of medical models
    Stage 2 (A.I.2): Secondary analysis for low-confidence cases
    
    All models are loaded locally for privacy and security.
    """
    
    def __init__(self):
        """
        Initialize all AI models and components
        
        This method loads all the required AI models including:
        - BioBERT for medical text analysis
        - Clinical BERT for severity assessment
        - Sentence Transformers for similarity matching
        - Image analysis models for medical image processing
        - Question generators and feedback systems
        """
        print("🚀 Initializing Free Open-Source AI Diagnosis System...")
        
        # Core medical analysis models
        self.biobert = BioBERTDiagnosis()                    # Primary medical text analysis
        self.clinical_bert = ClinicalBERTAnalysis()          # Severity assessment
        self.sentence_transformer = SentenceTransformerDiagnosis()  # Case similarity
        
        # Supporting analysis components
        self.question_generator = DynamicQuestionGenerator() # Follow-up question generation
        self.image_expert = ImageAnalysisExpert()           # Medical image analysis
        self.general_expert = GeneralAIExpert()             # General medical Q&A
        
        # Secondary analysis system (A.I.2)
        self.secondary_ai = SecondaryAIAnalyzer()           # Follow-up analysis
        
        # Recommendation and feedback systems
        self.feedback_generator = ActionableFeedbackGenerator()     # Action recommendations
        self.medication_expert = MedicationRecommendationExpert()  # Medication advice
        
        
        print("✅ System ready!")

    def comprehensive_diagnosis(self, symptoms: str, context: str = "") -> Dict:
        """
        A.I.(1) - Primary diagnosis using ensemble of medical models
        
        This method implements the first stage of diagnosis by combining results
        from multiple specialized medical AI models to provide a comprehensive
        initial assessment.
        
        Args:
            symptoms (str): Patient-reported symptoms description
            context (str): Additional context like patient history, test results
            
        Returns:
            Dict: Diagnosis result containing:
                - diagnosis: Primary diagnosis suggestion
                - confidence: Confidence score (0.0-1.0)
                - severity: Assessed severity level
                - similar_case: Similar medical cases from knowledge base
                - reasoning: Explanation of the analysis process
                - ai_stage: Identifies this as "primary" analysis
        """
        self.log_step("🔍 A.I.(1) Starting comprehensive analysis", f"Symptoms: {symptoms}, Context: {context}")
        
        # Emergency detection - highest priority check
        # This must be performed first to identify life-threatening conditions
        emergency_alert = self.feedback_generator.generate_emergency_recommendations(symptoms)
        
        # Parallel model analysis for comprehensive assessment
        biobert_result = self.biobert.diagnose(symptoms, context)              # Medical text analysis
        severity_result = self.clinical_bert.analyze_severity(symptoms)        # Severity assessment
        similarity_result = self.sentence_transformer.find_similar_cases(symptoms)  # Case similarity
        
        # Weighted ensemble confidence calculation
        # BioBERT gets highest weight (50%) as primary medical model
        # Clinical BERT gets 20% for severity assessment
        # Sentence Transformer gets 30% for case similarity
        base_confidence = (biobert_result['confidence'] * 0.5 + 
                          severity_result['confidence'] * 0.2 + 
                          similarity_result['confidence'] * 0.3)
        
        # Context bonus calculation
        # Additional context information increases confidence up to 10%
        context_bonus = min(len(context.split()) * 0.01, 0.1) if context else 0
        
        # Final confidence with severity adjustments
        final_confidence = base_confidence + context_bonus
        
        # Severity-based confidence adjustments
        if severity_result['severity'] == 'severe': 
            final_confidence = min(final_confidence + 0.1, 0.95)  # Cap at 95%
        elif severity_result['severity'] == 'mild': 
            final_confidence = max(final_confidence - 0.1, 0.3)   # Floor at 30%
        
        # Construct comprehensive result dictionary
        result = {
            "diagnosis": biobert_result['diagnosis'],
            "confidence": final_confidence,
            "severity": severity_result['severity'],
            "similar_case": similarity_result['similar_case'],
            "reasoning": f"A.I.(1) Multi-model analysis: BioBERT ({biobert_result['confidence']:.2f}), ClinicalBERT ({severity_result['confidence']:.2f}), SentenceTransformer ({similarity_result['confidence']:.2f}), Context bonus: {context_bonus:.2f}",
            "ai_stage": "primary"
        }
        
        # Override with emergency alert if critical condition detected
        if emergency_alert:
            result["emergency_alert"] = emergency_alert
            result["severity"] = "emergency"
        
        return result

    def enhanced_diagnosis_with_context(self, symptoms: str, all_context: str, previous_diagnosis: Dict) -> Dict:
        """
        Re-run A.I.(1) with enhanced context after A.I.(2) gathers more info
        
        This method re-executes the primary diagnosis with additional context
        gathered from follow-up questions and secondary analysis. It tracks
        confidence improvements to demonstrate the value of additional information.
        
        Args:
            symptoms (str): Original patient symptoms
            all_context (str): Accumulated context from all interactions
            previous_diagnosis (Dict): Previous diagnosis result for comparison
            
        Returns:
            Dict: Updated diagnosis with improvement metrics
        """
        self.log_step("🔄 A.I.(1) Re-analysis with enhanced context", f"Previous confidence: {previous_diagnosis['confidence']:.2f}")
        
        # Re-run primary diagnosis with all accumulated context
        # This leverages the enhanced information to improve accuracy
        new_result = self.comprehensive_diagnosis(symptoms, all_context)
        
        # Calculate and track confidence improvement
        # This demonstrates the value of additional context gathering
        confidence_improvement = new_result['confidence'] - previous_diagnosis['confidence']
        new_result['improvement'] = confidence_improvement
        new_result['reasoning'] += f" | Confidence improved by {confidence_improvement:.2f}"
        
        return new_result

    def secondary_analysis(self, symptoms: str, initial_diagnosis: Dict) -> Dict:
        """
        A.I.(2) - Secondary analysis for follow-up questions and test suggestions
        
        This method implements the second stage of analysis, activated when the
        primary diagnosis confidence is below the threshold. It generates targeted
        follow-up questions and suggests additional tests to gather more information.
        
        Args:
            symptoms (str): Original patient symptoms
            initial_diagnosis (Dict): Result from primary A.I.(1) analysis
            
        Returns:
            Dict: Secondary analysis results containing:
                - follow_up_questions: Targeted questions to ask the patient
                - suggested_tests: Additional tests or observations needed
                - ai_stage: Identifies this as "secondary" analysis
                - reasoning: Explanation of the secondary analysis process
        """
        self.log_step("🔍 A.I.(2) Secondary analysis", f"Initial confidence: {initial_diagnosis['confidence']:.2f}")
        
        # Generate targeted follow-up questions based on initial diagnosis
        # These questions are designed to clarify ambiguous symptoms
        follow_up_questions = self.secondary_ai.generate_follow_up_questions(
            symptoms, initial_diagnosis['diagnosis'], initial_diagnosis['confidence']
        )
        
        # Suggest additional tests or observations
        # These help gather objective data to improve diagnosis accuracy
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
        """
        Integrate image analysis into the diagnostic flow
        
        This method analyzes medical images and contextualizes them with current
        symptoms and patient history. It uses the image analysis expert to extract
        visual information and the secondary AI to relate findings to symptoms.
        
        Args:
            image: PIL Image object to analyze
            current_context (str): Current symptoms and patient context
            
        Returns:
            str: Formatted analysis combining image description and medical relevance
        """
        # Primary image analysis using specialized image models
        image_description = self.image_expert.analyze_image(image)
        
        # Use A.I.(2) to relate image findings to current symptoms
        # This provides medical context to the visual analysis
        if self.secondary_ai.analyzer:
            try:
                # Create contextualized prompt for medical relevance analysis
                prompt = f"How does this image relate to these symptoms: {current_context}. Image shows: {image_description}"
                result = self.secondary_ai.analyzer(prompt, max_new_tokens=100, temperature=0.3)
                contextualized_analysis = result[0]["generated_text"]
                return f"Image analysis: {image_description}. Medical relevance: {contextualized_analysis}"
            except (KeyError, IndexError, AttributeError) as e:
                # Handle data access errors gracefully
                print(f"Error accessing AI analysis result: {e}")
                return f"Image analysis: {image_description}"
            except Exception as e:
                # Handle any other unexpected errors
                print(f"Unexpected error during image contextualization: {e}")
                return f"Image analysis: {image_description}"
        
        # Fallback to basic image description if contextualization fails
        return f"Image analysis: {image_description}"

    def get_enhanced_medication_recommendations(self, diagnosis: str, symptoms: str, severity: str, current_medications: list = None) -> Dict:
        """
        Get AI-powered medication recommendations with safety checks
        
        This method provides personalized medication recommendations based on the
        diagnosis, symptoms, and severity. It includes safety checks for drug
        interactions and contraindications.
        
        Args:
            diagnosis (str): Primary diagnosis from the system
            symptoms (str): Patient-reported symptoms
            severity (str): Assessed severity level (mild, moderate, severe)
            current_medications (list, optional): List of current medications
            
        Returns:
            Dict: Comprehensive medication recommendations including:
                - AI-generated personalized recommendations
                - Evidence-based database matches
                - Safety warnings and contraindications
                - Drug interaction alerts
        """
        self.log_step("💊 Getting enhanced medication recommendations", f"Diagnosis: {diagnosis}")
        
        # Get personalized recommendations from specialized medication AI expert
        # This combines AI analysis with evidence-based medical databases
        recommendations = self.medication_expert.generate_personalized_recommendations(
            diagnosis, symptoms, severity
        )
        
        # Add drug interaction warnings if current medications are provided
        # This is critical for patient safety and preventing adverse reactions
        if current_medications:
            interaction_warning = self.medication_expert.get_drug_interactions_warning(current_medications)
            recommendations["drug_interactions"] = interaction_warning
        
        return recommendations


    def log_step(self, step: str, details: str = ""):
        """
        Log diagnostic steps with timestamps for debugging and audit trail
        
        This method provides consistent logging throughout the diagnosis process
        to track the system's decision-making steps and timing.
        
        Args:
            step (str): Description of the current step
            details (str, optional): Additional details about the step
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {step}"
        if details: 
            log_entry += f": {details}"
        print(log_entry)
