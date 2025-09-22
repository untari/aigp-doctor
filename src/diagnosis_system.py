
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
import logging
import traceback
import time
import gc
import psutil
import re

# Configure logging for robustness
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_gp_doctor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
from src.models.voice_system import VoiceSystem

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
        
        # Voice system for speech-to-text and text-to-speech capabilities
        self.voice_system = VoiceSystem()                          # Complete voice interface
        
        # Rate limiting and performance tracking
        self.last_request_time = 0
        self.request_count = 0
        self.min_request_interval = 1.0  # Minimum 1 second between requests

        print("✅ System ready with voice capabilities!")

    def _check_rate_limit(self) -> bool:
        """Check if request rate limit is exceeded"""
        current_time = time.time()
        if current_time - self.last_request_time < self.min_request_interval:
            return False
        self.last_request_time = current_time
        self.request_count += 1
        return True

    def _monitor_memory_usage(self):
        """Monitor and log memory usage"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            if memory_mb > 2048:  # Alert if using more than 2GB
                logger.warning(f"High memory usage: {memory_mb:.1f}MB")
                gc.collect()  # Force garbage collection
        except Exception as e:
            logger.debug(f"Memory monitoring failed: {str(e)}")

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
        try:
            # Rate limiting check
            if not self._check_rate_limit():
                logger.warning("Rate limit exceeded")
                return self._create_fallback_diagnosis("Please wait before making another request")

            # Memory monitoring
            self._monitor_memory_usage()

            logger.info(f"Starting comprehensive diagnosis for symptoms: {symptoms[:50]}...")
            start_time = time.time()

            self.log_step("🔍 A.I.(1) Starting comprehensive analysis", f"Symptoms: {symptoms}, Context: {context}")

            # Input validation and sanitization
            symptoms = self._validate_and_sanitize_input(symptoms)
            context = self._validate_and_sanitize_input(context)

            # Check input quality
            is_valid, validation_message = self._check_input_quality(symptoms)
            if not is_valid:
                logger.warning(f"Invalid input provided: {validation_message}")
                return self._create_fallback_diagnosis(validation_message)

            # Emergency detection - highest priority check with error handling
            emergency_alert = None
            try:
                emergency_alert = self.feedback_generator.generate_emergency_recommendations(symptoms)
                logger.debug("Emergency detection completed successfully")
            except Exception as e:
                logger.error(f"Emergency detection failed: {str(e)}")
                emergency_alert = {"alert": False, "reason": "", "action": ""}

            # Parallel model analysis with individual error handling
            results = {}

            # BioBERT analysis with fallback
            try:
                results['biobert'] = self.biobert.diagnose(symptoms, context)
                logger.debug("BioBERT analysis completed")
            except Exception as e:
                logger.error(f"BioBERT analysis failed: {str(e)}")
                results['biobert'] = {'diagnosis': 'Analysis Error', 'confidence': 0.3, 'reasoning': 'BioBERT unavailable'}

            # Clinical BERT analysis with fallback
            try:
                results['clinical_bert'] = self.clinical_bert.analyze_severity(symptoms)
                logger.debug("Clinical BERT analysis completed")
            except Exception as e:
                logger.error(f"Clinical BERT analysis failed: {str(e)}")
                results['clinical_bert'] = {'severity': 'Unknown', 'confidence': 0.3, 'reasoning': 'Clinical BERT unavailable'}

            # Sentence Transformer analysis with fallback
            try:
                results['sentence_transformer'] = self.sentence_transformer.find_similar_cases(symptoms)
                logger.debug("Sentence Transformer analysis completed")
            except Exception as e:
                logger.error(f"Sentence Transformer analysis failed: {str(e)}")
                results['sentence_transformer'] = {'similar_case': 'No similar cases found', 'confidence': 0.3, 'reasoning': 'Sentence Transformer unavailable'}

            # Extract results with safe defaults
            biobert_result = results.get('biobert', {'diagnosis': 'Requires Further Evaluation', 'confidence': 0.3, 'reasoning': 'Analysis unavailable'})
            severity_result = results.get('clinical_bert', {'severity': 'Unknown', 'confidence': 0.3, 'reasoning': 'Severity analysis unavailable'})
            similarity_result = results.get('sentence_transformer', {'similar_case': 'No similar cases', 'confidence': 0.3, 'reasoning': 'Case similarity unavailable'})

            # Weighted ensemble confidence calculation with safe access
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
            # Handle both string and dict emergency alerts
            if emergency_alert:
                if isinstance(emergency_alert, str) and emergency_alert.strip():
                    # String format emergency alert
                    result["emergency_alert"] = {
                        "alert": True,
                        "message": emergency_alert,
                        "reason": "Emergency keywords detected"
                    }
                    result["severity"] = "emergency"
                    logger.warning(f"Emergency condition detected: {emergency_alert[:100]}")
                elif isinstance(emergency_alert, dict) and emergency_alert.get("alert", False):
                    # Dict format emergency alert
                    result["emergency_alert"] = emergency_alert
                    result["severity"] = "emergency"
                    logger.warning(f"Emergency condition detected: {emergency_alert.get('reason', 'Unknown')}")

            # Log performance metrics
            end_time = time.time()
            logger.info(f"Comprehensive diagnosis completed in {end_time - start_time:.2f}s with confidence {final_confidence:.2f}")

            return result

        except Exception as e:
            logger.error(f"Critical error in comprehensive diagnosis: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return self._create_fallback_diagnosis(f"System temporarily unavailable: {str(e)[:50]}")

    def _create_fallback_diagnosis(self, error_message: str) -> Dict:
        """Create a safe fallback diagnosis when main system fails"""
        logger.warning(f"Using fallback diagnosis: {error_message}")
        return {
            "diagnosis": "System Analysis Unavailable",
            "confidence": 0.2,
            "severity": "unknown",
            "similar_case": "Please consult healthcare provider",
            "reasoning": f"Fallback response: {error_message}",
            "ai_stage": "fallback",
            "emergency_alert": {"alert": False, "reason": "System error", "action": "Seek medical attention if symptoms are severe"}
        }

    def _validate_and_sanitize_input(self, text: str, max_length: int = 1000) -> str:
        """Validate and sanitize user input for security and reliability"""
        if not text:
            return ""

        # Remove dangerous characters and trim
        # Remove potential script injection attempts
        text = re.sub(r'<[^>]*>', '', text)  # Remove HTML tags
        text = re.sub(r'[^\w\s\-.,?!:;()\'/"]', '', text)  # Keep only safe characters

        # Limit length to prevent memory issues
        text = text[:max_length]

        # Normalize whitespace
        text = ' '.join(text.split())

        logger.debug(f"Input sanitized: original length {len(text)}, final length {len(text)}")
        return text.strip()

    def _check_input_quality(self, symptoms: str) -> tuple[bool, str]:
        """Check if input symptoms are of sufficient quality for analysis"""
        if not symptoms or len(symptoms.strip()) < 3:
            return False, "Please provide more detailed symptom description"

        # Check for common non-medical inputs
        non_medical_patterns = [
            r'\b(hello|hi|hey|test|testing)\b',
            r'\b(how are you|what is|who are)\b',
            r'\b(weather|news|sports)\b'
        ]

        for pattern in non_medical_patterns:
            if re.search(pattern, symptoms.lower()):
                return False, "Please describe medical symptoms rather than general questions"

        # Check if it contains at least one potential symptom word
        symptom_words = [
            'pain', 'ache', 'hurt', 'fever', 'headache', 'nausea', 'tired', 'fatigue',
            'cough', 'shortness', 'breathing', 'chest', 'stomach', 'back', 'dizzy',
            'swelling', 'rash', 'itching', 'burning', 'numbness', 'tingling'
        ]

        if not any(word in symptoms.lower() for word in symptom_words):
            return False, "Please describe specific medical symptoms you're experiencing"

        return True, "Input appears to be valid medical symptoms"

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
                result = self.secondary_ai.analyzer(prompt, max_new_tokens=100)
                full_response = result[0]["generated_text"]

                # Extract only the new generated text after the prompt
                if prompt in full_response:
                    contextualized_analysis = full_response.replace(prompt, "").strip()
                else:
                    contextualized_analysis = full_response

                return f"Image analysis: {image_description}\n\nMedical relevance: {contextualized_analysis}"
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

    def voice_diagnosis(self, audio_file_path: str) -> Dict:
        """
        Complete voice diagnosis pipeline: STT → AI Analysis → TTS Response
        
        This method implements the complete voice consultation workflow:
        1. Speech-to-Text: Transcribe patient's voice input using Whisper
        2. AI Analysis: Process symptoms through the diagnosis system
        3. Text-to-Speech: Generate and speak the diagnosis response
        
        Args:
            audio_file_path (str): Path to recorded audio file
            
        Returns:
            Dict: Complete voice diagnosis result containing:
                - transcription: What the patient said
                - transcription_confidence: Confidence in transcription accuracy
                - diagnosis: Medical diagnosis result from AI analysis
                - voice_response: Text that was spoken back to patient
                - error: Error message if something went wrong
        """
        self.log_step("🎤 Starting voice diagnosis pipeline", f"Audio file: {audio_file_path}")
        
        # Step 1: Speech-to-Text - Transcribe patient's voice
        self.log_step("🔄 Transcribing patient voice input")
        symptoms_text, transcription_confidence = self.voice_system.transcribe_audio(audio_file_path)
        
        # Check transcription quality
        if transcription_confidence < 0.3:
            error_msg = "Could not understand audio clearly. Please try speaking again."
            self.log_step("❌ Voice transcription failed", f"Confidence: {transcription_confidence:.2%}")
            return {
                "error": error_msg,
                "transcription": symptoms_text,
                "transcription_confidence": transcription_confidence,
                "diagnosis": None,
                "voice_response": None
            }
        
        self.log_step("✅ Voice transcription successful", f"Patient said: '{symptoms_text}' (confidence: {transcription_confidence:.1%})")
        
        # Step 2: AI Analysis - Process the transcribed symptoms
        self.log_step("🔍 Processing symptoms through AI diagnosis system")
        diagnosis_result = self.comprehensive_diagnosis(symptoms_text)
        
        # Step 3: Prepare voice response for natural speech
        voice_response = self._format_diagnosis_for_speech(diagnosis_result)
        
        # Step 4: Text-to-Speech - Speak the response (non-blocking)
        self.log_step("🔊 Speaking diagnosis response to patient")
        self.voice_system.speak(voice_response, wait=False)
        
        # Compile complete result
        complete_result = {
            "transcription": symptoms_text,
            "transcription_confidence": transcription_confidence,
            "diagnosis": diagnosis_result,
            "voice_response": voice_response,
            "error": None
        }
        
        self.log_step("✅ Voice diagnosis pipeline complete", 
                     f"Diagnosis: {diagnosis_result['diagnosis']} ({diagnosis_result['confidence']:.1%} confidence)")
        
        return complete_result

    def _format_diagnosis_for_speech(self, diagnosis: Dict) -> str:
        """
        Format diagnosis result for natural speech output
        
        This method converts the technical diagnosis output into natural,
        conversational language suitable for speaking to patients.
        
        Args:
            diagnosis (Dict): Diagnosis result from the AI system
            
        Returns:
            str: Formatted text optimized for speech synthesis
        """
        confidence_percent = int(diagnosis['confidence'] * 100)
        
        # Start with diagnosis statement
        response = f"Based on your symptoms, my analysis suggests {diagnosis['diagnosis']} "
        response += f"with {confidence_percent} percent confidence. "
        
        # Add severity-appropriate messaging
        severity = diagnosis.get('severity', 'unknown')
        if severity == 'emergency' or diagnosis.get('emergency_alert'):
            response = "Emergency condition detected! " + response
            response += "You should seek immediate medical attention. "
        elif severity == 'severe':
            response += "This appears to be a serious condition requiring prompt medical evaluation. "
        elif severity == 'moderate':
            response += "This condition should be evaluated by a healthcare provider soon. "
        else:
            response += "This appears to be a mild condition. "
        
        # Add specific emergency alerts if present
        if diagnosis.get('emergency_alert'):
            emergency_msg = diagnosis['emergency_alert'].get('message', '')
            if emergency_msg:
                response += f"Important: {emergency_msg} "
        
        # Add confidence-based recommendations
        if diagnosis['confidence'] < 0.6:
            response += "However, I recommend getting additional symptoms checked for a more accurate diagnosis. "
        elif diagnosis['confidence'] > 0.8:
            response += "This assessment has high confidence based on your symptoms. "
        
        # Add general medical disclaimer
        response += "Please remember, this is an AI assessment for educational purposes only "
        response += "and should not replace professional medical advice from a qualified healthcare provider."
        
        return response

    def speak_diagnosis(self, diagnosis: Dict, wait: bool = False):
        """
        Speak a diagnosis result using text-to-speech
        
        Args:
            diagnosis (Dict): Diagnosis result to speak
            wait (bool): Whether to wait for speech to complete
        """
        voice_response = self._format_diagnosis_for_speech(diagnosis)
        self.voice_system.speak(voice_response, wait=wait)
        
    def record_and_diagnose(self, duration: int = 15) -> Dict:
        """
        Record audio from microphone and perform complete voice diagnosis
        
        This is a convenience method that combines audio recording with
        the complete voice diagnosis pipeline.
        
        Args:
            duration (int): Maximum recording duration in seconds
            
        Returns:
            Dict: Complete voice diagnosis result
        """
        self.log_step("🎤 Starting voice recording for diagnosis", f"Max duration: {duration}s")
        
        # Record audio
        audio_file = self.voice_system.record_audio(duration=duration)
        
        if not audio_file:
            return {
                "error": "Failed to record audio. Please check your microphone.",
                "transcription": None,
                "transcription_confidence": 0.0,
                "diagnosis": None,
                "voice_response": None
            }
        
        # Process the recorded audio
        result = self.voice_diagnosis(audio_file)
        
        # Clean up temporary audio file
        try:
            import os
            if os.path.exists(audio_file):
                os.remove(audio_file)
        except:
            pass  # Ignore cleanup errors
        
        return result
    
    def stop_speaking(self):
        """
        Stop any ongoing speech output
        """
        self.voice_system.stop_speaking()
        
    def get_voice_system_status(self) -> Dict:
        """
        Get the status of the voice system components
        
        Returns:
            Dict: Status information for voice system
        """
        return self.voice_system.is_system_ready()
