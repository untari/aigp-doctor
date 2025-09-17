

"""
Gradio Web Interface for AI GP Doctor
=====================================

This module creates the web-based user interface for the AI GP Doctor system
using Gradio. It provides a chat-style interface for medical consultation
with support for text input, image uploads, and interactive buttons.

Key Features:
- Professional medical-themed UI design
- Chat-style conversation flow
- Image upload and analysis capabilities
- Dynamic button interactions
- State management for multi-turn conversations
- Comprehensive medical disclaimers

Security Considerations:
- File upload handling needs validation
- Input sanitization for user messages
- No authentication mechanism (for demo purposes)
"""

import gradio as gr
from PIL import Image
from typing import Dict

# Import the core diagnosis system
from src.diagnosis_system import DiagnosisSystem
from image_handler import MedicalImageHandler

# Initialize the global diagnosis system instance
# This is shared across all user sessions
free_diagnosis_system = DiagnosisSystem()

# Initialize the secure medical image handler
medical_image_handler = MedicalImageHandler()

def create_chatbot_interface():
    """
    Creates and returns the professional Gradio Chatbot interface
    
    This function constructs the complete web interface including:
    - CSS styling for medical theme
    - Chat interface components
    - Input controls and buttons
    - State management system
    - Event handlers for interactions
    
    Returns:
        gr.Blocks: Complete Gradio interface ready for launch
    """
    # Custom CSS for modern medical chat interface
    # Defines the visual styling and responsive design
    custom_css = """
    body {
        background: #f5f7fa !important;
    }
    
    .main-container {
        max-width: 100%;
        margin: 0;
        background: transparent;
    }
    
    .header-section {
        text-align: center;
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        backdrop-filter: blur(10px);
        margin-bottom: 0.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .header-title {
        color: white;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.85rem;
        font-weight: 300;
        margin-bottom: 0.2rem;
    }
    
    .chat-container {
        background: white;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        margin: 0.5rem;
        overflow: hidden;
    }
    
    .gradio-container {
        background: transparent !important;
    }
    
    /* Chat bubbles styling */
    .message.user {
        background: #e3f2fd;
        color: #1565c0;
        border-radius: 18px 18px 6px 18px;
        margin-left: 20%;
        border: 1px solid #bbdefb;
    }
    
    .message.bot {
        background: #f8f9fa;
        color: #2c3e50;
        border-radius: 18px 18px 18px 6px;
        margin-right: 20%;
        border: 1px solid #e9ecef;
    }
    
    /* Input styling */
    .input-row {
        background: white;
        border-top: 1px solid #e9ecef;
        padding: 1rem;
    }
    
    .message-input {
        border: 2px solid #e9ecef;
        border-radius: 25px;
        padding: 0.75rem 1.25rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .message-input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Button styling */
    .action-button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .action-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .upload-button {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .upload-button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(17, 153, 142, 0.3);
    }
    
    /* Medical status indicators */
    .confidence-high { color: #27ae60; font-weight: bold; }
    .confidence-medium { color: #f39c12; font-weight: bold; }
    .confidence-low { color: #e74c3c; font-weight: bold; }
    
    .ai-stage-primary { 
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    .ai-stage-secondary {
        background: linear-gradient(90deg, #11998e, #38ef7d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    """

    with gr.Blocks(css=custom_css, theme=gr.themes.Base(), title="AI GP Doctor") as demo:
        case_state = gr.State({
            "stage": "AWAITING_SYMPTOMS",
            "symptoms": "",
            "context_history": [],
            "last_diagnosis": None
        })

        # Compact Professional Header
        gr.HTML("""
            <div class="header-section">
                <h1 class="header-title">🩺 AI GP Doctor</h1>
                <p class="header-subtitle">AI-Powered Medical Diagnosis Assistant</p>
                <p style="color: rgba(255,255,255,0.8); font-size: 0.8rem; margin: 0;">
                    BioBERT • Clinical BERT • Medical Image Analysis
                </p>
            </div>
        """)

        # Main chat interface
        with gr.Column(elem_classes="chat-container"):
            chatbot = gr.Chatbot(
                label="",
                show_label=False,
                bubble_full_width=False, 
                height=600,
                elem_classes="chat-messages",
                avatar_images=(None, None),
                show_copy_button=True,
                type='messages'
            )

            # Input section with voice support
            with gr.Row(elem_classes="input-row"):
                with gr.Column(scale=3):
                    txt_input = gr.Textbox(
                        show_label=False, 
                        placeholder="💬 Describe your symptoms, ask questions, or provide additional information...",
                        container=False,
                        elem_classes="message-input"
                    )
                with gr.Column(scale=1, min_width=100):
                    with gr.Row():
                        upload_btn = gr.UploadButton(
                            "📁 Medical Image",
                            file_types=["image"],
                            file_count="single",
                            elem_classes="upload-button",
                            size="sm"
                        )
                        voice_btn = gr.Button(
                            "🎤 Voice",
                            elem_classes="upload-button",
                            size="sm"
                        )

            # Image upload feedback area (initially hidden)
            with gr.Row(visible=False, elem_classes="upload-feedback") as image_feedback_area:
                image_status = gr.Markdown("", elem_classes="upload-status")

            # Image preview area (initially hidden)
            with gr.Row(visible=False, elem_classes="image-preview") as image_preview_area:
                with gr.Column(scale=1):
                    image_preview = gr.Image(
                        label="Uploaded Medical Image",
                        show_label=True,
                        interactive=False,
                        height=200,
                        elem_classes="medical-image-preview"
                    )
                with gr.Column(scale=2):
                    image_info = gr.Markdown("", elem_classes="image-info")

            # Hidden audio input for voice functionality (simplified like image upload)
            audio_input = gr.Audio(
                type="filepath",
                visible=False
            )

            # Action buttons with modern styling
            with gr.Row(elem_classes="action-buttons") as action_buttons:
                ask_questions_btn = gr.Button(
                    "🤔 Ask me clarifying questions", 
                    visible=False,
                    elem_classes="action-button",
                    size="sm"
                )
                provide_info_btn = gr.Button(
                    "📋 I'll provide more info/test results", 
                    visible=False,
                    elem_classes="action-button",
                    size="sm"
                )
                explain_btn = gr.Button(
                    "💡 Explain this diagnosis", 
                    visible=False,
                    elem_classes="action-button",
                    size="sm"
                )
                medication_btn = gr.Button(
                    "💊 Get medication recommendations", 
                    visible=False,
                    elem_classes="action-button",
                    size="sm"
                )

        def unified_handler(message, image_upload, history, state):
            """
            Unified event handler for user interactions
            
            This function processes text messages and image uploads,
            managing the conversation flow and state transitions.
            
            Args:
                message (str): User text input
                image_upload: Uploaded image file
                history (list): Chat conversation history
                state (dict): Current conversation state
                
            Returns:
                tuple: Updated history, state, and button visibility updates
            """
            history = history or []
            bot_message = ""

            # Initialize button visibility updates
            # These control which action buttons are shown to the user
            ask_q_update = gr.update(visible=False)
            provide_info_update = gr.update(visible=False)
            explain_update = gr.update(visible=False)
            medication_update = gr.update(visible=False)

            # Handle Image Upload with enhanced security and validation
            if image_upload is not None:
                # Use secure image handler for validation and preprocessing
                validation_result = medical_image_handler.validate_and_process_image(image_upload.name)

                if not validation_result['is_valid']:
                    # Image validation failed - show error message
                    error_msg = f"❌ **Image Upload Failed**\n\n{validation_result['error_message']}"
                    if validation_result.get('warnings'):
                        error_msg += f"\n\n**Warnings:**\n" + "\n".join(f"• {w}" for w in validation_result['warnings'])
                    error_msg += "\n\n💡 **Tips for better images:**\n• Use JPEG, PNG, or TIFF format\n• Keep file size under 15MB\n• Ensure image is clear and well-lit\n• Minimum size: 100x100 pixels"

                    history.append({"role": "assistant", "content": error_msg})
                    return history, state, ask_q_update, provide_info_update, explain_update, medication_update

                # Image is valid - proceed with analysis
                history.append({"role": "user", "content": {"path": image_upload.name}})
                processed_image = validation_result['processed_image']
                current_context = " ".join(state.get("context_history", []))
                description = free_diagnosis_system.analyze_image_in_context(processed_image, current_context)
                state["context_history"].append(description)

                # Enhanced success message with image info
                success_msg = f"✅ **Medical Image Analyzed Successfully**\n\n{description}.\n\nWhat are the primary symptoms?"
                if validation_result.get('warnings'):
                    success_msg += f"\n\n**Processing Notes:**\n" + "\n".join(f"• {w}" for w in validation_result['warnings'])

                state["stage"] = "AWAITING_SYMPTOMS"
                history.append({"role": "assistant", "content": success_msg})
                return history, state, ask_q_update, provide_info_update, explain_update, medication_update


            # Add user message to conversation history
            if message:
                history.append({"role": "user", "content": message})

            # Enhanced State Machine Logic with A.I.(1) and A.I.(2) routing
            # This implements the conversation flow state machine
            stage = state["stage"]

            if stage == "AWAITING_SYMPTOMS":
                # Differentiate between medical questions and symptom descriptions
                # This allows the system to handle both educational queries and diagnosis
                question_indicators = [
                    "what is", "what are", "how does", "why does", "explain", "difference between",
                    "tell me about", "can you explain", "what's the", "how to", "when should"
                ]
                
                is_question = any(indicator in message.lower() for indicator in question_indicators)
                
                if is_question:
                    # Handle as a general medical question using the general AI expert
                    try:
                        bot_message = free_diagnosis_system.general_expert.answer(message)
                        bot_message += "\n\n---\n\n💡 **Need a diagnosis?** Describe your symptoms and I'll help analyze them."
                    except Exception as e:
                        bot_message = "I'm having trouble processing your question right now. Please describe your symptoms for a medical diagnosis."
                    # Stay in AWAITING_SYMPTOMS state for potential follow-up
                else:
                    # Handle as symptoms - proceed with formal diagnosis workflow
                    state["symptoms"] = message
                    state["context_history"].append(f"Symptoms: {message}")
                    bot_message = "Thank you. Any pre-existing conditions, allergies, or relevant patient history? (e.g., 'diabetic, allergic to penicillin') If not, just say 'none')."
                    state["stage"] = "AWAITING_HISTORY"

            elif stage == "AWAITING_HISTORY" or stage == "AWAITING_HISTORY_VOICE":
                try:
                    state["context_history"].append(f"Patient History: {message}")
                    full_context = " ".join(state["context_history"])
                    # A.I.(1) - Primary diagnosis
                    diagnosis_result = free_diagnosis_system.comprehensive_diagnosis(state["symptoms"], full_context)
                    state["last_diagnosis"] = diagnosis_result

                    if diagnosis_result['confidence'] > 0.75:
                        bot_message = format_diagnosis_output(diagnosis_result, is_final=True)
                        state["stage"] = "DIAGNOSIS_COMPLETE"
                    else:
                        # A.I.(2) - Secondary analysis for low confidence
                        secondary_result = free_diagnosis_system.secondary_analysis(state["symptoms"], diagnosis_result)
                        state["secondary_analysis"] = secondary_result

                        bot_message = format_diagnosis_output(diagnosis_result, is_final=False)
                        bot_message += f"\n\n🤖 **A.I.(2) Suggests:**\n"
                        bot_message += f"**Questions:** {', '.join(secondary_result['follow_up_questions'])}\n"
                        bot_message += f"**Tests:** {', '.join(secondary_result['suggested_tests'])}\n"
                        bot_message += "\nWhat would you like to do?"
                        state["stage"] = "AWAITING_CLARIFICATION_CHOICE"
                except Exception as e:
                    bot_message = "I'm having trouble analyzing your symptoms right now. Please try describing them again or restart the conversation."
                    state["stage"] = "AWAITING_SYMPTOMS"

            elif stage == "AWAITING_CLARIFICATION_CHOICE":
                bot_message = "Please choose an option below or provide additional information directly."

            elif stage == "AWAITING_INFO":
                state["context_history"].append(f"Additional Info: {message}")
                full_context = " ".join(state["context_history"])
                
                # A.I.(1) re-analysis with enhanced context
                diagnosis_result = free_diagnosis_system.enhanced_diagnosis_with_context(
                    state["symptoms"], full_context, state["last_diagnosis"]
                )
                state["last_diagnosis"] = diagnosis_result
                
                if diagnosis_result['confidence'] > 0.75:
                    bot_message = format_diagnosis_output(diagnosis_result, is_final=True)
                    if diagnosis_result.get('improvement', 0) > 0:
                        bot_message += f"\n✨ **Confidence improved by {diagnosis_result['improvement']:.1%} with additional context!**"
                    state["stage"] = "DIAGNOSIS_COMPLETE"
                else:
                    # Run A.I.(2) again for more suggestions
                    secondary_result = free_diagnosis_system.secondary_analysis(state["symptoms"], diagnosis_result)
                    state["secondary_analysis"] = secondary_result
                    
                    bot_message = format_diagnosis_output(diagnosis_result, is_final=False)
                    bot_message += f"\n\n🤖 **A.I.(2) Additional Suggestions:**\n"
                    bot_message += f"**Questions:** {', '.join(secondary_result['follow_up_questions'])}\n"
                    bot_message += f"**Tests:** {', '.join(secondary_result['suggested_tests'])}\n"
                    bot_message += "You can provide more information or I can ask more specific questions."
                    state["stage"] = "AWAITING_CLARIFICATION_CHOICE"

            elif stage == "DIAGNOSIS_COMPLETE" and message:
                # Check if this is a new symptom description (start new diagnosis) or just a question
                symptom_indicators = [
                    "i have", "i'm experiencing", "i feel", "my", "pain", "hurt", "ache", "fever",
                    "headache", "nausea", "dizzy", "tired", "cough", "sore", "swollen", "rash",
                    "bleeding", "vomiting", "diarrhea", "constipation", "shortness", "breathing",
                    "chest pain", "back pain", "stomach", "throat", "runny nose", "congestion"
                ]

                is_new_symptoms = any(indicator in message.lower() for indicator in symptom_indicators)

                if is_new_symptoms:
                    # Reset state for new diagnosis
                    state = {
                        "stage": "AWAITING_SYMPTOMS",
                        "context_history": []
                    }
                    # Handle as new symptoms - proceed with formal diagnosis workflow
                    state["symptoms"] = message
                    state["context_history"].append(f"Symptoms: {message}")
                    bot_message = "Thank you. Any pre-existing conditions, allergies, or relevant patient history? (e.g., 'diabetic, allergic to penicillin') If not, just say 'none')."
                    state["stage"] = "AWAITING_HISTORY"
                else:
                    # Handle as general medical question
                    try:
                        bot_message = free_diagnosis_system.general_expert.answer(message)
                        bot_message += "\n\n---\n\n💡 **Need a new diagnosis?** Describe your symptoms and I'll help analyze them."
                    except Exception as e:
                        bot_message = "I'm having trouble answering that question. Please describe your symptoms for a new medical diagnosis."

            if bot_message:
                history.append({"role": "assistant", "content": bot_message})

            # Update button visibility based on state
            if state["stage"] == "AWAITING_CLARIFICATION_CHOICE":
                ask_q_update = gr.update(visible=True)
                provide_info_update = gr.update(visible=True)
            if state["stage"] == "DIAGNOSIS_COMPLETE":
                explain_update = gr.update(visible=True)
                medication_update = gr.update(visible=True)
            
            return history, state, ask_q_update, provide_info_update, explain_update, medication_update

        def on_ask_questions_click(state, history):
            """
            Handler for "Ask me clarifying questions" button
            
            This function generates and presents follow-up questions to gather
            additional information when the initial diagnosis confidence is low.
            
            Args:
                state (dict): Current conversation state
                history (list): Chat conversation history
                
            Returns:
                tuple: Updated history, state, and button visibility
            """
            history = history or []
            
            # Defensive check to ensure diagnosis exists before asking questions
            if not state.get("last_diagnosis"):
                bot_message = "I can't ask questions yet. Please describe the symptoms first."
                history.append({"role": "assistant", "content": bot_message})
                return history, state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

            # Use A.I.(2) generated questions if available, otherwise fallback to general questions
            if state.get("secondary_analysis") and state["secondary_analysis"].get("follow_up_questions"):
                questions = state["secondary_analysis"]["follow_up_questions"]
                bot_message = f"🤖 **A.I.(2) asks:** Please answer these specific questions:\n• {questions[0]}\n• {questions[1] if len(questions) > 1 else 'Any additional symptoms or changes?'}"
            else:
                # Fallback to general question generation
                try:
                    diagnosis = state["last_diagnosis"]["diagnosis"]
                    questions = free_diagnosis_system.question_generator.generate_questions(diagnosis)
                    bot_message = f"Of course. Please answer the following:\n• {questions[0]}\n• {questions[1]}"
                except Exception as e:
                    bot_message = "Let me ask some general questions:\n• How long have you had these symptoms?\n• Are the symptoms getting better, worse, or staying the same?"
            
            history.append({"role": "assistant", "content": bot_message})
            state["stage"] = "AWAITING_INFO"
            # Hide buttons after click to prevent multiple activations
            return history, state, gr.update(visible=False), gr.update(visible=False)

        def on_provide_info_click(state, history):
            history = history or []
            bot_message = "Please provide any additional information, test results, or observations."
            history.append({"role": "assistant", "content": bot_message})
            state["stage"] = "AWAITING_INFO"
            # Hide buttons after click
            return history, state, gr.update(visible=False), gr.update(visible=False)

        def on_explain_click(state, history):
            history = history or []
            if not state.get("last_diagnosis"):
                bot_message = "No diagnosis has been made yet."
                history.append({"role": "assistant", "content": bot_message})
                return history, state, gr.update(visible=False)
            diagnosis = state["last_diagnosis"]["diagnosis"]
            try:
                explanation = free_diagnosis_system.general_expert.answer(f"Please provide a detailed explanation of {diagnosis}.")
            except Exception as e:
                explanation = f"I'm having trouble generating a detailed explanation right now. {diagnosis} is the current diagnosis. Please consult with a healthcare professional for detailed information."
            history.append({"role": "assistant", "content": explanation})
            return history, state

        def on_medication_click(state, history):
            history = history or []
            if not state.get("last_diagnosis"):
                bot_message = "No diagnosis has been made yet."
                history.append({"role": "assistant", "content": bot_message})
                return history, state
                
            diagnosis = state["last_diagnosis"]["diagnosis"]
            symptoms = state.get("symptoms", "")
            severity = state["last_diagnosis"].get("severity", "mild")

            try:
                # Get enhanced medication recommendations
                med_recommendations = free_diagnosis_system.get_enhanced_medication_recommendations(
                    diagnosis, symptoms, severity
                )
            except Exception as e:
                med_recommendations = {
                    'error': True,
                    'message': "I'm having trouble generating medication recommendations right now. Please consult with a healthcare professional for appropriate treatment options."
                }
            
            # Format the medication recommendations
            bot_message = format_medication_recommendations(med_recommendations, diagnosis)
            history.append({"role": "assistant", "content": bot_message})
            return history, state

        def process_voice_input_simple(history, state):
            """
            Voice input handler with follow-up questions - matches text chat flow
            Two-stage voice process: 1) Record symptoms → ask follow-up 2) Record history → diagnosis + TTS
            """
            history = history or []
            
            # Check if we're waiting for medical history response
            if state.get("stage") == "AWAITING_HISTORY_VOICE":
                # Validate required state for medical history collection
                if not state.get("symptoms"):
                    history.append({"role": "assistant", "content": "❌ Error: No symptoms recorded. Please start over by describing your symptoms first."})
                    state["stage"] = "AWAITING_SYMPTOMS"
                    return history, state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                # Second voice input - for medical history
                try:
                    recording_msg = """🎤 **RECORDING MEDICAL HISTORY**

🔴 **Recording up to 15 seconds** (auto-stops when you finish)  

💡 **What to say**:  
• "None" (if no conditions)
• "Diabetic, allergic to penicillin"  
• "High blood pressure, heart condition"

*🔊 Listening for your medical history...*"""
                    
                    history.append({"role": "assistant", "content": recording_msg})
                    
                    # Record and transcribe medical history
                    audio_file = free_diagnosis_system.voice_system.record_audio(duration=15)
                    if not audio_file:
                        history.append({"role": "assistant", "content": "❌ Recording failed. Please try again or type your response."})
                        return history, state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                    
                    transcription, confidence = free_diagnosis_system.voice_system.transcribe_audio(audio_file)
                    
                    if confidence < 0.4:
                        history.append({"role": "assistant", "content": f"❌ Could not clearly understand your response (confidence: {confidence:.1%}). Please try again."})
                        return history, state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                    
                    # Add transcribed history as user message
                    history.append({"role": "user", "content": f"🎤 Medical history: '{transcription}'"})
                    
                    # Process through unified handler to get final diagnosis
                    updated_history, updated_state, ask_q_update, provide_info_update, explain_update, medication_update = unified_handler(
                        transcription, None, history, state
                    )
                    
                    # Speak the final diagnosis
                    try:
                        if updated_state.get("last_diagnosis"):
                            free_diagnosis_system.speak_diagnosis(updated_state["last_diagnosis"], wait=False)
                            if updated_history and updated_history[-1]["role"] == "assistant":
                                ai_message = updated_history[-1]["content"]
                                updated_history[-1] = {"role": "assistant", "content": f"{ai_message}\n\n🔊 *AI is speaking this diagnosis*"}
                    except Exception as tts_error:
                        print(f"TTS error (non-critical): {tts_error}")
                    
                    return updated_history, updated_state, ask_q_update, provide_info_update, explain_update, medication_update
                    
                except Exception as e:
                    error_msg = f"❌ Voice processing error: {str(e)}"
                    history.append({"role": "assistant", "content": error_msg})
                    return history, state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            
            # First voice input - for symptoms
            # Check voice system availability
            if not free_diagnosis_system.voice_system or not free_diagnosis_system.voice_system.recognizer:
                history.append({"role": "assistant", "content": "❌ Voice system not available. Please check your microphone and audio setup, or use text input instead."})
                return history, state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                
            try:
                # Show immediate recording status with clear instructions
                recording_msg = """🎤 **VOICE RECORDING ACTIVE**

🔴 **Recording up to 20 seconds** (auto-stops when you finish)  
⏸️ **Stops automatically** after 2 seconds of silence  
⏱️ You have **5 seconds** to start speaking

💡 **Examples of what to say**:  
• "I have chest pain and shortness of breath"  
• "My head hurts and I feel dizzy and nauseous"  
• "I have a fever, cough, and sore throat"  
• "I've been experiencing back pain for 3 days"

🎯 **How it works**: Speak normally into your microphone. When you pause for 2 seconds, recording stops automatically. No rush!

*🔊 Listening now... start speaking whenever you're ready*"""
                
                history.append({"role": "assistant", "content": recording_msg})
                
                # Record and transcribe symptoms only
                audio_file = free_diagnosis_system.voice_system.record_audio(duration=20)
                
                if not audio_file:
                    history.append({"role": "assistant", "content": "❌ Recording failed. Please check your microphone and try again."})
                    return history, state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                
                # Transcribe the recorded audio
                transcription, confidence = free_diagnosis_system.voice_system.transcribe_audio(audio_file)
                    
            except Exception as e:
                error_msg = f"❌ Voice processing system error: {str(e)}"
                print(f"Voice processing error: {e}")
                history.append({"role": "assistant", "content": error_msg})
                return history, state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            
            if not transcription or transcription.startswith("❌"):
                history.append({"role": "assistant", "content": f"❌ Voice processing failed: {transcription}. Please try again."})
                return history, state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            
            # Enhanced transcription validation  
            transcription_clean = transcription.lower().strip()
            repeated_words = ['thank you', 'thank you.', 'thanks', 'hello', 'hello.', '', 'uh', 'um', 'hmm']
            
            # Check for poor transcription quality
            is_poor_quality = (
                confidence < 0.4 or  # Low confidence
                len(transcription.split()) < 2 or  # Too short
                transcription_clean in repeated_words or  # Repeated meaningless words
                len(set(transcription.split())) < len(transcription.split()) * 0.7  # Too repetitive
            )
            
            if is_poor_quality:
                error_msg = f"❌ Transcription unclear (confidence: {confidence:.1%}). "
                if confidence < 0.4:
                    error_msg += "Please speak more clearly and loudly. "
                if len(transcription.split()) < 2:
                    error_msg += "Please speak longer phrases. "
                if transcription_clean in repeated_words:
                    error_msg += "No clear medical symptoms detected. "
                
                error_msg += "\n💡 Try: 'I have chest pain and fever' or 'My head hurts and I feel dizzy'"
                history.append({"role": "assistant", "content": error_msg})
                return history, state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            
            try:
                # Add transcribed symptoms as user message
                history.append({"role": "user", "content": f"🎤 Symptoms: '{transcription}'"})
                
                # Store symptoms and ask for medical history (follow text chat flow)
                state["symptoms"] = transcription
                state["context_history"] = state.get("context_history", [])
                state["context_history"].append(f"Symptoms: {transcription}")
                state["stage"] = "AWAITING_HISTORY_VOICE"  # Set voice-specific stage
                
                # Ask for medical history via TTS and text
                follow_up_msg = """Thank you. Any pre-existing conditions, allergies, or relevant patient history? 

🎤 **Click the Voice button again to respond**, or type your answer:

💡 **Examples**:
• "None" (if no conditions)
• "Diabetic, allergic to penicillin"  
• "High blood pressure, heart condition"
• "Asthma, no known allergies"

If you have no medical history, just say **"none"**."""
                
                history.append({"role": "assistant", "content": follow_up_msg})
                
                # Speak the question for better voice experience
                try:
                    spoken_question = "Thank you. Do you have any pre-existing conditions, allergies, or relevant medical history? If not, just say none."
                    free_diagnosis_system.voice_system.speak(spoken_question, wait=False)
                except Exception as tts_error:
                    print(f"TTS error (non-critical): {tts_error}")
                
                return history, state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                
            except Exception as e:
                error_msg = f"❌ AI processing error: {str(e)}"
                print(f"AI processing error: {e}")
                history.append({"role": "assistant", "content": error_msg})
                return history, state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)


        # Link handlers to events
        txt_input.submit(lambda msg, hist, st: unified_handler(msg, None, hist, st), [txt_input, chatbot, case_state], [chatbot, case_state, ask_questions_btn, provide_info_btn, explain_btn, medication_btn], queue=False).then(lambda: "", None, txt_input)
        upload_btn.upload(lambda img, hist, st: unified_handler(None, img, hist, st), [upload_btn, chatbot, case_state], [chatbot, case_state, ask_questions_btn, provide_info_btn, explain_btn, medication_btn], queue=False)
        voice_btn.click(process_voice_input_simple, [chatbot, case_state], [chatbot, case_state, ask_questions_btn, provide_info_btn, explain_btn, medication_btn], queue=False)

        ask_questions_btn.click(on_ask_questions_click, [case_state, chatbot], [chatbot, case_state, ask_questions_btn, provide_info_btn])
        provide_info_btn.click(on_provide_info_click, [case_state, chatbot], [chatbot, case_state, ask_questions_btn, provide_info_btn])
        explain_btn.click(on_explain_click, [case_state, chatbot], [chatbot, case_state])
        medication_btn.click(on_medication_click, [case_state, chatbot], [chatbot, case_state])


        # Professional examples section
        with gr.Column(elem_classes="examples-section"):
            gr.HTML("""
                <div style="padding: 1rem; background: #f8f9fa; border-radius: 10px; margin: 1rem;">
                    <h3 style="color: #2c3e50; margin-bottom: 0.5rem; font-size: 1.1rem;">💡 Example Queries</h3>
                    <p style="color: #6c757d; font-size: 0.9rem; margin: 0;">
Try these sample inputs to get started:</p>
                </div>
            """)
            
            gr.Examples(
                examples=[
                    ["I have a persistent cough, high fever, and chest pain for 3 days."],
                    ["Severe headache, nausea, and sensitivity to light since this morning."],
                    ["What is the difference between a viral and bacterial infection?"],
                    ["Skin rash on arms with itching, appeared after eating shellfish."],
                    ["Joint pain and stiffness in hands, worse in the morning."]
                ],
                inputs=txt_input,
                elem_id="examples-component"
            )

        # Medical disclaimer footer
        gr.HTML("""
            <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 10px; padding: 1rem; margin: 1rem; text-align: center;">
                <p style="color: #856404; margin: 0; font-size: 0.9rem;">
                    <strong>⚠️ Medical Disclaimer:</strong> This AI system is for educational and research purposes only. 
                    It is not a substitute for professional medical advice, diagnosis, or treatment. 
                    Always consult qualified healthcare providers for medical concerns.
                </p>
            </div>
        """)

        # Add some bottom spacing
        gr.HTML('<div style="height: 2rem;"></div>')

    return demo

def format_diagnosis_output(result: Dict, is_final: bool) -> str:
    """
    Format diagnosis output with professional medical styling
    
    This function creates a visually appealing and informative display
    of diagnosis results with appropriate medical styling and color coding.
    
    Args:
        result (Dict): Diagnosis result from the AI system
        is_final (bool): Whether this is a final or preliminary diagnosis
        
    Returns:
        str: HTML-formatted diagnosis display with styling
    """
    confidence = result['confidence']
    
    # Check for emergency first
    if result.get('emergency_alert'):
        return f"""
<div style="border: 3px solid #dc3545; background: #f8d7da; padding: 1.5rem; border-radius: 12px; margin: 1rem 0; box-shadow: 0 4px 12px rgba(220,53,69,0.3);">
{result['emergency_alert']}
</div>
"""
    
    # Determine confidence styling and icon
    if confidence > 0.8:
        confidence_class = "confidence-high"
        confidence_icon = "🟢"
        status_color = "#27ae60"
    elif confidence > 0.6:
        confidence_class = "confidence-medium"
        confidence_icon = "🟡"
        status_color = "#f39c12"
    else:
        confidence_class = "confidence-low"
        confidence_icon = "🔴"
        status_color = "#e74c3c"
    
    # AI Stage indicator
    ai_stage = result.get('ai_stage', 'primary')
    stage_emoji = "🤖" if ai_stage == "primary" else "🔍"
    stage_text = "A.I.(1) Primary Analysis" if ai_stage == "primary" else "A.I.(2) Secondary Analysis"
    
    if is_final:
        header = f"🎯 **FINAL DIAGNOSIS** {confidence_icon}"
        status_bg = "#d4edda"
        border_color = "#27ae60"
        recommendation = free_diagnosis_system.feedback_generator.generate_feedback(result['diagnosis'])
    else:
        header = f"⚠️ **PRELIMINARY DIAGNOSIS** {confidence_icon}"
        status_bg = "#fff3cd"
        border_color = "#ffc107"
        recommendation = ""

    # Format improvement if available
    improvement_text = ""
    if result.get('improvement', 0) > 0:
        improvement_text = f"\n✨ **Confidence Improved:** +{result['improvement']:.1%} with additional context"

    output = f"""
<div style="border-left: 4px solid {border_color}; background: {status_bg}; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
<h3 style="color: {status_color}; margin: 0 0 1rem 0; font-size: 1.2rem;">{header}</h3>

<div style="background: white; padding: 1rem; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
<table style="width: 100%; border-collapse: collapse;">
<tr><td style="padding: 0.5rem 0; font-weight: bold; color: #2c3e50;">🩺 Primary Diagnosis:</td><td style="padding: 0.5rem 0;">{result['diagnosis']}</td></tr>
<tr><td style="padding: 0.5rem 0; font-weight: bold; color: #2c3e50;">📊 Confidence Level:</td><td style="padding: 0.5rem 0; color: {status_color}; font-weight: bold;">{result['confidence']:.1%}</td></tr>
<tr><td style="padding: 0.5rem 0; font-weight: bold; color: #2c3e50;">⚡ Severity Assessment:</td><td style="padding: 0.5rem 0;">{result['severity'].title()}</td></tr>
<tr><td style="padding: 0.5rem 0; font-weight: bold; color: #2c3e50;">{stage_emoji} Analysis Stage:</td><td style="padding: 0.5rem 0;">{stage_text}</td></tr>
</table>
</div>

<div style="background: #f8f9fa; padding: 0.75rem; border-radius: 6px; margin: 1rem 0; border-left: 3px solid #6c757d;">
<strong>📋 Similar Medical Knowledge:</strong><br>
<em>"{result['similar_case']}"</em>
</div>

<div style="background: #e9ecef; padding: 0.75rem; border-radius: 6px; margin: 1rem 0;">
<strong>🧠 AI Reasoning:</strong><br>
<em>{result['reasoning']}</em>
</div>

{improvement_text}

{recommendation}
</div>
"""
    return output.strip()


def format_medication_recommendations(recommendations: Dict, diagnosis: str) -> str:
    """
    Format medication recommendations from the AI expert
    
    This function creates a comprehensive display of medication recommendations
    including AI-generated advice, database matches, safety warnings, and
    drug interaction alerts.
    
    Args:
        recommendations (Dict): Medication recommendations from the system
        diagnosis (str): Primary diagnosis for context
        
    Returns:
        str: Formatted medication recommendations with safety information
    """
    output = f"💊 **ENHANCED MEDICATION RECOMMENDATIONS FOR {diagnosis.upper()}**\n\n"
    
    # AI Recommendations section
    if recommendations.get("ai_recommendations"):
        output += "🤖 **AI-Powered Recommendations:**\n"
        output += f"```\n{recommendations['ai_recommendations']}\n```\n\n"
    
    # Database recommendations
    if recommendations.get("database_match"):
        output += "📋 **Evidence-Based Recommendations:**\n"
        for rec in recommendations["database_match"]:
            output += f"• {rec}\n"
        output += "\n"
    
    # Safety assessment
    if recommendations.get("safety_check"):
        safety = recommendations["safety_check"]
        urgency_color = {"emergency": "🚨", "urgent": "⚠️", "routine": "✅"}
        urgency_icon = urgency_color.get(safety.get("urgency_level", "routine"), "✅")
        
        output += f"{urgency_icon} **Safety Assessment:** {safety.get('urgency_level', 'routine').title()}\n"
        
        if safety.get("warnings"):
            for warning in safety["warnings"]:
                output += f"• {warning}\n"
        
        if safety.get("general_advice"):
            output += f"• {safety['general_advice']}\n"
        output += "\n"
    
    # Drug interactions
    if recommendations.get("drug_interactions"):
        output += f"{recommendations['drug_interactions']}\n\n"
    
    # Source information
    source = recommendations.get("source", "AI + Database")
    output += f"📊 **Source:** {source}\n\n"
    
    # Comprehensive disclaimer
    output += """🚨 **CRITICAL MEDICAL DISCLAIMER:**
• These recommendations are for educational purposes only
• AI-generated advice cannot replace professional medical consultation
• Always verify dosages and contraindications with healthcare providers
• Seek immediate medical attention if symptoms worsen
• Contact your doctor before starting any new medication
• This system is not a substitute for professional medical care"""
    
    return output
