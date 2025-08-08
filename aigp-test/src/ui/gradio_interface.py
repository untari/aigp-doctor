

import gradio as gr
from PIL import Image
from typing import Dict

from src.diagnosis_system import DiagnosisSystem

free_diagnosis_system = DiagnosisSystem()

def create_chatbot_interface():
    """Creates and returns the professional Gradio Chatbot interface."""
    # Custom CSS for modern chat interface
    custom_css = """
    body {
        background: #f5f7fa !important;
    }
    
    .main-container {
        max-width: 800px;
        margin: 0 auto;
        background: transparent;
    }
    
    .header-section {
        text-align: center;
        padding: 1rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .header-title {
        color: white;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.95rem;
        font-weight: 300;
        margin-bottom: 0.3rem;
    }
    
    .chat-container {
        background: white;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        margin: 1rem;
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
                height=450,
                elem_classes="chat-messages",
                avatar_images=(None, None),
                show_copy_button=True
            )

            # Input section
            with gr.Row(elem_classes="input-row"):
                with gr.Column(scale=4):
                    txt_input = gr.Textbox(
                        show_label=False, 
                        placeholder="💬 Describe your symptoms, ask questions, or provide additional information...",
                        container=False,
                        elem_classes="message-input"
                    )
                with gr.Column(scale=1, min_width=120):
                    upload_btn = gr.UploadButton(
                        "📁 Upload Image", 
                        file_types=["image"],
                        elem_classes="upload-button",
                        size="sm"
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

        def unified_handler(message, image_upload, history, state):
            history = history or []
            bot_message = ""

            # This will hold the updates for the buttons
            ask_q_update = gr.update(visible=False)
            provide_info_update = gr.update(visible=False)
            explain_update = gr.update(visible=False)

            # 1. Handle Image Upload with enhanced context integration
            if image_upload is not None:
                history.append((None, (image_upload.name,)))
                pil_image = Image.open(image_upload.name)
                current_context = " ".join(state.get("context_history", []))
                description = free_diagnosis_system.analyze_image_in_context(pil_image, current_context)
                state["context_history"].append(description)
                bot_message = f"I've analyzed the image in context. {description}. What are the primary symptoms?"
                state["stage"] = "AWAITING_SYMPTOMS"
                history.append((None, bot_message))
                return history, state, ask_q_update, provide_info_update, explain_update

            # Add user message to history
            if message:
                history.append((message, None))

            # 2. Enhanced State Machine Logic with A.I.(1) and A.I.(2) routing
            stage = state["stage"]

            if stage == "AWAITING_SYMPTOMS":
                # Check if this is a general medical question rather than symptoms
                question_indicators = [
                    "what is", "what are", "how does", "why does", "explain", "difference between",
                    "tell me about", "can you explain", "what's the", "how to", "when should"
                ]
                
                is_question = any(indicator in message.lower() for indicator in question_indicators)
                
                if is_question:
                    # Handle as a general medical question
                    bot_message = free_diagnosis_system.general_expert.answer(message)
                    bot_message += "\n\n---\n\n💡 **Need a diagnosis?** Describe your symptoms and I'll help analyze them."
                    # Stay in AWAITING_SYMPTOMS state for potential follow-up
                else:
                    # Handle as symptoms - proceed with diagnosis
                    state["symptoms"] = message
                    state["context_history"].append(f"Symptoms: {message}")
                    bot_message = "Thank you. Any pre-existing conditions, allergies, or relevant patient history? (e.g., 'diabetic, allergic to penicillin') If not, just say 'none')."
                    state["stage"] = "AWAITING_HISTORY"

            elif stage == "AWAITING_HISTORY":
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
                 bot_message = free_diagnosis_system.general_expert.answer(message)

            if bot_message:
                history.append((None, bot_message))

            # Update button visibility based on state
            if state["stage"] == "AWAITING_CLARIFICATION_CHOICE":
                ask_q_update = gr.update(visible=True)
                provide_info_update = gr.update(visible=True)
            if state["stage"] == "DIAGNOSIS_COMPLETE":
                explain_update = gr.update(visible=True)

            return history, state, ask_q_update, provide_info_update, explain_update

        def on_ask_questions_click(state, history):
            history = history or []
            # Defensive check
            if not state.get("last_diagnosis"):
                bot_message = "I can't ask questions yet. Please describe the symptoms first."
                history.append((None, bot_message))
                return history, state, gr.update(visible=False), gr.update(visible=False)

            # Use A.I.(2) generated questions if available, otherwise fallback
            if state.get("secondary_analysis") and state["secondary_analysis"].get("follow_up_questions"):
                questions = state["secondary_analysis"]["follow_up_questions"]
                bot_message = f"🤖 **A.I.(2) asks:** Please answer these specific questions:\n• {questions[0]}\n• {questions[1] if len(questions) > 1 else 'Any additional symptoms or changes?'}"
            else:
                diagnosis = state["last_diagnosis"]["diagnosis"]
                questions = free_diagnosis_system.question_generator.generate_questions(diagnosis)
                bot_message = f"Of course. Please answer the following:\n• {questions[0]}\n• {questions[1]}"
            
            history.append((None, bot_message))
            state["stage"] = "AWAITING_INFO"
            # Hide buttons after click
            return history, state, gr.update(visible=False), gr.update(visible=False)

        def on_provide_info_click(state, history):
            history = history or []
            bot_message = "Please provide any additional information, test results, or observations."
            history.append((None, bot_message))
            state["stage"] = "AWAITING_INFO"
            # Hide buttons after click
            return history, state, gr.update(visible=False), gr.update(visible=False)

        def on_explain_click(state, history):
            history = history or []
            if not state.get("last_diagnosis"):
                bot_message = "No diagnosis has been made yet."
                history.append((None, bot_message))
                return history, state
            diagnosis = state["last_diagnosis"]["diagnosis"]
            explanation = free_diagnosis_system.general_expert.answer(f"Please provide a detailed explanation of {diagnosis}.")
            history.append((None, explanation))
            return history, state

        # Link handlers to events
        txt_input.submit(lambda msg, hist, st: unified_handler(msg, None, hist, st), [txt_input, chatbot, case_state], [chatbot, case_state, ask_questions_btn, provide_info_btn, explain_btn], queue=False).then(lambda: "", None, txt_input)
        upload_btn.upload(lambda img, hist, st: unified_handler(None, img, hist, st), [upload_btn, chatbot, case_state], [chatbot, case_state, ask_questions_btn, provide_info_btn, explain_btn], queue=False)

        ask_questions_btn.click(on_ask_questions_click, [case_state, chatbot], [chatbot, case_state, ask_questions_btn, provide_info_btn])
        provide_info_btn.click(on_provide_info_click, [case_state, chatbot], [chatbot, case_state, ask_questions_btn, provide_info_btn])
        explain_btn.click(on_explain_click, [case_state, chatbot], [chatbot, case_state])

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
    """Format diagnosis output with professional medical styling"""
    confidence = result['confidence']
    
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
