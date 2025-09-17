#!/usr/bin/env python3
"""
AI GP Doctor - Modular Version
Free Open-Source Medical Diagnosis System
University Thesis Project

Entry point for the modular AI GP Doctor system.
This application provides AI-powered medical diagnosis assistance using multiple
open-source models including BioBERT, Clinical BERT, and image analysis models.
"""

import sys
import os

# Add src directory to Python path to enable module imports
# This allows importing from the src package structure
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the main UI interface factory function
from src.ui.gradio_interface import create_chatbot_interface

if __name__ == "__main__":
    # Application startup messages for user feedback
    print("🚀 Starting AI GP Doctor - Modular Version...")
    print("📦 Loading models (this may take a few minutes on first run)...")
    
    # Create the Gradio chatbot interface with all components
    # This initializes all AI models and UI components
    chatbot_demo = create_chatbot_interface()
    
    # Launch the web interface
    # share=False: Prevents automatic public sharing for security
    # debug=False: Disables debug mode to prevent information disclosure
    chatbot_demo.launch(share=False, debug=False)
