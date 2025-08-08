#!/usr/bin/env python3
"""
AI GP Doctor - Modular Version
Free Open-Source Medical Diagnosis System
University Thesis Project

Entry point for the modular AI GP Doctor system.
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ui.gradio_interface import create_chatbot_interface

if __name__ == "__main__":
    print("🚀 Starting AI GP Doctor - Modular Version...")
    print("📦 Loading models (this may take a few minutes on first run)...")
    
    # Create and launch the interface
    chatbot_demo = create_chatbot_interface()
    chatbot_demo.launch(share=True, debug=True)