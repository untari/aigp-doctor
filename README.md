# AI GP Doctor - Modular Version

## 📋 Project Description

AI GP Doctor is a **free, open-source medical diagnosis system** developed as a university thesis project. This modular application provides AI-powered medical diagnosis assistance using multiple open-source models including BioBERT, Clinical BERT, and specialized image analysis models.

### 🎯 Key Features

- **Two-Stage AI Diagnosis System**
  - **A.I.(1)**: Primary analysis using ensemble of specialized medical models
  - **A.I.(2)**: Secondary analysis for follow-up questions and context refinement

- **Multi-Modal Analysis**
  - Text-based symptom analysis
  - Medical image processing and analysis
  - Voice input/output capabilities

- **Privacy-First Design**
  - All models run locally without external API calls
  - No data sent to third-party services
  - Secure image processing with validation

- **Professional Web Interface**
  - Chat-style conversation flow
  - Medical-themed UI design
  - Image upload capabilities
  - Interactive diagnosis workflow

### 🏗️ Architecture

The system integrates multiple specialized AI models:

- **BioBERT**: Medical text analysis and understanding
- **Clinical BERT**: Clinical note processing and medical reasoning
- **Sentence Transformers**: Semantic similarity and context matching
- **Image Analysis Models**: Medical image interpretation
- **Voice System**: Speech-to-text and text-to-speech capabilities
- **Dynamic Question Generator**: Intelligent follow-up questions
- **Medication Expert**: medication interaction and recommendation analysis

## 🚀 Local Setup

### Prerequisites

- **Python 3.8+** (Python 3.12 recommended)
- **4GB+ RAM** (8GB+ recommended for optimal performance)
- **2GB+ free disk space** (for model downloads)

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd aigp-doctor
   ```

2. **Create Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **First-Time Model Download**
   ```bash
   # Models will be automatically downloaded on first run
   # This may take 5-15 minutes depending on internet speed
   python3 main.py
   ```

### 📦 Dependencies

The project uses the following key dependencies:

- **gradio**: Web interface framework
- **torch**: PyTorch for deep learning models
- **transformers**: Hugging Face transformers library
- **sentence-transformers**: Semantic text analysis
- **Pillow**: Image processing
- **openai-whisper**: Speech recognition
- **pyttsx3**: Text-to-speech synthesis
- **coqui-tts**: Advanced text-to-speech
- **librosa**: Audio processing
- **python-magic**: File type detection

### 🖥️ Running the Application

1. **Start the Application**
   ```bash
   python3 main.py
   ```

2. **Access the Interface**
   - Open your web browser
   - Navigate to `http://localhost:7860`
   - The interface will be available once models are loaded

3. **First Launch**
   - Initial startup may take 5-15 minutes for model downloads
   - Subsequent launches will be much faster
   - Watch the console for loading progress

## 📁 Project Structure

```
aigp-doctor/
├── main.py                          # Application entry point
├── requirements.txt                 # Python dependencies
├── image_handler.py                # Secure image processing
├── src/
│   ├── diagnosis_system.py         # Core diagnosis orchestrator
│   ├── models/                     # AI model implementations
│   │   ├── biobert.py              # BioBERT medical analysis
│   │   ├── clinical_bert.py        # Clinical BERT processing
│   │   ├── sentence_transformer.py # Semantic analysis
│   │   ├── question_generator.py   # Dynamic questioning
│   │   ├── image_expert.py         # Medical image analysis
│   │   ├── voice_system.py         # Speech processing
│   │   ├── general_expert.py       # General medical reasoning
│   │   ├── secondary_analyzer.py   # Secondary AI analysis
│   │   ├── feedback_generator.py   # Actionable feedback
│   │   └── medication_expert.py    # Drug recommendations
│   └── ui/
│       └── gradio_interface.py     # Web interface
└── test_voice.py                   # Voice system testing
```

## 🔧 Configuration

### System Requirements
- **Minimum**: 4GB RAM, 2-core CPU
- **Recommended**: 8GB+ RAM, 4-core CPU with GPU support
- **Storage**: 2GB for models and dependencies

### Voice Features (Optional)
For voice input/output capabilities:
- **Linux**: `sudo apt-get install portaudio19-dev`
- **macOS**: `brew install portaudio`
- **Windows**: Usually works out of the box

### GPU Acceleration (Optional)
For faster model inference:
```bash
# For NVIDIA GPUs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## ⚠️ Important Notes

### Medical Disclaimer
- This system is for **educational and research purposes only**
- **NOT intended for actual medical diagnosis**
- Always consult qualified healthcare professionals
- Not a replacement for professional medical advice

### Privacy & Security
- All processing happens locally on your machine
- No data is sent to external servers
- Images are processed securely with validation
- No persistent storage of medical data

### Performance
- First run requires model downloads (5-15 minutes)
- Subsequent runs start in 30-60 seconds
- GPU acceleration recommended for faster inference
- Memory usage: 2-4GB during operation

## 🤝 Contributing

This is a university thesis project. For academic purposes:
- Follow the existing code structure
- Maintain security best practices
- Document any changes thoroughly
- Ensure all models remain open-source

## 🆘 Troubleshooting

### Common Issues

1. **"Module not found" errors**
   ```bash
   # Ensure virtual environment is activated
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Memory errors during model loading**
   - Close other applications to free RAM
   - Consider using a machine with more memory
   - Models require 2-4GB RAM to load

3. **Slow performance**
   - Enable GPU acceleration if available
   - Ensure sufficient system resources
   - Close unnecessary background applications

4. **Voice features not working**
   - Install system audio dependencies
   - Check microphone permissions
   - Verify audio device availability

### Getting Help

For technical issues:
1. Check the console output for error messages
2. Verify all dependencies are installed correctly
3. Ensure Python version compatibility (3.8+)
4. Review the troubleshooting steps above

---

**🎓 University Thesis Project** | **🚀 Free & Open Source** | **🔒 Privacy-First Design**
