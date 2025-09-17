"""
Voice System Module for AI GP Doctor
====================================

This module implements a complete voice system with Speech-to-Text (STT) using OpenAI Whisper
and Text-to-Speech (TTS) capabilities for natural voice interactions during medical consultations.

Key Features:
- STT: OpenAI Whisper with medical context optimization
- TTS: pyttsx3 with medical voice configuration
- Medical terminology correction and enhancement
- Asynchronous speech processing to prevent UI blocking
- Audio quality enhancement and noise reduction
- Voice activity detection for better user experience

Medical Context:
- Optimized transcription prompts for medical terminology
- Common medical term correction (diabeetus -> diabetes, etc.)
- Emergency detection integration
- Natural speech formatting for medical responses

Security & Privacy:
- All voice processing runs locally (no external API calls)
- Temporary audio files are automatically cleaned up
- No voice data is transmitted or stored remotely
"""

import whisper
import pyttsx3
import speech_recognition as sr
import tempfile
import threading
import queue
import numpy as np
import os
import re
from typing import Optional, Tuple, Dict
import time

# Try to import Coqui TTS for better voice quality
try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
    print("🎵 Coqui TTS available - will use high-quality voices")
except ImportError:
    TTS = None
    COQUI_AVAILABLE = False
    print("💡 Coqui TTS not installed. Install with: pip install coqui-tts")

class VoiceSystem:
    """
    Complete voice system with Whisper STT and pyttsx3 TTS
    
    This class provides a comprehensive voice interface for the AI GP Doctor system,
    enabling natural voice consultations with high-quality transcription and speech synthesis.
    """
    
    def __init__(self):
        """
        Initialize the voice system with STT and TTS capabilities
        
        This method loads the Whisper model for speech-to-text, initializes the TTS engine,
        and sets up audio recording capabilities with medical optimization settings.
        """
        print("🎤 Initializing Voice System...")
        
        # Initialize Whisper for STT with medical optimization
        try:
            print("🔄 Loading Whisper model (this may take a few minutes)...")
            # Use 'small' model for better balance of accuracy and speed
            # Options: tiny (~39MB), base (~74MB), small (~244MB), medium (~769MB)
            self.whisper_model = whisper.load_model("small")
            print("✅ Whisper STT model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading Whisper: {e}")
            print("💡 Try: pip install openai-whisper torch")
            self.whisper_model = None
        
        # Initialize TTS engines with priority: Coqui TTS > pyttsx3
        self.coqui_tts = None
        self.tts_engine = None
        
        # Try to initialize Coqui TTS first (higher quality)
        if COQUI_AVAILABLE:
            try:
                print("🔄 Loading Coqui TTS model (high-quality voice)...")
                # Try multiple models in order of preference for medical speech
                models_to_try = [
                    "tts_models/en/ljspeech/tacotron2-DDC",  # High quality, clear pronunciation
                    "tts_models/en/ljspeech/tacotron2-DCA",  # Alternative tacotron model  
                    "tts_models/en/ljspeech/glow-tts",       # Fast, good quality
                    "tts_models/en/ljspeech/fast_pitch",     # Very fast, reasonable quality
                ]
                
                for model_name in models_to_try:
                    try:
                        print(f"🔄 Trying model: {model_name}")
                        self.coqui_tts = TTS(model_name=model_name, progress_bar=False)
                        print(f"✅ Successfully loaded: {model_name}")
                        break
                    except Exception as model_error:
                        print(f"⚠️ Model {model_name} failed: {model_error}")
                        continue
                
                if not self.coqui_tts:
                    raise Exception("All Coqui TTS models failed to load")
                print("✅ Coqui TTS loaded successfully! Using high-quality voice synthesis.")
            except Exception as e:
                print(f"⚠️ Coqui TTS initialization failed: {e}")
                print("💡 Falling back to pyttsx3...")
                self.coqui_tts = None
        
        # Initialize pyttsx3 as fallback or primary if Coqui not available
        try:
            self.tts_engine = pyttsx3.init()
            self._configure_tts_for_medical_use()
            if self.coqui_tts:
                print("✅ pyttsx3 initialized as backup TTS engine")
            else:
                print("✅ pyttsx3 TTS engine initialized (primary)")
        except Exception as e:
            print(f"❌ Error initializing pyttsx3: {e}")
            print("💡 Try: pip install pyttsx3")
            self.tts_engine = None
        
        # Initialize speech recognizer for microphone input
        try:
            self.recognizer = sr.Recognizer()
            
            # Try to initialize microphone with error handling for different systems
            try:
                self.microphone = sr.Microphone()
                print("✅ Default microphone found!")
            except Exception as mic_error:
                # Try to find available microphones
                try:
                    mic_list = sr.Microphone.list_microphone_names()
                    if mic_list:
                        print(f"💡 Available microphones: {len(mic_list)} found")
                        self.microphone = sr.Microphone(device_index=0)  # Use first available
                        print("✅ Using first available microphone!")
                    else:
                        raise Exception("No microphones detected")
                except Exception:
                    print(f"❌ Microphone initialization failed: {mic_error}")
                    self.microphone = None
                    self.recognizer = None
                    return
            
            # Configure recognizer for medical consultations with improved sensitivity
            if self.recognizer and self.microphone:
                self.recognizer.energy_threshold = 200  # Lower threshold for better sensitivity
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.pause_threshold = 2.0   # Longer pauses for medical descriptions (2 seconds of silence to stop)
                self.recognizer.operation_timeout = None
                
                # Set additional attributes only if they exist (version compatibility)
                if hasattr(self.recognizer, 'phrase_threshold'):
                    self.recognizer.phrase_threshold = 0.3  # More responsive to speech start
                if hasattr(self.recognizer, 'non_speaking_duration'):
                    self.recognizer.non_speaking_duration = 0.5  # Detect speech gaps better
                
                print("✅ Microphone system configured and ready!")
            
        except Exception as e:
            print(f"❌ Error initializing speech recognition: {e}")
            print("💡 Try: pip install pyaudio speech_recognition")
            print("💡 On Linux: sudo apt-get install python3-pyaudio portaudio19-dev")
            print("💡 On macOS: brew install portaudio")
            self.recognizer = None
            self.microphone = None
        
        # TTS queue for non-blocking speech (asynchronous operation)
        self.tts_queue = queue.Queue()
        self.tts_thread = None
        self.is_speaking = False
        
        # Medical terminology corrections database
        self.medical_corrections = self._load_medical_corrections()
        
        # Audio enhancement settings
        self.audio_enhancement_enabled = True
        
        print("🎉 Voice System initialization complete!")
        
    def _configure_tts_for_medical_use(self):
        """
        Configure TTS engine specifically for medical consultations
        
        This method optimizes the TTS voice properties for clear, professional
        medical communication with appropriate speed and tone.
        """
        if not self.tts_engine:
            return
            
        try:
            # Get available voices
            voices = self.tts_engine.getProperty('voices')
            
            # Try to find the clearest voice for medical use
            # Preference order: Clear/Professional voices > English quality > Gender
            best_voice = None
            fallback_voice = None
            quality_voice = None
            
            for voice in voices:
                voice_name = voice.name.lower()
                voice_id = voice.id.lower()
                
                # Look for high-quality/clear voices first (prioritize clarity over gender)
                if any(term in voice_name for term in ['english', 'en-us', 'en_us']):
                    # Priority 1: Professional/clear sounding voices
                    if any(term in voice_name for term in ['zira', 'david', 'mark', 'hazel', 'eva', 'aria']):
                        if any(term in voice_name for term in ['zira', 'hazel', 'eva', 'aria']):  # Known clear voices
                            best_voice = voice.id
                            break
                        elif not quality_voice:
                            quality_voice = voice.id
                    # Priority 2: Any English voice as fallback
                    elif not fallback_voice:
                        fallback_voice = voice.id
            
            # Set the best available voice with preference for clarity
            selected_voice = best_voice or quality_voice or fallback_voice
            if selected_voice:
                self.tts_engine.setProperty('voice', selected_voice)
                print(f"🔊 Selected voice for medical consultations")
            
            # Configure speech parameters for clear medical speech
            # Rate: Slower for medical terminology and better comprehension
            self.tts_engine.setProperty('rate', 110)  # Words per minute (slower for medical clarity)
            
            # Volume: Optimized for clarity (slightly higher for better audibility)
            self.tts_engine.setProperty('volume', 0.95)  # 95% volume for clarity
            
            # Add prosody control for less robotic speech (if supported by engine)
            try:
                # Some TTS engines support pitch variation
                if hasattr(self.tts_engine, 'setProperty'):
                    # Try to set pitch variation for more natural sound
                    voices = self.tts_engine.getProperty('voices')
                    if voices and len(voices) > 0:
                        current_voice = self.tts_engine.getProperty('voice')
                        # Enable inflection if supported
                        self.tts_engine.setProperty('inflection', True) if hasattr(self.tts_engine, 'setProperty') else None
                        print("🎵 Enhanced prosody settings applied")
            except Exception:
                pass  # Prosody controls not supported by this engine
            
        except Exception as e:
            print(f"⚠️ TTS configuration warning: {e}")
    
    def _load_medical_corrections(self) -> Dict[str, str]:
        """
        Load medical terminology corrections for common transcription errors
        
        Returns:
            Dict[str, str]: Mapping of commonly misheard terms to correct medical terms
        """
        return {
            # Common medical term corrections
            "diabeetus": "diabetes",
            "diabetus": "diabetes",
            "diabetic": "diabetic",
            "ammonia": "pneumonia", 
            "new monia": "pneumonia",
            "pneumatic": "pneumonia",
            "high pretension": "hypertension",
            "hyper tension": "hypertension",
            "heart burn": "heartburn",
            "migrane": "migraine",
            "migranes": "migraines",
            "migrain": "migraine",
            "short of breath": "shortness of breath",
            "shot of breath": "shortness of breath",
            "stomach egg": "stomach ache",
            "stomach ache": "stomach ache",
            "throw up": "vomiting",
            "threw up": "vomited",
            "throwing up": "vomiting",
            "feel nauseous": "nauseous",
            "feel dizzy": "dizzy",
            "chest paid": "chest pain",
            "back paid": "back pain",
            "head egg": "headache",
            "head ache": "headache",
            "sore throat": "sore throat",
            "sort throat": "sore throat",
            "runny knows": "runny nose",
            "runny news": "runny nose",
            "stuffy knows": "stuffy nose",
            "can't breathe": "cannot breathe",
            "cant breathe": "cannot breathe",
            "difficulty breathing": "difficulty breathing",
            "trouble breathing": "difficulty breathing",
            "fever": "fever",
            "temperature": "fever",
            "hot": "fever",
            "cold": "cold",
            "flu": "flu",
            "influenza": "influenza",
            "cough": "cough",
            "coughing": "coughing",
            "sneeze": "sneezing",
            "sneezing": "sneezing",
            "allergy": "allergy",
            "allergies": "allergies",
            "allergic": "allergic",
            "asthma": "asthma",
            "asmatic": "asthma"
        }
    
    def transcribe_audio(self, audio_file_path: str) -> Tuple[str, float]:
        """
        Transcribe audio file using Whisper with medical context optimization
        
        This method uses OpenAI Whisper to convert speech to text with special
        optimization for medical terminology and consultation context.
        
        Args:
            audio_file_path (str): Path to the audio file to transcribe
            
        Returns:
            Tuple[str, float]: (transcription_text, confidence_score)
                - transcription_text: The transcribed text with medical corrections
                - confidence_score: Confidence estimate (0.0-1.0)
        """
        if not self.whisper_model:
            return "Whisper model not available. Please install openai-whisper.", 0.0
        
        if not os.path.exists(audio_file_path):
            return "Audio file not found.", 0.0
        
        try:
            # Validate audio quality before processing
            audio_quality_check = self._validate_audio_quality(audio_file_path)
            if not audio_quality_check['is_valid']:
                return f"Poor audio quality: {audio_quality_check['reason']}", 0.1
            
            # Enhanced audio preprocessing if available
            processed_audio_path = self._enhance_audio_quality(audio_file_path)
            
            # Minimal medical context prompt to avoid contamination
            # Avoid specific words that might bias transcription
            medical_prompt = (
                "Listen carefully to the patient speaking about their health symptoms. "
                "Transcribe exactly what is said, focusing on medical complaints and symptoms."
            )
            
            print("🔄 Transcribing audio with medical context...")
            
            # Add debug information about audio
            print(f"🎧 Audio file: {processed_audio_path}")
            print(f"📊 File size: {os.path.getsize(processed_audio_path)} bytes")
            
            # Try transcription without any prompt first to avoid contamination
            print("🔍 Attempting transcription without prompt...")
            result = self.whisper_model.transcribe(
                processed_audio_path,
                language="en",  # English for medical consultations
                temperature=0.0,  # Minimum temperature for maximum consistency
                best_of=1,  # Single attempt to avoid hallucination
                beam_size=1,  # Simple beam search to reduce hallucination
                word_timestamps=True,  # Enable word-level timestamps
                condition_on_previous_text=False,  # Don't use previous context
                fp16=False,  # Use full precision for better accuracy
                no_speech_threshold=0.8,  # Very high threshold to avoid transcribing noise
                logprob_threshold=-0.5,  # Higher threshold for word confidence
                compression_ratio_threshold=2.0,  # Strict detection of repetitive text
                verbose=True  # Enable debug output to see what's happening
            )
            
            # Debug output
            print(f"🔍 Raw Whisper result: '{result.get('text', 'NO TEXT')}'")
            if 'segments' in result:
                print(f"🔍 Number of segments: {len(result['segments'])}")
                for i, segment in enumerate(result['segments'][:3]):  # Show first 3 segments
                    print(f"🔍 Segment {i}: '{segment.get('text', '')}' (confidence estimate based on logprob)")
            
            # Check if result looks like hallucination
            raw_text = result["text"].strip().lower()
            hallucination_phrases = [
                'thank you for watching', 'thanks for watching', 'subscribe', 'like and subscribe',
                'don\'t forget to', 'see you next time', 'goodbye', 'thanks for listening',
                'that\'s all for today', 'catch you later'
            ]
            
            is_likely_hallucination = any(phrase in raw_text for phrase in hallucination_phrases)
            
            if is_likely_hallucination:
                print(f"⚠️ Detected likely hallucination: '{raw_text}'")
                print("🔄 Retrying with stricter settings...")
                
                # Retry with even stricter settings
                result = self.whisper_model.transcribe(
                    processed_audio_path,
                    language="en",
                    temperature=0.0,
                    initial_prompt="",  # Completely empty prompt
                    no_speech_threshold=0.9,  # Extremely high threshold
                    logprob_threshold=0.0,  # Very high confidence required
                    compression_ratio_threshold=1.8,  # Strict repetition detection
                    condition_on_previous_text=False,
                    verbose=True
                )
                print(f"🔄 Retry result: '{result.get('text', 'NO TEXT')}'")
                
                # If still hallucinating, return error
                retry_text = result["text"].strip().lower()
                if any(phrase in retry_text for phrase in hallucination_phrases):
                    return "Could not clearly understand speech - please speak directly into microphone about your symptoms", 0.1
            
            # Extract transcription text
            transcription = result["text"].strip()
            
            # Apply medical terminology corrections
            corrected_transcription = self._correct_medical_terms(transcription)
            
            # Clean up the transcription
            final_transcription = self._clean_transcription(corrected_transcription)
            
            # Calculate confidence score based on various factors
            confidence = self._calculate_transcription_confidence(result, final_transcription)
            
            print(f"✅ Transcription complete: {confidence:.1%} confidence")
            
            # Clean up temporary processed audio file safely
            try:
                if processed_audio_path != audio_file_path and os.path.exists(processed_audio_path):
                    os.remove(processed_audio_path)
            except Exception as cleanup_error:
                print(f"⚠️ Warning: Could not clean up temporary file: {cleanup_error}")
            
            return final_transcription, confidence
            
        except ImportError as e:
            print(f"❌ Missing dependencies for transcription: {e}")
            return "Please install required packages: pip install openai-whisper torch", 0.0
        except FileNotFoundError as e:
            print(f"❌ Audio file access error: {e}")
            return "Audio file not found or inaccessible", 0.0
        except MemoryError as e:
            print(f"❌ Not enough memory for transcription: {e}")
            return "Insufficient memory for audio processing. Try shorter recordings.", 0.0
        except RuntimeError as e:
            print(f"❌ Whisper runtime error: {e}")
            if "CUDA" in str(e):
                return "GPU processing failed, trying CPU mode...", 0.0
            return f"Transcription processing failed: {str(e)}", 0.0
        except Exception as e:
            print(f"❌ Unexpected transcription error: {e}")
            print(f"❌ Error type: {type(e).__name__}")
            return f"Transcription failed: {str(e)}", 0.0
    
    def _validate_audio_quality(self, audio_file_path: str) -> dict:
        """
        Validate audio quality before sending to Whisper
        
        Args:
            audio_file_path (str): Path to audio file
            
        Returns:
            dict: {'is_valid': bool, 'reason': str, 'duration': float, 'volume': float}
        """
        try:
            import wave
            import numpy as np
            
            # Open and analyze the audio file
            with wave.open(audio_file_path, 'rb') as wav_file:
                frames = wav_file.readframes(-1)
                sound_info = wav_file.getparams()
                frame_rate = sound_info.framerate
                
                # Convert to numpy array for analysis
                audio_data = np.frombuffer(frames, dtype=np.int16)
                duration = len(audio_data) / frame_rate
                
                # Calculate audio metrics
                rms_volume = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
                max_amplitude = np.max(np.abs(audio_data))
                
                # Quality checks
                if duration < 0.5:
                    return {'is_valid': False, 'reason': 'Audio too short (less than 0.5s)', 'duration': duration, 'volume': rms_volume}
                
                if duration > 30:
                    return {'is_valid': False, 'reason': 'Audio too long (over 30s)', 'duration': duration, 'volume': rms_volume}
                
                if rms_volume < 100:  # Very quiet audio
                    return {'is_valid': False, 'reason': 'Audio too quiet - please speak louder', 'duration': duration, 'volume': rms_volume}
                
                if max_amplitude < 500:  # Almost no signal
                    return {'is_valid': False, 'reason': 'No clear speech detected - check microphone', 'duration': duration, 'volume': rms_volume}
                
                # Check for silence (all near-zero values)
                non_silent_ratio = np.mean(np.abs(audio_data) > 200)
                if non_silent_ratio < 0.1:  # Less than 10% non-silent
                    return {'is_valid': False, 'reason': 'Mostly silence detected - speak closer to microphone', 'duration': duration, 'volume': rms_volume}
                
                # Check for extremely quiet or noisy audio that might cause hallucination
                if rms_volume > 5000:  # Too loud, might be distorted
                    return {'is_valid': False, 'reason': 'Audio too loud/distorted - speak at normal volume', 'duration': duration, 'volume': rms_volume}
                
                # Check for very short audio that might cause hallucination
                if duration < 1.0:
                    return {'is_valid': False, 'reason': 'Recording too short - speak for at least 2-3 seconds', 'duration': duration, 'volume': rms_volume}
                
                # Additional quality metrics
                # Check for consistent volume (not just peaks)
                volume_std = np.std(np.abs(audio_data))
                if volume_std < 50:  # Very flat audio, might be noise
                    return {'is_valid': False, 'reason': 'Audio appears to be background noise - speak clearly into microphone', 'duration': duration, 'volume': rms_volume}
                
                print(f"✅ Audio quality OK: {duration:.1f}s, volume: {rms_volume:.0f}, speech: {non_silent_ratio:.1%}, variation: {volume_std:.0f}")
                return {'is_valid': True, 'reason': 'Good quality', 'duration': duration, 'volume': rms_volume, 'variation': volume_std}
                
        except ImportError:
            # If wave module not available, skip validation
            return {'is_valid': True, 'reason': 'Validation skipped', 'duration': 0, 'volume': 0}
        except Exception as e:
            print(f"⚠️ Audio validation error: {e}")
            return {'is_valid': True, 'reason': 'Validation failed but proceeding', 'duration': 0, 'volume': 0}
    
    def _enhance_audio_quality(self, audio_file_path: str) -> str:
        """
        Enhance audio quality for better transcription accuracy
        
        Args:
            audio_file_path (str): Path to original audio file
            
        Returns:
            str: Path to enhanced audio file (or original if enhancement fails)
        """
        if not self.audio_enhancement_enabled:
            return audio_file_path
            
        try:
            import librosa
            import soundfile as sf
            
            # Load audio with optimal sample rate for Whisper (16kHz)
            audio, sr = librosa.load(audio_file_path, sr=16000)
            
            # Basic audio enhancement
            # Trim silence from beginning and end
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
            
            # Normalize audio volume
            audio_normalized = librosa.util.normalize(audio_trimmed)
            
            # Apply gentle noise reduction (simple spectral gating)
            # This is a basic implementation - could be enhanced with more sophisticated methods
            if len(audio_normalized) > sr:  # Only if audio is longer than 1 second
                # Simple noise reduction: attenuate quiet sections
                threshold = np.percentile(np.abs(audio_normalized), 10)
                audio_normalized = np.where(np.abs(audio_normalized) < threshold, 
                                           audio_normalized * 0.1, audio_normalized)
            
            # Save enhanced audio
            enhanced_path = audio_file_path.replace('.wav', '_enhanced.wav')
            sf.write(enhanced_path, audio_normalized, sr)
            
            print("🎵 Audio enhanced for better transcription")
            return enhanced_path
            
        except ImportError:
            print("💡 Install librosa and soundfile for audio enhancement: pip install librosa soundfile")
            return audio_file_path
        except Exception as e:
            print(f"⚠️ Audio enhancement failed, using original: {e}")
            return audio_file_path
    
    def _correct_medical_terms(self, text: str) -> str:
        """
        Correct common medical transcription errors
        
        Args:
            text (str): Original transcription text
            
        Returns:
            str: Text with medical term corrections applied
        """
        corrected_text = text.lower()
        
        # Apply medical corrections (case-insensitive)
        for wrong_term, correct_term in self.medical_corrections.items():
            # Use word boundaries to avoid partial word replacements
            pattern = r'\b' + re.escape(wrong_term) + r'\b'
            corrected_text = re.sub(pattern, correct_term, corrected_text, flags=re.IGNORECASE)
        
        # Restore original capitalization for sentence beginnings
        sentences = corrected_text.split('. ')
        capitalized_sentences = [sentence.capitalize() for sentence in sentences]
        
        return '. '.join(capitalized_sentences)
    
    def _clean_transcription(self, text: str) -> str:
        """
        Clean up transcription text for medical use
        
        Args:
            text (str): Raw transcription text
            
        Returns:
            str: Cleaned transcription text
        """
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common punctuation issues
        cleaned = re.sub(r'\s+([,.!?])', r'\1', cleaned)  # Remove space before punctuation
        cleaned = re.sub(r'([.!?])\s*([a-z])', r'\1 \2'.capitalize(), cleaned)  # Capitalize after sentences
        
        # Ensure the transcription ends with appropriate punctuation
        if cleaned and not cleaned[-1] in '.!?':
            cleaned += '.'
        
        return cleaned
    
    def _calculate_transcription_confidence(self, whisper_result: dict, transcription: str) -> float:
        """
        Calculate confidence score for transcription quality
        
        Args:
            whisper_result (dict): Raw Whisper transcription result
            transcription (str): Final processed transcription
            
        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        base_confidence = 0.7  # Base confidence for Whisper
        
        # Factors that increase confidence
        word_count = len(transcription.split())
        if word_count >= 5:
            base_confidence += 0.1  # Bonus for sufficient content
        
        if word_count >= 10:
            base_confidence += 0.1  # Additional bonus for detailed descriptions
            
        # Check for medical terms (indicates relevant content)
        medical_terms = ['pain', 'fever', 'headache', 'cough', 'nausea', 'tired', 'dizzy', 'sore', 'ache']
        medical_term_count = sum(1 for term in medical_terms if term in transcription.lower())
        if medical_term_count > 0:
            base_confidence += min(medical_term_count * 0.05, 0.15)  # Up to 15% bonus
        
        # Penalty for very short transcriptions
        if word_count < 3:
            base_confidence -= 0.2
        
        # Cap confidence at 95% (never 100% certain)
        return min(base_confidence, 0.95)
    
    def speak(self, text: str, wait: bool = True, priority: str = "normal"):
        """
        Convert text to speech with medical optimization
        Uses Coqui TTS for high quality, falls back to pyttsx3
        
        Args:
            text (str): Text to speak
            wait (bool): If True, wait for speech to complete. If False, speak in background
            priority (str): Priority level ("emergency", "high", "normal", "low")
        """
        if not self.coqui_tts and not self.tts_engine:
            print("❌ No TTS engines available - speech skipped")
            return
        
        if not text or not text.strip():
            print("❌ No text provided for speech")
            return
        
        try:
            # Clean and prepare text for medical speech
            speech_text = self._prepare_text_for_medical_speech(text)
            
            if not speech_text or not speech_text.strip():
                print("❌ Text processing resulted in empty speech")
                return
                
        except Exception as e:
            print(f"❌ Error preparing text for speech: {e}")
            speech_text = text  # Fallback to original text
        
        if wait:
            # Synchronous speech (blocks until complete)
            self._speak_synchronous(speech_text)
        else:
            # Asynchronous speech (non-blocking)
            self._speak_asynchronous(speech_text, priority)
    
    def _prepare_text_for_medical_speech(self, text: str) -> str:
        """
        Prepare text for natural and clear medical speech output
        
        Args:
            text (str): Raw text to be spoken
            
        Returns:
            str: Text optimized for speech synthesis
        """
        # Remove markdown formatting that would be spoken literally
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold** -> bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *italic* -> italic
        text = text.replace('#', '').replace('`', '')    # Remove code markers
        
        # Add natural speech patterns to make voice less robotic
        # Replace formal medical language with conversational equivalents
        natural_replacements = {
            'indicates': 'suggests',
            'demonstrates': 'shows',
            'exhibits': 'has',
            'manifests': 'shows signs of',
            'suggests the presence of': 'points to',
            'is consistent with': 'looks like',
            'based on the analysis': 'from what I can see',
            'the assessment reveals': 'it appears that',
            'clinical findings suggest': 'the signs point to',
            'it is recommended': 'I recommend',
            'you should consider': 'you might want to',
            'immediate medical attention': 'seeing a doctor right away',
            'seek medical consultation': 'talk to a healthcare provider'
        }
        
        for formal, casual in natural_replacements.items():
            text = re.sub(r'\b' + re.escape(formal) + r'\b', casual, text, flags=re.IGNORECASE)
        
        # Replace medical emoji with spoken equivalents
        emoji_replacements = {
            "🩺": "Medical diagnosis: ",
            "💊": "Medication recommendation: ",
            "🏥": "Hospital recommendation: ",
            "⚠️": "Important warning: ",
            "🚨": "Emergency alert: ",
            "📊": "Analysis shows ",
            "🤖": "AI analysis indicates ",
            "✅": "Confirmed: ",
            "❌": "Not indicated: ",
            "🔍": "Upon examination: ",
            "📋": "Assessment: ",
            "💡": "Recommendation: ",
            "🎯": "Key finding: "
        }
        
        for emoji, replacement in emoji_replacements.items():
            text = text.replace(emoji, replacement)
        
        # Add natural pauses and rhythm for human-like speech
        # Use longer pauses to prevent rushed sentence endings
        text = text.replace('.', '.   ')   # Longer pause after sentences (3 spaces)
        text = text.replace(', and', ',  and')   # Natural pause before "and"
        text = text.replace(', but', ',  but')   # Natural pause before "but"
        text = text.replace(', so', ',  so')     # Natural pause before "so"
        text = text.replace(', however', ',  however')  # Pause before transitions
        text = text.replace(', therefore', ',  therefore')  # Pause before conclusions
        text = text.replace(':', ':   ')   # Longer pause after colons for emphasis
        text = text.replace(';', ';   ')   # Longer pause after semicolons
        text = text.replace('!', '!   ')   # Longer pause after exclamations
        text = text.replace('?', '?   ')   # Longer pause after questions
        
        # Special handling for sentence endings to prevent rushing
        text = text.replace('. The', '.   The')   # Extra pause before new sentences
        text = text.replace('. You', '.   You')   # Extra pause before addressing user
        text = text.replace('. I', '.   I')       # Extra pause before AI statements
        
        # Add natural conversation starters and transitions
        conversation_starters = {
            'Based on your symptoms': 'Looking at your symptoms',
            'According to the analysis': 'From what I can tell',
            'The diagnosis shows': 'It looks like you have',
            'My assessment is': 'I think you might have',
            'The recommendation is': 'What I\'d suggest is'
        }
        
        for formal, natural in conversation_starters.items():
            text = text.replace(formal, natural)
        
        # Expand medical abbreviations for clarity and better pronunciation
        medical_abbreviations = {
            'BP': 'blood pressure',
            'HR': 'heart rate', 
            'RR': 'respiratory rate',
            'temp': 'temperature',
            'mg': 'milligrams',
            'ml': 'milliliters', 
            'cc': 'cubic centimeters',
            'IV': 'intravenous',
            'IM': 'intramuscular',
            'PO': 'by mouth',
            'bid': 'twice daily',
            'tid': 'three times daily',
            'qid': 'four times daily',
            'OTC': 'over the counter',
            'Rx': 'prescription',
            'dx': 'diagnosis',
            'hx': 'history',
            'sx': 'symptoms',
            'tx': 'treatment',
            'pt': 'patient',
            'Dr.': 'Doctor',
            'vs': 'versus',
            'w/': 'with',
            'w/o': 'without',
            'c/o': 'complains of',
            'SOB': 'shortness of breath',
            'N/V': 'nausea and vomiting',
            'UTI': 'urinary tract infection',
            'URI': 'upper respiratory infection',
            'GI': 'gastrointestinal',
            'CV': 'cardiovascular',
            'resp': 'respiratory'
        }
        
        for abbrev, expansion in medical_abbreviations.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        
        # Improve sentence structure for natural conversation flow
        # Break down long, complex sentences into shorter, more natural ones
        text = self._break_long_sentences(text)
        
        # Add natural conversation fillers and transitions for less robotic speech
        text = self._add_conversation_markers(text)
        
        # Add emphasis for important medical terms with slight pauses
        important_terms = [
            'emergency', 'urgent', 'serious', 'severe', 'critical', 'dangerous',
            'diagnosis', 'medication', 'treatment', 'prescription', 'dosage',
            'symptoms', 'condition', 'infection', 'disease', 'syndrome'
        ]
        
        for term in important_terms:
            # Add slight pause before and after important medical terms
            pattern = r'\b(' + re.escape(term) + r')\b'
            replacement = r' \1 '
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Improve number pronunciation and add natural rhythm for medical context
        # Replace percentages for better speech with natural phrasing
        text = re.sub(r'(\d+)%', r'\1 percent', text)
        text = re.sub(r'(\d+) percent confidence', r'\1 percent confidence, which is pretty good', text)
        
        # Replace decimal numbers for clarity
        text = re.sub(r'(\d+)\.(\d+)', r'\1 point \2', text)
        # Replace ranges for better pronunciation
        text = re.sub(r'(\d+)-(\d+)', r'\1 to \2', text)
        
        # Add natural rhythm patterns and word emphasis
        # Emphasize key medical terms with natural speech patterns
        emphasis_patterns = {
            r'\b(mild|moderate|severe)\b': r'quite \1',
            r'\b(common|rare|typical)\b': r'fairly \1', 
            r'\b(likely|unlikely)\b': r'pretty \1',
            r'\b(recommend|suggest)\b': r'would \1',
            r'\b(important|crucial|vital)\b': r'really \1'
        }
        
        for pattern, replacement in emphasis_patterns.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Add natural speech rhythm with connecting words
        text = re.sub(r'\. ([A-Z])', r'. Now, \1', text)  # Add "Now" for transitions
        text = re.sub(r'\bAlso\b', 'Also,', text)  # Add comma after "Also"
        text = re.sub(r'\bHowever\b', 'However,', text)  # Add comma after "However"
        
        # Clean up extra whitespace (but preserve medical pauses)
        text = re.sub(r' +', ' ', text.strip())
        
        return text
    
    def _break_long_sentences(self, text: str) -> str:
        """
        Break long sentences into shorter, more natural chunks for speech
        
        Args:
            text (str): Input text with potentially long sentences
            
        Returns:
            str: Text with improved sentence structure for natural speech
        """
        # Split sentences that are too long (over 20 words)
        sentences = text.split('. ')
        improved_sentences = []
        
        for sentence in sentences:
            words = sentence.split()
            if len(words) > 20:
                # Find natural break points in long sentences
                break_points = ['and', 'but', 'however', 'therefore', 'also', 'additionally']
                
                for i, word in enumerate(words):
                    if word.lower() in break_points and i > 8:  # Don't break too early
                        # Split at this point
                        first_part = ' '.join(words[:i])
                        second_part = ' '.join(words[i:])
                        improved_sentences.extend([first_part, second_part])
                        break
                else:
                    # No natural break found, keep as is
                    improved_sentences.append(sentence)
            else:
                improved_sentences.append(sentence)
        
        return '. '.join(improved_sentences)
    
    def _add_conversation_markers(self, text: str) -> str:
        """
        Add natural conversation markers to make speech sound less robotic
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with conversation markers for natural flow
        """
        # Add natural introductory phrases
        if text.startswith('Your diagnosis'):
            text = 'So, ' + text.lower()
        elif text.startswith('The analysis'):
            text = 'Well, ' + text.lower()
        elif text.startswith('You have'):
            text = 'It looks like ' + text.lower()
        
        # Add natural transitions between ideas
        text = text.replace('. Additionally', '. Also')
        text = text.replace('. Furthermore', '. Plus')
        text = text.replace('. Moreover', '. And')
        text = text.replace('. In conclusion', '. So overall')
        text = text.replace('. It is important', '. Now, it\'s important')
        
        # Add natural speech patterns and conversational markers
        text = text.replace('You should', 'You might want to')
        text = text.replace('It is necessary', 'You\'ll need to')
        text = text.replace('Please ensure', 'Make sure you')
        text = text.replace('It appears that', 'It looks like')
        text = text.replace('The data suggests', 'From what I can see')
        text = text.replace('In my opinion', 'I think')
        text = text.replace('Based on this information', 'From this')
        
        # Add empathetic conversational markers for medical context
        if 'pain' in text.lower():
            text = text.replace('You have pain', 'I can see you\'re experiencing some pain')
        if 'worry' in text.lower() or 'concern' in text.lower():
            text = text.replace('There is no need to worry', 'Try not to worry too much')
            text = text.replace('This is concerning', 'This is something we should keep an eye on')
        
        # Add natural conversation enders
        if text.strip().endswith('.'):
            if 'recommend' in text.lower() or 'suggest' in text.lower():
                text = text.rstrip('.') + ', okay?'
            elif 'mild' in text.lower() or 'minor' in text.lower():
                text = text.rstrip('.') + ', so that\'s good news.'
        
        return text
    
    def _speak_synchronous(self, text: str):
        """
        Speak text synchronously (blocking)
        Uses Coqui TTS if available, falls back to pyttsx3
        
        Args:
            text (str): Text to speak
        """
        if not text or not text.strip():
            print("❌ No text to speak")
            return
            
        try:
            self.is_speaking = True
            
            # Validate text length to prevent extremely long speech
            if len(text) > 5000:  # Limit to ~5000 characters
                text = text[:5000] + "... text truncated for speech."
                print("⚠️ Text truncated for speech synthesis")
            
            # Try Coqui TTS first (higher quality)
            if self.coqui_tts:
                try:
                    print("🎵 Using Coqui TTS (high-quality)")
                    # Generate audio to temporary file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                        temp_audio_path = temp_audio.name
                    
                    # Generate speech with Coqui TTS
                    self.coqui_tts.tts_to_file(text=text, file_path=temp_audio_path)
                    
                    # Play the generated audio
                    self._play_audio_file(temp_audio_path)
                    
                    # Clean up temporary file safely
                    try:
                        import time
                        time.sleep(0.1)  # Small delay to ensure audio playback is complete
                        if os.path.exists(temp_audio_path):
                            os.remove(temp_audio_path)
                    except Exception as cleanup_error:
                        print(f"⚠️ Warning: Could not clean up temporary audio file: {cleanup_error}")
                        # Schedule cleanup for later if immediate cleanup fails
                        threading.Timer(5.0, lambda: self._delayed_cleanup(temp_audio_path)).start()
                    
                    self.is_speaking = False
                    return
                    
                except Exception as coqui_error:
                    print(f"⚠️ Coqui TTS failed: {coqui_error}, falling back to pyttsx3...")
            
            # Fallback to pyttsx3
            if self.tts_engine:
                print("🔊 Using pyttsx3 (backup)")
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            else:
                print("❌ No TTS engines available")
                
            self.is_speaking = False
            
        except RuntimeError as e:
            print(f"❌ TTS runtime error: {e}")
            if "driver" in str(e).lower():
                print("💡 Try restarting the application or checking audio drivers")
            self.is_speaking = False
        except OSError as e:
            print(f"❌ TTS audio system error: {e}")
            print("💡 Check that speakers/audio output is working")
            self.is_speaking = False
        except Exception as e:
            print(f"❌ Unexpected TTS error: {e}")
            print(f"❌ Error type: {type(e).__name__}")
            self.is_speaking = False
    
    def _speak_asynchronous(self, text: str, priority: str = "normal"):
        """
        Speak text asynchronously (non-blocking)
        
        Args:
            text (str): Text to speak
            priority (str): Priority level for queue management
        """
        # Add to appropriate queue based on priority
        if priority == "emergency":
            # Emergency messages go to front of queue
            temp_queue = queue.Queue()
            temp_queue.put((text, priority))
            while not self.tts_queue.empty():
                temp_queue.put(self.tts_queue.get())
            self.tts_queue = temp_queue
        else:
            self.tts_queue.put((text, priority))
        
        # Start TTS worker thread if not already running
        if not self.tts_thread or not self.tts_thread.is_alive():
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()
    
    def _tts_worker(self):
        """
        Background worker thread for asynchronous TTS processing
        """
        while not self.tts_queue.empty():
            try:
                text, priority = self.tts_queue.get(timeout=1)
                self._speak_synchronous(text)
                self.tts_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                print(f"❌ TTS worker error: {e}")
    
    def _play_audio_file(self, audio_file_path: str):
        """
        Play audio file using system audio player
        Cross-platform audio playback for Coqui TTS generated files
        
        Args:
            audio_file_path (str): Path to the audio file to play
        """
        try:
            import platform
            import subprocess
            
            system = platform.system().lower()
            
            if system == "windows":
                # Windows: use built-in audio player
                import winsound
                winsound.PlaySound(audio_file_path, winsound.SND_FILENAME)
            elif system == "darwin":  # macOS
                subprocess.run(["afplay", audio_file_path], check=True)
            elif system == "linux":
                # Try multiple Linux audio players in order of preference
                audio_players = [
                    ("aplay", ["-q"]),           # ALSA player (most common)
                    ("paplay", []),              # PulseAudio player
                    ("ffplay", ["-nodisp", "-autoexit", "-v", "quiet"]),  # FFmpeg player
                    ("play", []),                # SoX player
                    ("mplayer", ["-really-quiet"]),  # MPlayer
                ]
                played = False
                for player, args in audio_players:
                    try:
                        cmd = [player] + args + [audio_file_path]
                        subprocess.run(cmd, check=True, 
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
                        played = True
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                        continue
                
                if not played:
                    print("⚠️ No suitable audio player found on Linux.")
                    print("💡 Install one of: sudo apt-get install alsa-utils pulseaudio-utils ffmpeg sox mplayer")
            else:
                print(f"⚠️ Unsupported system for audio playback: {system}")
                
        except ImportError as e:
            print(f"⚠️ Audio playback dependency missing: {e}")
        except Exception as e:
            print(f"⚠️ Audio playback error: {e}")
    
    def _delayed_cleanup(self, file_path: str):
        """
        Delayed cleanup of temporary audio files
        
        Args:
            file_path (str): Path to file to clean up
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🧹 Delayed cleanup successful: {file_path}")
        except Exception as e:
            print(f"⚠️ Delayed cleanup also failed: {e}")
    
    def record_audio(self, duration: int = 15, auto_stop: bool = True) -> Optional[str]:
        """
        Record audio from microphone with voice activity detection
        
        Args:
            duration (int): Maximum recording duration in seconds
            auto_stop (bool): Automatically stop when silence is detected
            
        Returns:
            Optional[str]: Path to recorded audio file or None if failed
        """
        if not self.recognizer or not self.microphone:
            print("❌ Microphone system not available")
            return None
        
        try:
            with self.microphone as source:
                print("🎤 Calibrating microphone for optimal recording...")
                
                # Get baseline noise level
                print("🔇 Measuring background noise... (stay quiet for 1 second)")
                initial_energy = self.recognizer.energy_threshold
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                adjusted_energy = self.recognizer.energy_threshold
                
                print(f"🔊 Energy threshold adjusted: {initial_energy} → {adjusted_energy}")
                
                # If the adjustment made it too sensitive or not sensitive enough, override
                if adjusted_energy > 1000:  # Too high, won't pick up normal speech
                    self.recognizer.energy_threshold = 300
                    print("🔧 Threshold too high, setting to 300 for better sensitivity")
                elif adjusted_energy < 100:  # Too low, will pick up everything
                    self.recognizer.energy_threshold = 200  
                    print("🔧 Threshold too low, setting to 200 to reduce noise")
                
                print(f"🔴 RECORDING ACTIVE! (up to {duration} seconds, or until 2 seconds of silence)")
                print("💬 Describe your symptoms: 'I have chest pain and shortness of breath'") 
                print("🎯 Speak normally into your microphone - recording will auto-stop when you finish")
                print(f"📊 Sensitivity level: {self.recognizer.energy_threshold}")
                print("⏸️ The recording stops automatically after 2 seconds of silence")
                
                # Record with improved settings for better audio capture
                # Give user plenty of time to start speaking and describe symptoms
                audio = self.recognizer.listen(
                    source, 
                    timeout=5,  # Give user 5 seconds to start speaking
                    phrase_time_limit=duration  # Allow full duration for complete description
                )
                
                print("✅ Recording complete!")
                
                # Save to temporary file with error handling
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        audio_data = audio.get_wav_data()
                        if len(audio_data) < 1000:  # Less than ~0.1 seconds of audio
                            print("❌ Recorded audio too short - no speech detected")
                            return None
                        temp_file.write(audio_data)
                        temp_file_path = temp_file.name
                    
                    # Verify file was written correctly
                    if os.path.exists(temp_file_path) and os.path.getsize(temp_file_path) > 0:
                        print(f"💾 Audio saved successfully: {temp_file_path} ({os.path.getsize(temp_file_path)} bytes)")
                        return temp_file_path
                    else:
                        print("❌ Failed to save audio file")
                        return None
                        
                except Exception as save_error:
                    print(f"❌ Error saving audio: {save_error}")
                    return None
                    
        except sr.WaitTimeoutError:
            print("❌ No speech detected within time limit - try speaking immediately after recording starts")
            return None
        except sr.RequestError as e:
            print(f"❌ Could not request results from speech recognition service: {e}")
            return None
        except OSError as e:
            print(f"❌ Microphone access error: {e}")
            print("💡 Check microphone permissions and ensure it's not being used by another app")
            return None
        except Exception as e:
            print(f"❌ Recording error: {e}")
            print("💡 Try restarting the application or checking your microphone setup")
            return None
    
    def stop_speaking(self):
        """
        Stop any ongoing speech immediately
        """
        if self.tts_engine:
            try:
                self.tts_engine.stop()
                print("🔇 Speech stopped")
            except:
                pass
        
        # Clear the TTS queue
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
            except queue.Empty:
                break
        
        self.is_speaking = False
    
    def is_system_ready(self) -> Dict[str, bool]:
        """
        Check if all voice system components are ready
        
        Returns:
            Dict[str, bool]: Status of each component
        """
        return {
            "whisper_stt": self.whisper_model is not None,
            "coqui_tts": self.coqui_tts is not None,
            "pyttsx3_tts": self.tts_engine is not None,
            "microphone": self.recognizer is not None and self.microphone is not None,
            "overall_ready": all([
                self.whisper_model is not None,
                (self.coqui_tts is not None or self.tts_engine is not None),
                self.recognizer is not None
            ])
        }
    
    def get_system_info(self) -> Dict[str, str]:
        """
        Get information about the voice system configuration
        
        Returns:
            Dict[str, str]: System configuration information
        """
        info = {
            "whisper_model": "small (244MB)" if self.whisper_model else "Not loaded",
            "primary_tts": "Coqui TTS (high-quality)" if self.coqui_tts else "pyttsx3 (basic)" if self.tts_engine else "Not available",
            "backup_tts": "pyttsx3" if self.coqui_tts and self.tts_engine else "None",
            "audio_enhancement": "Enabled" if self.audio_enhancement_enabled else "Disabled",
            "medical_corrections": f"{len(self.medical_corrections)} terms loaded"
        }
        
        if self.tts_engine:
            try:
                voices = self.tts_engine.getProperty('voices')
                current_voice = self.tts_engine.getProperty('voice')
                info["available_voices"] = len(voices)
                info["current_voice"] = "Medical optimized"
            except:
                info["voice_info"] = "Default system voice"
        
        return info
    
    def get_detailed_system_status(self) -> Dict[str, any]:
        """
        Get detailed system status for troubleshooting
        
        Returns:
            Dict[str, any]: Comprehensive system status information
        """
        status = {
            'timestamp': time.time(),
            'overall_ready': False,
            'components': {},
            'errors': [],
            'warnings': [],
            'recommendations': [],
            'platform_info': {}
        }
        
        # Add platform information
        try:
            import platform
            status['platform_info'] = {
                'system': platform.system(),
                'platform': platform.platform(),
                'python_version': platform.python_version()
            }
        except Exception:
            pass
        
        # Check Whisper STT
        try:
            if self.whisper_model:
                status['components']['whisper_stt'] = {
                    'status': 'ready',
                    'model_size': 'small',
                    'memory_usage': 'unknown'
                }
            else:
                status['components']['whisper_stt'] = {'status': 'failed', 'reason': 'Model not loaded'}
                status['errors'].append('Whisper model not available')
                status['recommendations'].append('Install: pip install openai-whisper torch')
        except Exception as e:
            status['components']['whisper_stt'] = {'status': 'error', 'error': str(e)}
            status['errors'].append(f'Whisper error: {e}')
        
        # Check Coqui TTS (primary)
        try:
            if self.coqui_tts:
                status['components']['coqui_tts'] = {
                    'status': 'ready',
                    'engine': 'Coqui TTS',
                    'quality': 'high'
                }
            else:
                if COQUI_AVAILABLE:
                    status['components']['coqui_tts'] = {'status': 'failed', 'reason': 'Model loading failed'}
                    status['warnings'].append('Coqui TTS model failed to load - using fallback')
                else:
                    status['components']['coqui_tts'] = {'status': 'not_installed', 'reason': 'Package not available'}
                    status['recommendations'].append('Install high-quality TTS: pip install coqui-tts')
        except Exception as e:
            status['components']['coqui_tts'] = {'status': 'error', 'error': str(e)}
            status['errors'].append(f'Coqui TTS error: {e}')

        # Check pyttsx3 TTS (fallback)
        try:
            if self.tts_engine:
                voices = self.tts_engine.getProperty('voices') if self.tts_engine else []
                status['components']['pyttsx3_tts'] = {
                    'status': 'ready',
                    'engine': 'pyttsx3',
                    'voices_available': len(voices) if voices else 0,
                    'role': 'primary' if not self.coqui_tts else 'backup'
                }
            else:
                status['components']['pyttsx3_tts'] = {'status': 'failed', 'reason': 'Engine not initialized'}
                if not self.coqui_tts:  # Only error if no TTS available at all
                    status['errors'].append('No TTS engines available')
                    status['recommendations'].append('Install: pip install pyttsx3')
        except Exception as e:
            status['components']['pyttsx3_tts'] = {'status': 'error', 'error': str(e)}
            status['errors'].append(f'pyttsx3 error: {e}')
        
        # Check Microphone
        try:
            if self.recognizer and self.microphone:
                status['components']['microphone'] = {
                    'status': 'ready',
                    'energy_threshold': self.recognizer.energy_threshold,
                    'dynamic_threshold': self.recognizer.dynamic_energy_threshold
                }
            else:
                status['components']['microphone'] = {'status': 'failed', 'reason': 'Not initialized'}
                status['errors'].append('Microphone system not available')
                status['recommendations'].append('Install: pip install pyaudio speech_recognition')
        except Exception as e:
            status['components']['microphone'] = {'status': 'error', 'error': str(e)}
            status['errors'].append(f'Microphone error: {e}')
        
        # Overall status
        ready_components = sum(1 for comp in status['components'].values() if comp.get('status') == 'ready')
        total_components = len(status['components'])
        status['overall_ready'] = ready_components == total_components and total_components > 0
        status['readiness_score'] = ready_components / total_components if total_components > 0 else 0.0
        
        # Add warnings for partial functionality
        if ready_components > 0 and ready_components < total_components:
            status['warnings'].append('System partially functional - some features may not work')
        
        return status
    
    def cleanup(self):
        """
        Clean up voice system resources
        """
        print("🧹 Starting voice system cleanup...")
        
        # Stop any ongoing speech
        try:
            self.stop_speaking()
        except Exception as e:
            print(f"⚠️ Error stopping speech: {e}")
        
        # Clean up TTS engine
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except Exception as e:
                print(f"⚠️ Error stopping TTS engine: {e}")
        
        # Clean up any temporary files
        try:
            import tempfile
            import glob
            temp_dir = tempfile.gettempdir()
            temp_audio_files = glob.glob(os.path.join(temp_dir, "tmp*.wav"))
            for file_path in temp_audio_files:
                try:
                    os.remove(file_path)
                except:
                    pass
            if temp_audio_files:
                print(f"🗑️ Cleaned up {len(temp_audio_files)} temporary audio files")
        except Exception as e:
            print(f"⚠️ Error cleaning temporary files: {e}")
        
        print("✅ Voice system cleanup complete")