#!/usr/bin/env python3
"""
Voice System Testing Script
===========================

This script tests the complete voice capabilities of the AI GP Doctor system
including Speech-to-Text (STT) with Whisper, Text-to-Speech (TTS) with pyttsx3,
and the integrated voice diagnosis pipeline.

Test Coverage:
- Voice system initialization and component loading
- Text-to-Speech functionality with medical optimization
- Audio recording and microphone setup
- Speech-to-Text transcription with medical context
- Complete voice diagnosis pipeline
- Error handling and fallback mechanisms

Usage:
    python test_voice.py

Requirements:
    - All voice system dependencies installed (see requirements.txt)
    - Working microphone for audio recording tests
    - Audio output (speakers/headphones) for TTS tests
"""

import sys
import os
import time
import tempfile

# Add src directory to path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_voice_system_components():
    """Test individual voice system components"""
    print("=" * 60)
    print("🎤 TESTING VOICE SYSTEM COMPONENTS")
    print("=" * 60)
    
    try:
        from src.models.voice_system import VoiceSystem
        
        print("✅ Voice system module imported successfully")
        
        # Initialize voice system
        print("\n🔄 Initializing voice system...")
        voice = VoiceSystem()
        
        # Check system status
        print("\n📊 Voice system status:")
        status = voice.is_system_ready()
        for component, ready in status.items():
            status_icon = "✅" if ready else "❌"
            print(f"  {status_icon} {component}: {'Ready' if ready else 'Not available'}")
        
        # Get system information
        print("\n📋 Voice system configuration:")
        info = voice.get_system_info()
        for key, value in info.items():
            print(f"  • {key}: {value}")
        
        return voice if status['overall_ready'] else None
        
    except ImportError as e:
        print(f"❌ Error importing voice system: {e}")
        return None
    except Exception as e:
        print(f"❌ Error initializing voice system: {e}")
        return None

def test_text_to_speech(voice_system):
    """Test Text-to-Speech functionality"""
    print("\n" + "=" * 60)
    print("🔊 TESTING TEXT-TO-SPEECH (TTS)")
    print("=" * 60)
    
    if not voice_system:
        print("❌ Voice system not available - skipping TTS tests")
        return
    
    # Test basic TTS
    print("\n🔊 Testing basic TTS...")
    test_message = "Hello, I am your AI GP Doctor. How can I help you today?"
    voice_system.speak(test_message, wait=True)
    print("✅ Basic TTS test completed")
    
    # Test medical TTS
    print("\n🔊 Testing medical TTS with diagnosis...")
    medical_message = """
    Based on your symptoms, my analysis suggests a viral upper respiratory infection 
    with 85 percent confidence. This appears to be a mild condition. 
    Please remember, this is an AI assessment for educational purposes only.
    """
    voice_system.speak(medical_message, wait=True)
    print("✅ Medical TTS test completed")
    
    # Test asynchronous TTS
    print("\n🔊 Testing asynchronous TTS...")
    voice_system.speak("This message is spoken asynchronously.", wait=False)
    time.sleep(2)  # Give time for async speech
    print("✅ Asynchronous TTS test completed")

def test_audio_recording(voice_system):
    """Test audio recording functionality"""
    print("\n" + "=" * 60)
    print("🎙️ TESTING AUDIO RECORDING")
    print("=" * 60)
    
    if not voice_system:
        print("❌ Voice system not available - skipping recording tests")
        return None
    
    print("\n🎤 Testing microphone setup...")
    if not voice_system.recognizer or not voice_system.microphone:
        print("❌ Microphone system not available")
        return None
    
    print("✅ Microphone system ready")
    
    # Interactive recording test
    print("\n🎤 Interactive recording test")
    print("📝 Instructions:")
    print("   - Speak clearly into your microphone")
    print("   - Describe medical symptoms (e.g., 'I have a headache and fever')")
    print("   - Recording will last up to 5 seconds")
    
    user_input = input("\n🤔 Press Enter to start recording, or 's' to skip: ")
    if user_input.lower() == 's':
        print("⏭️ Skipping recording test")
        return None
    
    print("\n🔴 Recording in 3 seconds...")
    time.sleep(1)
    print("🔴 Recording in 2 seconds...")
    time.sleep(1)
    print("🔴 Recording in 1 second...")
    time.sleep(1)
    
    # Record audio
    audio_file = voice_system.record_audio(duration=5)
    
    if audio_file:
        print(f"✅ Recording successful! Audio saved to: {audio_file}")
        return audio_file
    else:
        print("❌ Recording failed")
        return None

def test_speech_to_text(voice_system, audio_file=None):
    """Test Speech-to-Text transcription"""
    print("\n" + "=" * 60)
    print("🎯 TESTING SPEECH-TO-TEXT (STT)")
    print("=" * 60)
    
    if not voice_system:
        print("❌ Voice system not available - skipping STT tests")
        return
    
    if not voice_system.whisper_model:
        print("❌ Whisper model not loaded - skipping STT tests")
        return
    
    if audio_file and os.path.exists(audio_file):
        print(f"\n🔄 Transcribing recorded audio: {audio_file}")
        
        # Test transcription
        transcription, confidence = voice_system.transcribe_audio(audio_file)
        
        print(f"\n📝 Transcription Results:")
        print(f"  • Text: '{transcription}'")
        print(f"  • Confidence: {confidence:.1%}")
        
        if confidence > 0.5:
            print("✅ STT test successful!")
        else:
            print("⚠️ STT test completed but with low confidence")
        
        # Clean up audio file
        try:
            os.remove(audio_file)
            print(f"🧹 Cleaned up audio file: {audio_file}")
        except:
            pass
            
        return transcription, confidence
    else:
        print("⏭️ No audio file available - skipping STT tests")
        return None, 0.0

def test_complete_voice_diagnosis():
    """Test the complete voice diagnosis pipeline"""
    print("\n" + "=" * 60)
    print("🩺 TESTING COMPLETE VOICE DIAGNOSIS PIPELINE")
    print("=" * 60)
    
    try:
        from src.diagnosis_system import DiagnosisSystem
        
        print("🔄 Initializing complete diagnosis system...")
        diagnosis_system = DiagnosisSystem()
        
        # Check voice system status
        voice_status = diagnosis_system.get_voice_system_status()
        print(f"\n📊 Voice system status: {voice_status}")
        
        if not voice_status.get('overall_ready', False):
            print("❌ Voice system not fully ready - skipping pipeline test")
            return
        
        # Test voice diagnosis with simulated audio (if available)
        print("\n🎤 Voice Diagnosis Pipeline Test")
        print("📝 This test would normally:")
        print("   1. Record patient symptoms via microphone")
        print("   2. Transcribe speech to text using Whisper")
        print("   3. Process symptoms through AI diagnosis system")
        print("   4. Generate voice response using TTS")
        
        # Test TTS response formatting
        print("\n🔊 Testing diagnosis speech formatting...")
        
        # Create a sample diagnosis for TTS testing
        sample_diagnosis = {
            'diagnosis': 'Viral Upper Respiratory Infection',
            'confidence': 0.82,
            'severity': 'mild',
            'similar_case': 'Common cold with typical symptoms',
            'reasoning': 'Symptoms consistent with viral infection pattern'
        }
        
        # Test speech formatting
        speech_text = diagnosis_system._format_diagnosis_for_speech(sample_diagnosis)
        print(f"\n📝 Formatted speech response:")
        print(f"'{speech_text}'")
        
        # Speak the diagnosis
        print(f"\n🔊 Speaking formatted diagnosis...")
        diagnosis_system.speak_diagnosis(sample_diagnosis, wait=True)
        
        print("✅ Complete voice diagnosis pipeline test completed!")
        
    except ImportError as e:
        print(f"❌ Error importing diagnosis system: {e}")
    except Exception as e:
        print(f"❌ Error testing voice diagnosis pipeline: {e}")

def test_error_handling():
    """Test error handling and fallback mechanisms"""
    print("\n" + "=" * 60)
    print("⚠️ TESTING ERROR HANDLING")
    print("=" * 60)
    
    try:
        from src.models.voice_system import VoiceSystem
        
        voice_system = VoiceSystem()
        
        # Test transcription with non-existent file
        print("\n🔍 Testing transcription with invalid file...")
        transcription, confidence = voice_system.transcribe_audio("nonexistent_file.wav")
        print(f"  Result: '{transcription}' (confidence: {confidence:.1%})")
        
        # Test TTS with empty text
        print("\n🔊 Testing TTS with empty text...")
        voice_system.speak("", wait=True)
        
        # Test stop speaking
        print("\n🔇 Testing stop speaking functionality...")
        voice_system.speak("This is a long message that should be interrupted.", wait=False)
        time.sleep(1)
        voice_system.stop_speaking()
        
        print("✅ Error handling tests completed")
        
    except Exception as e:
        print(f"❌ Error during error handling tests: {e}")

def run_all_tests():
    """Run all voice system tests"""
    print("🎉 STARTING AI GP DOCTOR VOICE SYSTEM TESTS")
    print("=" * 60)
    
    start_time = time.time()
    
    # Test 1: Component initialization
    voice_system = test_voice_system_components()
    
    # Test 2: Text-to-Speech
    test_text_to_speech(voice_system)
    
    # Test 3: Audio recording
    recorded_audio = test_audio_recording(voice_system)
    
    # Test 4: Speech-to-Text
    transcription, confidence = test_speech_to_text(voice_system, recorded_audio)
    
    # Test 5: Complete voice diagnosis pipeline
    test_complete_voice_diagnosis()
    
    # Test 6: Error handling
    test_error_handling()
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("🎊 VOICE SYSTEM TESTS COMPLETED")
    print("=" * 60)
    print(f"⏱️ Total test duration: {duration:.1f} seconds")
    
    if voice_system:
        print("\n📊 Final System Status:")
        final_status = voice_system.get_system_info()
        for key, value in final_status.items():
            print(f"  • {key}: {value}")
    
    print("\n💡 Next Steps:")
    print("  1. Install any missing dependencies if tests failed")
    print("  2. Test the complete system: python main.py")
    print("  3. Try voice consultation in the web interface")
    print("  4. Report any issues on GitHub")
    
    print(f"\n🎉 Voice system testing complete!")

if __name__ == "__main__":
    # Check if running in proper environment
    if not os.path.exists("requirements.txt"):
        print("❌ Error: Please run this script from the project root directory")
        print("   Usage: python test_voice.py")
        sys.exit(1)
    
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⏹️ Tests interrupted by user")
        print("👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error during testing: {e}")
        print("💡 Please check your installation and try again")
        sys.exit(1)