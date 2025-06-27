
import gradio as gr
from transformers import pipeline
from PIL import Image
import json

class InputProcessor:
    def __init__(self):
        # Whisper for speech-to-text conversion
        self.stt = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

    def process_inputs(self, audio_file, text_input, image_file):
        """Process all input modalities and convert to standardized format"""
        processed_data = {
            'text_data': '',
            'image_data': None,
            'has_audio': False,
            'has_text': False,
            'has_image': False
        }

        # Process audio input
        if audio_file is not None:
            print(f"Processing audio file: {audio_file}")
            try:
                transcription = self.stt(audio_file)["text"]
                processed_data['text_data'] = transcription
                processed_data['has_audio'] = True
                print(f"Audio transcribed to: {transcription}")
            except Exception as e:
                print(f"Error transcribing audio: {e}")
                processed_data['text_data'] = f"Error processing audio: {str(e)}"

        # Process text input
        if text_input and text_input.strip():
            # Combine with audio transcription if both exist
            if processed_data['text_data']:
                processed_data['text_data'] += " " + text_input
                print(f"Combined audio + text: {processed_data['text_data']}")
            else:
                processed_data['text_data'] = text_input
            processed_data['has_text'] = True

        # Process image input
        if image_file is not None:
            processed_data['image_data'] = image_file
            processed_data['has_image'] = True
            print(f"Image processed: {image_file}")

        return processed_data
