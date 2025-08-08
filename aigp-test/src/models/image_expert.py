

import torch
from transformers import pipeline
from PIL import Image

class ImageAnalysisExpert:
    """Analyzes medical images using optimized lightweight vision models."""
    def __init__(self):
        print("🔄 Loading Medical Image Analysis Expert (Optimized)...")
        try:
            # Using lighter BLIP model for faster loading (990MB vs 5.4GB)
            self.image_analyzer = pipeline(
                "image-to-text", 
                model="Salesforce/blip-image-captioning-base",
                model_kwargs={"torch_dtype": torch.float16}
            )
            print("✅ Medical Image Analysis Expert (BLIP-Fast) loaded successfully!")
            self.model_type = "blip"
        except Exception as e:
            print(f"❌ Error loading BLIP. Trying ultra-light alternative: {e}")
            try:
                # Ultra-lightweight fallback (500MB)
                self.image_analyzer = pipeline(
                    "image-to-text", 
                    model="microsoft/git-base-coco",
                    model_kwargs={"torch_dtype": torch.float16}
                )
                print("✅ Ultra-light Image Analysis Expert loaded!")
                self.model_type = "git"
            except Exception as e2:
                print(f"❌ Complete failure loading image models: {e2}")
                self.image_analyzer = None
                self.model_type = None

    def analyze_image(self, image: Image.Image) -> str:
        if not self.image_analyzer:
            return "Medical image analysis is currently unavailable."
        
        try:
            print("🔍 Analyzing medical image...")
            
            # Basic analysis - faster with lighter models
            result = self.image_analyzer(image)
            base_description = result[0]["generated_text"]
            
            # Enhanced analysis for better medical context
            combined = f"Medical image analysis: {base_description}"
            
            # Add medical context and structure the response
            structured_analysis = self._structure_medical_analysis(combined)
            
            print("✅ Medical image analysis completed!")
            return structured_analysis
            
        except Exception as e:
            print(f"Error during medical image analysis: {e}")
            return "I encountered an error trying to analyze the medical image."

    def _structure_medical_analysis(self, raw_description: str) -> str:
        """Structure the image analysis in a medical format"""
        
        # Define medical keywords for categorization
        anatomical_keywords = ["chest", "lung", "heart", "bone", "skull", "spine", "joint", "organ", "tissue", "skin"]
        abnormal_keywords = ["lesion", "mass", "fracture", "inflammation", "swelling", "abnormal", "irregular", "shadow", "opacity"]
        
        analysis = f"🩺 **Medical Image Analysis:**\n\n"
        analysis += f"**Visual Description:** {raw_description}\n\n"
        
        # Check for anatomical structures
        found_anatomy = [kw for kw in anatomical_keywords if kw.lower() in raw_description.lower()]
        if found_anatomy:
            analysis += f"**Anatomical Structures Identified:** {', '.join(found_anatomy)}\n\n"
        
        # Check for potential abnormalities
        found_abnormal = [kw for kw in abnormal_keywords if kw.lower() in raw_description.lower()]
        if found_abnormal:
            analysis += f"**Potential Findings:** {', '.join(found_abnormal)}\n\n"
        
        # Add standard medical disclaimer
        analysis += "**⚠️ Medical Disclaimer:** This is an AI-generated image description for educational purposes only. "
        analysis += "Professional medical interpretation by qualified healthcare providers is required for clinical decisions."
        
        return analysis

