

"""
Medical Image Analysis Expert Module
====================================

This module provides AI-powered analysis of medical images using lightweight
vision models. It combines image-to-text generation with medical context
analysis to provide clinically relevant image descriptions.

Key Features:
- Lightweight vision models for fast processing
- Medical context integration
- Structured medical image reporting
- Fallback model support for robustness
- Comprehensive medical disclaimers

Models Used:
- Primary: Salesforce/blip-image-captioning-base (990MB)
- Fallback: microsoft/git-base-coco (500MB)

IMPORTANT: AI image analysis is for educational purposes only and cannot
replace professional medical image interpretation by qualified radiologists
or healthcare providers.
"""

import torch
from transformers import pipeline
from PIL import Image

class ImageAnalysisExpert:
    """
    Medical image analysis expert using vision-language models
    
    This class analyzes medical images and provides structured medical
    descriptions with appropriate clinical context and disclaimers.
    """
    def __init__(self):
        """
        Initialize the medical image analysis expert with fallback models
        
        This method attempts to load vision-language models in order of preference:
        1. Primary: BLIP (Salesforce/blip-image-captioning-base) - 990MB
        2. Fallback: GIT (microsoft/git-base-coco) - 500MB
        3. None: Graceful degradation if both models fail
        
        The models are loaded with float16 precision to reduce memory usage
        while maintaining acceptable accuracy for medical image analysis.
        """
        print("🔄 Loading Medical Image Analysis Expert (Optimized)...")
        try:
            # Primary model: BLIP (Bootstrapping Language-Image Pre-training)
            # Using lighter BLIP model for faster loading (990MB vs 5.4GB)
            # This model provides good balance of accuracy and performance
            self.image_analyzer = pipeline(
                "image-to-text", 
                model="Salesforce/blip-image-captioning-base",
                model_kwargs={"torch_dtype": torch.float16}  # Half precision for memory efficiency
            )
            print("✅ Medical Image Analysis Expert (BLIP-Fast) loaded successfully!")
            self.model_type = "blip"
        except Exception as e:
            print(f"❌ Error loading BLIP. Trying ultra-light alternative: {e}")
            try:
                # Fallback model: GIT (GenerativeImage2Text) 
                # Ultra-lightweight fallback (500MB) for resource-constrained environments
                # Provides basic image captioning capabilities
                self.image_analyzer = pipeline(
                    "image-to-text", 
                    model="microsoft/git-base-coco",
                    model_kwargs={"torch_dtype": torch.float16}
                )
                print("✅ Ultra-light Image Analysis Expert loaded!")
                self.model_type = "git"
            except Exception as e2:
                # Complete failure - system will operate without image analysis
                print(f"❌ Complete failure loading image models: {e2}")
                self.image_analyzer = None
                self.model_type = None

    def analyze_image(self, image: Image.Image) -> str:
        """
        Analyze medical images and provide structured clinical descriptions
        
        This method processes medical images through vision-language models
        to generate clinically relevant descriptions with medical context
        and appropriate safety disclaimers.
        
        Args:
            image (Image.Image): PIL Image object containing the medical image
            
        Returns:
            str: Structured medical image analysis with clinical formatting
                 and safety disclaimers, or error message if analysis fails
        """
        # Check if image analysis models are available
        if not self.image_analyzer:
            return "Medical image analysis is currently unavailable."
        
        try:
            print("🔍 Analyzing medical image...")
            
            # Generate basic image description using vision-language model
            # This provides the raw visual analysis of the medical image
            result = self.image_analyzer(image)
            base_description = result[0]["generated_text"]
            
            # Prepare description for medical structuring
            # This combines the raw AI output with medical context
            combined = f"Medical image analysis: {base_description}"
            
            # Structure the analysis with medical formatting and keywords
            # This adds clinical context and safety disclaimers
            structured_analysis = self._structure_medical_analysis(combined)
            
            print("✅ Medical image analysis completed!")
            return structured_analysis
            
        except Exception as e:
            # Handle any errors during image processing gracefully
            print(f"Error during medical image analysis: {e}")
            return "I encountered an error trying to analyze the medical image."

    def _structure_medical_analysis(self, raw_description: str) -> str:
        """
        Structure the image analysis in a professional medical format
        
        This method transforms raw AI image descriptions into structured medical
        reports with clinical formatting, keyword extraction, and safety disclaimers.
        It identifies anatomical structures and potential abnormalities mentioned
        in the description for better clinical presentation.
        
        Args:
            raw_description (str): Raw image description from the vision model
            
        Returns:
            str: Professionally formatted medical image analysis report
        """
        
        # Define medical keyword databases for clinical categorization
        # These keywords help identify relevant medical content in the image description
        anatomical_keywords = ["chest", "lung", "heart", "bone", "skull", "spine", "joint", "organ", "tissue", "skin"]
        abnormal_keywords = ["lesion", "mass", "fracture", "inflammation", "swelling", "abnormal", "irregular", "shadow", "opacity"]
        
        # Initialize the structured medical report with professional formatting
        analysis = f"🩺 **Medical Image Analysis:**\n\n"
        analysis += f"**Visual Description:** {raw_description}\n\n"
        
        # Identify and highlight anatomical structures mentioned in the description
        # This helps clinicians quickly identify what body systems are visible
        found_anatomy = [kw for kw in anatomical_keywords if kw.lower() in raw_description.lower()]
        if found_anatomy:
            analysis += f"**Anatomical Structures Identified:** {', '.join(found_anatomy)}\n\n"
        
        # Identify and highlight potential abnormal findings
        # This draws attention to possible pathological findings
        found_abnormal = [kw for kw in abnormal_keywords if kw.lower() in raw_description.lower()]
        if found_abnormal:
            analysis += f"**Potential Findings:** {', '.join(found_abnormal)}\n\n"
        
        # Add mandatory medical disclaimer for safety and legal compliance
        # This is critical for educational AI systems providing medical information
        analysis += "**⚠️ Medical Disclaimer:** This is an AI-generated image description for educational purposes only. "
        analysis += "Professional medical interpretation by qualified healthcare providers is required for clinical decisions."
        
        return analysis

