"""
Medical Image Upload Handler
============================

This module provides comprehensive medical image upload handling with security validation,
preprocessing, and user experience enhancements for the AI GP Doctor system.

Key Features:
- Security validation (file type, size, integrity, basic malware detection)
- Image preprocessing for optimal AI analysis
- Error handling with user-friendly messages
- Support for multiple medical image formats
- Image quality enhancement and optimization
- Batch upload capabilities

Security Features:
- File type validation with magic number checking
- Size limits to prevent DoS attacks
- Basic malware scanning using file signatures
- Image integrity verification
- Sanitization of EXIF data for privacy

Author: AI GP Doctor Development Team
License: Open Source for Educational Use
"""

import os
import mimetypes
import hashlib
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import tempfile
import io

# Try to import python-magic, fallback to mimetypes if not available
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    print("⚠️ python-magic not available - using basic file type detection")

class MedicalImageHandler:
    """
    Comprehensive medical image upload handler with security and preprocessing

    This class handles all aspects of medical image upload including validation,
    security checks, preprocessing, and optimization for AI analysis.
    """

    def __init__(self):
        """Initialize the medical image handler with security and processing settings"""

        # Security settings
        self.max_file_size = 15 * 1024 * 1024  # 15MB max file size
        self.min_file_size = 1024  # 1KB minimum (prevents empty files)

        # Supported medical image formats
        self.allowed_formats = {
            'JPEG': ['.jpg', '.jpeg'],
            'PNG': ['.png'],
            'TIFF': ['.tiff', '.tif'],  # Medical DICOM exports often use TIFF
            'BMP': ['.bmp'],
            'WEBP': ['.webp']
        }

        # MIME types for validation
        self.allowed_mime_types = {
            'image/jpeg',
            'image/png',
            'image/tiff',
            'image/bmp',
            'image/webp'
        }

        # Malware signature patterns (basic detection)
        self.suspicious_patterns = [
            b'<script',  # Embedded scripts
            b'javascript:',  # JavaScript protocols
            b'<?php',  # PHP code
            b'<%',  # ASP/JSP code
            b'eval(',  # Code evaluation
        ]

        # Image processing settings for medical analysis
        self.target_size = (512, 512)  # Optimal size for AI models
        self.min_dimensions = (100, 100)  # Minimum useful size
        self.max_dimensions = (4096, 4096)  # Maximum to prevent memory issues

        print("🛡️ Medical Image Handler initialized with security validation")

    def validate_and_process_image(self, file_path: str) -> Dict[str, Union[str, bool, Image.Image]]:
        """
        Complete validation and processing pipeline for medical images

        Args:
            file_path (str): Path to the uploaded image file

        Returns:
            Dict containing validation results and processed image
        """
        result = {
            'is_valid': False,
            'processed_image': None,
            'error_message': '',
            'warnings': [],
            'file_info': {},
            'security_checks': {}
        }

        try:
            # Step 1: Basic file validation
            file_validation = self._validate_file_basics(file_path)
            if not file_validation['is_valid']:
                result['error_message'] = file_validation['error']
                return result

            result['file_info'] = file_validation['info']

            # Step 2: Security validation
            security_validation = self._validate_security(file_path)
            if not security_validation['is_valid']:
                result['error_message'] = security_validation['error']
                result['security_checks'] = security_validation
                return result

            result['security_checks'] = security_validation

            # Step 3: Image format and integrity validation
            image_validation = self._validate_image_integrity(file_path)
            if not image_validation['is_valid']:
                result['error_message'] = image_validation['error']
                return result

            # Step 4: Load and preprocess image
            preprocessing_result = self._preprocess_medical_image(file_path)
            if not preprocessing_result['is_valid']:
                result['error_message'] = preprocessing_result['error']
                return result

            # Success - image is valid and processed
            result['is_valid'] = True
            result['processed_image'] = preprocessing_result['image']
            result['warnings'] = preprocessing_result.get('warnings', [])

            print(f"✅ Medical image validated and processed successfully")

        except Exception as e:
            result['error_message'] = f"Unexpected error during image processing: {str(e)}"
            print(f"❌ Image processing error: {e}")

        return result

    def _validate_file_basics(self, file_path: str) -> Dict[str, Union[bool, str, dict]]:
        """Validate basic file properties"""

        if not os.path.exists(file_path):
            return {'is_valid': False, 'error': 'File not found'}

        if not os.path.isfile(file_path):
            return {'is_valid': False, 'error': 'Path is not a file'}

        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size < self.min_file_size:
            return {'is_valid': False, 'error': 'File too small - may be corrupted'}

        if file_size > self.max_file_size:
            size_mb = file_size / (1024 * 1024)
            return {'is_valid': False, 'error': f'File too large ({size_mb:.1f}MB). Maximum allowed: 15MB'}

        # Check file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        allowed_extensions = []
        for format_exts in self.allowed_formats.values():
            allowed_extensions.extend(format_exts)

        if file_ext not in allowed_extensions:
            return {'is_valid': False, 'error': f'Unsupported file format. Allowed: {", ".join(allowed_extensions)}'}

        # Get file info
        file_info = {
            'size_bytes': file_size,
            'size_mb': round(file_size / (1024 * 1024), 2),
            'extension': file_ext,
            'filename': os.path.basename(file_path)
        }

        return {'is_valid': True, 'info': file_info}

    def _validate_security(self, file_path: str) -> Dict[str, Union[bool, str]]:
        """Perform security validation on the uploaded file"""

        try:
            # Check MIME type using python-magic if available, otherwise use mimetypes
            if MAGIC_AVAILABLE:
                try:
                    file_mime = magic.from_file(file_path, mime=True)
                    if file_mime not in self.allowed_mime_types:
                        return {'is_valid': False, 'error': f'File type not allowed: {file_mime}'}
                except Exception as e:
                    print(f"⚠️ Magic detection failed: {e}, using fallback")
                    file_mime, _ = mimetypes.guess_type(file_path)
                    if file_mime not in self.allowed_mime_types:
                        return {'is_valid': False, 'error': f'Cannot verify file type: {file_mime}'}
            else:
                # Fallback to mimetypes if python-magic is not available
                file_mime, _ = mimetypes.guess_type(file_path)
                if file_mime not in self.allowed_mime_types:
                    return {'is_valid': False, 'error': f'File type not supported: {file_mime}'}

            # Basic malware scanning - check for suspicious patterns
            with open(file_path, 'rb') as f:
                file_content = f.read(8192)  # Read first 8KB for pattern matching

                for pattern in self.suspicious_patterns:
                    if pattern in file_content:
                        return {'is_valid': False, 'error': 'File contains suspicious content and cannot be processed'}

            # Calculate file hash for integrity
            file_hash = self._calculate_file_hash(file_path)

            return {
                'is_valid': True,
                'mime_type': file_mime,
                'file_hash': file_hash,
                'security_scan': 'passed'
            }

        except Exception as e:
            return {'is_valid': False, 'error': f'Security validation failed: {str(e)}'}

    def _validate_image_integrity(self, file_path: str) -> Dict[str, Union[bool, str]]:
        """Validate that the file is a valid, uncorrupted image"""

        try:
            # Try to open and verify the image
            with Image.open(file_path) as img:
                # Verify image can be loaded
                img.verify()

            # Re-open for dimension checks (verify() closes the file)
            with Image.open(file_path) as img:
                width, height = img.size

                # Check minimum dimensions
                if width < self.min_dimensions[0] or height < self.min_dimensions[1]:
                    return {'is_valid': False, 'error': f'Image too small ({width}x{height}). Minimum: {self.min_dimensions[0]}x{self.min_dimensions[1]}'}

                # Check maximum dimensions
                if width > self.max_dimensions[0] or height > self.max_dimensions[1]:
                    return {'is_valid': False, 'error': f'Image too large ({width}x{height}). Maximum: {self.max_dimensions[0]}x{self.max_dimensions[1]}'}

                # Check if image has reasonable aspect ratio (not extremely thin/wide)
                aspect_ratio = max(width, height) / min(width, height)
                if aspect_ratio > 10:  # Very extreme aspect ratio
                    return {'is_valid': False, 'error': 'Image aspect ratio too extreme - may not be a medical image'}

            return {'is_valid': True}

        except Image.UnidentifiedImageError:
            return {'is_valid': False, 'error': 'File is not a valid image or is corrupted'}
        except Exception as e:
            return {'is_valid': False, 'error': f'Image validation failed: {str(e)}'}

    def _preprocess_medical_image(self, file_path: str) -> Dict[str, Union[bool, str, Image.Image, list]]:
        """Preprocess medical image for optimal AI analysis"""

        try:
            warnings = []

            # Load image
            with Image.open(file_path) as img:
                # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
                if img.mode != 'RGB':
                    if img.mode == 'RGBA':
                        # Handle transparency by adding white background
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                        img = background
                        warnings.append('Image had transparency - converted to RGB with white background')
                    else:
                        img = img.convert('RGB')
                        warnings.append(f'Converted from {img.mode} to RGB for processing')

                # Remove EXIF data for privacy (creates a copy without metadata)
                data = list(img.getdata())
                img_without_exif = Image.new(img.mode, img.size)
                img_without_exif.putdata(data)
                img = img_without_exif

                # Enhance image quality for medical analysis
                img = self._enhance_medical_image(img)

                # Resize to optimal dimensions if needed
                original_size = img.size
                if img.size != self.target_size:
                    # Use high-quality resampling for medical images
                    img = img.resize(self.target_size, Image.Resampling.LANCZOS)
                    warnings.append(f'Resized from {original_size[0]}x{original_size[1]} to {self.target_size[0]}x{self.target_size[1]} for optimal AI analysis')

                return {
                    'is_valid': True,
                    'image': img,
                    'warnings': warnings
                }

        except Exception as e:
            return {'is_valid': False, 'error': f'Image preprocessing failed: {str(e)}'}

    def _enhance_medical_image(self, img: Image.Image) -> Image.Image:
        """Enhance medical image quality for better AI analysis"""

        try:
            # Enhance contrast slightly for better feature detection
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.1)  # 10% contrast boost

            # Enhance sharpness slightly for edge detection
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.05)  # 5% sharpness boost

            # Apply gentle noise reduction
            img = img.filter(ImageFilter.SMOOTH_MORE)

            return img

        except Exception:
            # If enhancement fails, return original image
            return img

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file for integrity verification"""

        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def get_supported_formats_info(self) -> Dict[str, list]:
        """Get information about supported image formats"""
        return {
            'formats': self.allowed_formats,
            'max_size_mb': self.max_file_size // (1024 * 1024),
            'optimal_dimensions': self.target_size,
            'max_dimensions': self.max_dimensions
        }

    def batch_validate_images(self, file_paths: List[str]) -> Dict[str, dict]:
        """Validate multiple images at once (for batch upload)"""

        results = {}
        for i, file_path in enumerate(file_paths):
            results[f'image_{i+1}'] = self.validate_and_process_image(file_path)

        return results