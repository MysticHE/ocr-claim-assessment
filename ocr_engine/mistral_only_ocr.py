import os
import base64
import time
from typing import Dict, List, Any, Optional
from PIL import Image
import io

try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    print("Mistral AI not available. Install with: pip install mistralai")

from config.settings import Config

class MistralOnlyOCREngine:
    """Streamlined OCR engine using only Mistral AI for maximum performance"""
    
    def __init__(self):
        """Initialize Mistral-only OCR engine"""
        self.api_key = Config.MISTRAL_API_KEY
        self.client = None
        
        if not MISTRAL_AVAILABLE:
            print("⚠️  Mistral AI library not available")
            self.mistral_available = False
            return
        
        if not self.api_key:
            print("⚠️  MISTRAL_API_KEY environment variable not set")
            self.mistral_available = False
            return
        
        try:
            self.client = Mistral(api_key=self.api_key)
            self.mistral_available = True
            print("✓ Mistral client initialized successfully")
        except Exception as e:
            print(f"⚠️  Failed to initialize Mistral client: {e}")
            self.mistral_available = False
            self.client = None
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 string for Mistral API"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if too large (Mistral has size limits)
                max_size = 1024
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # Convert to base64
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                buffer.seek(0)
                
                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return f"data:image/jpeg;base64,{image_data}"
                
        except Exception as e:
            raise ValueError(f"Error encoding image: {str(e)}")
    
    def create_ocr_prompt(self, languages: List[str], include_structure: bool = True) -> str:
        """Create OCR prompt for Mistral"""
        language_list = ", ".join(languages) if len(languages) > 1 else languages[0] if languages else "English"
        
        base_prompt = f"""
Extract all text from this image accurately. The text may be in {language_list}.

Instructions:
1. Extract ALL visible text, including headers, body text, numbers, labels, and any other textual content
2. Maintain the original text structure and formatting as much as possible
3. If text appears to be in multiple languages, identify and extract all of them
4. Include any numbers, dates, amounts, codes, or identifiers you see
5. For forms or structured documents, preserve the relationship between labels and values
6. If text is unclear or partially obscured, make your best attempt and indicate uncertainty with [unclear] or [partially visible]
"""
        
        if include_structure:
            base_prompt += """
7. If the document appears to be a medical claim, invoice, or official document, pay special attention to:
   - Patient/claimant names and IDs
   - Diagnosis codes and descriptions
   - Treatment dates and details
   - Monetary amounts and currencies
   - Doctor/provider information
   - Policy or reference numbers

Format your response as clean, readable text. If there are multiple sections, separate them clearly.
"""
        
        return base_prompt.strip()
    
    def extract_text_with_mistral(self, image_path: str, languages: List[str]) -> Dict[str, Any]:
        """Extract text using Mistral AI vision model"""
        try:
            start_time = time.time()
            
            # Encode image
            base64_image = self.encode_image_to_base64(image_path)
            
            # Create prompt
            prompt = self.create_ocr_prompt(languages)
            
            # Make API call
            response = self.client.chat.complete(
                model="pixtral-12b-2409",
                messages=[
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": base64_image
                            }
                        ]
                    }
                ]
            )
            
            # Extract text from response
            extracted_text = response.choices[0].message.content.strip()
            processing_time = (time.time() - start_time) * 1000
            
            # Calculate confidence and detect language
            confidence = self._calculate_confidence(extracted_text)
            detected_language = self._detect_language(extracted_text)
            
            return {
                'success': True,
                'text': extracted_text,
                'confidence': confidence,
                'language': detected_language,
                'processing_time_ms': int(processing_time),
                'engine': 'mistral',
                'word_count': len(extracted_text.split()) if extracted_text else 0
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'confidence': 0.0,
                'language': languages[0] if languages else 'en',
                'processing_time_ms': int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0,
                'engine': 'mistral',
                'error_type': 'api_error'
            }
    
    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence score based on text characteristics"""
        if not text:
            return 0.0
        
        confidence = 0.6  # Higher base confidence for Mistral AI
        
        # Increase confidence for longer text
        if len(text) > 100:
            confidence += 0.2
        elif len(text) > 50:
            confidence += 0.1
        
        # Increase confidence for structured text
        if any(indicator in text.lower() for indicator in ['date:', 'amount:', 'patient:', 'diagnosis:', '$', '#']):
            confidence += 0.15
        
        # Decrease confidence if uncertainty markers present
        uncertainty_markers = ['[unclear]', '[partially visible]', '???', 'unclear', 'illegible']
        uncertainty_count = sum(1 for marker in uncertainty_markers if marker in text.lower())
        confidence -= uncertainty_count * 0.1
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on text characteristics"""
        if not text:
            return 'en'
        
        # Check for Chinese characters
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return 'zh'
        
        # Check for Korean characters
        if any('\uac00' <= char <= '\ud7af' for char in text):
            return 'ko'
        
        # Check for Japanese characters
        if any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in text):
            return 'ja'
        
        # Check for Arabic characters
        if any('\u0600' <= char <= '\u06ff' for char in text):
            return 'ar'
        
        # Check for Thai characters
        if any('\u0e00' <= char <= '\u0e7f' for char in text):
            return 'th'
        
        # Check for Tamil characters
        if any('\u0b80' <= char <= '\u0bff' for char in text):
            return 'ta'
        
        # Default to English
        return 'en'
    
    def process_image(self, image_path: str, languages: List[str]) -> Dict[str, Any]:
        """Main method to process image with Mistral OCR only"""
        try:
            # Validate input
            if not os.path.exists(image_path):
                return {
                    'success': False,
                    'error': 'Image file not found',
                    'text': '',
                    'confidence': 0.0,
                    'engine': 'mistral',
                    'error_type': 'file_not_found'
                }
            
            if not self.mistral_available:
                return {
                    'success': False,
                    'error': 'Mistral AI is not available. Please check your API key and internet connection.',
                    'text': '',
                    'confidence': 0.0,
                    'engine': 'mistral',
                    'error_type': 'service_unavailable'
                }
            
            if not languages:
                languages = ['en']  # Default to English
            
            # Process with Mistral
            result = self.extract_text_with_mistral(image_path, languages)
            
            # Add metadata
            result['total_languages_attempted'] = len(languages)
            result['requested_languages'] = languages
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Processing failed: {str(e)}',
                'text': '',
                'confidence': 0.0,
                'engine': 'mistral',
                'error_type': 'processing_error'
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the OCR engine"""
        try:
            # Test Mistral AI connection with a simple request
            from PIL import Image
            import tempfile
            
            # Create a simple test image with text
            test_image = Image.new('RGB', (200, 100), color='white')
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(test_image)
            draw.text((10, 30), "TEST", fill='black')
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                test_image.save(tmp_file.name)
                
                start_time = time.time()
                
                # Test the connection
                result = self.extract_text_with_mistral(tmp_file.name, ['en'])
                
                response_time = int((time.time() - start_time) * 1000)
                
                # Clean up test image
                os.unlink(tmp_file.name)
                
                return {
                    'status': 'healthy' if result.get('success', False) else 'unhealthy',
                    'mistral_available': self.mistral_available,
                    'api_key_configured': bool(self.api_key),
                    'response_time_ms': response_time,
                    'test_result': result.get('success', False),
                    'error': result.get('error') if not result.get('success', False) else None,
                    'timestamp': time.time()
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'mistral_available': self.mistral_available,
                'api_key_configured': bool(self.api_key),
                'error': str(e),
                'timestamp': time.time()
            }

# Alias for backward compatibility
StreamlinedOCREngine = MistralOnlyOCREngine