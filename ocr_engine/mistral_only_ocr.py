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
            print("Warning: Mistral AI library not available")
            self.mistral_available = False
            return
        
        if not self.api_key:
            print("Warning: MISTRAL_API_KEY environment variable not set")
            self.mistral_available = False
            return
        
        try:
            self.client = Mistral(api_key=self.api_key)
            self.mistral_available = True
            print("✓ Mistral client initialized successfully")
            
            # Test API connection
            self._test_api_connection()
        except Exception as e:
            print(f"Warning: Failed to initialize Mistral client: {e}")
            self.mistral_available = False
            self.client = None
    
    def _test_api_connection(self):
        """Test basic API connection without making a full request"""
        try:
            # Simple test - just check if we can create a client and it's valid
            if self.client and self.api_key:
                print("✓ Mistral API connection test passed")
        except Exception as e:
            print(f"Warning: Mistral API connection test failed: {e}")
            self.mistral_available = False
    
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
    
    def _encode_image_to_base64_for_ocr(self, image_path: str) -> str:
        """Encode image to base64 string for Mistral OCR API following official documentation"""
        try:
            # Follow Mistral documentation format exactly
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
                
                # Determine format based on file extension
                file_extension = image_path.lower().split('.')[-1]
                if file_extension in ['jpg', 'jpeg']:
                    return f"data:image/jpeg;base64,{image_data}"
                elif file_extension == 'png':
                    return f"data:image/png;base64,{image_data}"
                else:
                    # Default to PNG for other image formats
                    return f"data:image/png;base64,{image_data}"
                
        except Exception as e:
            raise ValueError(f"Error encoding image for OCR: {str(e)}")
    
    def _encode_pdf_to_base64(self, pdf_path: str) -> str:
        """Encode PDF file to base64 string for Mistral OCR API"""
        try:
            with open(pdf_path, 'rb') as pdf_file:
                pdf_data = pdf_file.read()
                pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
                return f"data:application/pdf;base64,{pdf_base64}"
                
        except Exception as e:
            raise ValueError(f"Error encoding PDF: {str(e)}")
    
    def create_ocr_prompt(self, languages: List[str], include_structure: bool = True) -> str:
        """Create OCR prompt for Mistral with English output requirement"""
        
        base_prompt = """
Extract all text from this image accurately and provide the output in English.

Instructions:
1. Extract ALL visible text, including headers, body text, numbers, labels, and any other textual content
2. If the original text is in a non-English language, translate it to English while preserving meaning
3. Maintain the original document structure and formatting as much as possible
4. Include any numbers, dates, amounts, codes, or identifiers you see (keep these in their original format)
5. For forms or structured documents, preserve the relationship between labels and values
6. If text is unclear or partially obscured, make your best attempt and indicate uncertainty with [unclear] or [partially visible]
7. IMPORTANT: Always provide the final output in English, regardless of the source language
"""
        
        if include_structure:
            base_prompt += """
8. If the document appears to be a medical claim, invoice, or official document, pay special attention to:
   - Patient/claimant names and IDs
   - Diagnosis codes and descriptions (translate descriptions to English)
   - Treatment dates and details
   - Monetary amounts and currencies
   - Doctor/provider information
   - Policy or reference numbers

Format your response as clean, readable English text. If there are multiple sections, separate them clearly.
Translate any non-English content to English while preserving the document structure.
"""
        
        return base_prompt.strip()
    
    def extract_text_with_mistral(self, file_path: str, languages: List[str]) -> Dict[str, Any]:
        """Extract text using Mistral OCR API for both PDFs and images"""
        try:
            start_time = time.time()
            
            # Use OCR API for both PDFs and images
            return self._extract_with_ocr_api(file_path, languages, start_time)
                
        except Exception as e:
            return self._handle_extraction_error(e, languages, start_time)
    
    def _extract_with_ocr_api(self, file_path: str, languages: List[str], start_time: float) -> Dict[str, Any]:
        """Extract text from both images and PDFs using Mistral OCR API following official documentation"""
        try:
            # Determine file type and encode appropriately
            file_extension = file_path.lower().split('.')[-1]
            
            if file_extension == 'pdf':
                # Encode PDF to base64
                document_base64 = self._encode_pdf_to_base64(file_path)
                print(f"PDF encoded - length: {len(document_base64)}, prefix: {document_base64[:50]}...")
            else:
                # Encode image to base64 following Mistral documentation format
                document_base64 = self._encode_image_to_base64_for_ocr(file_path)
                print(f"Image encoded - length: {len(document_base64)}, prefix: {document_base64[:50]}...")
            
            # Make API call to Mistral OCR endpoint following official documentation
            response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": document_base64
                },
                include_image_base64=True
            )
            
            # Extract text from OCR response
            extracted_text = ""
            if response.pages:
                # Combine text from all pages
                extracted_text = "\n\n".join([
                    page.markdown for page in response.pages if hasattr(page, 'markdown') and page.markdown
                ])
            
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
                'engine': 'mistral_ocr',
                'word_count': len(extracted_text.split()) if extracted_text else 0
            }
            
        except Exception as e:
            # Add detailed error logging for debugging OCR API issues
            error_details = {
                'error_message': str(e),
                'error_type': type(e).__name__,
                'file_extension': file_path.lower().split('.')[-1] if file_path else 'unknown'
            }
            print(f"OCR API Error Details: {error_details}")
            return self._handle_extraction_error(e, languages, start_time)
    
    def _extract_from_image_with_ocr_focus(self, image_path: str, languages: List[str], start_time: float) -> Dict[str, Any]:
        """Extract text from images using Mistral Vision API with OCR-focused prompts for raw text extraction"""
        try:
            # Encode image for Vision API (standard quality is sufficient for Vision API)
            base64_image = self.encode_image_to_base64(image_path)
            
            # Use OCR-focused prompt for raw text extraction
            prompt = self.create_ocr_prompt(languages, include_structure=True)
            
            # Make API call to Vision API with OCR-focused configuration
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
                                "image_url": {
                                    "url": base64_image
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,  # Low temperature for consistent OCR extraction
                max_tokens=2048   # Sufficient for most OCR text
            )
            
            # Extract and clean the text response
            extracted_text = response.choices[0].message.content.strip()
            
            # Clean up conversational elements that might appear in Vision API responses
            extracted_text = self._clean_vision_response_for_ocr(extracted_text)
            
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
                'engine': 'mistral_vision_ocr',
                'word_count': len(extracted_text.split()) if extracted_text else 0
            }
            
        except Exception as e:
            return self._handle_extraction_error(e, languages, start_time)
    
    def _clean_vision_response_for_ocr(self, text: str) -> str:
        """Clean Vision API response to extract only the OCR text content"""
        if not text:
            return text
        
        # Remove common conversational prefixes that Vision API might add
        conversational_prefixes = [
            "The text in this image says:",
            "The text in the image is:",
            "I can see the following text:",
            "The extracted text is:",
            "Here is the text from the image:",
            "The text content is:",
            "Text extracted:",
            "The document contains:",
            "I can read:"
        ]
        
        cleaned_text = text
        for prefix in conversational_prefixes:
            if cleaned_text.lower().startswith(prefix.lower()):
                cleaned_text = cleaned_text[len(prefix):].strip()
                break
        
        # Remove quotes if the entire response is wrapped in quotes
        if ((cleaned_text.startswith('"') and cleaned_text.endswith('"')) or 
            (cleaned_text.startswith("'") and cleaned_text.endswith("'"))):
            cleaned_text = cleaned_text[1:-1].strip()
        
        return cleaned_text
    
    def _extract_from_image_with_vision_api(self, image_path: str, languages: List[str], start_time: float) -> Dict[str, Any]:
        """Extract text from images using Mistral Vision API (Pixtral)"""
        try:
            # Encode image
            base64_image = self.encode_image_to_base64(image_path)
            
            # Create prompt
            prompt = self.create_ocr_prompt(languages)
            
            # Make API call according to Mistral API docs
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
                                "image_url": {
                                    "url": base64_image
                                }
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
            error_message = str(e)
            error_type = 'api_error'
            
            # Categorize specific errors
            if 'api_key' in error_message.lower() or 'authentication' in error_message.lower():
                error_type = 'authentication_error'
                error_message = "Invalid or missing Mistral API key"
            elif 'rate limit' in error_message.lower() or 'quota' in error_message.lower():
                error_type = 'rate_limit_error'
                error_message = "Mistral API rate limit exceeded"
            elif 'network' in error_message.lower() or 'connection' in error_message.lower():
                error_type = 'network_error'
                error_message = "Network connection error to Mistral API"
            elif 'model' in error_message.lower():
                error_type = 'model_error'
                error_message = "Mistral model error - check model name"
            
            print(f"Mistral API Error: {error_message} (Type: {error_type})")
            
            return {
                'success': False,
                'error': error_message,
                'text': '',
                'confidence': 0.0,
                'language': languages[0] if languages else 'en',
                'processing_time_ms': int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0,
                'engine': 'mistral',
                'error_type': error_type
            }
    
    def _handle_extraction_error(self, e: Exception, languages: List[str], start_time: float) -> Dict[str, Any]:
        """Handle extraction errors with categorization"""
        error_message = str(e)
        error_type = 'api_error'
        
        # Categorize specific errors
        if 'api_key' in error_message.lower() or 'authentication' in error_message.lower():
            error_type = 'authentication_error'
            error_message = "Invalid or missing Mistral API key"
        elif 'rate limit' in error_message.lower() or 'quota' in error_message.lower():
            error_type = 'rate_limit_error'
            error_message = "Mistral API rate limit exceeded"
        elif 'network' in error_message.lower() or 'connection' in error_message.lower():
            error_type = 'network_error'
            error_message = "Network connection error to Mistral API"
        elif 'model' in error_message.lower():
            error_type = 'model_error'
            error_message = "Mistral model error - check model name"
        elif 'file' in error_message.lower():
            error_type = 'file_error'
            error_message = "File processing error - check file format and accessibility"
        
        print(f"Mistral API Error: {error_message} (Type: {error_type})")
        
        return {
            'success': False,
            'error': error_message,
            'text': '',
            'confidence': 0.0,
            'language': languages[0] if languages else 'en',
            'processing_time_ms': int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0,
            'engine': 'mistral',
            'error_type': error_type
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
                languages = []  # Auto-detection mode with English output
            
            # Process with Mistral
            result = self.extract_text_with_mistral(image_path, languages)
            
            # Add metadata
            result['auto_detection_mode'] = len(languages) == 0
            result['output_language'] = 'en'  # Always English output
            result['total_languages_attempted'] = len(languages) if languages else 'auto'
            
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