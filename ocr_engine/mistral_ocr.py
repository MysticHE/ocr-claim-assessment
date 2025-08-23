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

class MistralOCREngine:
    """Mistral AI vision-based OCR engine for enhanced text extraction"""
    
    def __init__(self):
        """Initialize Mistral OCR engine"""
        self.api_key = Config.MISTRAL_API_KEY
        
        if not MISTRAL_AVAILABLE:
            raise ImportError("Mistral AI is not installed. Please install with: pip install mistralai")
        
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is required")
        
        try:
            self.client = Mistral(api_key=self.api_key)
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Mistral client: {str(e)}")
    
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
                model="mistral-small-latest",  # Use appropriate vision model when available
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
            extracted_text = ""
            if response.choices and len(response.choices) > 0:
                extracted_text = response.choices[0].message.content.strip()
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Calculate confidence based on response quality
            confidence = self._calculate_confidence(extracted_text)
            
            return {
                'success': True,
                'text': extracted_text,
                'confidence': confidence,
                'language': languages[0] if languages else 'en',
                'detected_language': self._detect_language(extracted_text),
                'processing_time_ms': processing_time,
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
                'engine': 'mistral'
            }
    
    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence score based on text characteristics"""
        if not text:
            return 0.0
        
        confidence = 0.5  # Base confidence
        
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
        
        # Simple heuristic-based language detection
        # This is basic - in production, you might want to use a proper language detection library
        
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
        """Main method to process image with Mistral OCR"""
        try:
            # Validate input
            if not os.path.exists(image_path):
                return {
                    'success': False,
                    'error': 'Image file not found',
                    'text': '',
                    'confidence': 0.0,
                    'engine': 'mistral'
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
                'error': f'Unexpected error during Mistral OCR processing: {str(e)}',
                'text': '',
                'confidence': 0.0,
                'detected_language': languages[0] if languages else 'en',
                'engine': 'mistral'
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on Mistral OCR engine"""
        try:
            # Test API connection with a simple request
            response = self.client.chat.complete(
                model="mistral-small-latest",
                messages=[
                    {
                        "role": "user",
                        "content": "Hello, this is a health check."
                    }
                ],
                max_tokens=10
            )
            
            api_healthy = bool(response and response.choices)
            
            return {
                'status': 'healthy' if api_healthy else 'unhealthy',
                'mistral_available': MISTRAL_AVAILABLE,
                'api_key_configured': bool(self.api_key),
                'api_connection': api_healthy,
                'model': 'mistral-small-latest'
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'mistral_available': MISTRAL_AVAILABLE,
                'api_key_configured': bool(self.api_key),
                'api_connection': False
            }

class HybridOCREngine:
    """Hybrid OCR engine that combines EasyOCR and Mistral AI for best results"""
    
    def __init__(self):
        """Initialize hybrid OCR engine with lazy EasyOCR loading"""
        self.mistral_available = False
        self.easyocr_reader = None
        self.easyocr_initialization_attempted = False
        
        # Initialize Mistral OCR (primary engine)
        try:
            self.mistral_engine = MistralOCREngine()
            self.mistral_available = True
        except Exception as e:
            print(f"Mistral OCR not available in hybrid engine: {e}")
            self.mistral_engine = None
        
        # Don't initialize EasyOCR here - use lazy loading to save memory
        # EasyOCR will be initialized only when Mistral AI fails
        
        if not self.mistral_available:
            print("Warning: Primary OCR engine (Mistral AI) not available. EasyOCR will be used as fallback.")
    
    def process_image(self, image_path: str, languages: List[str], 
                     use_both: bool = True) -> Dict[str, Any]:
        """Process image using available OCR engines"""
        results = {}
        
        # Try Mistral OCR first (primary engine)
        if self.mistral_available:
            mistral_result = self.mistral_engine.process_image(image_path, languages)
            results['mistral'] = mistral_result
            
            # If Mistral succeeded and we don't need both engines, return result
            if mistral_result.get('success', False) and not use_both:
                return self._combine_results(results, image_path, languages)
        
        # Try EasyOCR as fallback only if Mistral failed or is unavailable
        need_easyocr = (
            not self.mistral_available or 
            (self.mistral_available and not results.get('mistral', {}).get('success', False)) or
            use_both
        )
        
        if need_easyocr:
            # Lazy initialization of EasyOCR
            if not self._ensure_easyocr_initialized():
                # EasyOCR initialization failed, return Mistral result if available
                if results:
                    return self._combine_results(results, image_path, languages)
                else:
                    return {
                        'success': False,
                        'error': 'No OCR engines available',
                        'text': '',
                        'confidence': 0.0,
                        'engine': 'hybrid'
                    }
            
            easyocr_result = self._process_with_easyocr(image_path, languages)
            results['easyocr'] = easyocr_result
        
        # Combine results or return best available
        return self._combine_results(results, image_path, languages)
    
    def _ensure_easyocr_initialized(self) -> bool:
        """Lazy initialization of EasyOCR reader"""
        if self.easyocr_reader is not None:
            return True
        
        if self.easyocr_initialization_attempted:
            return False
        
        self.easyocr_initialization_attempted = True
        
        try:
            print("Initializing EasyOCR (this may take a moment to download models)...")
            import easyocr
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
            print("EasyOCR initialized successfully")
            return True
        except Exception as e:
            print(f"Failed to initialize EasyOCR: {e}")
            self.easyocr_reader = None
            return False
    
    def _process_with_easyocr(self, image_path: str, languages: List[str]) -> Dict[str, Any]:
        """Process image with EasyOCR"""
        import time
        
        try:
            start_time = time.time()
            
            # EasyOCR expects language codes like ['en', 'ch_sim', 'ko']
            easyocr_langs = self._convert_to_easyocr_langs(languages)
            
            # Create reader with specified languages if different from current
            if set(easyocr_langs) != set(['en']):
                try:
                    import easyocr
                    reader = easyocr.Reader(easyocr_langs, gpu=False)
                except:
                    reader = self.easyocr_reader  # Fallback to default reader
            else:
                reader = self.easyocr_reader
            
            # Process image
            results = reader.readtext(image_path)
            
            # Extract text from results
            extracted_texts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                if text.strip():
                    extracted_texts.append(text.strip())
                    confidences.append(confidence)
            
            full_text = ' '.join(extracted_texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return {
                'success': True,
                'text': full_text,
                'confidence': avg_confidence,
                'language': languages[0] if languages else 'en',
                'detected_language': self._detect_language(full_text),
                'processing_time_ms': processing_time,
                'engine': 'easyocr',
                'word_count': len(full_text.split()) if full_text else 0
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'confidence': 0.0,
                'language': languages[0] if languages else 'en',
                'processing_time_ms': int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0,
                'engine': 'easyocr'
            }
    
    def _convert_to_easyocr_langs(self, languages: List[str]) -> List[str]:
        """Convert language codes to EasyOCR format"""
        lang_map = {
            'en': 'en',
            'zh': 'ch_sim',
            'zh-CN': 'ch_sim', 
            'zh-TW': 'ch_tra',
            'ja': 'ja',
            'ko': 'ko',
            'ar': 'ar',
            'th': 'th',
            'ta': 'ta',
            'es': 'es',
            'fr': 'fr',
            'de': 'de',
            'it': 'it',
            'pt': 'pt',
            'ru': 'ru'
        }
        
        easyocr_langs = []
        for lang in languages:
            mapped_lang = lang_map.get(lang.lower(), 'en')
            if mapped_lang not in easyocr_langs:
                easyocr_langs.append(mapped_lang)
        
        return easyocr_langs or ['en']
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on text characteristics"""
        if not text:
            return 'en'
        
        # Simple heuristic-based language detection
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return 'zh'
        if any('\uac00' <= char <= '\ud7af' for char in text):
            return 'ko'  
        if any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in text):
            return 'ja'
        if any('\u0600' <= char <= '\u06ff' for char in text):
            return 'ar'
        if any('\u0e00' <= char <= '\u0e7f' for char in text):
            return 'th'
        if any('\u0b80' <= char <= '\u0bff' for char in text):
            return 'ta'
        
        return 'en'
    
    def _combine_results(self, results: Dict[str, Any], image_path: str, 
                        languages: List[str]) -> Dict[str, Any]:
        """Combine results from multiple OCR engines"""
        if not results:
            return {
                'success': False,
                'error': 'No OCR engines available',
                'text': '',
                'confidence': 0.0,
                'engine': 'hybrid'
            }
        
        # If only one engine available, return its result
        if len(results) == 1:
            result = list(results.values())[0]
            result['engine'] = 'hybrid'
            result['engines_used'] = list(results.keys())
            return result
        
        # Compare results and choose the best one
        best_result = None
        best_score = 0
        
        for engine_name, result in results.items():
            if result.get('success', False):
                # Score based on confidence and text length
                score = (result.get('confidence', 0) * 0.7 + 
                        (min(len(result.get('text', '')), 1000) / 1000) * 0.3)
                
                if score > best_score:
                    best_score = score
                    best_result = result
                    best_result['primary_engine'] = engine_name
        
        if best_result:
            best_result['engine'] = 'hybrid'
            best_result['engines_used'] = list(results.keys())
            best_result['all_results'] = results
            return best_result
        
        # If all failed, return the first error
        first_result = list(results.values())[0]
        first_result['engine'] = 'hybrid'
        first_result['engines_used'] = list(results.keys())
        return first_result