import os
import time
from typing import Dict, List, Any, Optional
import numpy as np
from PIL import Image
import cv2

try:
    from paddleocr import PaddleOCR
    PADDLE_OCR_AVAILABLE = True
except ImportError:
    PADDLE_OCR_AVAILABLE = False
    print("PaddleOCR not available. Install with: pip install paddleocr")

from config.settings import Config
from ocr_engine.language_support import LanguageMapper

class PaddleOCREngine:
    """PaddleOCR engine for multi-language text extraction"""
    
    def __init__(self):
        """Initialize PaddleOCR engine"""
        self.language_mapper = LanguageMapper()
        self.ocr_instances = {}  # Cache OCR instances for different languages
        
        if not PADDLE_OCR_AVAILABLE:
            raise ImportError("PaddleOCR is not installed. Please install with: pip install paddleocr")
        
        # Initialize default English OCR
        try:
            self.ocr_instances['en'] = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=False,  # Set to True if GPU available
                show_log=False
            )
        except Exception as e:
            print(f"Warning: Could not initialize default PaddleOCR: {e}")
    
    def _get_ocr_instance(self, language: str) -> Optional[PaddleOCR]:
        """Get or create OCR instance for specific language"""
        # Map common language codes to PaddleOCR format
        paddle_lang = self.language_mapper.to_paddle_format(language)
        
        if paddle_lang not in self.ocr_instances:
            try:
                self.ocr_instances[paddle_lang] = PaddleOCR(
                    use_angle_cls=True,
                    lang=paddle_lang,
                    use_gpu=False,  # Set to True if GPU available
                    show_log=False
                )
            except Exception as e:
                print(f"Error initializing OCR for language {paddle_lang}: {e}")
                # Fallback to English if specific language fails
                if paddle_lang != 'en' and 'en' in self.ocr_instances:
                    print(f"Falling back to English OCR for language {language}")
                    return self.ocr_instances['en']
                return None
        
        return self.ocr_instances[paddle_lang]
    
    def preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Preprocess image for better OCR results"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                # Try with PIL if OpenCV fails
                pil_image = Image.open(image_path)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # Apply threshold
            _, threshold = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return threshold
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def extract_text_single_language(self, image_path: str, language: str) -> Dict[str, Any]:
        """Extract text using single language OCR"""
        try:
            start_time = time.time()
            
            # Get OCR instance for language
            ocr_instance = self._get_ocr_instance(language)
            if not ocr_instance:
                return {
                    'success': False,
                    'error': f'OCR instance not available for language: {language}',
                    'text': '',
                    'confidence': 0.0,
                    'boxes': [],
                    'processing_time_ms': 0
                }
            
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            input_image = processed_image if processed_image is not None else image_path
            
            # Perform OCR
            results = ocr_instance.ocr(input_image, cls=True)
            
            if not results or not results[0]:
                return {
                    'success': True,
                    'text': '',
                    'confidence': 0.0,
                    'boxes': [],
                    'language': language,
                    'processing_time_ms': int((time.time() - start_time) * 1000)
                }
            
            # Parse results
            extracted_text = []
            boxes = []
            confidences = []
            
            for line in results[0]:
                if len(line) >= 2:
                    bbox, (text, confidence) = line
                    extracted_text.append(text)
                    boxes.append({
                        'bbox': bbox,
                        'text': text,
                        'confidence': confidence
                    })
                    confidences.append(confidence)
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Join extracted text
            full_text = '\n'.join(extracted_text)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return {
                'success': True,
                'text': full_text,
                'confidence': avg_confidence,
                'boxes': boxes,
                'language': language,
                'processing_time_ms': processing_time,
                'word_count': len(full_text.split()) if full_text else 0
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'confidence': 0.0,
                'boxes': [],
                'processing_time_ms': int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0
            }
    
    def extract_text_multi_language(self, image_path: str, languages: List[str]) -> Dict[str, Any]:
        """Extract text using multiple languages and combine results"""
        start_time = time.time()
        all_results = {}
        best_result = None
        best_confidence = 0.0
        
        for language in languages:
            result = self.extract_text_single_language(image_path, language)
            all_results[language] = result
            
            if result['success'] and result['confidence'] > best_confidence:
                best_confidence = result['confidence']
                best_result = result
                best_result['detected_language'] = language
        
        if not best_result:
            return {
                'success': False,
                'error': 'No successful OCR results from any language',
                'text': '',
                'confidence': 0.0,
                'boxes': [],
                'detected_language': languages[0] if languages else 'en',
                'processing_time_ms': int((time.time() - start_time) * 1000),
                'all_results': all_results
            }
        
        # Add combined processing time
        best_result['processing_time_ms'] = int((time.time() - start_time) * 1000)
        best_result['all_results'] = all_results
        
        return best_result
    
    def process_image(self, image_path: str, languages: List[str]) -> Dict[str, Any]:
        """Main method to process image with OCR"""
        try:
            # Validate input
            if not os.path.exists(image_path):
                return {
                    'success': False,
                    'error': 'Image file not found',
                    'text': '',
                    'confidence': 0.0
                }
            
            if not languages:
                languages = ['en']  # Default to English
            
            # Validate supported languages
            supported_languages = []
            for lang in languages:
                if self.language_mapper.is_supported(lang):
                    supported_languages.append(lang)
                else:
                    print(f"Warning: Language {lang} not supported, skipping")
            
            if not supported_languages:
                supported_languages = ['en']  # Fallback to English
            
            # Process with single or multiple languages
            if len(supported_languages) == 1:
                result = self.extract_text_single_language(image_path, supported_languages[0])
                result['detected_language'] = supported_languages[0]
            else:
                result = self.extract_text_multi_language(image_path, supported_languages)
            
            # Add metadata
            result['total_languages_attempted'] = len(supported_languages)
            result['requested_languages'] = languages
            result['engine'] = 'paddleocr'
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Unexpected error during OCR processing: {str(e)}',
                'text': '',
                'confidence': 0.0,
                'boxes': [],
                'detected_language': languages[0] if languages else 'en',
                'engine': 'paddleocr'
            }
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return self.language_mapper.get_supported_languages()
    
    def cleanup(self):
        """Cleanup OCR instances to free memory"""
        try:
            for lang, instance in self.ocr_instances.items():
                del instance
            self.ocr_instances.clear()
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on OCR engine"""
        try:
            # Test with a simple image (create a simple test image)
            import tempfile
            
            # Create a simple test image
            test_image = Image.new('RGB', (200, 100), color='white')
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                test_image.save(tmp_file.name)
                
                # Test OCR
                result = self.extract_text_single_language(tmp_file.name, 'en')
                
                # Cleanup test image
                os.unlink(tmp_file.name)
                
                return {
                    'status': 'healthy' if result['success'] else 'unhealthy',
                    'paddle_ocr_available': PADDLE_OCR_AVAILABLE,
                    'cached_languages': list(self.ocr_instances.keys()),
                    'supported_languages_count': len(self.get_supported_languages()),
                    'test_result': result['success']
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'paddle_ocr_available': PADDLE_OCR_AVAILABLE,
                'cached_languages': list(self.ocr_instances.keys()),
                'supported_languages_count': len(self.get_supported_languages()) if hasattr(self, 'language_mapper') else 0
            }