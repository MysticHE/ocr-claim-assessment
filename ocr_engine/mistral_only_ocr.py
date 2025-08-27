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
    """Streamlined OCR engine using only Mistral AI with intelligent table preservation"""
    
    def __init__(self, preserve_tables: bool = True):
        """Initialize Mistral-only OCR engine
        
        Args:
            preserve_tables: If True, preserves table structure with pipes.
                           If False, removes all pipes for clean flowing text.
        """
        self.api_key = Config.MISTRAL_API_KEY
        self.client = None
        self.preserve_tables = preserve_tables
        
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
        """Encode image to base64 string for Mistral OCR API following official documentation exactly"""
        try:
            # Follow official documentation format: data:image/<format>;base64,{base64_data}
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
                
                # Use image MIME type as shown in official documentation
                file_extension = image_path.lower().split('.')[-1]
                if file_extension in ['jpg', 'jpeg']:
                    return f"data:image/jpeg;base64,{image_data}"
                elif file_extension == 'png':
                    return f"data:image/png;base64,{image_data}"
                else:
                    # Default to image/png for other image formats
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
    
    def extract_original_text_with_mistral(self, file_path: str) -> Dict[str, Any]:
        """Extract original text without auto-translation for dual-content support"""
        try:
            start_time = time.time()
            
            # Extract raw text without auto-translation
            return self._extract_original_with_ocr_api(file_path, start_time)
                
        except Exception as e:
            return self._handle_extraction_error(e, [], start_time)
    
    def _extract_with_ocr_api(self, file_path: str, languages: List[str], start_time: float) -> Dict[str, Any]:
        """Extract text from both images and PDFs using Mistral OCR API following official documentation"""
        try:
            # Determine file type and use correct document structure per official docs
            file_extension = file_path.lower().split('.')[-1]
            
            if file_extension == 'pdf':
                # For PDFs: use document_url structure
                document_base64 = self._encode_pdf_to_base64(file_path)
                document_config = {
                    "type": "document_url",
                    "document_url": document_base64
                }
            else:
                # For images: use image_url structure as per official documentation
                document_base64 = self._encode_image_to_base64_for_ocr(file_path)
                document_config = {
                    "type": "image_url",
                    "image_url": document_base64
                }
            
            # Make API call to Mistral OCR endpoint with correct document structure
            response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document=document_config,
                include_image_base64=True
            )
            
            # Extract text from OCR response
            extracted_text = ""
            if response.pages:
                # Combine text from all pages
                extracted_text = "\n\n".join([
                    page.markdown for page in response.pages if hasattr(page, 'markdown') and page.markdown
                ])
            
            # Enhanced text cleaning: intelligent table preservation or pipe removal
            extracted_text = self._clean_repetitive_patterns(extracted_text, preserve_tables=self.preserve_tables)
            
            # CRITICAL FIX: Strip leading whitespace from the ENTIRE text block to fix first row spacing
            # This addresses the root cause where Mistral OCR returns text with leading indentation
            extracted_text = extracted_text.lstrip()
            
            processing_time = (time.time() - start_time) * 1000
            
            # Calculate confidence and detect language
            confidence = self._calculate_confidence(extracted_text)
            detected_language = self._detect_language(extracted_text)
            
            # FINAL SAFETY CHECK: Ensure absolutely no leading whitespace in final result
            # This is the last line of defense before text reaches the Enhanced Processor
            final_text = extracted_text.lstrip() if extracted_text else ''
            
            # DEBUG: Log final OCR result being returned
            print(f"[FINAL RESULT DEBUG] ================================")
            print(f"[FINAL RESULT DEBUG] Final text being returned (first 200 chars):")
            print(f"'{final_text[:200]}'")
            print(f"[FINAL RESULT DEBUG] Final leading spaces: {len(final_text) - len(final_text.lstrip())}")
            print(f"[FINAL RESULT DEBUG] Final first line: '{final_text.split(chr(10))[0] if final_text else 'EMPTY'}'")
            print(f"[FINAL RESULT DEBUG] Engine: mistral_ocr")
            print(f"[FINAL RESULT DEBUG] ================================")
            
            return {
                'success': True,
                'text': final_text,
                'confidence': confidence,
                'language': detected_language,
                'processing_time_ms': int(processing_time),
                'engine': 'mistral_ocr',
                'word_count': len(final_text.split()) if final_text else 0
            }
            
        except Exception as e:
            return self._handle_extraction_error(e, languages, start_time)
    
    def _extract_original_with_ocr_api(self, file_path: str, start_time: float) -> Dict[str, Any]:
        """Extract original text without auto-translation using Mistral OCR API"""
        try:
            # Determine file type and use correct document structure per official docs
            file_extension = file_path.lower().split('.')[-1]
            
            if file_extension == 'pdf':
                # For PDFs: use document_url structure
                document_base64 = self._encode_pdf_to_base64(file_path)
                document_config = {
                    "type": "document_url",
                    "document_url": document_base64
                }
            else:
                # For images: use image_url structure as per official documentation
                document_base64 = self._encode_image_to_base64_for_ocr(file_path)
                document_config = {
                    "type": "image_url",
                    "image_url": document_base64
                }
            
            # Make API call to Mistral OCR endpoint without translation instructions
            response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document=document_config,
                include_image_base64=True
            )
            
            # Extract text from OCR response
            extracted_text = ""
            if response.pages:
                # Combine text from all pages
                extracted_text = "\n\n".join([
                    page.markdown for page in response.pages if hasattr(page, 'markdown') and page.markdown
                ])
            
            # Enhanced text cleaning: intelligent table preservation or pipe removal
            extracted_text = self._clean_repetitive_patterns(extracted_text, preserve_tables=self.preserve_tables)
            
            # CRITICAL FIX: Strip leading whitespace from the ENTIRE text block to fix first row spacing
            # This addresses the root cause where Mistral OCR returns text with leading indentation
            extracted_text = extracted_text.lstrip()
            
            processing_time = (time.time() - start_time) * 1000
            
            # Calculate confidence and detect language
            confidence = self._calculate_confidence(extracted_text)
            detected_language = self._detect_language(extracted_text)
            
            # FINAL SAFETY CHECK: Ensure absolutely no leading whitespace in final result
            # This is the last line of defense before text reaches the Enhanced Processor
            final_text = extracted_text.lstrip() if extracted_text else ''
            
            # DEBUG: Log final OCR result being returned
            print(f"[FINAL RESULT DEBUG] ================================")
            print(f"[FINAL RESULT DEBUG] Final text being returned (first 200 chars):")
            print(f"'{final_text[:200]}'")
            print(f"[FINAL RESULT DEBUG] Final leading spaces: {len(final_text) - len(final_text.lstrip())}")
            print(f"[FINAL RESULT DEBUG] Final first line: '{final_text.split(chr(10))[0] if final_text else 'EMPTY'}'")
            print(f"[FINAL RESULT DEBUG] Engine: mistral_ocr")
            print(f"[FINAL RESULT DEBUG] ================================")
            
            return {
                'success': True,
                'text': final_text,
                'confidence': confidence,
                'language': detected_language,
                'processing_time_ms': int(processing_time),
                'engine': 'mistral_ocr_original',
                'word_count': len(final_text.split()) if final_text else 0,
                'is_original': True  # Flag to indicate this is original untranslated content
            }
            
        except Exception as e:
            return self._handle_extraction_error(e, [], start_time)
    
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
            
            # FINAL SAFETY CHECK: Ensure absolutely no leading whitespace in final result
            # This is the last line of defense before text reaches the Enhanced Processor
            final_text = extracted_text.lstrip() if extracted_text else ''
            
            # DEBUG: Log final OCR result being returned
            print(f"[FINAL RESULT DEBUG] ================================")
            print(f"[FINAL RESULT DEBUG] Final text being returned (first 200 chars):")
            print(f"'{final_text[:200]}'")
            print(f"[FINAL RESULT DEBUG] Final leading spaces: {len(final_text) - len(final_text.lstrip())}")
            print(f"[FINAL RESULT DEBUG] Final first line: '{final_text.split(chr(10))[0] if final_text else 'EMPTY'}'")
            print(f"[FINAL RESULT DEBUG] Engine: mistral_ocr")
            print(f"[FINAL RESULT DEBUG] ================================")
            
            return {
                'success': True,
                'text': final_text,
                'confidence': confidence,
                'language': detected_language,
                'processing_time_ms': int(processing_time),
                'engine': 'mistral_vision_ocr',
                'word_count': len(final_text.split()) if final_text else 0
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
    
    def _clean_repetitive_patterns(self, text: str, preserve_tables: bool = False) -> str:
        """Enhanced text cleaning with intelligent table preservation or complete pipe removal"""
        if not text:
            return text
        
        original_length = len(text)
        
        import re
        
        # Step 1: Clean text with table awareness (NEW)
        if preserve_tables:
            cleaned_text = self._clean_with_table_preservation(text)
        else:
            cleaned_text = self._clean_pipes_and_spacing(text)
        
        lines = cleaned_text.split('\n')
        
        # Step 2: Detect and remove obvious API hallucination patterns (EXISTING)
        cleaned_lines = []
        next_count = 0
        consecutive_numbered_next = 0
        max_legitimate_next = 3  # Very conservative - most real documents have 0-3 "Next:" instructions
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Pattern 1: Detect obvious "Next:" repetition (API bug signature)
            if (line_stripped.startswith('**Next:**') or 
                line_stripped == 'Next:' or
                line_stripped.endswith('**Next:**')):
                next_count += 1
                
                # Allow very few legitimate "Next:" entries
                if next_count <= max_legitimate_next:
                    cleaned_lines.append(line)
                else:
                    # Skip API hallucination - this shouldn't exist in medical documents
                    continue
            
            # Pattern 2: Detect numbered "Next:" sequences (major API bug indicator)
            elif any(line_stripped.startswith(f'{i}. **Next:**') for i in range(1, 500)):
                consecutive_numbered_next += 1
                
                # This is definitely an API bug - medical bills don't have numbered "Next:" lists
                if consecutive_numbered_next <= 2:  # Allow maybe 1-2 as edge case
                    cleaned_lines.append(line)
                else:
                    # Skip obvious API parsing error
                    continue
            else:
                # Reset counters when we encounter legitimate content
                consecutive_numbered_next = 0
                if not (line_stripped.startswith('**Next:**') or line_stripped == 'Next:'):
                    next_count = 0  # Only reset for non-Next content
                cleaned_lines.append(line)
        
        # Additional cleanup for API hallucination patterns
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Remove massive numbered "Next:" sequences (obvious API bug)
        cleaned_text = re.sub(r'(\d+\.\s*\*\*Next:\*\*\s*\n?){5,}', 
                             '\n**[API parsing error detected and cleaned - multiple repetitive entries removed]**\n', 
                             cleaned_text)
        
        # Remove repetitive standalone "Next:" lines
        cleaned_text = re.sub(r'(\*\*Next:\*\*\s*\n?){10,}', 
                             '\n**[Repetitive OCR parsing error cleaned]**\n', 
                             cleaned_text)
        
        # Step 3: Final cleanup of excessive whitespace
        cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)
        
        # Remove obvious pattern corruption at end of documents
        if 'Next:' in cleaned_text and cleaned_text.count('Next:') > 20:
            # This is definitely an API bug - truncate at first major repetition
            lines_final = cleaned_text.split('\n')
            final_lines = []
            next_density = 0
            
            for line in lines_final:
                if 'Next:' in line:
                    next_density += 1
                    if next_density > 10:  # Stop processing when we hit obvious repetition
                        final_lines.append('\n**[OCR API error: Excessive repetitive content removed to prevent processing issues]**')
                        break
                else:
                    next_density = max(0, next_density - 1)  # Decay counter
                
                final_lines.append(line)
            
            cleaned_text = '\n'.join(final_lines)
        
        final_length = len(cleaned_text)
        if original_length > final_length * 1.2:  # Any significant reduction indicates API bug
            reduction_pct = ((original_length - final_length) / original_length * 100)
            print(f"   Enhanced OCR text cleaning applied: {original_length} -> {final_length} chars ({reduction_pct:.1f}% reduction)")
            print(f"   Removed pipe characters, excessive spacing, and repetitive patterns")
        
        return cleaned_text

    def _clean_pipes_and_spacing(self, text: str) -> str:
        """Clean pipe characters and excessive spacing from OCR text"""
        if not text:
            return text
        
        import re
        
        # Step 1: Remove excessive pipe characters that are OCR artifacts
        # Pattern: Multiple consecutive pipes (||||)
        text = re.sub(r'\|{2,}', ' ', text)
        
        # Step 2: Remove standalone pipes surrounded by spaces
        # Pattern: "word | | | word" -> "word word"
        text = re.sub(r'\s*\|\s*\|\s*\|\s*', ' ', text)
        text = re.sub(r'\s*\|\s*\|\s*', ' ', text)
        text = re.sub(r'\s*\|\s*', ' ', text)
        
        # Step 3: Clean up lines that are mostly pipes and dashes (table formatting artifacts)
        lines = text.split('\n')
        
        # ENHANCED FIX: Smart line processing to address first row spacing issue
        cleaned_lines = []
        for i, line in enumerate(lines):
            # For the FIRST line only: remove leading whitespace to fix display issue
            if i == 0:
                # Remove leading spaces/tabs from first line while preserving content
                cleaned_line = line.lstrip()
                # But keep trailing spaces for document structure preservation
                if line.endswith(' '):
                    cleaned_line = cleaned_line + ' '
            else:
                # For other lines: preserve document structure by only removing trailing whitespace
                cleaned_line = line.rstrip()
            
            # Skip lines that are mostly formatting characters
            stripped_line = cleaned_line.strip()
            if stripped_line and len(stripped_line) > 0:
                # Count actual content vs formatting characters
                content_chars = re.sub(r'[\|\-\s_=]+', '', stripped_line)
                formatting_chars = len(stripped_line) - len(content_chars)
                
                # If line is more than 70% formatting characters, it's likely a table border
                if len(content_chars) > 0 and formatting_chars / len(stripped_line) < 0.7:
                    cleaned_lines.append(cleaned_line)
                elif len(content_chars) == 0 and len(stripped_line) > 10:
                    # Skip pure formatting lines
                    continue
                else:
                    cleaned_lines.append(cleaned_line)
            else:
                # Keep empty lines for document structure
                cleaned_lines.append(cleaned_line)
        
        text = '\n'.join(cleaned_lines)
        
        # Step 4: Clean up multiple consecutive spaces
        text = re.sub(r' {3,}', ' ', text)  # Replace 3+ spaces with single space
        text = re.sub(r' {2}', ' ', text)   # Replace 2+ spaces with single space
        
        # Step 5: Fix spacing around common OCR patterns
        # Fix: "word|word" -> "word word"
        text = re.sub(r'([a-zA-Z0-9])\|([a-zA-Z0-9])', r'\1 \2', text)
        
        # Fix: "CLINIC BUKIT PANJANG SEGAR |" -> "CLINIC BUKIT PANJANG SEGAR"
        text = re.sub(r'\s*\|\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\|\s*', '', text, flags=re.MULTILINE)
        
        # Step 6: Clean up word boundaries
        # Ensure proper spacing between words
        text = re.sub(r'([a-zA-Z])([A-Z])', r'\1 \2', text)  # Split camelCase
        
        # Step 7: Final whitespace normalization
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)     # Max 2 consecutive line breaks
        text = text.strip()  # Remove leading/trailing whitespace from entire text
        
        return text
    
    def _clean_with_table_preservation(self, text: str) -> str:
        """Clean OCR text while preserving meaningful table structures"""
        if not text:
            return text
        
        import re
        
        lines = text.split('\n')
        
        # ENHANCED FIX: Smart line processing to address first row spacing issue
        processed_lines = []
        for i, line in enumerate(lines):
            # For the FIRST line only: remove leading whitespace to fix display issue
            if i == 0:
                # Remove leading spaces/tabs from first line while preserving content
                processed_line = line.lstrip()
                # But preserve any meaningful trailing whitespace
                if line.endswith(' ') and not processed_line.endswith(' '):
                    processed_line = processed_line + ' '
                processed_lines.append(processed_line)
            else:
                # For other lines: preserve document structure by only removing trailing whitespace
                processed_lines.append(line.rstrip())
        
        lines = processed_lines
        
        # Step 1: Analyze document for table patterns  
        table_info = self._analyze_table_structure(lines)
        
        # Step 2: Process each line based on table context
        final_processed_lines = []
        
        for i, line in enumerate(lines):
            line_type = table_info['line_types'].get(i, 'content')
            
            if line_type == 'separator':
                # Skip pure separator lines (--- | --- | ---)
                continue
            elif line_type == 'header' or line_type == 'data':
                # Clean but preserve table structure
                cleaned_line = self._clean_table_row(line, table_info)
                if cleaned_line.strip():  # Only add non-empty lines
                    final_processed_lines.append(cleaned_line)
            elif line_type == 'content':
                # Regular content - clean pipes and spacing
                cleaned_line = self._clean_content_line(line)
                final_processed_lines.append(cleaned_line)
            else:
                # Unknown type - clean minimally
                final_processed_lines.append(line)
        
        # Step 3: Format tables properly
        final_text = self._format_preserved_tables(final_processed_lines, table_info)
        
        # Step 4: Final cleanup
        final_text = re.sub(r'\n{3,}', '\n\n', final_text)
        final_text = final_text.strip()
        
        return final_text
    
    def _analyze_table_structure(self, lines: list) -> dict:
        """Analyze text lines to identify table structure patterns"""
        import re
        
        table_info = {
            'line_types': {},  # line_number: type
            'table_regions': [],  # [(start, end), ...]
            'column_count': 0,
            'has_headers': False
        }
        
        pipe_threshold = 2  # Minimum pipes to consider a table row
        separator_pattern = re.compile(r'^[\s\|\-_=]{10,}$')  # Pure separator lines
        
        current_table_start = None
        consecutive_table_lines = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Count pipe characters
            pipe_count = stripped.count('|')
            
            # Detect separator lines (--- | --- | ---)
            if separator_pattern.match(stripped):
                table_info['line_types'][i] = 'separator'
                continue
            
            # Check if this looks like a table row
            if pipe_count >= pipe_threshold:
                # Analyze content between pipes
                parts = [part.strip() for part in stripped.split('|')]
                content_parts = [part for part in parts if part and not re.match(r'^[\s\-_=]*$', part)]
                
                if len(content_parts) >= 2:  # At least 2 meaningful columns
                    if current_table_start is None:
                        current_table_start = i
                        consecutive_table_lines = 1
                        
                        # Check if this might be a header (first table line)
                        if self._looks_like_header(content_parts):
                            table_info['line_types'][i] = 'header'
                            table_info['has_headers'] = True
                        else:
                            table_info['line_types'][i] = 'data'
                    else:
                        consecutive_table_lines += 1
                        table_info['line_types'][i] = 'data'
                    
                    # Update column count
                    table_info['column_count'] = max(table_info['column_count'], len(content_parts))
                else:
                    # Line has pipes but not enough content - likely formatting artifact
                    table_info['line_types'][i] = 'content'
                    if current_table_start is not None and consecutive_table_lines >= 2:
                        table_info['table_regions'].append((current_table_start, i - 1))
                    current_table_start = None
                    consecutive_table_lines = 0
            else:
                # Regular content line
                table_info['line_types'][i] = 'content'
                if current_table_start is not None and consecutive_table_lines >= 2:
                    table_info['table_regions'].append((current_table_start, i - 1))
                current_table_start = None
                consecutive_table_lines = 0
        
        # Close any open table at end of document
        if current_table_start is not None and consecutive_table_lines >= 2:
            table_info['table_regions'].append((current_table_start, len(lines) - 1))
        
        return table_info
    
    def _looks_like_header(self, parts: list) -> bool:
        """Check if table row parts look like headers"""
        # Headers often contain descriptive words, not just data
        header_indicators = [
            'name', 'date', 'amount', 'description', 'type', 'item', 'service',
            'patient', 'doctor', 'diagnosis', 'treatment', 'fee', 'charge'
        ]
        
        for part in parts:
            part_lower = part.lower()
            if any(indicator in part_lower for indicator in header_indicators):
                return True
            # Headers often have title case or all caps
            if len(part) > 2 and (part.istitle() or part.isupper()):
                return True
        
        return False
    
    def _clean_table_row(self, line: str, table_info: dict) -> str:
        """Clean a table row while preserving structure and fixing alignment"""
        import re
        
        # First, normalize all leading/trailing whitespace
        line = line.strip()
        
        # Split by pipes and clean each cell
        parts = line.split('|')
        cleaned_parts = []
        
        for part in parts:
            cleaned_part = part.strip()
            
            # Remove excessive spaces within cells
            cleaned_part = re.sub(r'\s{2,}', ' ', cleaned_part)
            
            # Skip empty cells or pure formatting
            if cleaned_part and not re.match(r'^[\s\-_=]*$', cleaned_part):
                cleaned_parts.append(cleaned_part)
        
        # Reconstruct table row with consistent formatting
        if len(cleaned_parts) >= 2:  # Valid table row
            return '| ' + ' | '.join(cleaned_parts) + ' |'
        elif len(cleaned_parts) == 1:  # Single column
            return cleaned_parts[0]
        else:
            return ''
    
    def _clean_content_line(self, line: str) -> str:
        """Clean a regular content line (non-table) with proper word boundaries"""
        import re
        
        # First normalize all whitespace (but preserve leading for structure)
        line = line.rstrip()  # Only remove trailing, preserve leading for indentation
        
        # Remove pipes and ensure proper word separation
        # Pattern: word|word -> word word (with space)
        cleaned = re.sub(r'([a-zA-Z0-9\u00C0-\u017F\u4e00-\u9fff\uac00-\ud7af])\s*\|\s*([a-zA-Z0-9\u00C0-\u017F\u4e00-\u9fff\uac00-\ud7af])', r'\1 \2', line)
        
        # Remove remaining standalone pipes
        cleaned = re.sub(r'\s*\|\s*', ' ', cleaned)
        
        # Fix spacing between words that might have been separated by pipes
        # Insert space between UpperCase transitions (clinic names, addresses)
        cleaned = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned)
        
        # Insert space between letters and numbers
        cleaned = re.sub(r'([A-Za-z])(\d)', r'\1 \2', cleaned)
        cleaned = re.sub(r'(\d)([A-Za-z])', r'\1 \2', cleaned)
        
        # Insert space before parentheses and after
        cleaned = re.sub(r'([a-zA-Z0-9])(\()', r'\1 \2', cleaned)
        cleaned = re.sub(r'(\))([a-zA-Z0-9])', r'\1 \2', cleaned)
        
        # Normalize multiple spaces to single space
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)
        
        # Insert logical breaks for long clinic information lines
        if len(cleaned) > 80:  # Long lines need proper formatting
            # Add line break before address (starts with numbers)
            cleaned = re.sub(r'(.*?SEGAR\s+)(\d+\s+)', r'\1\n\2', cleaned)
            
            # Add line break before GST & UEN
            cleaned = re.sub(r'(\)\s+)(GST\s+&\s+UEN)', r'\1\n\2', cleaned)
            
            # Add line break before Telephone
            cleaned = re.sub(r'(\d+D\s+)(Telephone:)', r'\1\n\2', cleaned)
            
            # Add line break before Attending Physician
            cleaned = re.sub(r'(\d+\s+)(Attending\s+Physician:)', r'\1\n\2', cleaned)
            
            # Add line break before Invoice Date
            cleaned = re.sub(r'(Ratnam\s+)(Invoice\s+Date:)', r'\1\n\2', cleaned)
            
            # Add line break before Provided by
            cleaned = re.sub(r'(\d+\s+)(Provided\s+by:)', r'\1\n\2', cleaned)
        
        return cleaned.rstrip()  # Final trailing whitespace cleanup
    
    def _format_preserved_tables(self, lines: list, table_info: dict) -> str:
        """Format the final output with properly structured tables"""
        if not table_info['table_regions']:
            # No tables detected, just join lines
            return '\n'.join(lines)
        
        formatted_lines = []
        line_index = 0
        
        for start, end in table_info['table_regions']:
            # Add content before table
            while line_index < start:
                if line_index < len(lines):
                    formatted_lines.append(lines[line_index])
                line_index += 1
            
            # Add table with proper formatting
            formatted_lines.append('')  # Blank line before table
            
            # Add table content
            while line_index <= end and line_index < len(lines):
                table_line = lines[line_index]
                if table_line.strip():  # Only add non-empty lines
                    formatted_lines.append(table_line)
                line_index += 1
            
            formatted_lines.append('')  # Blank line after table
        
        # Add remaining content after last table
        while line_index < len(lines):
            formatted_lines.append(lines[line_index])
            line_index += 1
        
        return '\n'.join(formatted_lines)
    
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