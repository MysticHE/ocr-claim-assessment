"""
OpenAI-powered translation service for medical insurance claim documents.
Provides context-aware translation preserving medical terminology and document structure.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any
from openai import OpenAI
import logging

@dataclass
class TranslationResult:
    """Result of translation operation"""
    success: bool = False
    original_text: str = ""
    translated_text: str = ""
    source_language: str = ""
    target_language: str = "en"
    provider: str = "openai"
    processing_time_ms: int = 0
    error: Optional[str] = None

class OpenAITranslator:
    """OpenAI GPT-4 translation service specialized for medical documents"""
    
    def __init__(self):
        self.client = None
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.logger = logging.getLogger(__name__)
        
        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
                self.logger.info("OpenAI translator initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI client: {e}")
        else:
            self.logger.warning("OpenAI API key not found - translation service disabled")
    
    def is_available(self) -> bool:
        """Check if OpenAI translation service is available"""
        return self.client is not None
    
    def translate(self, text: str, source_language: str, target_language: str = "en") -> TranslationResult:
        """
        Translate text using OpenAI GPT-4o-mini with medical context awareness
        
        Args:
            text: Text to translate
            source_language: Source language code (e.g., 'zh', 'ms', 'ta')
            target_language: Target language code (default: 'en')
            
        Returns:
            TranslationResult with translation details
        """
        if not self.is_available():
            return TranslationResult(
                success=False,
                original_text=text,
                error="OpenAI translation service not available"
            )
        
        start_time = time.time()
        
        # Create medical-context translation prompt
        prompt = self._create_medical_translation_prompt(text, source_language, target_language)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=4000,
                timeout=30
            )
            
            translated_text = response.choices[0].message.content.strip()
            processing_time = int((time.time() - start_time) * 1000)
            
            return TranslationResult(
                success=True,
                original_text=text,
                translated_text=translated_text,
                source_language=source_language,
                target_language=target_language,
                provider="openai",
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            self.logger.error(f"OpenAI translation failed: {e}")
            
            return TranslationResult(
                success=False,
                original_text=text,
                source_language=source_language,
                target_language=target_language,
                provider="openai",
                processing_time_ms=processing_time,
                error=f"Translation failed: {str(e)}"
            )
    
    def _create_medical_translation_prompt(self, text: str, source_lang: str, target_lang: str) -> str:
        """Create specialized prompt for medical document translation"""
        
        # Language name mapping
        lang_names = {
            'zh': 'Chinese',
            'ms': 'Malay',
            'ta': 'Tamil',
            'hi': 'Hindi',
            'th': 'Thai',
            'vi': 'Vietnamese',
            'id': 'Indonesian',
            'ko': 'Korean',
            'ja': 'Japanese',
            'en': 'English'
        }
        
        source_name = lang_names.get(source_lang, source_lang.upper())
        target_name = lang_names.get(target_lang, target_lang.upper())
        
        prompt = f"""You are a professional medical translator specializing in insurance claims and healthcare documents.

TRANSLATION TASK:
- Source language: {source_name}
- Target language: {target_name}
- Document type: Medical insurance claim/healthcare document

CRITICAL REQUIREMENTS:
1. Preserve ALL medical terminology accuracy
2. Keep ALL numbers, dates, amounts, and reference codes EXACTLY as they appear
3. Maintain document structure and formatting
4. Use appropriate medical English terminology
5. Preserve proper names (hospitals, doctors, patients) exactly as written
6. Keep currency symbols and amounts unchanged
7. Do not add explanations or notes

DOCUMENT TO TRANSLATE:
{text}

Provide ONLY the translation in {target_name}, maintaining the exact structure and format."""
        
        return prompt
    
    def test_connection(self) -> Dict[str, Any]:
        """Test OpenAI connection and return status"""
        if not self.is_available():
            return {
                "available": False,
                "error": "OpenAI API key not configured"
            }
        
        try:
            # Test with a simple translation
            test_result = self.translate("Hello", "en", "en")
            return {
                "available": True,
                "test_successful": test_result.success,
                "response_time_ms": test_result.processing_time_ms,
                "error": test_result.error
            }
        except Exception as e:
            return {
                "available": True,
                "test_successful": False,
                "error": str(e)
            }