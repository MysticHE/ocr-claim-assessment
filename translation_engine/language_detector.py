"""
Language detection service for identifying source language of OCR text.
Uses langdetect library for fast and reliable language identification.
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple
from langdetect import detect, detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Set seed for consistent results
DetectorFactory.seed = 0

@dataclass
class LanguageDetectionResult:
    """Result of language detection"""
    detected_language: str = "en"
    confidence: float = 0.0
    alternatives: List[Tuple[str, float]] = None
    is_reliable: bool = False
    error: Optional[str] = None

class LanguageDetector:
    """Fast language detection for OCR text"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Common languages in medical documents
        self.supported_languages = {
            'en': 'English',
            'zh': 'Chinese',  # Simplified Chinese
            'zh-cn': 'Chinese (Simplified)',
            'zh-tw': 'Chinese (Traditional)', 
            'ms': 'Malay',
            'ta': 'Tamil',
            'hi': 'Hindi',
            'th': 'Thai',
            'vi': 'Vietnamese',
            'id': 'Indonesian',
            'ko': 'Korean',
            'ja': 'Japanese',
            'ar': 'Arabic',
            'ur': 'Urdu',
            'bn': 'Bengali',
            'te': 'Telugu',
            'ml': 'Malayalam',
            'kn': 'Kannada',
            'gu': 'Gujarati',
            'pa': 'Punjabi',
            'mr': 'Marathi',
            'ne': 'Nepali',
            'si': 'Sinhala',
            'my': 'Myanmar',
            'km': 'Khmer',
            'lo': 'Lao',
            'tl': 'Tagalog',
            'fr': 'French',
            'de': 'German',
            'es': 'Spanish',
            'pt': 'Portuguese',
            'it': 'Italian',
            'ru': 'Russian'
        }
        
        # Minimum confidence threshold for reliable detection
        self.confidence_threshold = 0.8
        
        # Minimum text length for reliable detection
        self.min_text_length = 10
    
    def detect_language(self, text: str) -> LanguageDetectionResult:
        """
        Detect language of the given text
        
        Args:
            text: Text to analyze for language detection
            
        Returns:
            LanguageDetectionResult with detected language and confidence
        """
        if not text or len(text.strip()) < self.min_text_length:
            return LanguageDetectionResult(
                detected_language="en",
                confidence=0.0,
                is_reliable=False,
                error="Text too short for reliable detection"
            )
        
        try:
            # Get primary detection
            primary_lang = detect(text)
            
            # Get detailed probabilities
            lang_probabilities = detect_langs(text)
            
            # Find the confidence for the primary language
            primary_confidence = 0.0
            alternatives = []
            
            for lang_prob in lang_probabilities:
                if lang_prob.lang == primary_lang:
                    primary_confidence = lang_prob.prob
                else:
                    alternatives.append((lang_prob.lang, lang_prob.prob))
            
            # Sort alternatives by confidence
            alternatives = sorted(alternatives, key=lambda x: x[1], reverse=True)
            
            # Determine if detection is reliable
            is_reliable = (
                primary_confidence >= self.confidence_threshold and
                len(text.strip()) >= self.min_text_length and
                primary_lang in self.supported_languages
            )
            
            # Normalize Chinese language codes
            normalized_lang = self._normalize_language_code(primary_lang)
            
            self.logger.info(f"Language detected: {normalized_lang} (confidence: {primary_confidence:.2f})")
            
            return LanguageDetectionResult(
                detected_language=normalized_lang,
                confidence=primary_confidence,
                alternatives=alternatives[:3],  # Top 3 alternatives
                is_reliable=is_reliable
            )
            
        except LangDetectException as e:
            self.logger.warning(f"Language detection failed: {e}")
            return LanguageDetectionResult(
                detected_language="en",
                confidence=0.0,
                is_reliable=False,
                error=f"Detection failed: {str(e)}"
            )
        except Exception as e:
            self.logger.error(f"Unexpected error in language detection: {e}")
            return LanguageDetectionResult(
                detected_language="en",
                confidence=0.0,
                is_reliable=False,
                error=f"Unexpected error: {str(e)}"
            )
    
    def _normalize_language_code(self, lang_code: str) -> str:
        """Normalize language codes to standard format"""
        # Handle Chinese variants
        if lang_code in ['zh-cn', 'zh']:
            return 'zh'  # Simplified Chinese
        elif lang_code == 'zh-tw':
            return 'zh-tw'  # Traditional Chinese
        
        # Return as-is for other languages
        return lang_code
    
    def is_english(self, text: str) -> bool:
        """Quick check if text is primarily English"""
        if not text or len(text.strip()) < 5:
            return True  # Default to English for very short text
        
        try:
            result = self.detect_language(text)
            return result.detected_language == 'en' and result.confidence > 0.7
        except:
            return True  # Default to English on error
    
    def get_language_name(self, lang_code: str) -> str:
        """Get human-readable language name"""
        return self.supported_languages.get(lang_code, lang_code.upper())
    
    def is_supported(self, lang_code: str) -> bool:
        """Check if language is supported for translation"""
        return lang_code in self.supported_languages
    
    def get_supported_languages(self) -> dict:
        """Get dictionary of supported languages"""
        return self.supported_languages.copy()