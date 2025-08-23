from typing import Dict, List, Set

class LanguageMapper:
    """Maps language codes between different OCR engines and provides language support information"""
    
    def __init__(self):
        # PaddleOCR supported languages mapping
        self.paddle_language_map = {
            # Common language codes to PaddleOCR format
            'en': 'en',                    # English
            'english': 'en',
            'zh': 'ch',                    # Chinese Simplified
            'zh_cn': 'ch', 
            'zh_hans': 'ch',
            'chinese': 'ch',
            'ch_sim': 'ch',
            'zh_tw': 'chinese_cht',        # Chinese Traditional
            'zh_hant': 'chinese_cht',
            'chinese_cht': 'chinese_cht',
            'ch_tra': 'chinese_cht',
            'ms': 'ms',                    # Malay
            'malay': 'ms',
            'ta': 'ta',                    # Tamil
            'tamil': 'ta',
            'ko': 'korean',                # Korean
            'korean': 'korean',
            'kr': 'korean',
            'ja': 'japan',                 # Japanese
            'japanese': 'japan',
            'jp': 'japan',
            'fr': 'fr',                    # French
            'french': 'fr',
            'de': 'de',                    # German
            'german': 'de',
            'es': 'es',                    # Spanish
            'spanish': 'es',
            'it': 'it',                    # Italian
            'italian': 'it',
            'pt': 'pt',                    # Portuguese
            'portuguese': 'pt',
            'ru': 'ru',                    # Russian
            'russian': 'ru',
            'ar': 'ar',                    # Arabic
            'arabic': 'ar',
            'hi': 'hi',                    # Hindi
            'hindi': 'hi',
            'th': 'th',                    # Thai
            'thai': 'th',
            'vi': 'vi',                    # Vietnamese
            'vietnamese': 'vi',
            'nl': 'nl',                    # Dutch
            'dutch': 'nl',
            'pl': 'pl',                    # Polish
            'polish': 'pl',
            'tr': 'tr',                    # Turkish
            'turkish': 'tr',
            'sv': 'sv',                    # Swedish
            'swedish': 'sv',
            'no': 'no',                    # Norwegian
            'norwegian': 'no',
            'da': 'da',                    # Danish
            'danish': 'da',
            'fi': 'fi',                    # Finnish
            'finnish': 'fi',
            'hu': 'hu',                    # Hungarian
            'hungarian': 'hu',
            'cs': 'cs',                    # Czech
            'czech': 'cs',
            'sk': 'sk',                    # Slovak
            'slovak': 'sk',
            'bg': 'bg',                    # Bulgarian
            'bulgarian': 'bg',
            'hr': 'hr',                    # Croatian
            'croatian': 'hr',
            'sr': 'rs_cyrillic',          # Serbian
            'serbian': 'rs_cyrillic',
            'sl': 'sl',                    # Slovenian
            'slovenian': 'sl',
            'et': 'et',                    # Estonian
            'estonian': 'et',
            'lv': 'lv',                    # Latvian
            'latvian': 'lv',
            'lt': 'lt',                    # Lithuanian
            'lithuanian': 'lt',
            'ro': 'ro',                    # Romanian
            'romanian': 'ro',
            'mt': 'mt',                    # Maltese
            'maltese': 'mt',
            'is': 'is',                    # Icelandic
            'icelandic': 'is',
            'ga': 'ga',                    # Irish
            'irish': 'ga',
            'cy': 'cy',                    # Welsh
            'welsh': 'cy',
            'sq': 'sq',                    # Albanian
            'albanian': 'sq',
            'mk': 'mk',                    # Macedonian (if supported)
            'macedonian': 'mk',
            'be': 'be',                    # Belarusian
            'belarusian': 'be',
            'uk': 'uk',                    # Ukrainian
            'ukrainian': 'uk',
            'ka': 'ka',                    # Georgian
            'georgian': 'ka',
            'hy': 'hy',                    # Armenian (if supported)
            'armenian': 'hy',
            'he': 'he',                    # Hebrew (if supported)
            'hebrew': 'he',
            'fa': 'fa',                    # Persian/Farsi
            'persian': 'fa',
            'farsi': 'fa',
            'ur': 'ur',                    # Urdu
            'urdu': 'ur',
            'bn': 'bn',                    # Bengali (if supported)
            'bengali': 'bn',
            'gu': 'gu',                    # Gujarati (if supported)
            'gujarati': 'gu',
            'pa': 'pa',                    # Punjabi (if supported)
            'punjabi': 'pa',
            'te': 'te',                    # Telugu
            'telugu': 'te',
            'kn': 'kn',                    # Kannada (if supported)
            'kannada': 'kn',
            'ml': 'ml',                    # Malayalam (if supported)
            'malayalam': 'ml',
            'or': 'or',                    # Odia (if supported)
            'odia': 'or',
            'as': 'as',                    # Assamese (if supported)
            'assamese': 'as',
            'mr': 'mr',                    # Marathi
            'marathi': 'mr',
            'ne': 'ne',                    # Nepali
            'nepali': 'ne',
            'si': 'si',                    # Sinhala (if supported)
            'sinhala': 'si',
            'my': 'my',                    # Myanmar (if supported)
            'myanmar': 'my',
            'km': 'km',                    # Khmer (if supported)
            'khmer': 'km',
            'lo': 'lo',                    # Lao (if supported)
            'lao': 'lo',
            'ka': 'ka',                    # Georgian
            'mn': 'mn',                    # Mongolian
            'mongolian': 'mn',
            'ug': 'ug',                    # Uyghur
            'uyghur': 'ug',
            'uz': 'uz',                    # Uzbek
            'uzbek': 'uz',
            'kk': 'kk',                    # Kazakh (if supported)
            'kazakh': 'kk',
            'ky': 'ky',                    # Kyrgyz (if supported)
            'kyrgyz': 'ky',
            'tg': 'tg',                    # Tajik (if supported)
            'tajik': 'tg'
        }
        
        # Comprehensive list of PaddleOCR supported languages
        self.paddle_supported = {
            'en', 'ch', 'chinese_cht', 'ta', 'te', 'korean', 'japan',
            'it', 'xi', 'pu', 'ru', 'ar', 'hi', 'ug', 'fa', 'ur',
            'rs_cyrillic', 'oc', 'mr', 'ne', 'rsc', 'bg', 'uk', 'be',
            'te', 'kn', 'ch_tra', 'hi', 'mr', 'ne', 'la', 'fr', 'de',
            'es', 'pt', 'it', 'ru', 'bg', 'pl', 'cs', 'sk', 'hr',
            'sl', 'et', 'lv', 'lt', 'hu', 'ro', 'sq', 'is', 'ga',
            'cy', 'mt', 'fi', 'sv', 'no', 'da', 'nl', 'tr', 'vi',
            'th', 'ms', 'tl', 'id', 'sw', 'mi'
        }
        
        # Language display names
        self.language_names = {
            'en': 'English',
            'ch': 'Chinese (Simplified)',
            'chinese_cht': 'Chinese (Traditional)',
            'ms': 'Malay',
            'ta': 'Tamil',
            'korean': 'Korean',
            'japan': 'Japanese',
            'fr': 'French',
            'de': 'German',
            'es': 'Spanish',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'th': 'Thai',
            'vi': 'Vietnamese',
            'nl': 'Dutch',
            'pl': 'Polish',
            'tr': 'Turkish',
            'sv': 'Swedish',
            'no': 'Norwegian',
            'da': 'Danish',
            'fi': 'Finnish',
            'hu': 'Hungarian',
            'cs': 'Czech',
            'sk': 'Slovak',
            'bg': 'Bulgarian',
            'hr': 'Croatian',
            'rs_cyrillic': 'Serbian (Cyrillic)',
            'sl': 'Slovenian',
            'et': 'Estonian',
            'lv': 'Latvian',
            'lt': 'Lithuanian',
            'ro': 'Romanian',
            'mt': 'Maltese',
            'is': 'Icelandic',
            'ga': 'Irish',
            'cy': 'Welsh',
            'sq': 'Albanian',
            'be': 'Belarusian',
            'uk': 'Ukrainian',
            'ka': 'Georgian',
            'fa': 'Persian',
            'ur': 'Urdu',
            'te': 'Telugu',
            'mr': 'Marathi',
            'ne': 'Nepali',
            'mn': 'Mongolian',
            'ug': 'Uyghur',
            'uz': 'Uzbek',
            'tl': 'Tagalog',
            'id': 'Indonesian',
            'sw': 'Swahili',
            'mi': 'Maori'
        }
        
        # Priority languages for the claim system
        self.priority_languages = [
            'en',      # English - Primary business language
            'ch',      # Chinese (Simplified) - Major market
            'ms',      # Malay - Local language
            'ta',      # Tamil - Local language
            'korean',  # Korean - Required language
            'japan',   # Japanese - Business language
            'chinese_cht'  # Traditional Chinese
        ]
    
    def to_paddle_format(self, language_code: str) -> str:
        """Convert language code to PaddleOCR format"""
        normalized = language_code.lower().strip()
        return self.paddle_language_map.get(normalized, normalized)
    
    def is_supported(self, language_code: str) -> bool:
        """Check if language is supported by PaddleOCR"""
        paddle_format = self.to_paddle_format(language_code)
        return paddle_format in self.paddle_supported
    
    def get_display_name(self, language_code: str) -> str:
        """Get human-readable language name"""
        paddle_format = self.to_paddle_format(language_code)
        return self.language_names.get(paddle_format, language_code.capitalize())
    
    def get_supported_languages(self) -> List[str]:
        """Get list of all supported language codes"""
        return list(self.paddle_language_map.keys())
    
    def get_priority_languages(self) -> List[str]:
        """Get priority languages for the claim system"""
        return self.priority_languages.copy()
    
    def get_language_info(self, language_code: str) -> Dict[str, str]:
        """Get comprehensive language information"""
        paddle_format = self.to_paddle_format(language_code)
        return {
            'original_code': language_code,
            'paddle_format': paddle_format,
            'display_name': self.get_display_name(language_code),
            'supported': self.is_supported(language_code),
            'priority': paddle_format in self.priority_languages
        }
    
    def filter_supported_languages(self, language_codes: List[str]) -> List[str]:
        """Filter list to only include supported languages"""
        supported = []
        for code in language_codes:
            if self.is_supported(code):
                supported.append(code)
        return supported
    
    def suggest_alternatives(self, language_code: str) -> List[str]:
        """Suggest alternative language codes for unsupported ones"""
        if self.is_supported(language_code):
            return [language_code]
        
        alternatives = []
        code_lower = language_code.lower()
        
        # Look for partial matches
        for supported_code, paddle_format in self.paddle_language_map.items():
            if (code_lower in supported_code or 
                supported_code in code_lower or
                any(part in supported_code for part in code_lower.split('_'))):
                alternatives.append(supported_code)
        
        # If no alternatives found, suggest priority languages
        if not alternatives:
            alternatives = self.priority_languages[:3]
        
        return alternatives[:5]  # Limit to 5 suggestions
    
    def get_language_families(self) -> Dict[str, List[str]]:
        """Group languages by families/regions"""
        return {
            'East Asian': ['ch', 'chinese_cht', 'japan', 'korean'],
            'Southeast Asian': ['ms', 'th', 'vi', 'id', 'tl'],
            'South Asian': ['ta', 'te', 'hi', 'mr', 'ne', 'ur', 'bn'],
            'European': ['en', 'fr', 'de', 'es', 'it', 'pt', 'ru', 'pl', 'nl'],
            'Slavic': ['ru', 'bg', 'hr', 'cs', 'sk', 'sl', 'pl', 'uk', 'be'],
            'Nordic': ['sv', 'no', 'da', 'fi', 'is'],
            'Middle Eastern': ['ar', 'fa', 'he', 'ur'],
            'Other': ['sw', 'mi', 'ka', 'hy', 'mn', 'ug', 'uz']
        }
    
    def detect_language_family(self, language_code: str) -> str:
        """Detect which language family a language belongs to"""
        paddle_format = self.to_paddle_format(language_code)
        
        for family, languages in self.get_language_families().items():
            if paddle_format in languages:
                return family
        
        return 'Other'
    
    def get_recommended_for_region(self, region: str) -> List[str]:
        """Get recommended languages for a specific region"""
        region_map = {
            'asia': ['en', 'ch', 'korean', 'japan', 'ms', 'th', 'vi'],
            'southeast_asia': ['en', 'ms', 'th', 'vi', 'id', 'tl'],
            'south_asia': ['en', 'hi', 'ta', 'te', 'mr', 'ne', 'ur'],
            'east_asia': ['en', 'ch', 'chinese_cht', 'korean', 'japan'],
            'europe': ['en', 'fr', 'de', 'es', 'it', 'pt', 'ru'],
            'middle_east': ['en', 'ar', 'fa', 'he'],
            'africa': ['en', 'fr', 'ar', 'sw'],
            'americas': ['en', 'es', 'pt', 'fr']
        }
        
        region_key = region.lower().replace(' ', '_')
        return region_map.get(region_key, ['en'])  # Default to English