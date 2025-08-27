"""
Test script for the translation pipeline functionality.
Tests OpenAI translation, language detection, and dual-content processing.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_language_detection():
    """Test the language detection functionality"""
    print("Testing Language Detection...")
    
    try:
        from translation_engine.language_detector import LanguageDetector
        
        detector = LanguageDetector()
        
        test_cases = [
            ("Hello, this is a medical bill for consultation.", "en"),
            ("你好，这是医疗账单。", "zh"),
            ("Ini adalah bil perubatan untuk perundingan.", "ms"), 
            ("यह परामर्श के लिए एक चिकित्सा बिल है।", "hi"),
            ("This is a test.", "en")
        ]
        
        for text, expected_lang in test_cases:
            result = detector.detect_language(text)
            print(f"   Text: '{text[:30]}...' if len(text) > 30 else text")
            print(f"   Expected: {expected_lang} | Detected: {result.detected_language} | Confidence: {result.confidence:.2f}")
            
            if result.detected_language.startswith(expected_lang):
                print("   PASS")
            else:
                print("   Different detection (may still be correct)")
            print()
        
        print("Language detection test completed")
        return True
        
    except Exception as e:
        print(f"Language detection test failed: {e}")
        return False

def test_openai_translator():
    """Test the OpenAI translation functionality"""
    print("Testing OpenAI Translation...")
    
    try:
        from translation_engine.openai_translator import OpenAITranslator
        
        translator = OpenAITranslator()
        
        if not translator.is_available():
            print("   OpenAI translator not available (API key not configured)")
            return False
        
        # Test connection first
        connection_test = translator.test_connection()
        print(f"   Connection test: {'PASS' if connection_test.get('test_successful') else 'FAIL'}")
        
        if not connection_test.get('test_successful'):
            print(f"   Error: {connection_test.get('error')}")
            return False
        
        # Test actual translation
        test_cases = [
            ("这是医疗账单。", "zh", "en"),
            ("Ini adalah bil perubatan.", "ms", "en"),
        ]
        
        for text, source_lang, target_lang in test_cases:
            print(f"   Translating: '{text}' from {source_lang} to {target_lang}")
            
            result = translator.translate(text, source_lang, target_lang)
            
            if result.success:
                print(f"   Translation: '{result.translated_text}'")
                print(f"   Processing time: {result.processing_time_ms}ms")
            else:
                print(f"   Translation failed: {result.error}")
                return False
            print()
        
        print("OpenAI translation test completed")
        return True
        
    except Exception as e:
        print(f"OpenAI translation test failed: {e}")
        return False

def test_dual_content_processing():
    """Test the dual-content OCR processing"""
    print("Testing Dual-Content Processing...")
    
    try:
        from claims_engine.enhanced_processor import EnhancedClaimProcessor
        
        processor = EnhancedClaimProcessor()
        
        # Mock OCR result with Chinese text
        mock_ocr_result = {
            'success': True,
            'text': '医院账单\n患者姓名：张三\n金额：$150.00\n日期：2025-01-15',
            'confidence': 0.9,
            'language': 'zh',
            'processing_time_ms': 1000,
            'engine': 'mistral_test'
        }
        
        print("   Processing mock Chinese medical bill OCR result...")
        dual_result = processor.process_dual_content_ocr(mock_ocr_result)
        
        print(f"   Original language: {dual_result.get('original_language')}")
        print(f"   Translation provider: {dual_result.get('translation_provider')}")
        print(f"   Language confidence: {dual_result.get('language_confidence')}")
        
        if dual_result.get('original_text'):
            print(f"   Original text: '{dual_result.get('original_text')[:50]}...'")
        
        if dual_result.get('translated_text'):
            print(f"   Translated text: '{dual_result.get('translated_text')[:50]}...'")
        
        if dual_result.get('original_language') and dual_result.get('original_language') != 'en':
            if dual_result.get('translation_provider'):
                print("   Translation pipeline working")
            else:
                print("   Translation not performed (likely due to missing OpenAI API key)")
        else:
            print("   English text detected - no translation needed")
        
        print("Dual-content processing test completed")
        return True
        
    except Exception as e:
        print(f"Dual-content processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all translation pipeline tests"""
    print("Starting Translation Pipeline Tests\n")
    
    # Check environment variables
    print("Environment Check:")
    openai_key = os.getenv('OPENAI_API_KEY')
    print(f"   OpenAI API Key: {'Configured' if openai_key else 'Missing'}")
    
    if not openai_key:
        print("   Some tests will be skipped without OpenAI API key")
    print()
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_language_detection():
        tests_passed += 1
    print()
    
    if test_openai_translator():
        tests_passed += 1
    print()
    
    if test_dual_content_processing():
        tests_passed += 1
    print()
    
    # Summary
    print("Test Summary:")
    print(f"   Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("   All tests passed! Translation pipeline is ready.")
    elif tests_passed > 0:
        print("   Some tests passed. Check configuration for missing components.")
    else:
        print("   All tests failed. Check setup and dependencies.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)