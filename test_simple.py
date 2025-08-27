"""
Simple test for translation pipeline without Unicode issues
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all translation components can be imported"""
    print("Testing imports...")
    
    try:
        from translation_engine.language_detector import LanguageDetector
        from translation_engine.openai_translator import OpenAITranslator
        from claims_engine.enhanced_processor import EnhancedClaimProcessor
        print("   All imports successful")
        return True
    except Exception as e:
        print(f"   Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without non-ASCII text"""
    print("Testing basic functionality...")
    
    try:
        # Test language detector
        from translation_engine.language_detector import LanguageDetector
        detector = LanguageDetector()
        result = detector.detect_language("Hello, this is a test document.")
        print(f"   Language detection: {result.detected_language} (confidence: {result.confidence:.2f})")
        
        # Test OpenAI translator availability
        from translation_engine.openai_translator import OpenAITranslator
        translator = OpenAITranslator()
        print(f"   OpenAI translator available: {translator.is_available()}")
        
        # Test enhanced processor
        from claims_engine.enhanced_processor import EnhancedClaimProcessor
        processor = EnhancedClaimProcessor()
        print(f"   Translation capabilities: {processor.translation_available}")
        
        print("   Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"   Basic functionality test failed: {e}")
        return False

def main():
    """Run simple translation pipeline tests"""
    print("Starting Simple Translation Pipeline Test\n")
    
    tests_passed = 0
    total_tests = 2
    
    if test_imports():
        tests_passed += 1
    print()
    
    if test_basic_functionality():
        tests_passed += 1
    print()
    
    # Summary
    print("Summary:")
    print(f"   Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("   Translation pipeline components are working!")
    else:
        print("   Some issues found. Check setup and dependencies.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)