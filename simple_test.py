#!/usr/bin/env python3
"""
Simple test script for Enhanced OCR Claim Processing System
"""

import os
import sys
import time

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all imports are working"""
    print("Testing imports...")
    
    try:
        from ai_engines.document_classifier import DocumentClassifier
        print("  - Document Classifier: OK")
    except Exception as e:
        print(f"  - Document Classifier: FAIL - {e}")
        return False
    
    try:
        from ai_engines.quality_assessor import DocumentQualityAssessor
        print("  - Quality Assessor: OK")
    except Exception as e:
        print(f"  - Quality Assessor: FAIL - {e}")
        return False
    
    try:
        from claims_engine.enhanced_processor import EnhancedClaimProcessor
        print("  - Enhanced Processor: OK")
    except Exception as e:
        print(f"  - Enhanced Processor: FAIL - {e}")
        return False
    
    try:
        from ocr_engine.mistral_only_ocr import MistralOnlyOCREngine
        print("  - OCR Engine: OK")
    except Exception as e:
        print(f"  - OCR Engine: FAIL - {e}")
        return False
    
    return True

def test_document_classification():
    """Test document classification"""
    print("\nTesting Document Classification...")
    
    try:
        from ai_engines.document_classifier import DocumentClassifier
        classifier = DocumentClassifier()
        
        # Simple test case
        test_text = "RECEIPT\nTotal: $50.00\nPayment Method: Cash"
        result = classifier.classify_document(test_text)
        
        print(f"  - Classified as: {result.document_type.value}")
        print(f"  - Confidence: {result.confidence:.2f}")
        print("  - Classification: OK")
        return True
        
    except Exception as e:
        print(f"  - Classification: FAIL - {e}")
        return False

def test_enhanced_processing():
    """Test enhanced claim processing"""
    print("\nTesting Enhanced Processing...")
    
    try:
        from claims_engine.enhanced_processor import EnhancedClaimProcessor
        processor = EnhancedClaimProcessor()
        
        # Sample OCR result
        sample_ocr = {
            'text': 'Patient: John Doe\nAmount: $100.00\nDate: 2024-08-15',
            'confidence': 0.85,
            'engine': 'test'
        }
        
        result = processor.process_enhanced_claim(sample_ocr)
        
        print(f"  - Status: {result.get('status', 'unknown')}")
        print(f"  - Success: {result.get('success', False)}")
        print("  - Enhanced Processing: OK")
        return True
        
    except Exception as e:
        print(f"  - Enhanced Processing: FAIL - {e}")
        return False

def main():
    """Run tests"""
    print("Enhanced OCR System - Simple Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_document_classification,
        test_enhanced_processing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test failed with exception: {e}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests PASSED! System is ready.")
        return 0
    else:
        print("Some tests FAILED. Check configuration.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)