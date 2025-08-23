#!/usr/bin/env python3
"""
Debug script to test enhanced processor initialization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_enhanced_processor():
    print("Testing Enhanced Processor Initialization")
    print("=" * 50)
    
    try:
        print("1. Testing imports...")
        from claims_engine.enhanced_processor import EnhancedClaimProcessor
        print("+ Enhanced processor import successful")
        
        from ai_engines.document_classifier import DocumentClassifier
        print("+ Document classifier import successful")
        
        from ai_engines.quality_assessor import DocumentQualityAssessor
        print("+ Quality assessor import successful")
        
    except Exception as e:
        print(f"- Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        print("\n2. Testing enhanced processor initialization...")
        processor = EnhancedClaimProcessor()
        print("+ Enhanced processor initialized successfully")
        
    except Exception as e:
        print(f"- Enhanced processor initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        print("\n3. Testing AI engines lazy loading...")
        print(f"   Classifier available: {processor.classifier_available}")
        print(f"   Quality assessor available: {processor.quality_assessor_available}")
        
    except Exception as e:
        print(f"- AI engines test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        print("\n4. Testing mock enhanced processing...")
        mock_ocr_result = {
            'success': True,
            'text': 'Test medical receipt\nAmount: $50.00\nDate: 2024-01-01\nProvider: Test Clinic',
            'confidence': 0.95,
            'engine': 'mistral'
        }
        
        result = processor.process_enhanced_claim(mock_ocr_result, None)
        print(f"+ Enhanced processing completed")
        print(f"   Success: {result.get('success', False)}")
        print(f"   Status: {result.get('status', 'unknown')}")
        print(f"   Confidence: {result.get('confidence', 0)}")
        print(f"   Keys: {list(result.keys())}")
        
        return True
        
    except Exception as e:
        print(f"- Enhanced processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_processor()
    print(f"\n{'+ All tests passed!' if success else '- Tests failed!'}")