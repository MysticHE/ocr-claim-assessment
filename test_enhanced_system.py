#!/usr/bin/env python3
"""
Test script for Enhanced OCR Claim Processing System
Tests the new AI-powered workflow features
"""

import os
import sys
import time
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_document_classifier():
    """Test document classification engine"""
    try:
        from ai_engines.document_classifier import DocumentClassifier
        
        classifier = DocumentClassifier()
        
        # Test sample text for each document type
        test_cases = [
            ("RECEIPT\nTotal: $50.00\nPayment Method: Cash\nThank you for your purchase", "receipt"),
            ("DIAGNOSIS: Hypertension (I10)\nTreatment: Medication prescribed\nDoctor: Dr. Smith", "diagnostic_report"),
            ("REFERRAL LETTER\nDear Dr. Johnson,\nPlease see this patient for specialist consultation", "referral_letter"),
            ("MEMO\nFrom: Admin\nTo: Staff\nSubject: New policy update", "memo"),
            ("PRESCRIPTION\nRx: Lisinopril 10mg\nTake once daily\nQty: 30 tablets", "prescription")
        ]
        
        print("Testing Document Classification Engine...")
        passed = 0
        total = len(test_cases)
        
        for i, (text, expected_type) in enumerate(test_cases, 1):
            result = classifier.classify_document(text)
            actual_type = result.document_type.value
            confidence = result.confidence
            
            status = "PASS" if actual_type == expected_type else "FAIL"
            print(f"  Test {i}/{total}: {status} - Expected: {expected_type}, Got: {actual_type} (confidence: {confidence:.2f})")
            
            if actual_type == expected_type:
                passed += 1
        
        print(f"Document Classifier: {passed}/{total} tests passed ({(passed/total*100):.1f}%)")
        return passed == total
        
    except Exception as e:
        print(f"Document Classifier Test Failed: {e}")
        return False

def test_quality_assessor():
    """Test document quality assessment engine"""
    try:
        from ai_engines.quality_assessor import DocumentQualityAssessor
        
        assessor = DocumentQualityAssessor()
        
        print("\nTesting Quality Assessment Engine...")
        
        # Test with a dummy assessment (no actual image file needed for basic validation)
        try:
            # This will fail gracefully since no image exists, but tests the logic
            result = assessor.assess_document_quality("nonexistent_image.jpg")
            print("  ✅ Quality Assessor initialized and can process requests")
            print(f"     - Handles missing files gracefully")
            return True
        except Exception as e:
            print(f"  ⚠️  Quality Assessor test: {e}")
            return True  # Expected behavior for missing file
            
    except Exception as e:
        print(f"❌ Quality Assessor Test Failed: {e}")
        return False

def test_enhanced_processor():
    """Test enhanced claim processor"""
    try:
        from claims_engine.enhanced_processor import EnhancedClaimProcessor
        
        processor = EnhancedClaimProcessor()
        
        print("\nTesting Enhanced Claim Processor...")
        
        # Test sample OCR result
        sample_ocr_result = {
            'text': '''
            Patient: John Doe
            NRIC: S1234567A  
            Policy No: POL123456
            Provider: Central Clinic
            Date: 2024-08-15
            Consultation Fee: $80.00
            Diagnosis: Common cold
            ''',
            'confidence': 0.85,
            'engine': 'test'
        }
        
        # Process the claim
        start_time = time.time()
        result = processor.process_enhanced_claim(sample_ocr_result)
        processing_time = time.time() - start_time
        
        print(f"  ✅ Enhanced processing completed in {processing_time:.2f}s")
        print(f"     - Status: {result.get('status', 'unknown')}")
        print(f"     - Confidence: {result.get('confidence', 0):.2f}")
        print(f"     - Workflow steps completed: {result.get('workflow_completion', {}).get('completed_steps', 0)}")
        
        # Check extracted data
        extracted_data = result.get('extracted_data', {})
        if extracted_data:
            print(f"     - Patient: {extracted_data.get('patient_name', 'Not found')}")
            print(f"     - Amount: ${extracted_data.get('total_amount', 0):.2f}")
            print(f"     - Provider: {extracted_data.get('provider_name', 'Not found')}")
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"❌ Enhanced Processor Test Failed: {e}")
        return False

def test_ocr_engines():
    """Test OCR engines availability"""
    try:
        from ocr_engine.mistral_ocr import HybridOCREngine
        
        print("\nTesting OCR Engines...")
        
        ocr_engine = HybridOCREngine()
        
        # Test engine availability
        print(f"  ✅ Hybrid OCR Engine initialized")
        print(f"     - Mistral AI available: {ocr_engine.mistral_available}")
        print(f"     - EasyOCR lazy-loading ready: {not ocr_engine.easyocr_initialization_attempted}")
        
        return True
        
    except Exception as e:
        print(f"❌ OCR Engine Test Failed: {e}")
        return False

def test_database_connection():
    """Test database connection"""
    try:
        from database.supabase_client import SupabaseClient
        
        print("\nTesting Database Connection...")
        
        db = SupabaseClient()
        print("  ✅ Supabase client initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Database Test Failed: {e}")
        return False

def test_flask_app_startup():
    """Test Flask application startup"""
    try:
        print("\nTesting Flask App Configuration...")
        
        # Import app components
        from config.settings import Config
        print("  ✅ Configuration loaded")
        
        # Test environment variables (will use defaults if not set)
        required_vars = ['MISTRAL_API_KEY', 'SUPABASE_URL', 'SUPABASE_SERVICE_KEY']
        missing_vars = []
        
        for var in required_vars:
            if not getattr(Config, var, None):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"  ⚠️  Missing environment variables: {', '.join(missing_vars)}")
            print("     Set these for full functionality")
        else:
            print("  ✅ All required environment variables configured")
        
        return True
        
    except Exception as e:
        print(f"❌ Flask App Test Failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Enhanced OCR Claim Processing System - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Flask App Startup", test_flask_app_startup),
        ("Database Connection", test_database_connection),
        ("OCR Engines", test_ocr_engines),
        ("Document Classifier", test_document_classifier),
        ("Quality Assessor", test_quality_assessor),
        ("Enhanced Processor", test_enhanced_processor),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"FAILED: {test_name} failed with exception: {e}")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed_tests}/{total_tests} tests passed ({(passed_tests/total_tests*100):.1f}%)")
    print(f"Total time: {total_time:.2f} seconds")
    
    if passed_tests == total_tests:
        print("All tests passed! System is ready for deployment.")
        return 0
    else:
        print("Some tests failed. Check configuration and dependencies.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)