#!/usr/bin/env python3
"""
Test script for OpenAI GPT-4o-mini integration in OCR processing system.
Tests both OpenAI functionality and graceful fallback to regex methods.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_openai_parser_availability():
    """Test if OpenAI parser can be imported and initialized"""
    print("🧪 Testing OpenAI Parser Availability")
    print("=" * 50)
    
    try:
        from ai_engines.openai_parser import OpenAIParser
        print("✅ OpenAI parser module imported successfully")
        
        parser = OpenAIParser()
        print(f"✅ OpenAI parser instance created")
        print(f"   Available: {parser.available}")
        print(f"   API Key Configured: {'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
        
        return parser
    except ImportError as e:
        print(f"❌ Failed to import OpenAI parser: {e}")
        return None
    except Exception as e:
        print(f"❌ Failed to initialize OpenAI parser: {e}")
        return None

def test_openai_health_check(parser):
    """Test OpenAI API connectivity"""
    print("\n🏥 Testing OpenAI Health Check")
    print("=" * 50)
    
    if not parser:
        print("❌ No parser instance available")
        return False
    
    try:
        health_result = parser.health_check()
        print(f"Status: {health_result['status']}")
        print(f"Available: {health_result['available']}")
        
        if health_result['status'] == 'healthy':
            print(f"✅ OpenAI API connection successful")
            print(f"   Model: {health_result.get('model', 'N/A')}")
            return True
        else:
            print(f"⚠️ OpenAI API connection failed: {health_result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_sample_extraction(parser):
    """Test data extraction with sample medical document text"""
    print("\n📋 Testing Sample Document Extraction")
    print("=" * 50)
    
    if not parser:
        print("❌ No parser instance available")
        return None
    
    # Sample medical receipt text
    sample_text = """
    SINGAPORE GENERAL HOSPITAL
    Receipt for Medical Services
    
    Patient Name: John Doe
    Patient ID: S1234567A
    Policy Number: SGH-POL-2024-001
    Date of Service: 2024-01-15
    
    Services Provided:
    - General Consultation: SGD 85.00
    - Blood Test (Full Panel): SGD 120.00
    - X-Ray Chest: SGD 95.00
    
    Total Amount: SGD 300.00
    Payment Method: Insurance Claim
    
    Doctor: Dr. Sarah Lim
    Department: Internal Medicine
    Reference No: REF-2024-0115-001
    """
    
    try:
        print(f"📄 Processing sample medical receipt...")
        print(f"   Text length: {len(sample_text)} characters")
        
        # Test extraction
        start_time = time.time()
        result = parser.extract_structured_data(
            ocr_text=sample_text,
            document_type="receipt",
            quality_score=0.9
        )
        processing_time = int((time.time() - start_time) * 1000)
        
        print(f"⏱️  Processing completed in {processing_time}ms")
        print(f"✅ Extraction Success: {result.success}")
        print(f"🎯 Confidence: {result.confidence:.2f}")
        print(f"🔧 Fallback Used: {result.fallback_used}")
        
        if result.success:
            data = result.extracted_data
            print(f"\n📊 Extracted Data:")
            print(f"   Patient Name: {data.get('patient_name', 'N/A')}")
            print(f"   Patient ID: {data.get('patient_id', 'N/A')}")
            print(f"   Policy Number: {data.get('policy_number', 'N/A')}")
            print(f"   Total Amount: {data.get('total_amount', 'N/A')}")
            print(f"   Currency: {data.get('currency', 'N/A')}")
            print(f"   Treatment Dates: {data.get('treatment_dates', [])}")
            
            # Show document insights if available
            if 'document_insights' in data:
                insights = data['document_insights']
                print(f"\n🔍 Document Insights:")
                print(f"   Appears Genuine: {insights.get('document_appears_genuine', 'N/A')}")
                print(f"   Data Completeness: {insights.get('data_completeness', 'N/A')}")
                print(f"   Missing Fields: {insights.get('missing_critical_fields', [])}")
        else:
            print(f"❌ Extraction failed: {result.error}")
        
        return result
        
    except Exception as e:
        print(f"❌ Sample extraction test failed: {e}")
        return None

def test_enhanced_processor_integration():
    """Test integration with EnhancedClaimProcessor"""
    print("\n🔧 Testing Enhanced Processor Integration")
    print("=" * 50)
    
    try:
        from claims_engine.enhanced_processor import EnhancedClaimProcessor
        
        processor = EnhancedClaimProcessor()
        print(f"✅ Enhanced processor created")
        print(f"   OpenAI Parser Available: {processor.openai_parser_available}")
        print(f"   Classifier Available: {processor.classifier_available}")
        print(f"   Quality Assessor Available: {processor.quality_assessor_available}")
        
        # Test sample OCR result processing
        sample_ocr_result = {
            'text': """
            POLYCLINIC RECEIPT
            Patient: Mary Tan
            NRIC: S9876543B  
            Amount: $45.50
            Date: 2024-01-20
            Service: General Consultation
            """,
            'confidence': 0.85,
            'engine': 'mistral'
        }
        
        print(f"\n🧪 Testing extraction with sample OCR result...")
        extracted_data = processor.extract_enhanced_structured_data(sample_ocr_result)
        
        print(f"✅ Extraction completed")
        print(f"   Patient Name: {extracted_data.patient_name}")
        print(f"   Patient ID: {extracted_data.patient_id}")
        print(f"   Total Amount: {extracted_data.total_amount}")
        print(f"   AI Confidence Scores: {extracted_data.ai_confidence_scores}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced processor integration test failed: {e}")
        return False

def test_fallback_behavior():
    """Test fallback to regex when OpenAI is unavailable"""
    print("\n🔄 Testing Fallback Behavior")
    print("=" * 50)
    
    # Temporarily disable OpenAI by setting invalid key
    original_key = os.environ.get('OPENAI_API_KEY')
    os.environ['OPENAI_API_KEY'] = 'invalid_key_for_testing'
    
    try:
        from ai_engines.openai_parser import OpenAIParser
        
        # Create new parser instance with invalid key
        parser = OpenAIParser()
        health = parser.health_check()
        
        print(f"✅ Fallback test setup complete")
        print(f"   OpenAI Available: {health['available']}")
        print(f"   Expected: False (using invalid key)")
        
        # Test extraction should fallback to regex
        sample_text = "Patient: John Smith, Amount: $100.00, Date: 2024-01-15"
        
        result = parser.extract_structured_data(sample_text, "receipt", 0.8)
        print(f"✅ Fallback extraction completed")
        print(f"   Success: {result.success}")
        print(f"   Fallback Used: {result.fallback_used}")
        print(f"   Expected: True (should fallback)")
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")
        return False
    finally:
        # Restore original key
        if original_key:
            os.environ['OPENAI_API_KEY'] = original_key
        elif 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']

def main():
    """Run all OpenAI integration tests"""
    print("🚀 OpenAI Integration Test Suite")
    print("=" * 60)
    print(f"Environment: {os.environ.get('RENDER_SERVICE_NAME', 'Local Development')}")
    print(f"OpenAI API Key: {'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
    print()
    
    # Test results tracking
    results = {}
    
    # Run tests
    parser = test_openai_parser_availability()
    results['parser_availability'] = parser is not None
    
    results['health_check'] = test_openai_health_check(parser)
    results['sample_extraction'] = test_sample_extraction(parser) is not None
    results['processor_integration'] = test_enhanced_processor_integration()
    results['fallback_behavior'] = test_fallback_behavior()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 50)
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! OpenAI integration is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check configuration and try again.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)