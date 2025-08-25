#!/usr/bin/env python3

"""
Debug script to test enhanced processing validation issues
"""

import sys
sys.path.append('.')

from claims_engine.enhanced_processor import EnhancedClaimProcessor, EnhancedClaimData

def test_validation_debug():
    """Test validation with debug information"""
    processor = EnhancedClaimProcessor()
    print("Testing enhanced processor validation...")
    
    # Create test data similar to what might be extracted
    test_data = EnhancedClaimData()
    test_data.patient_name = "John Doe"
    test_data.provider_name = "Test Hospital"
    test_data.treatment_dates = ["2025-08-25"]
    test_data.total_amount = 100.0
    test_data.amounts = [100.0]
    test_data.document_type = "receipt"
    test_data.document_classification_confidence = 0.1  # Very low confidence
    test_data.quality_acceptable = False  # Poor quality
    
    print(f"\nTest data:")
    print(f"  Patient: {test_data.patient_name}")
    print(f"  Provider: {test_data.provider_name}")
    print(f"  Dates: {test_data.treatment_dates}")
    print(f"  Amount: {test_data.total_amount}")
    print(f"  Document type: {test_data.document_type}")
    print(f"  Classification confidence: {test_data.document_classification_confidence}")
    print(f"  Quality acceptable: {test_data.quality_acceptable}")
    
    # Test validation
    is_valid, issues = processor.validate_enhanced_claim_data(test_data)
    
    print(f"\nValidation results:")
    print(f"  Valid: {is_valid}")
    print(f"  Issues: {issues}")
    print(f"  Classification threshold: removed (no longer used)")
    
    return is_valid, issues

if __name__ == "__main__":
    test_validation_debug()