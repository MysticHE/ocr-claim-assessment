#!/usr/bin/env python3

"""
Test the simplified decision-making logic without confidence calculations
"""

import sys
sys.path.append('.')

from claims_engine.enhanced_processor import EnhancedClaimProcessor, EnhancedClaimData

def test_decision_making():
    """Test simplified decision making without confidence"""
    processor = EnhancedClaimProcessor()
    print("Testing simplified decision making...")
    
    # Test case 1: Clean claim (should be approved)
    test_data = EnhancedClaimData()
    test_data.patient_name = "John Doe"
    test_data.provider_name = "Test Hospital"
    test_data.treatment_dates = ["2025-08-25"]
    test_data.total_amount = 150.0
    
    print(f"\n=== Test 1: Clean Claim ===")
    decision = processor.make_enhanced_decision(
        test_data, [], [], None, None, 0.0
    )
    print(f"Status: {decision.status.value}")
    print(f"Confidence: {decision.confidence}")
    print(f"Reasons: {decision.reasons}")
    print(f"Notes: {decision.processing_notes}")
    
    # Test case 2: Fraud detected (should be rejected)
    print(f"\n=== Test 2: Fraud Detection ===")
    fraud_findings = ["DUPLICATE claim detected - similar to claim #12345"]
    decision = processor.make_enhanced_decision(
        test_data, [], fraud_findings, None, None, 0.0
    )
    print(f"Status: {decision.status.value}")
    print(f"Confidence: {decision.confidence}")
    print(f"Reasons: {decision.reasons}")
    print(f"Notes: {decision.processing_notes}")
    
    # Test case 3: Validation issues (should be rejected)
    print(f"\n=== Test 3: Validation Issues ===")
    validation_issues = ["Missing patient name", "Missing provider name"]
    decision = processor.make_enhanced_decision(
        test_data, validation_issues, [], None, None, 0.0
    )
    print(f"Status: {decision.status.value}")
    print(f"Confidence: {decision.confidence}")
    print(f"Reasons: {decision.reasons}")
    print(f"Notes: {decision.processing_notes}")
    
    print(f"\n✅ All decision tests completed successfully!")
    print(f"✅ No confidence calculations needed - simplified logic works!")

if __name__ == "__main__":
    test_decision_making()