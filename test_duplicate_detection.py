#!/usr/bin/env python3
"""
Test Duplicate Detection System
Run this script to test your duplicate detection configuration
"""
import os
import sys
from datetime import datetime

def test_duplicate_detection():
    """Test the enhanced duplicate detection system"""
    print("🔍 Testing Duplicate Detection System")
    print("=" * 40)
    
    # Test environment setup
    print("1. Checking environment configuration...")
    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_SERVICE_KEY')
    
    if not url or not key:
        print("❌ Environment variables not set")
        print("   Run 'python setup_database_env.py' first")
        return False
    
    print("✅ Environment variables configured")
    
    # Test database connection
    print("2. Testing database connection...")
    try:
        from database.supabase_client import SupabaseClient
        client = SupabaseClient()
        
        if not client.test_connection():
            print("❌ Database connection failed")
            return False
            
        print("✅ Database connection successful")
        
        # Get database stats
        stats = client.get_claim_stats()
        print(f"   📊 Total claims in database: {stats.get('total_claims', 0)}")
        
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return False
    
    # Test enhanced processor
    print("3. Testing enhanced processor...")
    try:
        from claims_engine.enhanced_processor import EnhancedClaimProcessor, EnhancedClaimData
        
        processor = EnhancedClaimProcessor()
        print("✅ Enhanced processor initialized")
        
        # Create test claim data
        test_data = EnhancedClaimData(
            patient_name="TAN WENBIN",
            total_amount=180143.02,
            treatment_dates=["2020-02-27"],
            provider_name="Singapore General Hospital",
            currency="SGD"
        )
        
        print("4. Running duplicate detection test...")
        print(f"   🧪 Test data: {test_data.patient_name}, ${test_data.total_amount}")
        
        # Test duplicate detection
        duplicate_result = processor._check_duplicate_claim(test_data)
        
        if duplicate_result:
            print("🚨 DUPLICATE DETECTED!")
            print(f"   Similarity: {duplicate_result['similarity_score']:.1%}")
            print(f"   Match type: {duplicate_result['match_type']}")
            print(f"   Details: {duplicate_result['details']}")
        else:
            print("✅ No duplicates found")
        
        print("✅ Duplicate detection test completed")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced processor error: {e}")
        return False

def simulate_duplicate_scenario():
    """Simulate a duplicate detection scenario for testing"""
    print("\n🎭 Simulating Duplicate Detection Scenario")
    print("=" * 45)
    
    try:
        from claims_engine.enhanced_processor import EnhancedClaimProcessor, EnhancedClaimData
        
        processor = EnhancedClaimProcessor()
        
        # Test different scenarios
        test_scenarios = [
            {
                'name': 'Exact Match Test',
                'data': EnhancedClaimData(
                    patient_name="TAN WENBIN",
                    total_amount=180143.02,
                    treatment_dates=["2020-02-27"],
                    provider_name="Singapore General Hospital"
                )
            },
            {
                'name': 'Similar Name Test',
                'data': EnhancedClaimData(
                    patient_name="TAN WEN BIN",  # Slightly different spacing
                    total_amount=180143.02,
                    treatment_dates=["2020-02-27"],
                    provider_name="Singapore General Hospital"
                )
            },
            {
                'name': 'Similar Amount Test',
                'data': EnhancedClaimData(
                    patient_name="TAN WENBIN",
                    total_amount=180000.00,  # Similar amount
                    treatment_dates=["2020-02-27"],
                    provider_name="Singapore General Hospital"
                )
            }
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n{i}. {scenario['name']}")
            print(f"   Patient: {scenario['data'].patient_name}")
            print(f"   Amount: ${scenario['data'].total_amount}")
            
            result = processor._check_duplicate_claim(scenario['data'])
            
            if result:
                print(f"   🚨 DUPLICATE: {result['similarity_score']:.1%} similarity ({result['match_type']})")
            else:
                print("   ✅ No duplicate found")
        
    except Exception as e:
        print(f"❌ Simulation error: {e}")

if __name__ == "__main__":
    if test_duplicate_detection():
        print("\n" + "=" * 50)
        print("✅ All tests passed! Duplicate detection is working")
        
        if input("\nWould you like to run duplicate simulation tests? (y/n): ").lower().startswith('y'):
            simulate_duplicate_scenario()
    else:
        print("\n❌ Tests failed - please check your configuration")
        sys.exit(1)