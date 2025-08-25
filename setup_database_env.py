#!/usr/bin/env python3
"""
Database Environment Setup Helper
Run this script to configure your Supabase environment variables for duplicate detection
"""
import os
import sys

def setup_database_environment():
    """Interactive setup for database environment variables"""
    print("Database Environment Setup for OCR System")
    print("=" * 50)
    
    # Check current environment
    current_url = os.environ.get('SUPABASE_URL')
    current_key = os.environ.get('SUPABASE_SERVICE_KEY')
    
    print(f"Current SUPABASE_URL: {'Set' if current_url else 'Not set'}")
    print(f"Current SUPABASE_SERVICE_KEY: {'Set' if current_key else 'Not set'}")
    print()
    
    if current_url and current_key:
        print("Environment variables are already configured!")
        
        # Test database connection
        try:
            from database.supabase_client import SupabaseClient
            client = SupabaseClient()
            if client.test_connection():
                print("Database connection test passed!")
                print("Duplicate detection should work properly now")
                return True
            else:
                print("Database connection test failed")
                print("   Please check your Supabase URL and service key")
                return False
        except Exception as e:
            print(f"Error testing database connection: {e}")
            return False
    
    print("To enable advanced duplicate detection, you need:")
    print("1. Your Supabase project URL (e.g., https://xyz.supabase.co)")
    print("2. Your Supabase service key (from your project settings)")
    print()
    
    # Get user input
    if input("Would you like to set up environment variables now? (y/n): ").lower().startswith('y'):
        url = input("Enter your Supabase URL: ").strip()
        key = input("Enter your Supabase Service Key: ").strip()
        
        if url and key:
            # Create .env file
            env_content = f"""# Supabase Database Configuration
SUPABASE_URL={url}
SUPABASE_SERVICE_KEY={key}
"""
            
            with open('.env', 'w') as f:
                f.write(env_content)
            
            print("\nCreated .env file with your database configuration")
            print("Note: You may need to restart your application to load the new environment variables")
            print("Keep your .env file secure and don't commit it to version control")
            
            # Try to load and test
            os.environ['SUPABASE_URL'] = url
            os.environ['SUPABASE_SERVICE_KEY'] = key
            
            try:
                from database.supabase_client import SupabaseClient
                client = SupabaseClient()
                if client.test_connection():
                    print("✅ Database connection test passed!")
                    return True
                else:
                    print("❌ Database connection test failed - please check your credentials")
                    return False
            except Exception as e:
                print(f"❌ Error testing database connection: {e}")
                return False
        else:
            print("❌ Invalid input - setup cancelled")
            return False
    else:
        print("⚠️ Setup cancelled - duplicate detection will use fallback mode only")
        return False

def check_database_status():
    """Check current database configuration status"""
    print("Database Configuration Status")
    print("-" * 30)
    
    # Environment variables
    url = os.environ.get('SUPABASE_URL')
    key = os.environ.get('SUPABASE_SERVICE_KEY')
    
    print(f"SUPABASE_URL: {'Set' if url else 'Missing'}")
    print(f"SUPABASE_SERVICE_KEY: {'Set' if key else 'Missing'}")
    
    if not url or not key:
        print("\nDatabase environment not configured")
        print("   Duplicate detection will use basic fallback mode only")
        print("   Run 'python setup_database_env.py' to configure")
        return False
    
    # Test connection
    try:
        from database.supabase_client import SupabaseClient
        client = SupabaseClient()
        
        if client.test_connection():
            print("Database connection: Working")
            
            # Get basic stats
            stats = client.get_claim_stats()
            print(f"Total claims in database: {stats.get('total_claims', 0)}")
            print("Advanced duplicate detection: Enabled")
            return True
        else:
            print("Database connection: Failed")
            print("   Check your Supabase URL and service key")
            return False
            
    except Exception as e:
        print(f"Database connection error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'status':
        check_database_status()
    else:
        setup_database_environment()