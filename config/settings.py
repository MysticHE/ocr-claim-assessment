import os
from typing import List

class Config:
    """Application configuration class with environment variable support"""
    
    # Security
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY') or 'dev-key-change-in-production'
    
    # Supabase Configuration
    SUPABASE_URL = os.environ.get('SUPABASE_URL')
    SUPABASE_ANON_KEY = os.environ.get('SUPABASE_ANON_KEY')
    SUPABASE_SERVICE_KEY = os.environ.get('SUPABASE_SERVICE_KEY')
    
    # OCR Configuration
    MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY')
    SUPPORTED_LANGUAGES = os.environ.get('SUPPORTED_LANGUAGES', 'en,ch_sim,ms,ta,korean').split(',')
    
    # OCR Engine Selection
    # Options: 'hybrid' (Mistral + EasyOCR fallback) or 'mistral_only' (streamlined)
    OCR_ENGINE_TYPE = os.environ.get('OCR_ENGINE_TYPE', 'hybrid')
    
    # File Upload Configuration
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', 16 * 1024 * 1024))  # 16MB
    # Mistral Pixtral vision model supported formats only
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    
    # Application Settings
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    ENVIRONMENT = os.environ.get('ENVIRONMENT', 'production')
    
    # Language Mappings for OCR
    LANGUAGE_MAPPINGS = {
        'en': 'English',
        'ch_sim': 'Chinese (Simplified)',
        'ch_tra': 'Chinese (Traditional)', 
        'ms': 'Malay',
        'ta': 'Tamil',
        'korean': 'Korean',
        'japan': 'Japanese',
        'fr': 'French',
        'de': 'German',
        'es': 'Spanish',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'ar': 'Arabic',
        'hi': 'Hindi'
    }
    
    @staticmethod
    def validate_required_vars():
        """Validate that all required environment variables are set"""
        required_vars = ['SUPABASE_URL', 'SUPABASE_SERVICE_KEY', 'FLASK_SECRET_KEY']
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    @staticmethod
    def create_upload_folder():
        """Create upload folder if it doesn't exist"""
        upload_dir = Config.UPLOAD_FOLDER
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir, exist_ok=True)
        return upload_dir