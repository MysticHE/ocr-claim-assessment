import os
import uuid
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import time
from datetime import datetime

from config.settings import Config
from database.supabase_client import SupabaseClient
from ocr_engine.mistral_only_ocr import MistralOnlyOCREngine
from claims_engine.processor import ClaimProcessor
from claims_engine.enhanced_processor import EnhancedClaimProcessor

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Create upload folder
Config.create_upload_folder()

# Print startup information
print("🚀 Starting Enhanced OCR Claim Processing System...")
print(f"   Environment: {os.environ.get('RENDER_SERVICE_NAME', 'Local')}")
print(f"   Python: {os.sys.version}")

# Check required environment variables
required_env_vars = ['MISTRAL_API_KEY', 'SUPABASE_URL', 'SUPABASE_SERVICE_KEY']
missing_vars = []
for var in required_env_vars:
    if not os.environ.get(var):
        missing_vars.append(var)

if missing_vars:
    print(f"⚠️  Missing required environment variables: {', '.join(missing_vars)}")
    print("   Application will run in limited mode.")

# Initialize services with proper error handling
db = None
ocr_engine = None
claim_processor = None
enhanced_claim_processor = None

print("\n🔧 Initializing services...")

try:
    db = SupabaseClient()
    print("✓ Database client initialized")
except Exception as e:
    print(f"✗ Database initialization failed: {e}")
    db = None

try:
    ocr_engine = MistralOnlyOCREngine()
    print("✓ Mistral OCR engine initialized")
except Exception as e:
    print(f"✗ OCR engine initialization failed: {e}")
    print(f"   Error type: {type(e).__name__}")
    ocr_engine = None

try:
    claim_processor = ClaimProcessor()
    print("✓ Claim processor initialized")
except Exception as e:
    print(f"✗ Claim processor initialization failed: {e}")
    claim_processor = None

try:
    enhanced_claim_processor = EnhancedClaimProcessor()
    print("✓ Enhanced claim processor initialized")
except Exception as e:
    print(f"✗ Enhanced claim processor initialization failed: {e}")
    enhanced_claim_processor = None

# Service availability checks
def check_services():
    """Check if required services are available"""
    issues = []
    if not db:
        issues.append("Database connection not available")
    if not ocr_engine:
        issues.append("OCR engine not available - check Mistral API key")
    if not enhanced_claim_processor:
        issues.append("Enhanced claim processor not available")
    
    if issues:
        print("⚠️  Service issues detected:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✓ All services initialized successfully")
    
    return len(issues) == 0

# Check services on startup
services_ready = check_services()

# Startup summary
print(f"\n📊 Startup Summary:")
print(f"   Services Ready: {services_ready}")
print(f"   Database: {'✓' if db else '✗'}")
print(f"   OCR Engine: {'✓' if ocr_engine else '✗'}")
print(f"   Claim Processor: {'✓' if claim_processor else '✗'}")
print(f"   Enhanced Processor: {'✓' if enhanced_claim_processor else '✗'}")

if not services_ready:
    print("\n⚠️  Some services failed to initialize, but application will continue.")
    print("   Check /health endpoint for detailed status.")

print(f"\n🌐 Application starting on port {os.environ.get('PORT', '5000')}...")
print("✅ Startup complete - Ready for requests")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def validate_file(file):
    """Validate uploaded file"""
    if not file or file.filename == '':
        return False, "No file selected"
    
    if not allowed_file(file.filename):
        return False, "File type not allowed. Supported: PDF, PNG, JPG, JPEG, GIF, WebP, TIFF, BMP"
    
    return True, "Valid file"

@app.route('/')
def index():
    """Main upload page"""
    return render_template('index.html', 
                         languages=Config.LANGUAGE_MAPPINGS,
                         max_file_size=Config.MAX_FILE_SIZE)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and OCR processing"""
    try:
        # Check if services are initialized
        if not all([db, ocr_engine, enhanced_claim_processor]):
            return jsonify({
                'success': False,
                'error': 'Services not properly initialized. Please try again later.'
            }), 500
        
        # Validate file
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        is_valid, message = validate_file(file)
        
        if not is_valid:
            return jsonify({'success': False, 'error': message}), 400
        
        # Get form data
        selected_languages = request.form.getlist('languages')
        if not selected_languages:
            selected_languages = ['en']  # Default to English
            
        # Generate unique claim ID
        claim_id = str(uuid.uuid4())
        
        # Secure filename and save
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{claim_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        file.save(filepath)
        
        # Record processing start time
        start_time = time.time()
        
        # Initial claim record
        claim_data = {
            'id': claim_id,
            'file_name': filename,
            'file_size': os.path.getsize(filepath),
            'claim_status': 'processing',
            'created_at': datetime.utcnow().isoformat(),
            'metadata': {
                'selected_languages': selected_languages,
                'original_filename': filename,
                'stored_filename': unique_filename
            }
        }
        
        # Save to database
        db.insert_claim(claim_data)
        
        # Process OCR with proper error handling
        ocr_results = ocr_engine.process_image(filepath, selected_languages)
        
        # Check if OCR was successful
        if not ocr_results.get('success', False):
            error_type = ocr_results.get('error_type', 'unknown')
            error_msg = ocr_results.get('error', 'OCR processing failed')
            
            # Update claim status to failed
            db.update_claim_status(
                claim_id, 
                'failed', 
                0.0,
                {
                    'error_message': error_msg,
                    'error_type': error_type,
                    'processing_time_ms': int((time.time() - start_time) * 1000),
                    'selected_languages': selected_languages
                }
            )
            
            # Log the error
            db.insert_log(claim_id, 'error', f'OCR processing failed: {error_msg}')
            
            # Clean up file
            try:
                os.remove(filepath)
            except:
                pass
            
            # Return user-friendly error message based on error type
            if error_type == 'service_unavailable':
                error_message = "OCR service is temporarily unavailable. Please ensure you have a valid Mistral API key configured. Check service status at /health endpoint."
            elif error_type == 'file_not_found':
                error_message = "The uploaded file could not be processed. Please try uploading the file again."
            elif error_type == 'api_error':
                error_message = "There was an issue connecting to the OCR service. Please check your internet connection and Mistral API key. Check service status at /health endpoint."
            else:
                error_message = f"OCR processing failed: {error_msg}. Check service status at /health endpoint."
            
            return jsonify({
                'success': False,
                'error': error_message,
                'error_type': error_type
            }), 400
        
        # Process claim logic with enhanced AI workflow
        claim_decision = enhanced_claim_processor.process_enhanced_claim(ocr_results, filepath)
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        # Update claim with enhanced results
        db.update_claim_status(
            claim_id, 
            claim_decision['status'], 
            claim_decision['confidence'],
            {
                'ocr_text': ocr_results.get('text', ''),
                'language_detected': ocr_results.get('detected_language', ''),
                'processing_time_ms': processing_time,
                'claim_amount': claim_decision.get('amount', 0),
                'decision_reasons': claim_decision.get('reasons', []),
                'enhanced_results': claim_decision if claim_decision.get('success', False) else None
            }
        )
        
        # Save OCR results
        db.insert_ocr_result({
            'claim_id': claim_id,
            'language_code': ','.join(selected_languages),
            'extracted_text': ocr_results.get('text', ''),
            'confidence_score': ocr_results.get('confidence', 0),
            'bounding_boxes': ocr_results.get('boxes', []),
            'processing_engine': ocr_results.get('engine', 'mistral')
        })
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
            
        return jsonify({
            'success': True,
            'claim_id': claim_id,
            'redirect_url': url_for('view_enhanced_results', claim_id=claim_id)
        })
        
    except RequestEntityTooLarge:
        return jsonify({
            'success': False,
            'error': f'File too large. Maximum size: {Config.MAX_FILE_SIZE // (1024*1024)}MB'
        }), 413
        
    except Exception as e:
        # Log error
        if db:
            try:
                db.insert_log(claim_id if 'claim_id' in locals() else None, 'error', str(e))
            except:
                pass
        
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred. Please try again.'
        }), 500

@app.route('/results/<claim_id>')
def view_results(claim_id):
    """Display processing results"""
    try:
        if not db:
            flash('Database service unavailable', 'error')
            return redirect(url_for('index'))
            
        # Get claim data
        claim_result = db.get_claim(claim_id)
        
        if not claim_result.data:
            flash('Claim not found', 'error')
            return redirect(url_for('index'))
            
        claim_data = claim_result.data[0]
        
        # Get OCR results
        ocr_result = db.get_ocr_result(claim_id)
        ocr_data = ocr_result.data[0] if ocr_result.data else {}
        
        return render_template('results.html', 
                             claim=claim_data, 
                             ocr=ocr_data,
                             languages=Config.LANGUAGE_MAPPINGS)
        
    except Exception as e:
        flash('Error retrieving results', 'error')
        return redirect(url_for('index'))

@app.route('/enhanced/<claim_id>')
def view_enhanced_results(claim_id):
    """Display enhanced processing results with AI workflow"""
    try:
        if not db:
            flash('Database service unavailable', 'error')
            return redirect(url_for('index'))
            
        # Get claim data
        claim_result = db.get_claim(claim_id)
        
        if not claim_result.data:
            flash('Claim not found', 'error')
            return redirect(url_for('index'))
            
        claim_data = claim_result.data[0]
        
        # Get OCR results
        ocr_result = db.get_ocr_result(claim_id)
        ocr_data = ocr_result.data[0] if ocr_result.data else {}
        
        # Try to parse enhanced data from metadata
        enhanced_data = None
        if claim_data.get('metadata') and isinstance(claim_data.get('metadata'), dict):
            enhanced_data = claim_data.get('metadata').get('enhanced_results')
        
        # Structure data as expected by template
        result = {
            'claim_id': claim_id,
            'status': claim_data.get('claim_status', 'unknown'),
            'confidence': claim_data.get('confidence_score', 0),
            'processing_time_ms': claim_data.get('metadata', {}).get('processing_time_ms', 0),
            'claim_amount': claim_data.get('claim_amount', 0),
            'ocr_text': claim_data.get('ocr_text', ''),
            'language_detected': claim_data.get('language_detected', ''),
            'file_name': claim_data.get('file_name', ''),
            'created_at': claim_data.get('created_at', ''),
            'decision_reasons': claim_data.get('metadata', {}).get('decision_reasons', []),
            
            # Enhanced AI results - properly map from enhanced_data
            'workflow_steps': enhanced_data.get('workflow_steps', []) if enhanced_data else [],
            'document_classification': enhanced_data.get('document_classification', {}) if enhanced_data else {},
            'quality_assessment': enhanced_data.get('quality_assessment', {}) if enhanced_data else {},
            'extracted_data': enhanced_data.get('extracted_data', {}) if enhanced_data else {},
            'fraud_findings': enhanced_data.get('fraud_findings', []) if enhanced_data else [],
            'validation_issues': enhanced_data.get('validation_issues', []) if enhanced_data else [],
            'ai_engines_used': enhanced_data.get('ai_engines_used', []) if enhanced_data else [],
            'workflow_completion': enhanced_data.get('workflow_completion', {}) if enhanced_data else {},
            
            # For backward compatibility and debugging
            'metadata': claim_data.get('metadata', {}),
            'enhanced_data_available': enhanced_data is not None
        }
        
        return render_template('enhanced_results.html', 
                             result=result,
                             languages=Config.LANGUAGE_MAPPINGS)
        
    except Exception as e:
        flash('Error retrieving enhanced results', 'error')
        return redirect(url_for('index'))

@app.route('/api/status/<claim_id>')
def get_claim_status(claim_id):
    """Get claim processing status via API"""
    try:
        if not db:
            return jsonify({'error': 'Database unavailable'}), 500
            
        result = db.get_claim(claim_id)
        if not result.data:
            return jsonify({'error': 'Claim not found'}), 404
            
        claim = result.data[0]
        return jsonify({
            'claim_id': claim_id,
            'status': claim.get('claim_status'),
            'confidence': claim.get('confidence_score'),
            'processing_time': claim.get('metadata', {}).get('processing_time_ms', 0)
        })
        
    except Exception as e:
        return jsonify({'error': 'Failed to get status'}), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': f'File too large. Maximum size: {Config.MAX_FILE_SIZE // (1024*1024)}MB'
    }), 413

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring OCR service status"""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': time.time(),
            'services': {}
        }
        
        # Check database connection
        if db:
            try:
                # Test database connection with a simple query
                # This will fail if database is not accessible
                db.supabase.table('claims').select('id').limit(1).execute()
                health_status['services']['database'] = {
                    'status': 'healthy',
                    'message': 'Database connection successful'
                }
            except Exception as e:
                health_status['status'] = 'unhealthy'
                health_status['services']['database'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        else:
            health_status['status'] = 'unhealthy'
            health_status['services']['database'] = {
                'status': 'unhealthy',
                'error': 'Database client not initialized'
            }
        
        # Check OCR engine
        if ocr_engine:
            ocr_health = ocr_engine.health_check()
            health_status['services']['ocr'] = ocr_health
            if ocr_health['status'] != 'healthy':
                health_status['status'] = 'unhealthy'
        else:
            health_status['status'] = 'unhealthy'
            health_status['services']['ocr'] = {
                'status': 'unhealthy',
                'error': 'OCR engine not initialized'
            }
        
        # Check processors
        health_status['services']['claim_processor'] = {
            'status': 'healthy' if claim_processor else 'unhealthy',
            'initialized': bool(claim_processor)
        }
        
        health_status['services']['enhanced_processor'] = {
            'status': 'healthy' if enhanced_claim_processor else 'unhealthy',
            'initialized': bool(enhanced_claim_processor)
        }
        
        return jsonify(health_status), 200 if health_status['status'] == 'healthy' else 503
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': time.time()
        }), 503

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

# Production deployment uses gunicorn - no local server needed