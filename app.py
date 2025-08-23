import os
import uuid
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import time
from datetime import datetime

from config.settings import Config
from database.supabase_client import SupabaseClient
from ocr_engine.mistral_ocr import HybridOCREngine
from claims_engine.processor import ClaimProcessor
from claims_engine.enhanced_processor import EnhancedClaimProcessor

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Create upload folder
Config.create_upload_folder()

# Initialize services
try:
    db = SupabaseClient()
    ocr_engine = HybridOCREngine()
    claim_processor = ClaimProcessor()
    enhanced_claim_processor = EnhancedClaimProcessor()
except Exception as e:
    print(f"Error initializing services: {e}")
    db = None
    ocr_engine = None
    claim_processor = None
    enhanced_claim_processor = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def validate_file(file):
    """Validate uploaded file"""
    if not file or file.filename == '':
        return False, "No file selected"
    
    if not allowed_file(file.filename):
        return False, "File type not allowed. Supported: PDF, PNG, JPG, JPEG, TIFF, BMP"
    
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
        
        # Process OCR
        ocr_results = ocr_engine.process_image(filepath, selected_languages)
        
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
            'processing_engine': 'paddleocr'
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
        
        return render_template('enhanced_results.html', 
                             claim=claim_data, 
                             ocr=ocr_data,
                             enhanced_data=enhanced_data,
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

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Validate environment variables in development
    if app.config['DEBUG']:
        try:
            Config.validate_required_vars()
        except ValueError as e:
            print(f"Configuration Error: {e}")
            print("Please set the required environment variables.")
    
    app.run(debug=app.config['DEBUG'], host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))