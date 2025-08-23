# OCR Claim Assessment System

A comprehensive multi-language OCR and claim processing system built with Flask, PaddleOCR, Mistral AI, and Supabase.

## Features

- **Multi-Language OCR**: Supports 80+ languages including English, Chinese, Malay, Tamil, Korean, Japanese, and more
- **AI-Powered Processing**: Advanced OCR with PaddleOCR and Mistral AI integration
- **Automated Claim Assessment**: Smart decision-making based on configurable business rules
- **Real-time Processing**: Fast document processing with progress tracking
- **Secure Database**: Supabase integration for reliable data storage
- **Responsive UI**: Mobile-friendly design with modern CSS

## Supported Languages

**Priority Languages:**
- English (en)
- Chinese Simplified (ch_sim) / Traditional (chinese_cht)
- Malay (ms)
- Tamil (ta)
- Korean (korean)
- Japanese (japan)

**Additional Languages:**
- French, German, Spanish, Italian, Portuguese, Russian
- Arabic, Hindi, Thai, Vietnamese, Dutch, Polish
- And 60+ more languages

## Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd ocr-claim-assessment
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment setup**
   ```bash
   cp .env.example .env
   # Edit .env with your actual configuration values
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

### Production Deployment (Render)

1. **Supabase Setup**
   - Create account at [supabase.com](https://supabase.com)
   - Create new project
   - Run the SQL schema from the setup guide
   - Get your project URL and API keys

2. **Render Deployment**
   - Connect GitHub repository to Render
   - Create new Web Service
   - Use `render.yaml` configuration
   - Set environment variables in Render dashboard

3. **Environment Variables (Render)**
   ```
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_ANON_KEY=your_anon_key
   SUPABASE_SERVICE_KEY=your_service_key
   MISTRAL_API_KEY=your_mistral_key
   FLASK_SECRET_KEY=your_secret_key_32_chars
   ```

## Architecture

```
ocr-claim-assessment/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── render.yaml           # Render deployment config
├── static/css/           # Stylesheets (main, components, responsive)
├── templates/            # HTML templates
├── config/               # Application configuration
├── database/             # Supabase client and models
├── ocr_engine/          # OCR processing (PaddleOCR, Mistral)
├── claims_engine/       # Business logic and claim processing
└── README.md            # This file
```

## Usage

1. **Upload Document**
   - Drag and drop or select file (PDF, PNG, JPG, JPEG, TIFF, BMP)
   - Maximum file size: 16MB

2. **Select Languages**
   - Choose one or more languages for OCR processing
   - System automatically detects best language match

3. **Processing**
   - OCR extracts text from document
   - Business rules engine evaluates claim
   - Decision made: Approved, Rejected, or Review Required

4. **Results**
   - View processing results and extracted data
   - Download or print results
   - Copy extracted text

## Configuration

### Business Rules

Edit `claims_engine/processor.py` to customize:
- Maximum/minimum claim amounts
- Auto-approval thresholds
- Required fields validation
- Diagnosis code patterns
- Suspicious pattern detection

### OCR Languages

Modify `config/settings.py`:
```python
SUPPORTED_LANGUAGES = 'en,ch_sim,ms,ta,korean,japan,fr,de,es'
```

### File Upload Limits

```python
MAX_FILE_SIZE = 16777216  # 16MB
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'}
```

## API Endpoints

- `GET /` - Main upload page
- `POST /upload` - Process document upload
- `GET /results/<claim_id>` - View processing results
- `GET /api/status/<claim_id>` - Get claim status (JSON)

## Database Schema

### Claims Table
```sql
CREATE TABLE claims (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  file_name TEXT NOT NULL,
  file_size INTEGER,
  language_detected TEXT,
  ocr_text TEXT,
  claim_amount DECIMAL(10,2),
  claim_status TEXT CHECK (claim_status IN ('approved', 'rejected', 'review')),
  confidence_score DECIMAL(3,2),
  processing_time_ms INTEGER,
  metadata JSONB
);
```

### OCR Results Table
```sql
CREATE TABLE ocr_results (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  claim_id UUID REFERENCES claims(id),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  language_code TEXT NOT NULL,
  extracted_text TEXT,
  confidence_score DECIMAL(3,2),
  bounding_boxes JSONB,
  processing_engine TEXT
);
```

## Security Features

- **No API keys in code**: All secrets stored in environment variables
- **File validation**: Type and size restrictions
- **Input sanitization**: Text cleaning and validation
- **Secure uploads**: Temporary file handling with cleanup
- **Database security**: Supabase Row Level Security ready

## Performance

- **OCR Processing**: 5-30 seconds depending on document complexity
- **Languages**: Optimized for English, Chinese, Malay, Tamil, Korean
- **File Limits**: 16MB maximum, multiple formats supported
- **Scalability**: Stateless design for horizontal scaling

## Troubleshooting

### Common Issues

1. **PaddleOCR Installation**
   ```bash
   pip install paddleocr==2.7.3
   # If issues with dependencies:
   pip install opencv-python-headless==4.8.1.78
   ```

2. **Mistral API Issues**
   - Verify API key is set correctly
   - Check rate limits and quotas
   - Ensure model availability

3. **Supabase Connection**
   - Verify URL and keys are correct
   - Check database schema is created
   - Ensure service role key has necessary permissions

4. **File Upload Issues**
   - Check file size limits
   - Verify file type is supported
   - Ensure upload directory permissions

### Performance Tips

- Use appropriate language selection (don't select all languages)
- Optimize image quality before upload
- Monitor Supabase usage quotas
- Use CDN for static assets in production

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests if applicable
5. Update documentation
6. Submit pull request

## License

This project is licensed under the MIT License.

## Support

For support and questions:
- Check the troubleshooting section
- Review configuration settings
- Verify environment variables
- Check Render deployment logs