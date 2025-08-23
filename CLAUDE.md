# Claude Code Project Instructions

## Project Overview
OCR Claim Assessment System - AI-powered document processing for insurance claims using OCR and natural language processing.

## Key Commands
- `npm run lint` - Not available (Python project)
- `npm run typecheck` - Not available (Python project)  
- `python -m pytest` - Run tests (if test framework is added)
- `gunicorn app:app` - Start production server

## Dependencies
- **Critical**: mistralai>=1.0.0, supabase>=2.0.0 (resolved httpx conflict)
- **OCR Engines**: Mistral AI (primary) + EasyOCR fallback (PaddleOCR completely removed)
- **Web Framework**: flask==3.0.0
- **Production**: gunicorn==21.2.0

## Environment Variables Required
- `MISTRAL_API_KEY` - Mistral AI API key for OCR processing
- `SUPABASE_URL` - Supabase project URL  
- `SUPABASE_SERVICE_KEY` - Supabase service key for database access

## Architecture
- **Frontend**: Flask templates with responsive CSS
- **OCR Processing**: HybridOCREngine (Mistral AI + EasyOCR fallback)
- **Database**: Supabase (PostgreSQL)
- **Deployment**: Render with Poetry/pip

## Recent Issues Resolved
- Fixed httpx dependency conflict between mistralai and supabase packages
- Removed PaddleOCR due to installation issues, switched to HybridOCREngine
- App now uses Mistral AI as primary OCR with EasyOCR fallback
- Fixed PaddleOCR warning messages in Render deployment logs by completely removing PaddleOCR initialization

## Testing Strategy
Currently no test framework configured. Consider adding pytest for testing OCR functionality and API endpoints.