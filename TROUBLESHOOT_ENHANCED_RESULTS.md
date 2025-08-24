# Enhanced Results Error Troubleshooting Guide

## Issue: "Error retrieving enhanced results"

### Most Common Causes & Solutions

### 1. **Missing Environment Variables** (Most Likely)
**Symptoms**: Enhanced results page shows generic error message
**Cause**: Missing API keys and database credentials

**Solution**:
```bash
# Set these environment variables:
MISTRAL_API_KEY=your_mistral_api_key_here
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your_service_key_here
```

**For Render Deployment**:
1. Go to your Render dashboard
2. Select your web service
3. Go to "Environment" tab
4. Add the missing environment variables

### 2. **Database Connection Issues**
**Symptoms**: Error mentions "database" or "connection"
**Cause**: Supabase connection failed

**Solution**:
- Verify Supabase URL and service key are correct
- Check Supabase project status (not paused/deleted)
- Ensure database tables exist (claims, ocr_results)

### 3. **Missing Enhanced Data**
**Symptoms**: Claim exists but enhanced results section is empty
**Cause**: Claim was processed before enhanced processing was implemented

**Solution**:
- Reprocess the claim by uploading the document again
- Enhanced processing only works for newly uploaded documents

### 4. **Template/Code Issues**
**Symptoms**: Specific error messages or server errors
**Cause**: Missing data fields or template rendering issues

**Solution**: 
- Use the debug endpoint to diagnose: `/debug/{claim_id}`
- Check server logs for specific error details

## Diagnostic Tools

### 1. Debug Endpoint
Visit: `https://your-app-url.com/debug/{claim_id}`
This will show:
- Environment variables status
- Service initialization status  
- Claim data availability
- Enhanced results data structure
- Specific error messages

### 2. Health Check
Visit: `https://your-app-url.com/health`
Shows overall system health and service status

### 3. Check Server Logs
**Render**: View logs in dashboard under "Logs" tab
**Local**: Check terminal output where `python app.py` is running

## Step-by-Step Diagnosis

### Step 1: Check Environment Variables
```bash
# Run this in your deployment environment
python -c "
import os
print('MISTRAL_API_KEY:', 'SET' if os.getenv('MISTRAL_API_KEY') else 'MISSING')
print('SUPABASE_URL:', 'SET' if os.getenv('SUPABASE_URL') else 'MISSING')
print('SUPABASE_SERVICE_KEY:', 'SET' if os.getenv('SUPABASE_SERVICE_KEY') else 'MISSING')
"
```

### Step 2: Test Database Connection
```bash
python -c "
from database.supabase_client import SupabaseClient
try:
    db = SupabaseClient()
    print('Database connection: SUCCESS')
except Exception as e:
    print(f'Database connection: FAILED - {e}')
"
```

### Step 3: Test Enhanced Processing
```bash
python simple_test.py
```

### Step 4: Check Specific Claim
1. Go to `/debug/{claim_id}` 
2. Look for:
   - `claim_data.exists: true`
   - `enhanced_data.exists: true`
   - `error: null`

## Quick Fixes

### Fix 1: Enhanced Error Handling (Already Applied)
The app now shows more specific error messages instead of generic "Error retrieving enhanced results"

### Fix 2: Add Missing Fields
```python
# Added to app.py in view_enhanced_results route
'duplicate_detected': enhanced_data.get('duplicate_detected', None) if enhanced_data else None,
```

### Fix 3: Template Safety
The enhanced_results.html template now handles missing data gracefully

## Testing Enhanced Results

### Create Test Claim
1. Upload any document image
2. Wait for processing to complete
3. Visit `/enhanced/{claim_id}`
4. Should see full interactive dashboard

### Verify Data Structure
Enhanced results should contain:
- `workflow_steps`: Processing timeline
- `document_classification`: Document type and confidence
- `quality_assessment`: Image quality metrics  
- `extracted_data`: OCR and structured data
- `fraud_findings`: Risk assessment
- `validation_issues`: Data validation results
- `duplicate_detected`: Duplicate claim detection (if any)

## Contact & Support

If issues persist after following this guide:
1. Check the debug endpoint output
2. Review server logs for specific errors
3. Verify all environment variables are set correctly
4. Ensure you're testing with newly processed claims (not old ones)

The enhanced results system requires all components to be properly configured and connected.