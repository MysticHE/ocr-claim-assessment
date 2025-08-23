# Complete Setup Guide: Supabase + Render Deployment

## Step 1: Supabase Database Setup

### 1.1 Create Supabase Project
1. Go to [https://supabase.com](https://supabase.com)
2. Sign up/Login with GitHub account
3. Click **"New Project"**
4. Fill in project details:
   - **Name**: `ocr-claim-assessment`
   - **Database Password**: Generate strong password (save it!)
   - **Region**: Choose closest to your users (e.g., Singapore)
5. Click **"Create new project"** (takes 2-3 minutes)

### 1.2 Database Schema Setup
1. Go to **SQL Editor** in Supabase dashboard
2. Click **"New Query"**
3. Copy and paste this SQL code:

```sql
-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Claims table
CREATE TABLE claims (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE,
  file_name TEXT NOT NULL,
  file_size INTEGER,
  language_detected TEXT,
  ocr_text TEXT,
  claim_amount DECIMAL(10,2),
  claim_status TEXT CHECK (claim_status IN ('processing', 'approved', 'rejected', 'review')),
  confidence_score DECIMAL(3,2),
  metadata JSONB
);

-- OCR results table  
CREATE TABLE ocr_results (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  claim_id UUID REFERENCES claims(id) ON DELETE CASCADE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL,
  language_code TEXT NOT NULL,
  extracted_text TEXT,
  confidence_score DECIMAL(3,2),
  bounding_boxes JSONB,
  processing_engine TEXT
);

-- Processing logs table
CREATE TABLE processing_logs (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  claim_id UUID REFERENCES claims(id) ON DELETE CASCADE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL,
  log_level TEXT,
  message TEXT,
  error_details JSONB
);

-- Create indexes for better performance
CREATE INDEX idx_claims_status ON claims(claim_status);
CREATE INDEX idx_claims_created_at ON claims(created_at);
CREATE INDEX idx_ocr_claim_id ON ocr_results(claim_id);
CREATE INDEX idx_logs_claim_id ON processing_logs(claim_id);
CREATE INDEX idx_logs_created_at ON processing_logs(created_at);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION trigger_set_timestamp()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = TIMEZONE('utc'::text, NOW());
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for updated_at
CREATE TRIGGER set_timestamp
  BEFORE UPDATE ON claims
  FOR EACH ROW
  EXECUTE PROCEDURE trigger_set_timestamp();
```

4. Click **"Run"** to execute the SQL
5. Verify tables are created in **Table Editor**

### 1.3 Get Supabase Credentials
1. Go to **Settings** → **API**
2. Copy these values (keep them secure!):
   - **Project URL**: `https://your-project-ref.supabase.co`
   - **anon public key**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`
   - **service_role secret key**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`

## Step 2: Mistral AI Setup

### 2.1 Get Mistral API Key
1. Go to [https://console.mistral.ai](https://console.mistral.ai)
2. Sign up for account
3. Go to **API Keys** section
4. Click **"Create new key"**
5. Copy the API key (starts with `ms-...`)

## Step 3: GitHub Repository Setup

### 3.1 Create Repository
1. Go to [GitHub](https://github.com)
2. Click **"New repository"**
3. Name: `ocr-claim-assessment`
4. Make it **Public** (required for Render free tier)
5. Don't initialize with README (we have one)

### 3.2 Push Code to GitHub
```bash
# In your project directory (C:\Users\keyqu\OCR)
git init
git add .
git commit -m "Initial commit: OCR Claim Assessment System"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ocr-claim-assessment.git
git push -u origin main
```

## Step 4: Render Deployment

### 4.1 Create Render Account
1. Go to [https://render.com](https://render.com)
2. Click **"Get Started for Free"**
3. **Sign up with GitHub** (recommended)
4. Authorize Render to access your repositories

### 4.2 Create Web Service
1. Click **"New +"** → **"Web Service"**
2. Click **"Connect account"** if not already connected
3. Find your `ocr-claim-assessment` repository
4. Click **"Connect"**

### 4.3 Configure Service Settings
Fill in these settings:

**Basic Settings:**
- **Name**: `ocr-claim-assessment`
- **Environment**: `Python 3`
- **Region**: `Singapore` (or closest to your users)
- **Branch**: `main`

**Build & Deploy Settings:**
- **Build Command**: 
  ```bash
  pip install --upgrade pip && pip install -r requirements.txt
  ```
- **Start Command**: 
  ```bash
  gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
  ```

**Plan:**
- Select **"Starter"** (free tier)

### 4.4 Environment Variables
In the **Environment Variables** section, add these:

| Key | Value | Notes |
|-----|--------|--------|
| `SUPABASE_URL` | `https://your-project-ref.supabase.co` | From Supabase Settings → API |
| `SUPABASE_ANON_KEY` | `eyJhbGciOiJIUzI1NiIs...` | anon public key from Supabase |
| `SUPABASE_SERVICE_KEY` | `eyJhbGciOiJIUzI1NiIs...` | service_role key from Supabase |
| `MISTRAL_API_KEY` | `ms-...` | From Mistral AI console |
| `FLASK_SECRET_KEY` | `your-random-32-char-string` | Generate random string |
| `SUPPORTED_LANGUAGES` | `en,ch_sim,ms,ta,korean,japan` | Comma-separated language codes |
| `MAX_FILE_SIZE` | `16777216` | 16MB in bytes |
| `UPLOAD_FOLDER` | `/tmp/uploads` | Render temp directory |
| `DEBUG` | `False` | Production setting |
| `ENVIRONMENT` | `production` | Environment identifier |

### 4.5 Generate Flask Secret Key
Generate a secure secret key:

**Option 1 - Python:**
```python
import secrets
print(secrets.token_urlsafe(32))
```

**Option 2 - Online Generator:**
Use [https://passwordsgenerator.net/](https://passwordsgenerator.net/) - generate 32 character string

### 4.6 Deploy
1. Click **"Create Web Service"**
2. Render will start building your application
3. Watch the deployment logs for any errors
4. Once successful, you'll get a URL like: `https://ocr-claim-assessment.onrender.com`

## Step 5: Testing Your Deployment

### 5.1 Basic Functionality Test
1. Visit your Render URL
2. Upload a test document (PDF or image with text)
3. Select languages (start with English)
4. Click **"Process Document"**
5. Verify results are displayed correctly

### 5.2 Database Verification
1. Go to Supabase **Table Editor**
2. Check **claims** table for new entries
3. Check **ocr_results** table for OCR data
4. Verify timestamps and data structure

### 5.3 Multi-Language Test
1. Upload documents with different languages
2. Test Chinese, Malay, Tamil, Korean text
3. Verify language detection works correctly
4. Check confidence scores are reasonable

## Step 6: Monitoring & Maintenance

### 6.1 Render Dashboard
- Monitor deployment logs
- Check resource usage
- Set up custom domain (paid plans)
- Monitor uptime

### 6.2 Supabase Dashboard
- Monitor database usage
- Check API request logs
- Set up Row Level Security (RLS) if needed
- Monitor storage usage

### 6.3 Error Handling
Common deployment issues:

**Build Fails:**
```bash
# Check requirements.txt versions
# Verify Python version compatibility
# Check build logs for specific error
```

**OCR Issues:**
```bash
# Verify PaddleOCR installation
# Check language model downloads
# Monitor processing timeouts
```

**Database Errors:**
```bash
# Verify Supabase credentials
# Check network connectivity
# Verify table schema matches code
```

## Step 7: Going Live Checklist

### Pre-Launch:
- [ ] All environment variables set correctly
- [ ] Database schema created and tested
- [ ] API keys are working and have sufficient quotas
- [ ] Upload limits are appropriate for your use case
- [ ] Error handling works correctly
- [ ] Security settings are production-ready

### Post-Launch:
- [ ] Monitor error logs regularly
- [ ] Set up backup procedures for database
- [ ] Document operational procedures
- [ ] Plan for scaling if needed
- [ ] Set up monitoring/alerting

## Cost Estimates

### Free Tier Limits:
**Render (Free):**
- 750 hours/month
- Sleeps after 15 minutes of inactivity
- 512MB RAM

**Supabase (Free):**
- 50,000 monthly active users
- 500MB database space
- 1GB bandwidth

**Mistral AI:**
- Pay-per-use pricing
- ~$0.25 per 1M tokens
- Estimate: $0.01-0.05 per document

### Upgrade Paths:
- **Render Starter**: $7/month (always-on, more resources)
- **Supabase Pro**: $25/month (more storage, advanced features)
- Consider costs as usage grows

## Security Checklist

### Environment Variables:
- [ ] No API keys in code repository
- [ ] All secrets stored in Render environment variables
- [ ] Secret keys are sufficiently random and long

### Application Security:
- [ ] File upload validation enabled
- [ ] File size limits enforced
- [ ] Only allowed file types accepted
- [ ] Input sanitization for all user data
- [ ] Temporary files are cleaned up

### Database Security:
- [ ] Service role key has minimum required permissions
- [ ] Consider enabling Row Level Security (RLS)
- [ ] Regular security updates applied
- [ ] Monitor for unusual activity

## Troubleshooting Common Issues

### Issue 1: "Module not found" errors
**Solution:** Check requirements.txt versions and Python compatibility

### Issue 2: Database connection fails
**Solution:** Verify Supabase URL and keys, check firewall settings

### Issue 3: OCR processing times out
**Solution:** Increase timeout settings, optimize image preprocessing

### Issue 4: File upload fails
**Solution:** Check file size limits, verify temporary directory permissions

### Issue 5: Languages not working
**Solution:** Verify PaddleOCR language models are downloading correctly

---

🎉 **Congratulations!** Your OCR Claim Assessment System is now live and ready to process multi-language claim documents!