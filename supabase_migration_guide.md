# Supabase Migration Guide for Enhanced AI System

## Current Status: ✅ MOSTLY COMPATIBLE

Your existing Supabase schema **already supports 90% of the enhanced features**! The new system is designed to be backward-compatible with your current setup.

## What's Already Working

✅ **Existing tables are compatible**:
- `claims` table with metadata JSONB field *(enhanced data will be stored here)*
- `ocr_results` table for OCR processing results
- `processing_logs` table for system logs

✅ **Enhanced data storage**:
- AI analysis results stored in existing `metadata` JSONB field
- Workflow steps and decision reasoning automatically saved
- No breaking changes to existing data

## Optional Enhancements (Recommended)

### 1. Add New Indexes for Performance (Optional but Recommended)

Run these SQL commands in your Supabase SQL editor to improve query performance:

```sql
-- Index for enhanced metadata queries
CREATE INDEX IF NOT EXISTS idx_claims_enhanced_metadata 
ON claims USING GIN ((metadata->'enhanced_results'));

-- Index for document classification results
CREATE INDEX IF NOT EXISTS idx_claims_document_type 
ON claims ((metadata->'enhanced_results'->'document_classification'->>'document_type'));

-- Index for quality scores
CREATE INDEX IF NOT EXISTS idx_claims_quality_score 
ON claims ((metadata->'enhanced_results'->'quality_assessment'->'quality_score'->>'overall_score'));

-- Index for AI confidence scores
CREATE INDEX IF NOT EXISTS idx_claims_ai_confidence 
ON claims ((metadata->'enhanced_results'->>'confidence'));
```

### 2. Add New Tables for Advanced Analytics (Optional)

If you want more detailed analytics, you can add these optional tables:

```sql
-- Optional: Document classification tracking
CREATE TABLE IF NOT EXISTS document_classifications (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    claim_id UUID NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    document_type TEXT NOT NULL,
    confidence FLOAT NOT NULL,
    reasoning JSONB,
    detected_features JSONB,
    processing_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Optional: Quality assessments tracking  
CREATE TABLE IF NOT EXISTS quality_assessments (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    claim_id UUID NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    overall_score FLOAT NOT NULL,
    readability_score FLOAT,
    clarity_score FLOAT,
    completeness_score FLOAT,
    issues_detected TEXT[],
    recommendations TEXT[],
    is_acceptable BOOLEAN,
    technical_metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Optional: Fraud detection tracking
CREATE TABLE IF NOT EXISTS fraud_detections (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    claim_id UUID NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    suspicious_findings TEXT[],
    fraud_score FLOAT,
    detection_methods TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add indexes for new tables
CREATE INDEX IF NOT EXISTS idx_document_classifications_claim_id ON document_classifications(claim_id);
CREATE INDEX IF NOT EXISTS idx_document_classifications_type ON document_classifications(document_type);
CREATE INDEX IF NOT EXISTS idx_quality_assessments_claim_id ON quality_assessments(claim_id);
CREATE INDEX IF NOT EXISTS idx_quality_assessments_score ON quality_assessments(overall_score);
CREATE INDEX IF NOT EXISTS idx_fraud_detections_claim_id ON fraud_detections(claim_id);
CREATE INDEX IF NOT EXISTS idx_fraud_detections_score ON fraud_detections(fraud_score);
```

### 3. Update RLS Policies (If Using Row Level Security)

If you're using RLS, add policies for new tables:

```sql
-- Enable RLS on new tables (if using RLS)
ALTER TABLE document_classifications ENABLE ROW LEVEL SECURITY;
ALTER TABLE quality_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE fraud_detections ENABLE ROW LEVEL SECURITY;

-- Example policies (adjust based on your auth setup)
CREATE POLICY "Users can view their document classifications" 
ON document_classifications FOR SELECT 
USING (auth.uid() = (SELECT user_id FROM claims WHERE id = claim_id));

CREATE POLICY "Users can view their quality assessments" 
ON quality_assessments FOR SELECT 
USING (auth.uid() = (SELECT user_id FROM claims WHERE id = claim_id));

CREATE POLICY "Users can view their fraud detections" 
ON fraud_detections FOR SELECT 
USING (auth.uid() = (SELECT user_id FROM claims WHERE id = claim_id));
```

## Migration Steps

### Step 1: **No Action Required** ✅
Your existing system will work immediately with the enhanced features! The new AI data is stored in your existing `metadata` JSONB field.

### Step 2: **Add Performance Indexes** (Recommended - 5 minutes)
1. Go to your Supabase dashboard
2. Navigate to SQL Editor  
3. Copy and paste the index creation SQL from section 1 above
4. Run the queries

### Step 3: **Add Analytics Tables** (Optional - 10 minutes)
Only if you want detailed analytics and reporting:
1. Copy and paste the table creation SQL from section 2 above
2. Run in your Supabase SQL editor

## What Data is Enhanced

The new system automatically stores enhanced data in your existing `metadata` field:

```json
{
  "enhanced_results": {
    "status": "approved",
    "confidence": 0.87,
    "workflow_steps": [...],
    "document_classification": {
      "document_type": "receipt", 
      "confidence": 0.92
    },
    "quality_assessment": {
      "quality_score": {
        "overall_score": 0.85,
        "readability_score": 0.88
      },
      "issues_detected": []
    },
    "extracted_data": {
      "patient_name": "John Doe",
      "total_amount": 150.00,
      "document_type": "receipt"
    },
    "fraud_findings": [],
    "processing_time_ms": 2341
  }
}
```

## Testing Your Migration

Run this simple test to verify everything works:

```bash
# Test the enhanced system
python simple_test.py

# Start your app and upload a test document
python app.py
# Visit: http://localhost:5000/enhanced/{claim_id}
```

## Rollback Plan

If you need to rollback (unlikely):
1. The enhanced features are additive - your original data is unchanged
2. Simply switch back to the original `app.py` import:
   ```python
   # Change this line back:
   claim_decision = claim_processor.process_claim(ocr_results)
   ```

## Summary

🎉 **You're ready to go!** Your existing Supabase setup works with the enhanced AI system immediately. The optional enhancements above will give you better performance and more detailed analytics, but they're not required for the system to function.

The enhanced AI features will start working as soon as you deploy the updated code!