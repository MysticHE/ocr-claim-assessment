"""
OpenAI GPT-4o-mini integration for intelligent OCR data extraction.
Replaces regex-based parsing with AI-powered natural language understanding.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

# Optional OpenAI import - graceful degradation if not available
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

@dataclass
class ExtractionResult:
    """Result from OpenAI data extraction"""
    success: bool
    extracted_data: Dict[str, Any]
    confidence: float
    processing_time_ms: int
    error: Optional[str] = None
    fallback_used: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'extracted_data': self.extracted_data,
            'confidence': self.confidence,
            'processing_time_ms': self.processing_time_ms,
            'error': self.error,
            'fallback_used': self.fallback_used
        }

class OpenAIParser:
    """OpenAI GPT-4o-mini powered intelligent data extraction"""
    
    def __init__(self):
        """Initialize OpenAI parser with lazy loading"""
        self._client = None
        self._initialization_attempted = False
        self._available = OPENAI_AVAILABLE
        
        # Extraction prompt template
        self.extraction_prompt = """
You are an expert data extraction system for insurance claims processing. Extract structured information from the following OCR text.

Document Type Context: {document_type}
Quality Score: {quality_score}

OCR Text:
{ocr_text}

Extract the following information and return as valid JSON only (no additional text):

{{
  "document_type": "claims|receipt|referral_letter|memo",
  "patient_name": "Full patient name or null",
  "patient_id": "Patient ID/NRIC/IC number or null", 
  "policy_number": "Insurance policy number or null",
  "claim_number": "Claim reference number or null",
  "provider_name": "Healthcare provider/clinic/hospital name or null",
  "diagnosis_codes": ["List of ICD-10 codes found or empty array"],
  "treatment_dates": ["YYYY-MM-DD format dates or empty array"],
  "visit_dates": ["YYYY-MM-DD format visit/appointment dates or empty array"],
  "document_date": "YYYY-MM-DD format document creation/submission date or null",
  "line_items": [
    {{
      "description": "Actual service name from document (e.g., 'Consultation Fee', 'Blood Test', 'X-Ray', 'Medication')",
      "amount": numeric_amount,
      "currency": "SGD|USD|MYR"
    }}
  ],
  "amounts": [list of numeric amounts found],
  "total_amount": numeric_total_or_null,
  "currency": "SGD|USD|MYR detected currency",
  "confidence": 0.95,
  "document_insights": {{
    "document_appears_genuine": true_or_false,
    "data_completeness": 0.85,
    "readability_assessment": "readable|partially_readable|unreadable",
    "suspicious_patterns": ["list of any suspicious elements"],
    "missing_critical_fields": ["list of missing required fields"],
    "extraction_notes": "Brief notes about the extraction process"
  }}
}}

Special Instructions for Document Classification and Extraction:
- Document types must be classified as: "claims", "receipt", "referral letter", or "memo" only
- Tax invoices, hospital bills, medical certificates = "claims" 
- Payment confirmations, paid receipts = "receipt"
- Medical referrals, specialist appointments = "referral letter"
- Internal notes, communications = "memo"
- Patient names often appear in billing/payment sections (e.g., "AMOUNT DUE: TAN WENBIN", "PAYMENTS TAN WENBIN")
- Look for ALL CAPS names in payment contexts and standalone on their own lines
- Singapore format: Look for patterns like "TAN WENBIN", "LEE MING HUA" (2-3 words, all caps)
- Extract ALL amounts found - no limits on claim amounts
- For SGH bills: Look for "AMOUNT DUE :" followed by patient name and final amount
- Visit dates: Extract dates related to actual medical visits/consultations/treatments
- Document date: Extract invoice date, bill date, document creation date, or admission/discharge dates
- Readability assessment: Based on document quality, text clarity, and completeness

Date Extraction Guidelines:
- treatment_dates: Dates when medical services were provided
- visit_dates: Same as treatment_dates but specifically for appointments/consultations
- document_date: When the document was created/issued (invoice date, admission date, etc.)
- Look for patterns: "Date:", "Invoice Date:", "Visit Date:", "Admission:", "Service Date:"

Line Items Extraction Guidelines:
- Extract actual service names from the document text, not generic descriptions
- Look for medical services: "Consultation Fee", "Blood Test", "X-Ray", "MRI Scan", "Medication", "Laboratory Test"
- Look for hospital services: "Ward Charges", "Nursing Care", "Surgery", "Anesthesia", "Room Charges"
- Look for itemized billing: Service names followed by amounts, prices in tables/lists
- Match each description with its corresponding amount from the document
- If service name is unclear, use the text that appears before the amount
- Common patterns: "Service Name ... Amount", "Description: Amount", "Item - Price"

Rules:
- Return ONLY valid JSON, no markdown or explanations
- Use null for missing values, not empty strings
- Dates must be in YYYY-MM-DD format
- Amounts should be numeric (not strings)
- Confidence should reflect extraction certainty (0.0-1.0)
- Be conservative with suspicious_patterns detection
- Focus on extracting actual data present in the text
- Pay special attention to names in payment/billing contexts
"""
    
    @property
    def client(self):
        """Lazy-loaded OpenAI client"""
        if not self._available:
            return None
            
        if self._client is None and not self._initialization_attempted:
            self._initialization_attempted = True
            try:
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    print("OpenAI API key not found in environment variables")
                    return None
                    
                print("Initializing OpenAI client for intelligent data extraction...")
                self._client = OpenAI(api_key=api_key)
                print("OpenAI client initialized successfully")
                
            except Exception as e:
                print(f"OpenAI client initialization failed: {e}")
                self._client = None
                
        return self._client
    
    @property
    def available(self):
        """Check if OpenAI parser is available"""
        return self._available and self.client is not None
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for OpenAI service"""
        if not self._available:
            return {
                'status': 'unavailable',
                'error': 'OpenAI library not installed',
                'available': False
            }
            
        if not self.client:
            return {
                'status': 'unavailable', 
                'error': 'OpenAI API key not configured or client initialization failed',
                'available': False
            }
        
        try:
            # Test API connectivity with a minimal request
            test_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5,
                timeout=5
            )
            
            if test_response:
                return {
                    'status': 'healthy',
                    'message': 'OpenAI API connection successful',
                    'available': True,
                    'model': 'gpt-4o-mini'
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': f'OpenAI API connection failed: {str(e)}',
                'available': False
            }
    
    def extract_structured_data(self, ocr_text: str, document_type: str = "unknown", 
                              quality_score: float = 0.5) -> ExtractionResult:
        """
        Extract structured data using OpenAI GPT-4o-mini
        
        Args:
            ocr_text: Raw OCR extracted text
            document_type: Type of document (receipt, invoice, etc.)
            quality_score: Document quality assessment score
            
        Returns:
            ExtractionResult with parsed data and metadata
        """
        start_time = time.time()
        
        if not self.available:
            return ExtractionResult(
                success=False,
                extracted_data={},
                confidence=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000),
                error="OpenAI parser not available - check API key and installation",
                fallback_used=True
            )
        
        try:
            # Prepare the extraction prompt
            prompt = self.extraction_prompt.format(
                document_type=document_type or "unknown",
                quality_score=quality_score,
                ocr_text=ocr_text[:8000]  # Limit input size for cost efficiency
            )
            
            print(f"Sending OCR text to OpenAI GPT-4o-mini for intelligent extraction...")
            print(f"   Document type: {document_type}")
            print(f"   Quality score: {quality_score}")
            print(f"   Text length: {len(ocr_text)} characters")
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a professional data extraction system that returns only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1000,
                temperature=0.1,  # Low temperature for consistent extraction
                timeout=15
            )
            
            # Parse response
            if response.choices and response.choices[0].message.content:
                raw_content = response.choices[0].message.content.strip()
                
                # Clean up response (remove markdown if present)
                if raw_content.startswith("```json"):
                    raw_content = raw_content.replace("```json", "").replace("```", "").strip()
                
                try:
                    extracted_data = json.loads(raw_content)
                    processing_time = int((time.time() - start_time) * 1000)
                    
                    print(f"OpenAI extraction completed successfully!")
                    print(f"   Processing time: {processing_time}ms")
                    print(f"   Confidence: {extracted_data.get('confidence', 0.0)}")
                    print(f"   Fields extracted: {len([k for k, v in extracted_data.items() if v and k != 'document_insights'])}")
                    
                    return ExtractionResult(
                        success=True,
                        extracted_data=extracted_data,
                        confidence=extracted_data.get('confidence', 0.8),
                        processing_time_ms=processing_time,
                        fallback_used=False
                    )
                    
                except json.JSONDecodeError as e:
                    print(f"OpenAI returned invalid JSON: {e}")
                    print(f"   Raw response: {raw_content[:200]}...")
                    
                    return ExtractionResult(
                        success=False,
                        extracted_data={},
                        confidence=0.0,
                        processing_time_ms=int((time.time() - start_time) * 1000),
                        error=f"Invalid JSON response from OpenAI: {str(e)}",
                        fallback_used=True
                    )
            else:
                return ExtractionResult(
                    success=False,
                    extracted_data={},
                    confidence=0.0,
                    processing_time_ms=int((time.time() - start_time) * 1000),
                    error="Empty response from OpenAI API",
                    fallback_used=True
                )
                
        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            print(f"OpenAI extraction failed: {e}")
            print(f"   Processing time: {processing_time}ms")
            
            # Determine error type for better user feedback
            if "timeout" in str(e).lower():
                error_msg = "OpenAI API request timeout - processing took too long"
            elif "rate limit" in str(e).lower():
                error_msg = "OpenAI API rate limit exceeded - please wait and try again"
            elif "insufficient_quota" in str(e).lower():
                error_msg = "OpenAI API quota exceeded - check your billing"
            else:
                error_msg = f"OpenAI API error: {str(e)}"
            
            return ExtractionResult(
                success=False,
                extracted_data={},
                confidence=0.0,
                processing_time_ms=processing_time,
                error=error_msg,
                fallback_used=True
            )
    
    def extract_with_context(self, ocr_result: Dict[str, Any], 
                           classification_result: Optional[Any] = None,
                           quality_result: Optional[Any] = None) -> ExtractionResult:
        """
        Extract data with additional context from document classification and quality assessment
        
        Args:
            ocr_result: OCR processing result
            classification_result: Document classification result  
            quality_result: Document quality assessment result
            
        Returns:
            ExtractionResult with enhanced context-aware extraction
        """
        
        # Prepare context information
        document_type = "unknown"
        quality_score = 0.5
        
        if classification_result:
            document_type = getattr(classification_result, 'document_type', 'unknown')
            if hasattr(document_type, 'value'):
                document_type = document_type.value
        
        if quality_result:
            quality_score = getattr(quality_result, 'quality_score', quality_score)
            if hasattr(quality_score, 'overall_score'):
                quality_score = quality_score.overall_score
        
        # Extract text from OCR result
        ocr_text = ocr_result.get('text', '')
        
        if not ocr_text.strip():
            return ExtractionResult(
                success=False,
                extracted_data={},
                confidence=0.0,
                processing_time_ms=0,
                error="No OCR text available for extraction",
                fallback_used=True
            )
        
        # Perform context-aware extraction
        return self.extract_structured_data(ocr_text, document_type, quality_score)
    
    def validate_extraction_result(self, result: ExtractionResult) -> Tuple[bool, List[str]]:
        """
        Validate the extraction result for completeness and consistency
        
        Args:
            result: ExtractionResult to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        if not result.success:
            return False, [result.error or "Extraction failed"]
        
        issues = []
        data = result.extracted_data
        
        # Check essential fields
        essential_fields = ['patient_name', 'total_amount']
        missing_essential = []
        
        for field in essential_fields:
            if not data.get(field):
                missing_essential.append(field)
        
        if missing_essential:
            issues.append(f"Missing essential fields: {', '.join(missing_essential)}")
        
        # Validate data types and formats
        if data.get('amounts') and not isinstance(data['amounts'], list):
            issues.append("Amounts field should be a list")
        
        if data.get('treatment_dates') and not isinstance(data['treatment_dates'], list):
            issues.append("Treatment dates field should be a list")
        
        if data.get('total_amount') and not isinstance(data['total_amount'], (int, float)):
            issues.append("Total amount should be numeric")
        
        if data.get('confidence') and (data['confidence'] < 0 or data['confidence'] > 1):
            issues.append("Confidence should be between 0.0 and 1.0")
        
        # Check extraction confidence threshold
        if result.confidence < 0.5:
            issues.append(f"Low extraction confidence: {result.confidence:.2f}")
        
        return len(issues) == 0, issues