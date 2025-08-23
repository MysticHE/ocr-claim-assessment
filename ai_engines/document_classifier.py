import re
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False

from config.settings import Config

class DocumentType(Enum):
    """Document type classifications for claims processing"""
    RECEIPT = "receipt"
    INVOICE = "invoice" 
    REFERRAL_LETTER = "referral_letter"
    MEMO = "memo"
    DIAGNOSTIC_REPORT = "diagnostic_report"
    PRESCRIPTION = "prescription"
    MEDICAL_CERTIFICATE = "medical_certificate"
    INSURANCE_FORM = "insurance_form"
    IDENTITY_DOCUMENT = "identity_document"
    UNKNOWN = "unknown"

@dataclass
class DocumentClassificationResult:
    """Result of document classification"""
    document_type: DocumentType
    confidence: float
    reasoning: List[str]
    detected_features: Dict[str, Any]
    processing_time_ms: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'document_type': self.document_type.value,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'detected_features': self.detected_features,
            'processing_time_ms': self.processing_time_ms
        }

class DocumentClassifier:
    """AI-powered document classifier for insurance claims"""
    
    def __init__(self):
        """Initialize document classifier"""
        self.mistral_available = False
        self.client = None
        
        # Try to initialize Mistral AI
        if MISTRAL_AVAILABLE and Config.MISTRAL_API_KEY:
            try:
                self.client = Mistral(api_key=Config.MISTRAL_API_KEY)
                self.mistral_available = True
            except Exception as e:
                print(f"Mistral AI not available for document classification: {e}")
        
        # Initialize classification patterns for rule-based fallback
        self.classification_patterns = self._load_classification_patterns()
    
    def _load_classification_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load patterns for rule-based document classification"""
        return {
            'receipt': {
                'keywords': [
                    'receipt', 'total', 'amount paid', 'payment', 'cash', 'credit card',
                    'invoice no', 'receipt no', 'transaction', 'subtotal', 'tax',
                    'gst', 'service charge', 'discount', 'balance due'
                ],
                'patterns': [
                    r'receipt\s*(?:no|#)\s*:?\s*[a-zA-Z0-9]+',
                    r'total\s*:?\s*\$?\d+\.?\d*',
                    r'amount\s*paid\s*:?\s*\$?\d+\.?\d*',
                    r'payment\s*method',
                    r'gst\s*:?\s*\$?\d+\.?\d*'
                ],
                'negative_keywords': ['diagnosis', 'prescription', 'referral', 'medical certificate'],
                'confidence_boost': 0.3
            },
            'invoice': {
                'keywords': [
                    'invoice', 'bill', 'billing', 'charges', 'fees', 'consultation fee',
                    'treatment cost', 'medical bill', 'hospital bill', 'clinic invoice',
                    'invoice date', 'due date', 'payment terms'
                ],
                'patterns': [
                    r'invoice\s*(?:no|#)\s*:?\s*[a-zA-Z0-9]+',
                    r'bill\s*(?:no|#)\s*:?\s*[a-zA-Z0-9]+',
                    r'consultation\s*fee\s*:?\s*\$?\d+\.?\d*',
                    r'treatment\s*(?:cost|fee)\s*:?\s*\$?\d+\.?\d*',
                    r'due\s*date'
                ],
                'negative_keywords': ['receipt no', 'amount paid'],
                'confidence_boost': 0.25
            },
            'referral_letter': {
                'keywords': [
                    'referral', 'refer', 'specialist', 'consultation', 'further treatment',
                    'recommend', 'suggested', 'please see', 'appointment', 'follow up',
                    'kindly arrange', 'request consultation', 'dear colleague'
                ],
                'patterns': [
                    r'dear\s+(?:dr|doctor|colleague)',
                    r'referral\s*(?:for|to)',
                    r'please\s*(?:see|arrange|refer)',
                    r'recommend\s*(?:that|further|specialist)',
                    r'kindly\s*(?:arrange|see|refer)'
                ],
                'negative_keywords': ['receipt', 'invoice', 'payment'],
                'confidence_boost': 0.35
            },
            'memo': {
                'keywords': [
                    'memo', 'memorandum', 'note', 'reminder', 'notice', 'internal',
                    'from:', 'to:', 'subject:', 'cc:', 'date:', 'regarding'
                ],
                'patterns': [
                    r'memo(?:randum)?\s*:',
                    r'from\s*:\s*.+',
                    r'to\s*:\s*.+',
                    r'subject\s*:\s*.+',
                    r'cc\s*:\s*.+',
                    r'regarding\s*:'
                ],
                'negative_keywords': ['diagnosis', 'treatment', 'patient'],
                'confidence_boost': 0.4
            },
            'diagnostic_report': {
                'keywords': [
                    'diagnosis', 'test results', 'lab results', 'x-ray', 'mri', 'ct scan',
                    'blood test', 'urine test', 'pathology', 'radiology', 'findings',
                    'impression', 'conclusion', 'abnormal', 'normal', 'positive', 'negative'
                ],
                'patterns': [
                    r'diagnosis\s*:?\s*.+',
                    r'test\s*results?\s*:',
                    r'lab\s*results?\s*:',
                    r'(?:x-?ray|mri|ct\s*scan|ultrasound)',
                    r'findings\s*:',
                    r'impression\s*:',
                    r'conclusion\s*:'
                ],
                'negative_keywords': ['receipt', 'invoice', 'payment'],
                'confidence_boost': 0.35
            },
            'prescription': {
                'keywords': [
                    'prescription', 'rx', 'medication', 'medicine', 'dosage', 'tablets',
                    'capsules', 'syrup', 'injection', 'take', 'times daily', 'mg', 'ml',
                    'prescriber', 'dispense', 'refill', 'generic', 'brand'
                ],
                'patterns': [
                    r'rx\s*:?\s*.+',
                    r'prescription\s*(?:no|#)',
                    r'\d+\s*(?:mg|ml|tablets?|capsules?)',
                    r'take\s*\d+.*(?:daily|times?)',
                    r'(?:morning|evening|night).*dose'
                ],
                'negative_keywords': ['receipt', 'invoice', 'total amount'],
                'confidence_boost': 0.3
            },
            'medical_certificate': {
                'keywords': [
                    'medical certificate', 'mc', 'sick leave', 'medical leave', 'unfit',
                    'certified', 'medical opinion', 'unable to work', 'rest', 'days off'
                ],
                'patterns': [
                    r'medical\s*certificate',
                    r'sick\s*leave',
                    r'(?:unfit|unable)\s*(?:for\s*work|to\s*work)',
                    r'\d+\s*days?\s*(?:rest|leave|off)',
                    r'certified\s*(?:that|to)'
                ],
                'negative_keywords': ['receipt', 'invoice', 'payment'],
                'confidence_boost': 0.4
            },
            'insurance_form': {
                'keywords': [
                    'claim form', 'insurance claim', 'policy number', 'member id',
                    'claimant', 'beneficiary', 'coverage', 'deductible', 'co-payment',
                    'claim number', 'incident date', 'accident', 'injury'
                ],
                'patterns': [
                    r'claim\s*(?:form|number)\s*:?\s*[a-zA-Z0-9]+',
                    r'policy\s*(?:no|number)\s*:?\s*[a-zA-Z0-9]+',
                    r'member\s*id\s*:?\s*[a-zA-Z0-9]+',
                    r'claimant\s*:?\s*.+',
                    r'incident\s*date'
                ],
                'negative_keywords': [],
                'confidence_boost': 0.3
            },
            'identity_document': {
                'keywords': [
                    'nric', 'passport', 'identity card', 'id card', 'identification',
                    'citizen', 'nationality', 'date of birth', 'place of birth',
                    'address', 'postal code'
                ],
                'patterns': [
                    r'nric\s*(?:no)?\s*:?\s*[a-zA-Z0-9]+',
                    r'passport\s*(?:no)?\s*:?\s*[a-zA-Z0-9]+',
                    r'identity\s*card',
                    r'date\s*of\s*birth',
                    r'nationality\s*:'
                ],
                'negative_keywords': ['treatment', 'diagnosis', 'medication'],
                'confidence_boost': 0.35
            }
        }
    
    def classify_document(self, ocr_text: str, image_path: Optional[str] = None) -> DocumentClassificationResult:
        """Classify document type based on OCR text and optionally image"""
        start_time = time.time()
        
        # Try AI-powered classification first
        if self.mistral_available and image_path:
            ai_result = self._classify_with_ai(ocr_text, image_path)
            processing_time = int((time.time() - start_time) * 1000)
            
            if ai_result['confidence'] >= 0.7:
                return DocumentClassificationResult(
                    document_type=DocumentType(ai_result['document_type']),
                    confidence=ai_result['confidence'],
                    reasoning=ai_result['reasoning'],
                    detected_features=ai_result['features'],
                    processing_time_ms=processing_time
                )
        
        # Fallback to rule-based classification
        rule_result = self._classify_with_rules(ocr_text)
        processing_time = int((time.time() - start_time) * 1000)
        
        return DocumentClassificationResult(
            document_type=DocumentType(rule_result['document_type']),
            confidence=rule_result['confidence'],
            reasoning=rule_result['reasoning'],
            detected_features=rule_result['features'],
            processing_time_ms=processing_time
        )
    
    def _classify_with_ai(self, ocr_text: str, image_path: str) -> Dict[str, Any]:
        """Classify document using Mistral AI"""
        try:
            # Create classification prompt
            prompt = f"""
Analyze this document image and extracted text to classify the document type.

Extracted Text:
{ocr_text[:2000]}  # Limit text length

Document Types:
1. receipt - Payment receipt with totals and transaction details
2. invoice - Bill or invoice for services with itemized charges
3. referral_letter - Medical referral to specialist or other provider
4. memo - Internal memo or note
5. diagnostic_report - Medical test results, lab reports, imaging results
6. prescription - Medication prescription with drugs and dosage
7. medical_certificate - Sick leave or medical fitness certificate
8. insurance_form - Insurance claim form or related documentation
9. identity_document - NRIC, passport, or ID verification
10. unknown - Cannot determine document type

Based on the visual layout and text content, classify this document and provide:
1. Document type (one of the above)
2. Confidence score (0.0-1.0)
3. Key features that led to this classification
4. Reasoning for the decision

Respond in JSON format:
{
  "document_type": "type_name",
  "confidence": 0.0,
  "features": ["feature1", "feature2"],
  "reasoning": ["reason1", "reason2"]
}
"""
            
            # Encode image
            import base64
            from PIL import Image
            import io
            
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                max_size = 1024
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                buffer.seek(0)
                
                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                base64_image = f"data:image/jpeg;base64,{image_data}"
            
            # Make API call
            response = self.client.chat.complete(
                model="mistral-small-latest",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": base64_image
                                }
                            }
                        ]
                    }
                ]
            )
            
            # Parse response
            if response.choices and len(response.choices) > 0:
                response_text = response.choices[0].message.content.strip()
                
                # Try to extract JSON from response
                try:
                    import json
                    # Find JSON in response
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        result = json.loads(json_str)
                        
                        return {
                            'document_type': result.get('document_type', 'unknown'),
                            'confidence': float(result.get('confidence', 0.5)),
                            'features': result.get('features', []),
                            'reasoning': result.get('reasoning', ['AI-powered classification'])
                        }
                except:
                    pass
            
            # Fallback if JSON parsing fails
            return {
                'document_type': 'unknown',
                'confidence': 0.3,
                'features': ['AI processing completed'],
                'reasoning': ['AI classification attempted but result unclear']
            }
            
        except Exception as e:
            return {
                'document_type': 'unknown',
                'confidence': 0.1,
                'features': ['AI processing failed'],
                'reasoning': [f'AI classification error: {str(e)}']
            }
    
    def _classify_with_rules(self, ocr_text: str) -> Dict[str, Any]:
        """Classify document using rule-based approach"""
        text_lower = ocr_text.lower()
        scores = {}
        all_features = {}
        all_reasoning = {}
        
        # Calculate scores for each document type
        for doc_type, patterns in self.classification_patterns.items():
            score = 0
            features = []
            reasoning = []
            
            # Check keywords
            keyword_matches = 0
            for keyword in patterns['keywords']:
                if keyword.lower() in text_lower:
                    keyword_matches += 1
                    features.append(f"Keyword: {keyword}")
            
            keyword_score = min(keyword_matches / len(patterns['keywords']), 1.0)
            score += keyword_score * 0.4
            
            if keyword_matches > 0:
                reasoning.append(f"Found {keyword_matches} relevant keywords")
            
            # Check patterns
            pattern_matches = 0
            for pattern in patterns['patterns']:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    pattern_matches += 1
                    features.append(f"Pattern: {pattern}")
            
            pattern_score = min(pattern_matches / len(patterns['patterns']), 1.0)
            score += pattern_score * 0.4
            
            if pattern_matches > 0:
                reasoning.append(f"Found {pattern_matches} structural patterns")
            
            # Check negative keywords (reduce score)
            negative_matches = 0
            for neg_keyword in patterns['negative_keywords']:
                if neg_keyword.lower() in text_lower:
                    negative_matches += 1
            
            if negative_matches > 0:
                score -= negative_matches * 0.1
                reasoning.append(f"Penalized for {negative_matches} conflicting indicators")
            
            # Apply confidence boost
            if score > 0:
                score += patterns['confidence_boost'] * (score / 0.8)  # Proportional boost
            
            # Ensure score is between 0 and 1
            score = max(0, min(1, score))
            
            scores[doc_type] = score
            all_features[doc_type] = features
            all_reasoning[doc_type] = reasoning
        
        # Find best match
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        
        # If no clear winner, classify as unknown
        if best_score < 0.3:
            return {
                'document_type': 'unknown',
                'confidence': best_score,
                'features': ['Insufficient distinctive features'],
                'reasoning': ['Document does not clearly match any known type']
            }
        
        return {
            'document_type': best_type,
            'confidence': best_score,
            'features': all_features[best_type],
            'reasoning': all_reasoning[best_type]
        }
    
    def get_document_requirements(self, document_type: DocumentType) -> Dict[str, Any]:
        """Get processing requirements for specific document type"""
        requirements = {
            DocumentType.RECEIPT: {
                'required_fields': ['amount', 'date', 'merchant'],
                'optional_fields': ['tax', 'payment_method', 'receipt_number'],
                'validation_rules': ['amount_positive', 'date_recent'],
                'processing_notes': 'Extract transaction details and verify payment amount'
            },
            DocumentType.INVOICE: {
                'required_fields': ['amount', 'date', 'provider', 'services'],
                'optional_fields': ['invoice_number', 'due_date', 'tax'],
                'validation_rules': ['amount_positive', 'date_valid', 'provider_verified'],
                'processing_notes': 'Verify service provider and itemized charges'
            },
            DocumentType.REFERRAL_LETTER: {
                'required_fields': ['patient_name', 'referring_doctor', 'specialist', 'reason'],
                'optional_fields': ['appointment_date', 'urgency'],
                'validation_rules': ['doctor_licensed', 'referral_valid'],
                'processing_notes': 'Verify medical referral chain and authorization'
            },
            DocumentType.DIAGNOSTIC_REPORT: {
                'required_fields': ['patient_name', 'test_type', 'results', 'date'],
                'optional_fields': ['doctor_name', 'lab_name', 'reference_ranges'],
                'validation_rules': ['results_format_valid', 'date_recent'],
                'processing_notes': 'Extract test results and clinical findings'
            },
            DocumentType.PRESCRIPTION: {
                'required_fields': ['patient_name', 'medication', 'dosage', 'doctor'],
                'optional_fields': ['pharmacy', 'refills', 'instructions'],
                'validation_rules': ['medication_valid', 'doctor_licensed'],
                'processing_notes': 'Verify prescription details and controlled substances'
            },
            DocumentType.MEDICAL_CERTIFICATE: {
                'required_fields': ['patient_name', 'doctor', 'period', 'condition'],
                'optional_fields': ['restrictions', 'follow_up'],
                'validation_rules': ['period_reasonable', 'doctor_licensed'],
                'processing_notes': 'Validate medical leave period and certification'
            },
            DocumentType.INSURANCE_FORM: {
                'required_fields': ['claimant_name', 'policy_number', 'incident_date'],
                'optional_fields': ['claim_number', 'witness', 'police_report'],
                'validation_rules': ['policy_active', 'incident_date_valid'],
                'processing_notes': 'Verify policy coverage and claim eligibility'
            },
            DocumentType.IDENTITY_DOCUMENT: {
                'required_fields': ['name', 'id_number', 'date_of_birth'],
                'optional_fields': ['address', 'nationality', 'expiry_date'],
                'validation_rules': ['id_format_valid', 'not_expired'],
                'processing_notes': 'Verify identity document authenticity and validity'
            }
        }
        
        return requirements.get(document_type, {
            'required_fields': [],
            'optional_fields': [],
            'validation_rules': [],
            'processing_notes': 'Unknown document type - manual review required'
        })