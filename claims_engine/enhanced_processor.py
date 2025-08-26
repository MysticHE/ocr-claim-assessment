import re
import json
import hashlib
import time
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict

from database.models import ClaimStatus, ClaimDecision

# Optional AI engines - only import if available to avoid deployment issues
try:
    from ai_engines.document_classifier import DocumentClassifier, DocumentType
    CLASSIFIER_AVAILABLE = True
except ImportError:
    CLASSIFIER_AVAILABLE = False
    DocumentType = None

try:
    from ai_engines.quality_assessor import DocumentQualityAssessor
    QUALITY_ASSESSOR_AVAILABLE = True
except ImportError:
    QUALITY_ASSESSOR_AVAILABLE = False

try:
    from ai_engines.openai_parser import OpenAIParser
    OPENAI_PARSER_AVAILABLE = True
except ImportError:
    OPENAI_PARSER_AVAILABLE = False

@dataclass
class EnhancedClaimData:
    """Enhanced claim data with AI analysis results"""
    # Original extracted data
    patient_name: Optional[str] = None
    patient_id: Optional[str] = None
    policy_number: Optional[str] = None
    claim_number: Optional[str] = None
    provider_name: Optional[str] = None
    diagnosis_codes: List[str] = None
    treatment_dates: List[str] = None
    amounts: List[float] = None
    total_amount: Optional[float] = None
    currency: str = 'SGD'
    
    # New fields for enhanced extraction
    visit_dates: List[str] = None  # Now represents "Date Recorded" - when the record/document was created
    document_date: Optional[str] = None  # Date document was created/issued
    line_items: List[Dict[str, Any]] = None
    
    # AI Analysis Results
    document_type: Optional[str] = None
    document_quality_score: float = 0.0
    quality_issues: List[str] = None
    quality_acceptable: bool = True
    
    # Workflow tracking
    processing_stage: str = "initiated"
    workflow_steps_completed: List[str] = None
    
    def __post_init__(self):
        if self.diagnosis_codes is None:
            self.diagnosis_codes = []
        if self.treatment_dates is None:
            self.treatment_dates = []
        if self.visit_dates is None:
            self.visit_dates = []
        if self.line_items is None:
            self.line_items = []
        if self.amounts is None:
            self.amounts = []
        if self.quality_issues is None:
            self.quality_issues = []
        if self.workflow_steps_completed is None:
            self.workflow_steps_completed = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization with error handling"""
        try:
            return asdict(self)
        except Exception as e:
            print(f"Warning: asdict() failed for EnhancedClaimData: {e}")
            # Fallback to manual conversion
            result = {}
            for field_name in self.__dataclass_fields__:
                try:
                    value = getattr(self, field_name, None)
                    if value is None:
                        result[field_name] = None
                    elif isinstance(value, (str, int, float, bool)):
                        result[field_name] = value
                    elif isinstance(value, list):
                        result[field_name] = list(value)
                    elif isinstance(value, dict):
                        result[field_name] = dict(value)
                    elif hasattr(value, 'isoformat'):  # datetime objects
                        result[field_name] = value.isoformat()
                    else:
                        result[field_name] = str(value)
                except Exception as field_error:
                    print(f"Warning: Failed to convert field {field_name}: {field_error}")
                    result[field_name] = f"conversion_error_{field_name}"
            return result

@dataclass
class WorkflowStep:
    """Individual workflow step tracking"""
    step_name: str
    status: str  # pending, in_progress, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    ai_engine_used: Optional[str] = None
    output_summary: Optional[str] = None
    issues_found: List[str] = None
    
    def __post_init__(self):
        if self.issues_found is None:
            self.issues_found = []

class EnhancedClaimProcessor:
    """Enhanced claim processor with AI-powered workflow"""
    
    def __init__(self):
        """Initialize enhanced claim processor with lazy loading"""
        self.load_business_rules()
        
        # Lazy loading for AI engines (initialize only when needed)
        self._document_classifier = None
        self._quality_assessor = None
        self._openai_parser = None
        self._classifier_initialization_attempted = False
        self._assessor_initialization_attempted = False
        self._openai_initialization_attempted = False
        
        # Streamlined workflow steps definition (8 essential steps)
        self.workflow_steps = [
            "document_upload",           # 1. File upload and initial processing
            "quality_assessment",        # 2. Document quality check
            "document_classification",   # 3. AI document type detection  
            "ocr_processing",           # 4. Text extraction
            "structured_extraction",    # 5. Data field extraction
            "data_validation",          # 6. Business rules validation
            "fraud_detection",          # 7. Fraud pattern detection
            "decision_generation"       # 8. Final claim decision (includes policy check)
        ]
        
        # Duplicate claim tracking (simple in-memory store)
        self.processed_claims_cache = {}
    
    def load_business_rules(self):
        """Load enhanced business rules"""
        self.rules = {
            # Remove all amount limits and thresholds
            'quality_threshold': 0.6,  # Minimum quality score
            # 'classification_confidence_threshold': removed - no longer used for validation
            'fraud_detection_enabled': True,
            'duplicate_check_enabled': True,
            'auto_reject_reasons': [
                'expired_policy',
                'invalid_diagnosis',
                'missing_information',
                'duplicate_claim',
                'poor_document_quality',
                'unrecognized_document_type'
            ],
            'required_fields': [
                'patient_name',
                'provider_name',
                'document_date'
            ],
            'document_type_requirements': {
                'receipt': ['amount', 'date', 'merchant'],
                'invoice': ['amount', 'date', 'provider', 'services'],
                'referral_letter': ['patient_name', 'referring_doctor', 'specialist'],
                'diagnostic_report': ['patient_name', 'test_type', 'results'],
                'prescription': ['patient_name', 'medication', 'dosage'],
                'medical_certificate': ['patient_name', 'doctor', 'period'],
                'insurance_form': ['claimant_name', 'policy_number', 'incident_date']
            },
            'suspicious_patterns': [
                r'\b(fraud|fake|false)\b',
                r'\b(duplicate|copy|photocopy)\b',
                r'\b(altered|modified|changed)\b'
            ]
        }
    
    @property
    def document_classifier(self):
        """Lazy-loaded document classifier"""
        if not CLASSIFIER_AVAILABLE:
            return None
            
        if self._document_classifier is None and not self._classifier_initialization_attempted:
            self._classifier_initialization_attempted = True
            try:
                print("Initializing document classifier...")
                self._document_classifier = DocumentClassifier()
                print("Document classifier initialized successfully")
            except Exception as e:
                print(f"Document classifier not available: {e}")
                self._document_classifier = None
        return self._document_classifier
    
    @property
    def classifier_available(self):
        """Check if document classifier is available"""
        return CLASSIFIER_AVAILABLE and self.document_classifier is not None
    
    @property
    def quality_assessor(self):
        """Lazy-loaded quality assessor"""
        if not QUALITY_ASSESSOR_AVAILABLE:
            return None
            
        if self._quality_assessor is None and not self._assessor_initialization_attempted:
            self._assessor_initialization_attempted = True
            try:
                print("Initializing quality assessor...")
                self._quality_assessor = DocumentQualityAssessor()
                print("Quality assessor initialized successfully")
            except Exception as e:
                print(f"Quality assessor not available: {e}")
                self._quality_assessor = None
        return self._quality_assessor
    
    @property
    def quality_assessor_available(self):
        """Check if quality assessor is available"""
        return QUALITY_ASSESSOR_AVAILABLE and self.quality_assessor is not None
    
    @property
    def openai_parser(self):
        """Lazy-loaded OpenAI parser"""
        if not OPENAI_PARSER_AVAILABLE:
            return None
            
        if self._openai_parser is None and not self._openai_initialization_attempted:
            self._openai_initialization_attempted = True
            try:
                print("Initializing OpenAI parser for intelligent data extraction...")
                self._openai_parser = OpenAIParser()
                print("OpenAI parser initialized successfully")
            except Exception as e:
                print(f"OpenAI parser not available: {e}")
                self._openai_parser = None
        return self._openai_parser
    
    @property
    def openai_parser_available(self):
        """Check if OpenAI parser is available and configured"""
        return OPENAI_PARSER_AVAILABLE and self.openai_parser is not None and self.openai_parser.available
    
    def process_enhanced_claim(self, ocr_result: Dict[str, Any], 
                             image_path: Optional[str] = None) -> Dict[str, Any]:
        """Main enhanced claim processing workflow"""
        start_time = time.time()
        
        print(f"Starting enhanced claim processing workflow")
        print(f"   OCR result keys: {list(ocr_result.keys()) if ocr_result else 'None'}")
        print(f"   Image path provided: {image_path is not None}")
        print(f"   Classifier available: {self.classifier_available}")
        print(f"   Quality assessor available: {self.quality_assessor_available}")
        print(f"   OpenAI parser available: {self.openai_parser_available}")
        
        # Determine AI capabilities
        has_advanced_ai = self.classifier_available and self.quality_assessor_available
        has_openai_extraction = self.openai_parser_available
        
        if has_openai_extraction:
            print("   Using OpenAI GPT-4o-mini for intelligent data extraction")
        elif has_advanced_ai:
            print("   Using advanced AI engines with regex-based extraction")
        else:
            print("   Running in lightweight fallback mode (regex-only extraction)")
        
        # Initialize workflow tracking
        workflow_steps = []
        current_step = None
        
        try:
            # Step 1: Document Upload (already completed)
            workflow_steps.append(WorkflowStep(
                step_name="document_upload",
                status="completed",
                start_time=datetime.now(),
                end_time=datetime.now(),
                output_summary="Document uploaded and OCR completed"
            ))
            
            # Step 2: Quality Assessment
            current_step = WorkflowStep(
                step_name="quality_assessment",
                status="in_progress",
                start_time=datetime.now(),
                ai_engine_used="quality_assessor"
            )
            
            quality_result = None
            if self.quality_assessor_available and image_path:
                quality_result = self.quality_assessor.assess_document_quality(
                    image_path, ocr_result.get('confidence')
                )
                current_step.output_summary = f"Quality score: {quality_result.quality_score.overall_score:.2f}"
                current_step.issues_found = quality_result.recommendations
                current_step.status = "completed"
            else:
                # Simple OCR-confidence based quality assessment fallback
                ocr_confidence = ocr_result.get('confidence', 0.5)
                quality_score = max(0.4, ocr_confidence)  # Minimum threshold
                current_step.output_summary = f"Simple quality assessment: {quality_score:.2f} (based on OCR confidence)"
                current_step.issues_found = ["Limited quality assessment in fallback mode"]
                current_step.status = "completed"
            
            current_step.end_time = datetime.now()
            workflow_steps.append(current_step)
            
            # Step 3: Document Classification
            current_step = WorkflowStep(
                step_name="document_classification",
                status="in_progress", 
                start_time=datetime.now(),
                ai_engine_used="document_classifier"
            )
            
            classification_result = None
            mistral_result = None
            
            # Try Mistral AI classifier first
            if self.classifier_available:
                mistral_result = self.document_classifier.classify_document(
                    ocr_result.get('text', ''), image_path
                )
                
                # Use Mistral result if confidence is high enough
                if mistral_result.confidence >= 0.5:  # Require at least 50% confidence
                    classification_result = mistral_result
                    current_step.output_summary = f"Mistral classified as: {mistral_result.document_type.value}"
                    current_step.ai_engine_used = "mistral_ai"
                    current_step.status = "completed"
                else:
                    # Mistral confidence too low, fall back to enhanced classification
                    doc_type = self._simple_document_classification(ocr_result.get('text', ''))
                    current_step.output_summary = f"Enhanced classification: {doc_type} (Mistral confidence too low: {mistral_result.confidence:.2f})"
                    current_step.ai_engine_used = "enhanced_rules"
                    current_step.status = "completed"
                    
                    # Create enhanced classification result
                    from collections import namedtuple
                    ClassificationResult = namedtuple('ClassificationResult', ['document_type', 'confidence'])
                    classification_result = ClassificationResult(document_type=doc_type, confidence=0.75)
            else:
                # Mistral not available, use enhanced classification
                doc_type = self._simple_document_classification(ocr_result.get('text', ''))
                current_step.output_summary = f"Enhanced classification: {doc_type} (Mistral not available)"
                current_step.ai_engine_used = "enhanced_rules"
                current_step.status = "completed"
                
                # Create enhanced classification result
                from collections import namedtuple
                ClassificationResult = namedtuple('ClassificationResult', ['document_type', 'confidence'])
                classification_result = ClassificationResult(document_type=doc_type, confidence=0.7)
            
            current_step.end_time = datetime.now()
            workflow_steps.append(current_step)
            
            # Step 4: OCR Processing (already completed)
            workflow_steps.append(WorkflowStep(
                step_name="ocr_processing",
                status="completed",
                start_time=datetime.now(),
                end_time=datetime.now(),
                ai_engine_used=ocr_result.get('engine', 'hybrid'),
                output_summary=f"Extracted {len(ocr_result.get('text', '').split())} words"
            ))
            
            # Step 5: Structured Data Extraction
            current_step = WorkflowStep(
                step_name="structured_extraction",
                status="in_progress",
                start_time=datetime.now()
            )
            
            extracted_data = self.extract_enhanced_structured_data(
                ocr_result, classification_result, quality_result
            )
            
            current_step.output_summary = f"Extracted {len([x for x in [extracted_data.patient_name, extracted_data.total_amount, extracted_data.provider_name] if x])} key fields"
            current_step.status = "completed"
            current_step.end_time = datetime.now()
            workflow_steps.append(current_step)
            
            # Step 6: Data Validation
            current_step = WorkflowStep(
                step_name="data_validation",
                status="in_progress",
                start_time=datetime.now()
            )
            
            is_valid, validation_issues = self.validate_enhanced_claim_data(extracted_data)
            current_step.issues_found = validation_issues
            current_step.output_summary = f"Validation {'passed' if is_valid else 'failed'} - {len(validation_issues)} issues"
            current_step.status = "completed"
            current_step.end_time = datetime.now()
            workflow_steps.append(current_step)
            
            # Step 7: Fraud Detection
            current_step = WorkflowStep(
                step_name="fraud_detection",
                status="in_progress",
                start_time=datetime.now()
            )
            
            fraud_findings = self.enhanced_fraud_detection(ocr_result, extracted_data)
            current_step.issues_found = fraud_findings
            
            # Create detailed fraud summary instead of just count
            if fraud_findings:
                fraud_summary = f"Fraud detected: {', '.join(fraud_findings[:2])}"
                if len(fraud_findings) > 2:
                    fraud_summary += f" + {len(fraud_findings) - 2} more"
            else:
                fraud_summary = "No fraud indicators detected"
            current_step.output_summary = fraud_summary
            current_step.status = "completed"
            current_step.end_time = datetime.now()
            workflow_steps.append(current_step)
            
            # Step 8: Claim Adjudication & Decision (combining policy_verification + claim_adjudication + decision_generation)
            current_step = WorkflowStep(
                step_name="decision_generation",
                status="in_progress",
                start_time=datetime.now(),
                ai_engine_used="rule_engine"
            )
            
            decision = self.make_enhanced_decision(
                extracted_data, validation_issues, fraud_findings, 
                quality_result, classification_result, ocr_result.get('confidence', 0.0)
            )
            
            current_step.output_summary = f"Decision: {decision.status.value}"
            current_step.status = "completed"
            current_step.end_time = datetime.now()
            workflow_steps.append(current_step)
            
            # Calculate total processing time
            total_processing_time = int((time.time() - start_time) * 1000)
            
            print(f"Enhanced processing workflow completed successfully")
            print(f"   Total processing time: {total_processing_time}ms")
            print(f"   Workflow steps completed: {len([s for s in workflow_steps if s.status == 'completed'])}/{len(workflow_steps)}")
            print(f"   Decision status: {decision.status.value}")
            
            # Compile results with robust JSON serialization
            try:
                # Safely convert extracted data
                try:
                    extracted_data_dict = extracted_data.to_dict()
                except Exception as e:
                    print(f"Warning: extracted_data.to_dict() failed: {e}")
                    extracted_data_dict = self._safe_dataclass_to_dict(extracted_data)
                
                # Safely convert classification result
                try:
                    classification_dict = self._classification_to_dict(classification_result) if classification_result else None
                except Exception as e:
                    print(f"Warning: classification conversion failed: {e}")
                    classification_dict = None
                
                # Safely convert quality result
                try:
                    quality_dict = quality_result.to_dict() if quality_result else None
                except Exception as e:
                    print(f"Warning: quality_result.to_dict() failed: {e}")
                    quality_dict = None
                
                # Safely convert workflow steps
                try:
                    workflow_steps_dict = [self._workflow_step_to_dict(step) for step in workflow_steps]
                except Exception as e:
                    print(f"Warning: workflow steps conversion failed: {e}")
                    workflow_steps_dict = []
                
                return {
                    'success': True,
                    'status': decision.status.value,
                    'confidence': decision.confidence,
                    'amount': decision.amount,
                    'reasons': decision.reasons,
                    'processing_notes': decision.processing_notes,
                    'total_processing_time_ms': total_processing_time,
                    
                    # Enhanced data
                    'extracted_data': extracted_data_dict,
                    'validation_issues': validation_issues,
                    'fraud_findings': fraud_findings,
                    'workflow_steps': workflow_steps_dict,
                    
                    # AI Analysis results
                    'document_classification': classification_dict,
                    'quality_assessment': quality_dict,
                    'ai_engines_used': [step.ai_engine_used for step in workflow_steps if step.ai_engine_used],
                    
                    # Workflow summary
                    'workflow_completion': {
                        'total_steps': len(workflow_steps),
                        'completed_steps': len([s for s in workflow_steps if s.status == "completed"]),
                        'failed_steps': len([s for s in workflow_steps if s.status == "failed"]),
                        'skipped_steps': len([s for s in workflow_steps if s.status == "skipped"])
                    }
                }
            except Exception as compilation_error:
                print(f"Critical error during result compilation: {compilation_error}")
                raise compilation_error  # Re-raise to trigger the outer exception handler
            
        except Exception as e:
            # Log error for debugging
            print(f"Enhanced processing error: {type(e).__name__}: {str(e)}")
            
            # Mark current step as failed if there was one in progress
            if current_step and current_step.status == "in_progress":
                current_step.status = "failed"
                current_step.end_time = datetime.now()
                current_step.issues_found = [f"Processing error: {str(e)}"]
            
            return {
                'success': False,
                'status': ClaimStatus.REVIEW.value,
                'confidence': 0.0,
                'amount': 0.0,
                'reasons': [f"Enhanced processing error: {str(e)}"],
                'processing_notes': "Claim requires manual review due to processing error",
                'error': str(e),
                'workflow_steps': [self._workflow_step_to_dict(step) for step in workflow_steps]
            }
    
    def extract_enhanced_structured_data(self, ocr_result: Dict[str, Any],
                                       classification_result=None,
                                       quality_result=None) -> EnhancedClaimData:
        """Enhanced structured data extraction with OpenAI intelligence and fallback to regex"""
        text = ocr_result.get('text', '')
        
        # Create enhanced claim data
        extracted = EnhancedClaimData()
        
        # Try OpenAI extraction first if available
        openai_result = None
        if self.openai_parser_available:
            try:
                print("Attempting OpenAI GPT-4o-mini data extraction...")
                openai_result = self.openai_parser.extract_with_context(
                    ocr_result, classification_result, quality_result
                )
                
                if openai_result.success:
                    print(f"OpenAI extraction successful (confidence: {openai_result.confidence:.2f})")
                    ai_data = openai_result.extracted_data
                    
                    # Map OpenAI results to EnhancedClaimData
                    extracted.patient_name = ai_data.get('patient_name')
                    extracted.patient_id = ai_data.get('patient_id')
                    extracted.policy_number = ai_data.get('policy_number')
                    extracted.claim_number = ai_data.get('claim_number')
                    extracted.provider_name = ai_data.get('provider_name')
                    extracted.diagnosis_codes = ai_data.get('diagnosis_codes', [])
                    extracted.treatment_dates = ai_data.get('treatment_dates', [])
                    extracted.amounts = ai_data.get('amounts', [])
                    extracted.total_amount = ai_data.get('total_amount')
                    extracted.currency = ai_data.get('currency', 'SGD')
                    
                    # Map new fields from OpenAI
                    extracted.visit_dates = ai_data.get('visit_dates', [])  # Now "Date Recorded"
                    extracted.document_date = ai_data.get('document_date')
                    extracted.line_items = ai_data.get('line_items', [])
                    
                    # AI confidence scores removed for simplified processing
                    
                    # Add document insights from OpenAI
                    if 'document_insights' in ai_data:
                        insights = ai_data['document_insights']
                        extracted.quality_issues = insights.get('missing_critical_fields', [])
                        
                        # Update quality based on OpenAI analysis
                        if quality_result is None:
                            # Use OpenAI's assessment if no computer vision quality result
                            extracted.quality_acceptable = insights.get('document_appears_genuine', True)
                            extracted.document_quality_score = insights.get('data_completeness', 0.8)
                    
                else:
                    print(f"OpenAI extraction failed: {openai_result.error}")
                    print("   Falling back to regex-based extraction...")
                    
            except Exception as e:
                print(f"OpenAI extraction error: {e}")
                print("   Falling back to regex-based extraction...")
        
        # Fallback to regex-based extraction if OpenAI failed or unavailable
        if openai_result is None or not openai_result.success:
            print("Using regex-based data extraction...")
            
            # Basic extraction (existing regex logic)
            extracted.patient_name = self._extract_patient_name(text)
            extracted.patient_id = self._extract_patient_id(text)
            extracted.policy_number = self._extract_policy_number(text)
            extracted.claim_number = self._extract_claim_number(text)
            extracted.provider_name = self._extract_provider_name(text)
            extracted.diagnosis_codes = self._extract_diagnosis_codes(text)
            extracted.treatment_dates = self._extract_dates(text)
            extracted.amounts = self._extract_amounts(text)
            extracted.total_amount = max(extracted.amounts) if extracted.amounts else None
            extracted.currency = self._extract_currency(text)
            extracted.line_items = self._extract_line_items(text)
            
            # If no line items found, create them from amounts
            if not extracted.line_items and extracted.amounts:
                extracted.line_items = self._create_line_items_from_amounts(text, extracted.amounts, extracted.currency)
            
            # Set confidence scores for regex extraction
            extracted.ai_confidence_scores = {
                'ocr_confidence': ocr_result.get('confidence', 0.0),
                'extraction_confidence': 0.6,  # Lower confidence for regex
                'extraction_method': 'regex_fallback'
            }
            
            print(f"Regex extraction completed - {len([x for x in [extracted.patient_name, extracted.total_amount, extracted.provider_name] if x])} key fields found")
        
        # Add AI analysis results from other engines
        if classification_result:
            # Handle both proper classification results and fallback string results
            if hasattr(classification_result.document_type, 'value'):
                extracted.document_type = classification_result.document_type.value
            else:
                extracted.document_type = classification_result.document_type
            extracted.document_classification_confidence = classification_result.confidence
        
        if quality_result:
            extracted.document_quality_score = quality_result.quality_score.overall_score
            extracted.quality_issues = [issue.value for issue in quality_result.issues_detected]
            extracted.quality_acceptable = quality_result.is_acceptable
        
        # AI confidence scores removed for simplified processing
        
        # Post-processing: Fill missing visit_dates (Date Recorded) with fallbacks
        if not extracted.visit_dates:
            if extracted.treatment_dates:
                extracted.visit_dates = extracted.treatment_dates
            elif extracted.document_date:
                extracted.visit_dates = [extracted.document_date]
        
        # Set processing stage
        extracted.processing_stage = "data_extracted"
        
        return extracted
    
    def validate_enhanced_claim_data(self, data: EnhancedClaimData) -> Tuple[bool, List[str]]:
        """Enhanced validation with document-type specific checks"""
        issues = []
        
        # Basic validation (updated required fields)
        if not data.patient_name:
            issues.append("Missing patient name")
        
        if not data.provider_name:
            issues.append("Missing provider name")
        
        # Check for document date - can be treatment date, visit date (date recorded), or document date
        document_date_available = bool(
            data.treatment_dates or 
            data.visit_dates or 
            data.document_date or
            getattr(data, 'dates', None)
        )
        if not document_date_available:
            issues.append("Missing date of document")
        
        # Document quality validation - REMOVED in Version 2.3.0
        # Quality assessment performed for informational purposes only
        # if not data.quality_acceptable:
        #     issues.append("Document quality below acceptable threshold")
        
        # Document type specific validation
        if data.document_type and data.document_type in self.rules['document_type_requirements']:
            required_fields = self.rules['document_type_requirements'][data.document_type]
            
            # Map document requirements to extracted data fields
            field_mapping = {
                'amount': data.total_amount or (data.amounts and data.amounts[0]),
                'date': (data.treatment_dates and data.treatment_dates[0]) or 
                       (data.visit_dates and data.visit_dates[0]) or
                       data.document_date,
                'document_date': (data.treatment_dates and data.treatment_dates[0]) or 
                               (data.visit_dates and data.visit_dates[0]) or
                               data.document_date,
                'patient_name': data.patient_name,
                'provider': data.provider_name,
                'provider_name': data.provider_name,
                'merchant': data.provider_name,  # Alias for provider
                'policy_number': data.policy_number,
                'claimant_name': data.patient_name,  # Alias for patient
            }
            
            for required_field in required_fields:
                if not field_mapping.get(required_field):
                    issues.append(f"Missing required field for {data.document_type}: {required_field}")
        
        # Classification confidence check - REMOVED for Version 2.3.0
        # All confidence thresholds removed to ensure enhanced processing completes successfully
        # Classification confidence now used for informational purposes only
        
        return len(issues) == 0, issues
    
    def enhanced_fraud_detection(self, ocr_result: Dict[str, Any], 
                               data: EnhancedClaimData) -> List[str]:
        """Simplified fraud detection - duplicate claims only"""
        suspicious_findings = []
        
        # Only check for duplicate submissions
        if self.rules['duplicate_check_enabled']:
            duplicate_result = self._check_duplicate_claim(data)
            if duplicate_result:
                duplicate_reason = f"DUPLICATE SUBMISSION: {duplicate_result['details']} ({duplicate_result['similarity_score']:.1%} match)"
                suspicious_findings.append(duplicate_reason)
                print(f"   DUPLICATE detected: {duplicate_result['match_type']} with {duplicate_result['similarity_score']:.1%} similarity")
            else:
                print("   No duplicates found in database")
        
        # Summary logging
        if suspicious_findings:
            print(f"FRAUD CHECK COMPLETE: {len(suspicious_findings)} duplicate(s) detected")
        else:
            print("FRAUD CHECK COMPLETE: No duplicate claims detected")
        
        return suspicious_findings
    
    def _check_duplicate_claim(self, data: EnhancedClaimData) -> Optional[Dict[str, Any]]:
        """Enhanced duplicate detection with fuzzy matching and similarity scoring"""
        try:
            # Enhanced database duplicate detection with better error handling
            from database.supabase_client import SupabaseClient
            
            # Reduced logging for production performance
            db_client = SupabaseClient()
            
            # Test database connection first
            if not db_client.test_connection():
                return self._fallback_duplicate_check(data)
            
            # Continue with database duplicate detection
            
            # Calculate multiple similarity metrics
            duplicates_found = []
            
            # 1. Exact hash matching using existing claims table
            # Use robust fields that are more likely to be consistent
            primary_key = f"{data.patient_name}|{data.total_amount}|{data.patient_id}|{data.claim_number}"
            claim_key = f"{data.patient_name}|{data.total_amount}|{data.treatment_dates}|{data.provider_name or 'unknown'}"
            
            # Create multiple hashes for different matching strategies
            primary_hash = hashlib.md5(primary_key.encode()).hexdigest()
            claim_hash = hashlib.md5(claim_key.encode()).hexdigest()
            
            # Debug logging for duplicate detection
            print(f"DUPLICATE CHECK:")
            print(f"   Patient: {data.patient_name}")
            print(f"   Amount: {data.total_amount}")
            print(f"   Patient ID: {data.patient_id}")
            print(f"   Claim Number: {data.claim_number}")
            print(f"   Primary Hash: {primary_hash}")
            print(f"   Claim Hash: {claim_hash}")
            
            # Search for duplicates with minimal logging
            if data.patient_name:
                similar_claims = self._find_similar_claims_by_text(db_client, data)
                duplicates_found.extend(similar_claims)
            
            if data.total_amount:
                proximity_matches = self._find_claims_by_proximity(db_client, data)
                duplicates_found.extend(proximity_matches)
            
            if duplicates_found:
                # Return the highest similarity match
                best_match = max(duplicates_found, key=lambda x: x['similarity_score'])
                print(f"DUPLICATE DETECTED! Best match: {best_match['similarity_score']:.1%} similarity")
                
                return {
                    'type': 'duplicate_detected',
                    'similarity_score': best_match['similarity_score'],
                    'match_type': best_match['match_type'],
                    'days_ago': best_match['days_ago'],
                    'matched_claim_id': best_match.get('claim_id'),
                    'details': best_match['details'],
                    'all_matches': duplicates_found[:3]  # Top 3 matches
                }
            else:
                print("No duplicates found in database")
                # Store claim data for future duplicate comparisons
                self._store_claim_for_comparison(db_client, data, claim_hash, primary_hash)
            
            return None
            
        except ValueError as ve:
            print(f"Database configuration error: {ve}")
            print("Please set SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables")
            return self._fallback_duplicate_check(data)
        except ConnectionError as ce:
            print(f"Database connection error: {ce}")
            return self._fallback_duplicate_check(data)
        except Exception as e:
            print(f"Unexpected error in duplicate detection: {e}")
            print("Falling back to in-memory duplicate checking")
            return self._fallback_duplicate_check(data)
    
    def _find_similar_claims_by_text(self, db_client, data: EnhancedClaimData) -> List[Dict]:
        """Find claims with similar text using fuzzy matching - works with existing claims table"""
        matches = []
        
        try:
            # Get recent claims (last 90 days) for comparison using your existing table structure
            query_result = db_client.supabase.table('claims').select(
                'id, created_at, claim_amount, metadata, file_name'
            ).gte(
                'created_at', 
                (datetime.now() - timedelta(days=90)).isoformat()
            ).execute()
            
            # Removed logging for production performance
            
            if query_result.data:
                for claim in query_result.data:
                    similarity_score = self._calculate_text_similarity_from_claims(data, claim)
                    
                    if similarity_score > 0.8:  # High similarity threshold
                        claim_date = datetime.fromisoformat(claim['created_at'].replace('Z', '+00:00'))
                        now_utc = datetime.now(timezone.utc)
                        days_ago = (now_utc - claim_date).days
                        
                        # Extract patient name from metadata if available
                        patient_from_db = "Unknown"
                        if claim.get('metadata') and isinstance(claim['metadata'], dict):
                            extracted_data = claim['metadata'].get('enhanced_results', {}).get('extracted_data', {})
                            patient_from_db = extracted_data.get('patient_name', 'Unknown')
                        
                        matches.append({
                            'similarity_score': similarity_score,
                            'match_type': 'fuzzy_text',
                            'days_ago': days_ago,
                            'claim_id': claim['id'],
                            'details': f"Similar patient name '{patient_from_db}' with {similarity_score:.1%} similarity"
                        })
                        
                        # Removed logging for production performance
            
        except Exception as e:
            print(f"      Error in text similarity search: {e}")
        
        return matches
    
    def _find_claims_by_proximity(self, db_client, data: EnhancedClaimData) -> List[Dict]:
        """Find claims with similar amounts and dates - works with existing claims table"""
        matches = []
        
        if not data.total_amount:
            print("      No amount provided for proximity matching")
            return matches
        
        try:
            # Look for claims with similar amounts (±10%) within recent timeframe
            amount_min = data.total_amount * 0.9
            amount_max = data.total_amount * 1.1
            
            print(f"      Searching for amounts between ${amount_min:.2f} and ${amount_max:.2f}")
            
            query_result = db_client.supabase.table('claims').select(
                'id, created_at, claim_amount, metadata, file_name'
            ).gte(
                'claim_amount', amount_min
            ).lte('claim_amount', amount_max).execute()
            
            print(f"      Found {len(query_result.data) if query_result.data else 0} claims in amount range")
            
            if query_result.data:
                for claim in query_result.data:
                    # Calculate date similarity based on submission date
                    claim_date = datetime.fromisoformat(claim['created_at'].replace('Z', '+00:00'))
                    now_utc = datetime.now(timezone.utc)
                    date_diff = abs((now_utc - claim_date).days)
                    
                    if date_diff <= 30:  # Within 30 days
                        amount_similarity = 1 - abs(data.total_amount - float(claim['claim_amount'])) / data.total_amount
                        date_similarity = max(0, 1 - date_diff / 30)
                        overall_similarity = (amount_similarity * 0.7) + (date_similarity * 0.3)
                        
                        if overall_similarity > 0.7:
                            matches.append({
                                'similarity_score': overall_similarity,
                                'match_type': 'amount_date_proximity',
                                'days_ago': date_diff,
                                'claim_id': claim['id'],
                                'details': f"Similar amount ${float(claim['claim_amount']):.2f} submitted {date_diff} days ago"
                            })
                            
                            print(f"      Proximity match: ${float(claim['claim_amount']):.2f} ({overall_similarity:.1%} similarity)")
        
        except Exception as e:
            print(f"      Error in proximity search: {e}")
        
        return matches
    
    def _calculate_text_similarity_from_claims(self, data: EnhancedClaimData, stored_claim: Dict) -> float:
        """Calculate text similarity from existing claims table structure"""
        try:
            # Extract patient name from metadata->enhanced_results->extracted_data
            stored_patient_name = None
            if stored_claim.get('metadata') and isinstance(stored_claim['metadata'], dict):
                enhanced_results = stored_claim['metadata'].get('enhanced_results', {})
                if isinstance(enhanced_results, dict):
                    extracted_data = enhanced_results.get('extracted_data', {})
                    if isinstance(extracted_data, dict):
                        stored_patient_name = extracted_data.get('patient_name')
            
            if data.patient_name and stored_patient_name:
                name1 = data.patient_name.lower().strip()
                name2 = stored_patient_name.lower().strip()
                
                # Calculate similarity ratio
                name_similarity = self._levenshtein_similarity(name1, name2)
                
                # Weight by other factors
                amount_match = 0
                if data.total_amount and stored_claim.get('claim_amount'):
                    amount_diff = abs(data.total_amount - float(stored_claim['claim_amount']))
                    amount_match = max(0, 1 - amount_diff / max(data.total_amount, float(stored_claim['claim_amount'])))
                
                # Combined similarity score
                return (name_similarity * 0.6) + (amount_match * 0.4)
            
        except Exception as e:
            print(f"      Error calculating text similarity: {e}")
        
        return 0.0
    
    def _calculate_text_similarity(self, data: EnhancedClaimData, stored_claim: Dict) -> float:
        """Calculate text similarity using multiple algorithms - legacy method"""
        try:
            # Simple Levenshtein distance for patient names
            if data.patient_name and stored_claim.get('metadata', {}).get('patient_name'):
                name1 = data.patient_name.lower().strip()
                name2 = stored_claim['metadata']['patient_name'].lower().strip()
                
                # Calculate similarity ratio
                name_similarity = self._levenshtein_similarity(name1, name2)
                
                # Weight by other factors
                amount_match = 0
                if data.total_amount and stored_claim.get('claim_amount'):
                    amount_diff = abs(data.total_amount - stored_claim['claim_amount'])
                    amount_match = max(0, 1 - amount_diff / max(data.total_amount, stored_claim['claim_amount']))
                
                # Combined similarity score
                return (name_similarity * 0.6) + (amount_match * 0.4)
            
        except Exception:
            pass
        
        return 0.0
    
    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Calculate Levenshtein similarity ratio"""
        if not s1 or not s2:
            return 0.0
        
        # Simple implementation of Levenshtein distance
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        
        if len(s2) == 0:
            return 0.0
        
        # Calculate similarity ratio
        distance = self._levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        
        return 1 - (distance / max_len)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        
        distances = range(len(s1) + 1)
        
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        
        return distances[-1]
    
    def _store_claim_for_comparison(self, db_client, data: EnhancedClaimData, claim_hash: str, primary_hash: str = None):
        """Store claim data for future duplicate comparisons"""
        try:
            # Store in duplicate tracking table (would need to create this table)
            duplicate_data = {
                'claim_hash': claim_hash,
                'primary_hash': primary_hash or claim_hash,
                'patient_name': data.patient_name,
                'patient_id': data.patient_id,
                'claim_number': data.claim_number,
                'claim_amount': data.total_amount,
                'provider_name': data.provider_name,
                'treatment_dates': str(data.treatment_dates),
                'created_at': datetime.now().isoformat(),
                'metadata': {
                    'document_type': data.document_type,
                    'quality_score': data.document_quality_score
                }
            }
            
            # In a real implementation, you'd create a duplicate_claims table
            # For now, store in memory cache as fallback
            self.processed_claims_cache[claim_hash] = {
                'timestamp': datetime.now(),
                'patient': data.patient_name,
                'amount': data.total_amount,
                'provider': data.provider_name
            }
            
        except Exception:
            pass
    
    def _fallback_duplicate_check(self, data: EnhancedClaimData) -> Optional[Dict]:
        """Fallback to simple hash-based duplicate detection"""
        try:
            # Use primary key for more reliable duplicate detection
            primary_key = f"{data.patient_name}|{data.total_amount}|{data.patient_id}|{data.claim_number}"
            claim_key = f"{data.patient_name}|{data.total_amount}|{data.treatment_dates}|{data.provider_name or 'unknown'}"
            
            primary_hash = hashlib.md5(primary_key.encode()).hexdigest()
            claim_hash = hashlib.md5(claim_key.encode()).hexdigest()
            
            # Check both primary hash (more reliable) and claim hash
            for hash_key, hash_type in [(primary_hash, 'primary_key'), (claim_hash, 'claim_key')]:
                if hash_key in self.processed_claims_cache:
                    stored_claim = self.processed_claims_cache[hash_key]
                    time_diff = datetime.now() - stored_claim['timestamp']
                    
                    if time_diff.days < 30:
                        return {
                            'type': 'duplicate_detected',
                            'similarity_score': 1.0,
                            'match_type': f'exact_hash_{hash_type}',
                            'days_ago': time_diff.days,
                            'details': f"Exact duplicate detected via {hash_type} (submitted {time_diff.days} days ago)"
                        }
            
            # Store both hashes for future comparison
            claim_data = {
                'timestamp': datetime.now(),
                'patient': data.patient_name,
                'amount': data.total_amount
            }
            self.processed_claims_cache[primary_hash] = claim_data
            self.processed_claims_cache[claim_hash] = claim_data
            
            return None
        except Exception:
            return None
    
    def make_enhanced_decision(self, data: EnhancedClaimData, validation_issues: List[str],
                             fraud_findings: List[str], quality_result=None,
                             classification_result=None, ocr_confidence: float = 0.0) -> ClaimDecision:
        """Simplified claim decision making without confidence calculations"""
        
        # PRIORITY 1: Duplicate claim rejection (highest priority)
        if fraud_findings:
            # Since fraud detection now only checks duplicates, all findings are duplicate-related
            return ClaimDecision(
                status=ClaimStatus.REJECTED,
                confidence=0.9,  # High confidence for duplicate detection
                amount=data.total_amount or 0,
                reasons=fraud_findings,
                processing_notes="Claim rejected due to duplicate submission detected"
            )
        
        # PRIORITY 2: Validation issues (only critical field validation)
        if validation_issues:
            return ClaimDecision(
                status=ClaimStatus.REJECTED,
                confidence=0.9,  # High confidence in validation logic
                amount=data.total_amount or 0,
                reasons=validation_issues,
                processing_notes="Claim rejected due to missing required information"
            )
        
        # All quality checks and confidence thresholds removed - approve clean claims
        claim_amount = data.total_amount or 0
        
        # Simplified approval: if no fraud and no validation issues, approve
        return ClaimDecision(
            status=ClaimStatus.APPROVED,
            confidence=0.85,  # Static confidence for approved claims
            amount=claim_amount,
            reasons=["Clean claim with all required information"],
            processing_notes="Approved - passed fraud detection and validation checks"
        )
    
    # Completeness score calculation removed - no longer used in simplified logic
    
    def _safe_dataclass_to_dict(self, obj) -> Dict[str, Any]:
        """Safely convert dataclass to dictionary with JSON serialization handling"""
        try:
            result = {}
            
            # Get all fields from the dataclass
            if hasattr(obj, '__dataclass_fields__'):
                for field_name in obj.__dataclass_fields__:
                    value = getattr(obj, field_name, None)
                    
                    # Handle different value types
                    if value is None:
                        result[field_name] = None
                    elif isinstance(value, (str, int, float, bool)):
                        result[field_name] = value
                    elif isinstance(value, list):
                        result[field_name] = list(value)  # Create a copy
                    elif isinstance(value, dict):
                        result[field_name] = dict(value)  # Create a copy
                    elif hasattr(value, 'isoformat'):  # datetime objects
                        result[field_name] = value.isoformat()
                    else:
                        # Convert to string as fallback
                        result[field_name] = str(value)
            else:
                # Fallback: convert to string representation
                result = {'data': str(obj)}
                
            return result
            
        except Exception as e:
            print(f"Error in _safe_dataclass_to_dict: {e}")
            return {'conversion_error': str(e)}
    
    def _workflow_step_to_dict(self, step: WorkflowStep) -> Dict[str, Any]:
        """Convert workflow step to dictionary"""
        return {
            'step_name': step.step_name,
            'status': step.status,
            'start_time': step.start_time.isoformat() if step.start_time else None,
            'end_time': step.end_time.isoformat() if step.end_time else None,
            'ai_engine_used': step.ai_engine_used,
            'output_summary': step.output_summary,
            'issues_found': step.issues_found or [],
            'duration_ms': int((step.end_time - step.start_time).total_seconds() * 1000) if step.start_time and step.end_time else None
        }
    
    # Include existing extraction methods from original processor
    def _extract_patient_name(self, text: str) -> Optional[str]:
        """Extract patient name from text"""
        # First try direct search for names in billing context (most reliable)
        billing_patterns = [
            r'amount\s+due\s*:?\s*([A-Z][A-Z\s]+?)(?:\s*:\s*|$)',  # "AMOUNT DUE : TAN WENBIN"
            r'payments?\s*:?\s*([A-Z][A-Z\s]+?)(?:\s*:\s*|\s*\d)',   # "PAYMENTS TAN WENBIN: 11,062"
            r'adjustment\s*:?\s*([A-Z][A-Z\s]+?)(?:\s*:\s*|\s*\d)',  # "ADJUSTMENT TAN WENBIN: 863"
        ]
        
        # Look for names in billing contexts first (most reliable)
        for pattern in billing_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                name = match.group(1).strip()
                name = re.sub(r'[^A-Za-z\s]', '', name)  # Remove numbers and special chars
                name = ' '.join(name.split())  # Normalize whitespace
                
                # Validate it's a proper name (2-3 words, each 2+ chars)
                words = name.split()
                if (len(words) >= 2 and len(words) <= 3 and 
                    all(len(word) >= 2 for word in words) and
                    not any(word.lower() in ['amount', 'due', 'total', 'before', 'after', 'tax'] for word in words)):
                    return name.title()
        
        # Second try: look for standalone names at beginning of lines
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            # Look for lines with just a name (like "TAN WENBIN" on its own line)
            if re.match(r'^[A-Z]{2,}\s+[A-Z]{2,}(?:\s+[A-Z]{2,})?$', line):
                words = line.split()
                if (len(words) >= 2 and len(words) <= 3 and 
                    all(len(word) >= 2 for word in words) and
                    # Exclude obvious non-names
                    not any(word.upper() in ['TAX', 'INVOICE', 'BILL', 'CASE', 'CUSTOMER', 'EXTERNAL', 
                                           'ADMISSION', 'DISCHARGE', 'BILLING', 'AMOUNT', 'TOTAL',
                                           'LABORATORY', 'INVESTIGATIONS', 'TREATMENT', 'SERVICES',
                                           'PROCEDURES', 'PROFESSIONAL', 'SURGICAL', 'OPERATION',
                                           'CONSUMABLES', 'DRUGS', 'PRESCRIPTIONS', 'WARD'] for word in words)):
                    return line.title()
        
        # Fallback: traditional patterns
        traditional_patterns = [
            r'patient\s*:?\s*([A-Za-z\s,]+?)(?:\n|$|[0-9])',
            r'name\s*:?\s*([A-Za-z\s,]+?)(?:\n|$|[0-9])',
            r'claimant\s*:?\s*([A-Za-z\s,]+?)(?:\n|$|[0-9])',
        ]
        
        for pattern in traditional_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                name = match.group(1).strip()
                name = re.sub(r'[^A-Za-z\s,]', '', name)
                name = ' '.join(name.split())
                if len(name) >= 4 and len(name) <= 50:
                    return name.title()
        
        return None
    
    def _extract_patient_id(self, text: str) -> Optional[str]:
        """Extract patient ID from text"""
        patterns = [
            r'patient\s*id\s*:?\s*([A-Za-z0-9\-]+)',
            r'id\s*:?\s*([A-Za-z0-9\-]+)',
            r'nric\s*:?\s*([A-Za-z0-9]+)',
            r'ic\s*:?\s*([A-Za-z0-9]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip().upper()
        return None
    
    def _extract_policy_number(self, text: str) -> Optional[str]:
        """Extract policy number from text"""
        patterns = [
            r'policy\s*(?:no|number)\s*:?\s*([A-Za-z0-9\-]+)',
            r'policy\s*:?\s*([A-Za-z0-9\-]+)',
            r'pol\s*(?:no|#)\s*:?\s*([A-Za-z0-9\-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip().upper()
        return None
    
    def _extract_claim_number(self, text: str) -> Optional[str]:
        """Extract claim number from text"""
        patterns = [
            r'claim\s*(?:no|number)\s*:?\s*([A-Za-z0-9\-]+)',
            r'claim\s*:?\s*([A-Za-z0-9\-]+)',
            r'ref\s*(?:no|#)\s*:?\s*([A-Za-z0-9\-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip().upper()
        return None
    
    def _extract_provider_name(self, text: str) -> Optional[str]:
        """Extract healthcare provider name from text"""
        patterns = [
            # Medical facility patterns
            r'clinic\s*:?\s*([A-Za-z\s&,.-]+?)(?:\n|$|[0-9])',
            r'hospital\s*:?\s*([A-Za-z\s&,.-]+?)(?:\n|$|[0-9])',
            r'medical\s*center\s*:?\s*([A-Za-z\s&,.-]+?)(?:\n|$|[0-9])',
            r'healthcare\s*:?\s*([A-Za-z\s&,.-]+?)(?:\n|$|[0-9])',
            
            # Doctor/practitioner patterns
            r'doctor\s*:?\s*([A-Za-z\s,.-]+?)(?:\n|$|[0-9])',
            r'dr\.?\s*([A-Za-z\s,.-]+?)(?:\n|$|[0-9])',
            r'physician\s*:?\s*([A-Za-z\s,.-]+?)(?:\n|$|[0-9])',
            
            # General provider patterns
            r'provider\s*:?\s*([A-Za-z\s&,.-]+?)(?:\n|$|[0-9])',
            r'practitioner\s*:?\s*([A-Za-z\s&,.-]+?)(?:\n|$|[0-9])',
            
            # Location-based patterns (common in Singapore)
            r'pte\s*ltd\s*([A-Za-z\s&,.-]+?)(?:\n|$)',
            r'([A-Za-z\s&,.-]+?)\s*pte\s*ltd',
            r'([A-Za-z\s&,.-]+?)\s*clinic',
            r'([A-Za-z\s&,.-]+?)\s*hospital',
            r'([A-Za-z\s&,.-]+?)\s*medical',
            
            # Address-based extraction (if provider appears before address)
            r'([A-Z][A-Za-z\s&,.-]+?)\s*(?:singapore|s\s*\d{6}|\d{6})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                provider = match.group(1).strip()
                # Clean up the provider name
                provider = re.sub(r'[^A-Za-z\s&,.-]', '', provider)
                provider = re.sub(r'\s+', ' ', provider)  # Normalize whitespace
                
                # Filter out common false positives
                false_positives = ['patient', 'claim', 'invoice', 'bill', 'receipt', 'total', 'amount']
                if len(provider) > 3 and not any(fp in provider.lower() for fp in false_positives):
                    return provider.title()
        
        return None
    
    def _extract_diagnosis_codes(self, text: str) -> List[str]:
        """Extract ICD-10 diagnosis codes from text"""
        patterns = [
            r'\b[A-Z]\d{2}(?:\.\d+)?\b',
            r'icd\s*:?\s*([A-Z]\d{2}(?:\.\d+)?)',
            r'diagnosis\s*code\s*:?\s*([A-Z]\d{2}(?:\.\d+)?)',
        ]
        
        codes = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                code = match.upper() if isinstance(match, str) else match
                if code not in codes:
                    codes.append(code)
        return codes
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract treatment dates from text"""
        date_patterns = [
            r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}',
            r'\d{2,4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}',
            r'\b\d{1,2}\s+\w+\s+\d{2,4}\b',
            r'\b\w+\s+\d{1,2},?\s+\d{2,4}\b'
        ]
        
        dates = []
        for date_pattern in date_patterns:
            matches = re.findall(date_pattern, text)
            for match in matches:
                normalized_date = self._normalize_date(match)
                if normalized_date and normalized_date not in dates:
                    dates.append(normalized_date)
        return dates
    
    def _normalize_date(self, date_str: str) -> Optional[str]:
        """Normalize date string to standard format"""
        try:
            formats = ['%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d', '%Y/%m/%d', 
                      '%d-%m-%Y', '%d.%m.%Y', '%d %B %Y', '%B %d, %Y']
            
            for fmt in formats:
                try:
                    parsed_date = datetime.strptime(date_str.strip(), fmt)
                    return parsed_date.strftime('%Y-%m-%d')
                except ValueError:
                    continue
            return None
        except Exception:
            return None
    
    def _extract_amounts(self, text: str) -> List[float]:
        """Extract monetary amounts from text"""
        patterns = [
            r'(?:S\$|SGD|USD|\$)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'(?:amount|total|cost|fee|charge)\s*:?\s*(?:S\$|SGD|USD|\$)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'\b(\d{1,3}(?:,\d{3})*(?:\.\d{2}))\s*(?:S\$|SGD|USD|\$)',
        ]
        
        amounts = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    clean_amount = match.replace(',', '')
                    amount = float(clean_amount)
                    if amount > 0:  # Remove max amount limit
                        amounts.append(amount)
                except ValueError:
                    continue
        return list(set(amounts))
    
    def _extract_currency(self, text: str) -> str:
        """Extract currency from text"""
        currency_symbols = ['$', 'SGD', 'USD', 'MYR', 'S$']
        for symbol in currency_symbols:
            if symbol in text:
                if symbol in ['$', 'S$', 'SGD']:
                    return 'SGD'
                elif symbol == 'USD':
                    return 'USD'
                elif symbol == 'MYR':
                    return 'MYR'
        return 'SGD'
    
    def _simple_document_classification(self, text: str) -> str:
        """Enhanced text-based document classification - returns: claims, receipt, referral_letter, or memo"""
        if not text:
            return "claims"  # Default to claims instead of unknown
            
        text_lower = text.lower()
        
        # Enhanced classification with stronger keywords and patterns
        
        # Claims - medical bills, invoices, hospital documents (highest priority)
        claims_keywords = [
            # Invoice indicators
            'tax invoice', 'invoice', 'bill', 'billing', 'amount due', 'total amount', 'total:',
            # Medical facility indicators  
            'hospital', 'singapore general hospital', 'sgh', 'clinic', 'medical center', 'polyclinic',
            # Medical services
            'consultation fee', 'medication', 'treatment', 'diagnosis', 'prescription', 'medical certificate',
            # Healthcare specific
            'patient', 'doctor', 'physician', 'medical', 'healthcare', 'ward', 'admission', 'discharge',
            # Financial indicators
            'gst', 'service charge', 'medisave', 'insurance claim', 'co-payment', 'deductible'
        ]
        
        # Receipt - payment confirmations (second priority)
        receipt_keywords = [
            'receipt', 'paid', 'payment received', 'payment made', 'transaction completed',
            'cash payment', 'card payment', 'credit card', 'nets', 'payment confirmation',
            'amount paid', 'balance paid', 'settled', 'payment successful'
        ]
        
        # Referral Letter - medical referrals (third priority, exclude if has billing terms)
        referral_keywords = [
            'referral', 'refer to', 'refer patient', 'specialist appointment', 'medical referral',
            'kindly arrange', 'please see', 'consultation with', 'appointment with specialist',
            'dear colleague', 'dear doctor'
        ]
        
        # Memo - internal communications (lowest priority)
        memo_keywords = [
            'memo', 'memorandum', 'note to', 'internal note', 'communication', 'notice',
            'from:', 'to:', 'subject:', 'regarding:', 'circular', 'announcement'
        ]
        
        # Count matches for each category
        claims_matches = sum(1 for keyword in claims_keywords if keyword in text_lower)
        receipt_matches = sum(1 for keyword in receipt_keywords if keyword in text_lower)  
        referral_matches = sum(1 for keyword in referral_keywords if keyword in text_lower)
        memo_matches = sum(1 for keyword in memo_keywords if keyword in text_lower)
        
        # Check for financial indicators that would exclude referral classification
        financial_indicators = ['invoice', 'bill', 'amount due', 'total:', 'fee', 'payment', 'charge']
        has_financial_terms = any(indicator in text_lower for indicator in financial_indicators)
        
        # Classification logic with scoring
        if claims_matches >= 2 or (claims_matches >= 1 and ('invoice' in text_lower or 'bill' in text_lower or 'amount due' in text_lower)):
            return "claims"
        elif receipt_matches >= 2 or (receipt_matches >= 1 and ('paid' in text_lower or 'receipt' in text_lower)):
            return "receipt"  
        elif referral_matches >= 1 and not has_financial_terms:
            return "referral_letter"
        elif memo_matches >= 2:
            return "memo"
        elif claims_matches > 0:
            return "claims"
        else:
            # If no clear matches, default to claims for medical contexts
            return "claims"
    
    def _extract_line_items(self, text: str) -> List[Dict[str, Any]]:
        """Extract line items with service descriptions and amounts from OCR text"""
        if not text:
            return []
            
        line_items = []
        lines = text.split('\n')
        
        # Common medical service patterns
        service_patterns = [
            # Direct service patterns with amounts
            r'([A-Za-z][^$]*?(?:fee|charge|cost|service|test|scan|consultation|medication|treatment|ward|room|nursing|laboratory|x-ray|mri|ct|blood|urine|injection|surgery|anesthesia))\s*[:\-]?\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            # Table-like patterns
            r'([A-Za-z][^$\d]*?)\s+\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*$',
            # Description followed by amount
            r'^([A-Za-z][^$\d]*?)(?:\s*-\s*|\s+)\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        ]
        
        # Medical service keywords for better matching
        medical_keywords = [
            'consultation', 'fee', 'charge', 'test', 'scan', 'x-ray', 'mri', 'ct', 'blood', 'urine',
            'medication', 'treatment', 'ward', 'room', 'nursing', 'laboratory', 'injection', 
            'surgery', 'anesthesia', 'diagnostic', 'therapy', 'examination', 'procedure'
        ]
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 10:  # Skip very short lines
                continue
                
            # Try each pattern
            for pattern in service_patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                for match in matches:
                    if len(match) == 2:
                        description, amount_str = match
                        description = description.strip()
                        
                        # Clean up description
                        description = re.sub(r'[:\-]+$', '', description).strip()
                        
                        # Skip if description is too short or doesn't contain medical terms
                        if len(description) < 3:
                            continue
                            
                        # Check if description contains medical keywords
                        has_medical_keyword = any(keyword in description.lower() for keyword in medical_keywords)
                        
                        # Parse amount
                        try:
                            amount = float(amount_str.replace(',', ''))
                            if amount > 0 and (has_medical_keyword or len(description) > 8):
                                # Capitalize first letter of each word
                                description = ' '.join(word.capitalize() for word in description.split())
                                
                                line_items.append({
                                    'description': description,
                                    'amount': amount,
                                    'currency': 'SGD'  # Default currency
                                })
                        except ValueError:
                            continue
        
        # Remove duplicates and sort by amount descending
        seen = set()
        unique_items = []
        for item in sorted(line_items, key=lambda x: x['amount'], reverse=True):
            key = (item['description'].lower(), item['amount'])
            if key not in seen:
                seen.add(key)
                unique_items.append(item)
        
        return unique_items[:10]  # Limit to 10 items
    
    def _create_line_items_from_amounts(self, text: str, amounts: List[float], currency: str) -> List[Dict[str, Any]]:
        """Create line items from amounts by trying to find nearby service descriptions"""
        if not amounts:
            return []
            
        line_items = []
        text_lines = text.split('\n')
        
        # Common medical services for fallback naming
        common_services = [
            'Consultation Fee', 'Medical Examination', 'Laboratory Test', 'Blood Test', 
            'X-Ray', 'Medication', 'Treatment Fee', 'Service Charge', 'Ward Charges',
            'Nursing Care', 'Diagnostic Test', 'Medical Procedure'
        ]
        
        for i, amount in enumerate(amounts[:10]):  # Limit to 10 items
            description = None
            
            # Try to find a description near this amount in the text
            amount_str = f"{amount:.2f}".replace('.00', '')  # Remove .00 for matching
            amount_patterns = [
                str(int(amount)) if amount == int(amount) else str(amount),
                f"{amount:,.2f}",
                f"{amount:,.0f}" if amount == int(amount) else f"{amount:.2f}",
                amount_str
            ]
            
            # Search for lines containing this amount
            for line in text_lines:
                for pattern in amount_patterns:
                    if pattern in line and len(line.strip()) > 10:
                        # Extract potential service name from the line
                        parts = line.split(pattern)
                        if len(parts) > 1:
                            potential_desc = parts[0].strip()
                            # Clean up the description
                            potential_desc = re.sub(r'^[:\-\s]+|[:\-\s]+$', '', potential_desc)
                            potential_desc = re.sub(r'\s+', ' ', potential_desc)
                            
                            if len(potential_desc) > 3 and len(potential_desc) < 50:
                                # Capitalize properly
                                description = ' '.join(word.capitalize() for word in potential_desc.split())
                                break
                
                if description:
                    break
            
            # Fallback to common service names
            if not description:
                if i < len(common_services):
                    description = common_services[i]
                else:
                    description = f"Medical Service {i + 1}"
            
            line_items.append({
                'description': description,
                'amount': amount,
                'currency': currency or 'SGD'
            })
        
        return line_items
    
    def _classification_to_dict(self, classification_result) -> Dict[str, Any]:
        """Convert classification result to dict, handling both namedtuple and object types"""
        if hasattr(classification_result, 'to_dict'):
            # Real DocumentClassificationResult object
            return classification_result.to_dict()
        else:
            # Namedtuple fallback result
            document_type = classification_result.document_type
            if hasattr(document_type, 'value'):
                document_type = document_type.value
                
            return {
                'document_type': document_type,
                'confidence': classification_result.confidence,
                'reasoning': ['Simple text-based classification'],
                'detected_features': [],
                'processing_time_ms': 0
            }