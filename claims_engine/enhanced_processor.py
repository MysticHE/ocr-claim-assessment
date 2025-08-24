import re
import json
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
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
    
    # AI Analysis Results
    document_type: Optional[str] = None
    document_classification_confidence: float = 0.0
    document_quality_score: float = 0.0
    quality_issues: List[str] = None
    quality_acceptable: bool = True
    
    # Workflow tracking
    processing_stage: str = "initiated"
    workflow_steps_completed: List[str] = None
    ai_confidence_scores: Dict[str, float] = None
    
    def __post_init__(self):
        if self.diagnosis_codes is None:
            self.diagnosis_codes = []
        if self.treatment_dates is None:
            self.treatment_dates = []
        if self.amounts is None:
            self.amounts = []
        if self.quality_issues is None:
            self.quality_issues = []
        if self.workflow_steps_completed is None:
            self.workflow_steps_completed = []
        if self.ai_confidence_scores is None:
            self.ai_confidence_scores = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

@dataclass
class WorkflowStep:
    """Individual workflow step tracking"""
    step_name: str
    status: str  # pending, in_progress, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    ai_engine_used: Optional[str] = None
    confidence_score: Optional[float] = None
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
        self._classifier_initialization_attempted = False
        self._assessor_initialization_attempted = False
        
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
            'max_claim_amount': 10000.00,
            'min_claim_amount': 1.00,
            'auto_approve_threshold': 500.00,
            'quality_threshold': 0.6,  # Minimum quality score
            'classification_confidence_threshold': 0.7,
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
                'treatment_date',
                'amount'
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
    
    def process_enhanced_claim(self, ocr_result: Dict[str, Any], 
                             image_path: Optional[str] = None) -> Dict[str, Any]:
        """Main enhanced claim processing workflow"""
        start_time = time.time()
        
        print(f"Starting enhanced claim processing workflow")
        print(f"   OCR result keys: {list(ocr_result.keys()) if ocr_result else 'None'}")
        print(f"   Image path provided: {image_path is not None}")
        print(f"   Classifier available: {self.classifier_available}")
        print(f"   Quality assessor available: {self.quality_assessor_available}")
        
        # Fallback mode detection
        fallback_mode = not (self.classifier_available and self.quality_assessor_available)
        if fallback_mode:
            print("   Running in lightweight fallback mode (no heavy AI dependencies)")
        
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
                current_step.confidence_score = quality_result.quality_score.overall_score
                current_step.output_summary = f"Quality score: {quality_result.quality_score.overall_score:.2f}"
                current_step.issues_found = quality_result.recommendations
                current_step.status = "completed"
            else:
                # Simple OCR-confidence based quality assessment fallback
                ocr_confidence = ocr_result.get('confidence', 0.5)
                quality_score = max(0.4, ocr_confidence)  # Minimum threshold
                current_step.confidence_score = quality_score
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
            if self.classifier_available:
                classification_result = self.document_classifier.classify_document(
                    ocr_result.get('text', ''), image_path
                )
                current_step.confidence_score = classification_result.confidence
                current_step.output_summary = f"Classified as: {classification_result.document_type.value}"
                current_step.status = "completed"
            else:
                # Simple text-based classification fallback
                doc_type = self._simple_document_classification(ocr_result.get('text', ''))
                current_step.confidence_score = 0.6  # Lower confidence for simple classification
                current_step.output_summary = f"Simple classification: {doc_type} (fallback mode)"
                current_step.status = "completed"
            
            current_step.end_time = datetime.now()
            workflow_steps.append(current_step)
            
            # Step 4: OCR Processing (already completed)
            workflow_steps.append(WorkflowStep(
                step_name="ocr_processing",
                status="completed",
                start_time=datetime.now(),
                end_time=datetime.now(),
                ai_engine_used=ocr_result.get('engine', 'hybrid'),
                confidence_score=ocr_result.get('confidence', 0.0),
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
            current_step.output_summary = f"Fraud check: {len(fraud_findings)} suspicious indicators"
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
            
            current_step.confidence_score = decision.confidence
            current_step.output_summary = f"Decision: {decision.status.value} (confidence: {decision.confidence:.2f})"
            current_step.status = "completed"
            current_step.end_time = datetime.now()
            workflow_steps.append(current_step)
            
            # Calculate total processing time
            total_processing_time = int((time.time() - start_time) * 1000)
            
            print(f"Enhanced processing workflow completed successfully")
            print(f"   Total processing time: {total_processing_time}ms")
            print(f"   Workflow steps completed: {len([s for s in workflow_steps if s.status == 'completed'])}/{len(workflow_steps)}")
            print(f"   Decision status: {decision.status.value}")
            print(f"   Decision confidence: {decision.confidence}")
            
            # Compile results
            return {
                'success': True,
                'status': decision.status.value,
                'confidence': decision.confidence,
                'amount': decision.amount,
                'reasons': decision.reasons,
                'processing_notes': decision.processing_notes,
                'total_processing_time_ms': total_processing_time,
                
                # Enhanced data
                'extracted_data': extracted_data.to_dict(),
                'validation_issues': validation_issues,
                'fraud_findings': fraud_findings,
                'workflow_steps': [self._workflow_step_to_dict(step) for step in workflow_steps],
                
                # AI Analysis results
                'document_classification': classification_result.to_dict() if classification_result else None,
                'quality_assessment': quality_result.to_dict() if quality_result else None,
                'ai_engines_used': [step.ai_engine_used for step in workflow_steps if step.ai_engine_used],
                
                # Workflow summary
                'workflow_completion': {
                    'total_steps': len(workflow_steps),
                    'completed_steps': len([s for s in workflow_steps if s.status == "completed"]),
                    'failed_steps': len([s for s in workflow_steps if s.status == "failed"]),
                    'skipped_steps': len([s for s in workflow_steps if s.status == "skipped"])
                }
            }
            
        except Exception as e:
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
        """Enhanced structured data extraction with AI context"""
        text = ocr_result.get('text', '')
        
        # Create enhanced claim data
        extracted = EnhancedClaimData()
        
        # Basic extraction (reuse existing logic)
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
        
        # Add AI analysis results
        if classification_result:
            extracted.document_type = classification_result.document_type.value
            extracted.document_classification_confidence = classification_result.confidence
        
        if quality_result:
            extracted.document_quality_score = quality_result.quality_score.overall_score
            extracted.quality_issues = [issue.value for issue in quality_result.issues_detected]
            extracted.quality_acceptable = quality_result.is_acceptable
        
        # Set AI confidence scores
        extracted.ai_confidence_scores = {
            'ocr_confidence': ocr_result.get('confidence', 0.0),
            'classification_confidence': classification_result.confidence if classification_result else 0.0,
            'quality_score': quality_result.quality_score.overall_score if quality_result else 0.5
        }
        
        # Set processing stage
        extracted.processing_stage = "data_extracted"
        
        return extracted
    
    def validate_enhanced_claim_data(self, data: EnhancedClaimData) -> Tuple[bool, List[str]]:
        """Enhanced validation with document-type specific checks"""
        issues = []
        
        # Basic validation (existing logic)
        if not data.patient_name:
            issues.append("Missing patient name")
        
        if not data.treatment_dates:
            issues.append("Missing treatment date")
        
        if not data.amounts and not data.total_amount:
            issues.append("Missing claim amount")
        
        # Document quality validation
        if not data.quality_acceptable:
            issues.append("Document quality below acceptable threshold")
        
        # Document type specific validation
        if data.document_type and data.document_type in self.rules['document_type_requirements']:
            required_fields = self.rules['document_type_requirements'][data.document_type]
            
            # Map document requirements to extracted data fields
            field_mapping = {
                'amount': data.total_amount or (data.amounts and data.amounts[0]),
                'date': data.treatment_dates and data.treatment_dates[0],
                'patient_name': data.patient_name,
                'provider': data.provider_name,
                'merchant': data.provider_name,  # Alias for provider
                'policy_number': data.policy_number,
                'claimant_name': data.patient_name,  # Alias for patient
            }
            
            for required_field in required_fields:
                if not field_mapping.get(required_field):
                    issues.append(f"Missing required field for {data.document_type}: {required_field}")
        
        # Classification confidence check
        if data.document_classification_confidence < self.rules['classification_confidence_threshold']:
            issues.append(f"Low document classification confidence: {data.document_classification_confidence:.2f}")
        
        return len(issues) == 0, issues
    
    def enhanced_fraud_detection(self, ocr_result: Dict[str, Any], 
                               data: EnhancedClaimData) -> List[str]:
        """Enhanced fraud detection with multiple techniques"""
        suspicious_findings = []
        
        # Existing pattern-based detection
        text = ocr_result.get('text', '').lower()
        for pattern in self.rules['suspicious_patterns']:
            if re.search(pattern, text, re.IGNORECASE):
                suspicious_findings.append(f"Suspicious pattern detected: {pattern}")
        
        # Duplicate claim detection
        if self.rules['duplicate_check_enabled']:
            duplicate_result = self._check_duplicate_claim(data)
            if duplicate_result:
                suspicious_findings.append(f"Duplicate: {duplicate_result['details']} (Similarity: {duplicate_result['similarity_score']:.1%})")
        
        # Quality-based fraud indicators
        if data.quality_issues:
            quality_fraud_indicators = [
                'overexposed', 'underexposed', 'compression_artifacts', 
                'partial_visibility', 'skewed'
            ]
            
            for issue in data.quality_issues:
                if issue in quality_fraud_indicators:
                    suspicious_findings.append(f"Potential document tampering: {issue}")
        
        # Amount-based anomaly detection
        if data.total_amount:
            if data.total_amount == int(data.total_amount) and data.total_amount > 100:
                suspicious_findings.append("Suspicious round amount for large claim")
        
        # OCR confidence anomaly
        ocr_confidence = ocr_result.get('confidence', 1.0)
        if ocr_confidence < 0.5:
            suspicious_findings.append(f"Very low OCR confidence may indicate document quality issues: {ocr_confidence:.2f}")
        
        return suspicious_findings
    
    def _check_duplicate_claim(self, data: EnhancedClaimData) -> Optional[Dict[str, Any]]:
        """Enhanced duplicate detection with fuzzy matching and similarity scoring"""
        try:
            # Store in database for persistent duplicate detection
            from database.supabase_client import SupabaseClient
            
            db_client = SupabaseClient()
            
            # Calculate multiple similarity metrics
            duplicates_found = []
            
            # 1. Exact hash matching (existing method)
            claim_key = f"{data.patient_name}|{data.total_amount}|{data.treatment_dates}|{data.provider_name}"
            claim_hash = hashlib.md5(claim_key.encode()).hexdigest()
            
            # 2. Fuzzy text matching for patient names
            if data.patient_name:
                similar_claims = self._find_similar_claims_by_text(db_client, data)
                duplicates_found.extend(similar_claims)
            
            # 3. Amount and date proximity matching
            proximity_matches = self._find_claims_by_proximity(db_client, data)
            duplicates_found.extend(proximity_matches)
            
            # Store current claim for future comparisons
            self._store_claim_for_comparison(db_client, data, claim_hash)
            
            if duplicates_found:
                # Return the highest similarity match
                best_match = max(duplicates_found, key=lambda x: x['similarity_score'])
                
                return {
                    'type': 'duplicate_detected',
                    'similarity_score': best_match['similarity_score'],
                    'match_type': best_match['match_type'],
                    'days_ago': best_match['days_ago'],
                    'matched_claim_id': best_match.get('claim_id'),
                    'details': best_match['details'],
                    'all_matches': duplicates_found[:3]  # Top 3 matches
                }
            
            return None
            
        except Exception as e:
            # Fallback to in-memory cache
            return self._fallback_duplicate_check(data)
    
    def _find_similar_claims_by_text(self, db_client, data: EnhancedClaimData) -> List[Dict]:
        """Find claims with similar text using fuzzy matching"""
        matches = []
        
        try:
            # Get recent claims (last 90 days) for comparison
            query_result = db_client.supabase.table('claims').select('*').gte(
                'created_at', 
                (datetime.now() - timedelta(days=90)).isoformat()
            ).execute()
            
            if query_result.data:
                for claim in query_result.data:
                    similarity_score = self._calculate_text_similarity(data, claim)
                    
                    if similarity_score > 0.8:  # High similarity threshold
                        days_ago = (datetime.now() - datetime.fromisoformat(claim['created_at'].replace('Z', '+00:00'))).days
                        
                        matches.append({
                            'similarity_score': similarity_score,
                            'match_type': 'fuzzy_text',
                            'days_ago': days_ago,
                            'claim_id': claim['id'],
                            'details': f"Similar text content with {similarity_score:.1%} similarity"
                        })
            
        except Exception:
            pass
        
        return matches
    
    def _find_claims_by_proximity(self, db_client, data: EnhancedClaimData) -> List[Dict]:
        """Find claims with similar amounts and dates"""
        matches = []
        
        if not data.total_amount or not data.treatment_dates:
            return matches
        
        try:
            # Look for claims with similar amounts (±10%) and recent dates (±7 days)
            amount_min = data.total_amount * 0.9
            amount_max = data.total_amount * 1.1
            
            query_result = db_client.supabase.table('claims').select('*').gte(
                'claim_amount', amount_min
            ).lte('claim_amount', amount_max).execute()
            
            if query_result.data:
                for claim in query_result.data:
                    # Calculate date similarity
                    claim_date = datetime.fromisoformat(claim['created_at'].replace('Z', '+00:00'))
                    date_diff = abs((datetime.now() - claim_date).days)
                    
                    if date_diff <= 30:  # Within 30 days
                        amount_similarity = 1 - abs(data.total_amount - claim['claim_amount']) / data.total_amount
                        date_similarity = max(0, 1 - date_diff / 30)
                        overall_similarity = (amount_similarity * 0.7) + (date_similarity * 0.3)
                        
                        if overall_similarity > 0.7:
                            matches.append({
                                'similarity_score': overall_similarity,
                                'match_type': 'amount_date_proximity',
                                'days_ago': date_diff,
                                'claim_id': claim['id'],
                                'details': f"Similar amount (${claim['claim_amount']}) within {date_diff} days"
                            })
        
        except Exception:
            pass
        
        return matches
    
    def _calculate_text_similarity(self, data: EnhancedClaimData, stored_claim: Dict) -> float:
        """Calculate text similarity using multiple algorithms"""
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
    
    def _store_claim_for_comparison(self, db_client, data: EnhancedClaimData, claim_hash: str):
        """Store claim data for future duplicate comparisons"""
        try:
            # Store in duplicate tracking table (would need to create this table)
            duplicate_data = {
                'claim_hash': claim_hash,
                'patient_name': data.patient_name,
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
            claim_key = f"{data.patient_name}|{data.total_amount}|{data.treatment_dates}|{data.provider_name}"
            claim_hash = hashlib.md5(claim_key.encode()).hexdigest()
            
            if claim_hash in self.processed_claims_cache:
                stored_claim = self.processed_claims_cache[claim_hash]
                time_diff = datetime.now() - stored_claim['timestamp']
                
                if time_diff.days < 30:
                    return {
                        'type': 'duplicate_detected',
                        'similarity_score': 1.0,
                        'match_type': 'exact_hash',
                        'days_ago': time_diff.days,
                        'details': f"Exact duplicate detected (submitted {time_diff.days} days ago)"
                    }
            else:
                self.processed_claims_cache[claim_hash] = {
                    'timestamp': datetime.now(),
                    'patient': data.patient_name,
                    'amount': data.total_amount
                }
            
            return None
        except Exception:
            return None
    
    def make_enhanced_decision(self, data: EnhancedClaimData, validation_issues: List[str],
                             fraud_findings: List[str], quality_result=None,
                             classification_result=None, ocr_confidence: float = 0.0) -> ClaimDecision:
        """Make enhanced claim decision with AI inputs"""
        
        # Calculate enhanced confidence score
        base_confidence = ocr_confidence * 0.3
        
        # Add classification confidence
        if classification_result:
            base_confidence += classification_result.confidence * 0.2
        
        # Add quality score
        if quality_result:
            base_confidence += quality_result.quality_score.overall_score * 0.3
        
        # Add data completeness score
        completeness_score = self._calculate_data_completeness_score(data)
        base_confidence += completeness_score * 0.2
        
        # Cap at 1.0
        base_confidence = min(1.0, base_confidence)
        
        reasons = []
        
        # Enhanced rejection logic
        if validation_issues:
            return ClaimDecision(
                status=ClaimStatus.REJECTED,
                confidence=base_confidence,
                amount=data.total_amount or 0,
                reasons=validation_issues,
                processing_notes="Claim rejected due to validation failures in enhanced processing"
            )
        
        # Quality-based rejection
        if quality_result and not quality_result.is_acceptable:
            return ClaimDecision(
                status=ClaimStatus.REJECTED,
                confidence=base_confidence * 0.7,
                amount=data.total_amount or 0,
                reasons=["Document quality below acceptable threshold"] + quality_result.recommendations[:3],
                processing_notes="Claim rejected due to poor document quality"
            )
        
        # Fraud-based review
        if fraud_findings:
            return ClaimDecision(
                status=ClaimStatus.REVIEW,
                confidence=base_confidence * 0.8,
                amount=data.total_amount or 0,
                reasons=fraud_findings,
                processing_notes="Claim flagged for manual review due to fraud indicators"
            )
        
        # Document type specific logic
        if classification_result and classification_result.confidence < 0.5:
            return ClaimDecision(
                status=ClaimStatus.REVIEW,
                confidence=base_confidence * 0.9,
                amount=data.total_amount or 0,
                reasons=["Uncertain document classification requires review"],
                processing_notes="Manual review required due to unclear document type"
            )
        
        # Enhanced approval logic
        claim_amount = data.total_amount or 0
        
        if claim_amount <= self.rules['auto_approve_threshold']:
            if base_confidence >= 0.8:
                reasons.append(f"Small claim auto-approved with high AI confidence")
                return ClaimDecision(
                    status=ClaimStatus.APPROVED,
                    confidence=base_confidence,
                    amount=claim_amount,
                    reasons=reasons,
                    processing_notes="Auto-approved by enhanced AI processing"
                )
        
        # Medium amounts with high confidence
        elif claim_amount <= self.rules['max_claim_amount']:
            if base_confidence >= 0.9:
                reasons.append("High confidence claim approved by AI analysis")
                return ClaimDecision(
                    status=ClaimStatus.APPROVED,
                    confidence=base_confidence,
                    amount=claim_amount,
                    reasons=reasons,
                    processing_notes="Approved based on comprehensive AI analysis"
                )
        
        # Default to review for safety
        reasons.append("Enhanced processing suggests manual review")
        return ClaimDecision(
            status=ClaimStatus.REVIEW,
            confidence=base_confidence,
            amount=claim_amount,
            reasons=reasons,
            processing_notes="Manual review recommended by enhanced AI processing"
        )
    
    def _calculate_data_completeness_score(self, data: EnhancedClaimData) -> float:
        """Calculate completeness score based on extracted data"""
        total_fields = 10  # Total expected fields
        populated_fields = 0
        
        if data.patient_name:
            populated_fields += 1
        if data.patient_id:
            populated_fields += 1
        if data.policy_number:
            populated_fields += 1
        if data.provider_name:
            populated_fields += 1
        if data.treatment_dates:
            populated_fields += 1
        if data.amounts:
            populated_fields += 1
        if data.diagnosis_codes:
            populated_fields += 1
        if data.claim_number:
            populated_fields += 1
        if data.currency:
            populated_fields += 1
        if data.total_amount:
            populated_fields += 1
        
        return populated_fields / total_fields
    
    def _workflow_step_to_dict(self, step: WorkflowStep) -> Dict[str, Any]:
        """Convert workflow step to dictionary"""
        return {
            'step_name': step.step_name,
            'status': step.status,
            'start_time': step.start_time.isoformat() if step.start_time else None,
            'end_time': step.end_time.isoformat() if step.end_time else None,
            'ai_engine_used': step.ai_engine_used,
            'confidence_score': step.confidence_score,
            'output_summary': step.output_summary,
            'issues_found': step.issues_found or [],
            'duration_ms': int((step.end_time - step.start_time).total_seconds() * 1000) if step.start_time and step.end_time else None
        }
    
    # Include existing extraction methods from original processor
    def _extract_patient_name(self, text: str) -> Optional[str]:
        """Extract patient name from text"""
        patterns = [
            r'patient\s*:?\s*([A-Za-z\s,]+?)(?:\n|$|[0-9])',
            r'name\s*:?\s*([A-Za-z\s,]+?)(?:\n|$|[0-9])',
            r'claimant\s*:?\s*([A-Za-z\s,]+?)(?:\n|$|[0-9])',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                name = match.group(1).strip()
                name = re.sub(r'[^A-Za-z\s,]', '', name)
                if len(name) > 3:
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
            r'clinic\s*:?\s*([A-Za-z\s&,.-]+?)(?:\n|$|[0-9])',
            r'hospital\s*:?\s*([A-Za-z\s&,.-]+?)(?:\n|$|[0-9])',
            r'doctor\s*:?\s*([A-Za-z\s,.-]+?)(?:\n|$|[0-9])',
            r'provider\s*:?\s*([A-Za-z\s&,.-]+?)(?:\n|$|[0-9])',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                provider = match.group(1).strip()
                provider = re.sub(r'[^A-Za-z\s&,.-]', '', provider)
                if len(provider) > 3:
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
                    if amount > 0 and amount <= self.rules['max_claim_amount']:
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
        """Simple text-based document classification fallback"""
        if not text:
            return "unknown"
            
        text_lower = text.lower()
        
        # Simple keyword-based classification
        if any(keyword in text_lower for keyword in ['receipt', 'paid', 'cash', 'card', 'total amount']):
            return "receipt"
        elif any(keyword in text_lower for keyword in ['invoice', 'bill', 'due', 'payment']):
            return "invoice"
        elif any(keyword in text_lower for keyword in ['referral', 'refer to', 'specialist']):
            return "referral_letter"
        elif any(keyword in text_lower for keyword in ['prescription', 'medication', 'dosage', 'rx']):
            return "prescription"
        elif any(keyword in text_lower for keyword in ['certificate', 'medical', 'doctor', 'clinic']):
            return "medical_certificate"
        elif any(keyword in text_lower for keyword in ['diagnosis', 'report', 'test', 'result']):
            return "diagnostic_report"
        elif any(keyword in text_lower for keyword in ['memo', 'memorandum', 'note']):
            return "memo"
        elif any(keyword in text_lower for keyword in ['insurance', 'claim', 'policy']):
            return "insurance_form"
        elif any(keyword in text_lower for keyword in ['id', 'identity', 'nric', 'passport']):
            return "identity_document"
        else:
            return "unknown"