import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from database.models import ClaimStatus, ClaimDecision

@dataclass
class ExtractedClaimData:
    """Structured data extracted from OCR text"""
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
    
    def __post_init__(self):
        if self.diagnosis_codes is None:
            self.diagnosis_codes = []
        if self.treatment_dates is None:
            self.treatment_dates = []
        if self.amounts is None:
            self.amounts = []

class ClaimProcessor:
    """Process OCR results and make claim decisions based on business rules"""
    
    def __init__(self):
        """Initialize claim processor with business rules"""
        self.load_business_rules()
    
    def load_business_rules(self):
        """Load business rules for claim processing"""
        # These rules would typically come from the Word document or a configuration file
        self.rules = {
            # Remove all amount limits
            'auto_reject_reasons': [
                'expired_policy',
                'invalid_diagnosis',
                'missing_information',
                'duplicate_claim'
            ],
            'required_fields': [
                'patient_name',
                'treatment_date',
                'amount'
            ],
            'valid_diagnosis_prefixes': [
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                'U', 'V', 'W', 'X', 'Y', 'Z'  # ICD-10 codes
            ],
            'suspicious_patterns': [
                r'\b(fraud|fake|false)\b',
                r'\b(duplicate|copy|photocopy)\b',
                r'\b(altered|modified|changed)\b'
            ],
            'currency_symbols': ['$', 'SGD', 'USD', 'MYR', 'S$'],
            'date_formats': [
                r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}',  # DD/MM/YYYY, MM/DD/YYYY
                r'\d{2,4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}',  # YYYY/MM/DD
                r'\b\d{1,2}\s+\w+\s+\d{2,4}\b',           # DD Month YYYY
                r'\b\w+\s+\d{1,2},?\s+\d{2,4}\b'          # Month DD, YYYY
            ]
        }
    
    def extract_structured_data(self, ocr_result: Dict[str, Any]) -> ExtractedClaimData:
        """Extract structured data from OCR text"""
        text = ocr_result.get('text', '').lower()
        original_text = ocr_result.get('text', '')
        
        extracted = ExtractedClaimData()
        
        # Extract patient name
        extracted.patient_name = self._extract_patient_name(original_text)
        
        # Extract patient ID
        extracted.patient_id = self._extract_patient_id(original_text)
        
        # Extract policy number
        extracted.policy_number = self._extract_policy_number(original_text)
        
        # Extract claim number
        extracted.claim_number = self._extract_claim_number(original_text)
        
        # Extract provider name
        extracted.provider_name = self._extract_provider_name(original_text)
        
        # Extract diagnosis codes
        extracted.diagnosis_codes = self._extract_diagnosis_codes(original_text)
        
        # Extract treatment dates
        extracted.treatment_dates = self._extract_dates(original_text)
        
        # Extract amounts
        extracted.amounts = self._extract_amounts(original_text)
        extracted.total_amount = max(extracted.amounts) if extracted.amounts else None
        
        # Extract currency
        extracted.currency = self._extract_currency(original_text)
        
        return extracted
    
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
                # Clean up the name
                name = re.sub(r'[^A-Za-z\s,]', '', name)
                if len(name) > 3:  # Minimum name length
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
                # Clean up the provider name
                provider = re.sub(r'[^A-Za-z\s&,.-]', '', provider)
                if len(provider) > 3:
                    return provider.title()
        
        return None
    
    def _extract_diagnosis_codes(self, text: str) -> List[str]:
        """Extract ICD-10 diagnosis codes from text"""
        # ICD-10 codes typically follow pattern: Letter followed by 2-3 digits, optional decimal and more digits
        patterns = [
            r'\b[A-Z]\d{2}(?:\.\d+)?\b',  # Standard ICD-10 format
            r'icd\s*:?\s*([A-Z]\d{2}(?:\.\d+)?)',  # Preceded by ICD
            r'diagnosis\s*code\s*:?\s*([A-Z]\d{2}(?:\.\d+)?)',  # Diagnosis code
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
        dates = []
        
        for date_pattern in self.rules['date_formats']:
            matches = re.findall(date_pattern, text)
            for match in matches:
                # Normalize date format
                normalized_date = self._normalize_date(match)
                if normalized_date and normalized_date not in dates:
                    dates.append(normalized_date)
        
        return dates
    
    def _normalize_date(self, date_str: str) -> Optional[str]:
        """Normalize date string to standard format"""
        try:
            # Try different parsing formats
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
            r'(?:S\$|SGD|USD|\$)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',  # Currency symbols
            r'(?:amount|total|cost|fee|charge)\s*:?\s*(?:S\$|SGD|USD|\$)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'\b(\d{1,3}(?:,\d{3})*(?:\.\d{2}))\s*(?:S\$|SGD|USD|\$)',  # Amount followed by currency
        ]
        
        amounts = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Clean and convert amount
                    clean_amount = match.replace(',', '')
                    amount = float(clean_amount)
                    if amount > 0:  # Remove amount limit
                        amounts.append(amount)
                except ValueError:
                    continue
        
        return list(set(amounts))  # Remove duplicates
    
    def _extract_currency(self, text: str) -> str:
        """Extract currency from text"""
        for symbol in self.rules['currency_symbols']:
            if symbol in text:
                if symbol in ['$', 'S$', 'SGD']:
                    return 'SGD'
                elif symbol == 'USD':
                    return 'USD'
                elif symbol == 'MYR':
                    return 'MYR'
        
        return 'SGD'  # Default currency
    
    def validate_claim_data(self, data: ExtractedClaimData) -> Tuple[bool, List[str]]:
        """Validate extracted claim data"""
        issues = []
        
        # Check required fields
        if not data.patient_name:
            issues.append("Missing patient name")
        
        if not data.treatment_dates:
            issues.append("Missing treatment date")
        
        if not data.amounts and not data.total_amount:
            issues.append("Missing claim amount")
        
        # Validate amounts
        # Remove all amount validation - no limits
        
        # Validate dates
        current_date = datetime.now()
        for date_str in data.treatment_dates:
            try:
                treatment_date = datetime.strptime(date_str, '%Y-%m-%d')
                if treatment_date > current_date:
                    issues.append("Treatment date in the future")
                elif (current_date - treatment_date).days > 365:  # 1 year limit
                    issues.append("Treatment date too old (>1 year)")
            except ValueError:
                issues.append(f"Invalid date format: {date_str}")
        
        # Validate diagnosis codes
        if data.diagnosis_codes:
            for code in data.diagnosis_codes:
                if not any(code.startswith(prefix) for prefix in self.rules['valid_diagnosis_prefixes']):
                    issues.append(f"Invalid diagnosis code format: {code}")
        
        return len(issues) == 0, issues
    
    def detect_suspicious_patterns(self, ocr_result: Dict[str, Any]) -> List[str]:
        """Detect suspicious patterns in the OCR text"""
        text = ocr_result.get('text', '').lower()
        suspicious_findings = []
        
        for pattern in self.rules['suspicious_patterns']:
            if re.search(pattern, text, re.IGNORECASE):
                suspicious_findings.append(f"Suspicious pattern detected: {pattern}")
        
        # Check OCR confidence
        confidence = ocr_result.get('confidence', 1.0)
        if confidence < 0.7:
            suspicious_findings.append(f"Low OCR confidence: {confidence:.2f}")
        
        # Check for duplicate detection within text
        lines = text.split('\n')
        unique_lines = set(lines)
        if len(lines) > len(unique_lines) * 1.5:  # Too many duplicate lines
            suspicious_findings.append("Possible duplicate or copied content")
        
        return suspicious_findings
    
    def make_decision(self, data: ExtractedClaimData, validation_issues: List[str], 
                     suspicious_findings: List[str], ocr_confidence: float) -> ClaimDecision:
        """Make claim processing decision based on business rules"""
        
        # Determine base confidence
        base_confidence = ocr_confidence * 0.7  # Start with OCR confidence
        
        # Adjust confidence based on data quality
        if data.patient_name:
            base_confidence += 0.1
        if data.policy_number:
            base_confidence += 0.1
        if data.treatment_dates:
            base_confidence += 0.1
        if data.amounts or data.total_amount:
            base_confidence += 0.1
        
        # Cap confidence at 1.0
        base_confidence = min(1.0, base_confidence)
        
        reasons = []
        
        # Auto-reject conditions
        if validation_issues:
            return ClaimDecision(
                status=ClaimStatus.REJECTED,
                confidence=base_confidence,
                amount=data.total_amount or (max(data.amounts) if data.amounts else 0),
                reasons=validation_issues,
                processing_notes="Claim rejected due to validation issues"
            )
        
        if suspicious_findings:
            return ClaimDecision(
                status=ClaimStatus.REVIEW,
                confidence=base_confidence * 0.8,  # Reduce confidence for suspicious claims
                amount=data.total_amount or (max(data.amounts) if data.amounts else 0),
                reasons=suspicious_findings,
                processing_notes="Claim requires manual review due to suspicious patterns"
            )
        
        # Decision based purely on confidence - no amount limits
        claim_amount = data.total_amount or (max(data.amounts) if data.amounts else 0)
        
        # Auto-approve based on high confidence only
        if base_confidence >= 0.8:
            reasons.append("Claim auto-approved based on high confidence")
            return ClaimDecision(
                status=ClaimStatus.APPROVED,
                confidence=base_confidence,
                amount=claim_amount,
                reasons=reasons,
                processing_notes="Auto-approved based on confidence score"
            )
        else:
            reasons.append("Manual review required due to lower confidence")
            return ClaimDecision(
                status=ClaimStatus.REVIEW,
                confidence=base_confidence,
                amount=claim_amount,
                reasons=reasons,
                processing_notes="Manual review required due to confidence below threshold"
            )
    
    def process_claim(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """Main method to process a claim from OCR results"""
        try:
            # Extract structured data
            extracted_data = self.extract_structured_data(ocr_result)
            
            # Validate data
            is_valid, validation_issues = self.validate_claim_data(extracted_data)
            
            # Detect suspicious patterns
            suspicious_findings = self.detect_suspicious_patterns(ocr_result)
            
            # Make decision
            decision = self.make_decision(
                extracted_data, 
                validation_issues, 
                suspicious_findings,
                ocr_result.get('confidence', 0.0)
            )
            
            # Return structured result
            return {
                'status': decision.status.value,
                'confidence': decision.confidence,
                'amount': decision.amount,
                'reasons': decision.reasons,
                'processing_notes': decision.processing_notes,
                'extracted_data': {
                    'patient_name': extracted_data.patient_name,
                    'patient_id': extracted_data.patient_id,
                    'policy_number': extracted_data.policy_number,
                    'claim_number': extracted_data.claim_number,
                    'provider_name': extracted_data.provider_name,
                    'diagnosis_codes': extracted_data.diagnosis_codes,
                    'treatment_dates': extracted_data.treatment_dates,
                    'amounts': extracted_data.amounts,
                    'total_amount': extracted_data.total_amount,
                    'currency': extracted_data.currency
                },
                'validation_issues': validation_issues,
                'suspicious_findings': suspicious_findings,
                'ocr_confidence': ocr_result.get('confidence', 0.0)
            }
            
        except Exception as e:
            return {
                'status': ClaimStatus.REVIEW.value,
                'confidence': 0.0,
                'amount': 0.0,
                'reasons': [f"Processing error: {str(e)}"],
                'processing_notes': "Claim requires manual review due to processing error",
                'error': str(e)
            }