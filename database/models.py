from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

class ClaimStatus(Enum):
    """Claim processing status enumeration"""
    PROCESSING = "processing"
    APPROVED = "approved"
    REJECTED = "rejected"
    REVIEW = "review"

class LogLevel(Enum):
    """Logging level enumeration"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    DEBUG = "debug"

@dataclass
class ClaimRecord:
    """Data model for claim records"""
    id: str
    file_name: str
    file_size: int
    claim_status: ClaimStatus
    created_at: datetime
    updated_at: Optional[datetime] = None
    language_detected: Optional[str] = None
    ocr_text: Optional[str] = None
    claim_amount: Optional[float] = None
    confidence_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            'id': self.id,
            'file_name': self.file_name,
            'file_size': self.file_size,
            'claim_status': self.claim_status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'language_detected': self.language_detected,
            'ocr_text': self.ocr_text,
            'claim_amount': self.claim_amount,
            'confidence_score': self.confidence_score,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClaimRecord':
        """Create instance from dictionary"""
        return cls(
            id=data['id'],
            file_name=data['file_name'],
            file_size=data['file_size'],
            claim_status=ClaimStatus(data['claim_status']),
            created_at=datetime.fromisoformat(data['created_at'].replace('Z', '+00:00')),
            updated_at=datetime.fromisoformat(data['updated_at'].replace('Z', '+00:00')) if data.get('updated_at') else None,
            language_detected=data.get('language_detected'),
            ocr_text=data.get('ocr_text'),
            claim_amount=data.get('claim_amount'),
            confidence_score=data.get('confidence_score'),
            metadata=data.get('metadata', {})
        )

@dataclass
class OCRResult:
    """Data model for OCR processing results"""
    id: Optional[str] = None
    claim_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    language_code: str = ""
    extracted_text: str = ""
    confidence_score: float = 0.0
    bounding_boxes: List[Dict[str, Any]] = field(default_factory=list)
    processing_engine: str = "paddleocr"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            'id': self.id,
            'claim_id': self.claim_id,
            'created_at': self.created_at.isoformat(),
            'language_code': self.language_code,
            'extracted_text': self.extracted_text,
            'confidence_score': self.confidence_score,
            'bounding_boxes': self.bounding_boxes,
            'processing_engine': self.processing_engine
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OCRResult':
        """Create instance from dictionary"""
        return cls(
            id=data.get('id'),
            claim_id=data['claim_id'],
            created_at=datetime.fromisoformat(data['created_at'].replace('Z', '+00:00')),
            language_code=data['language_code'],
            extracted_text=data['extracted_text'],
            confidence_score=data['confidence_score'],
            bounding_boxes=data.get('bounding_boxes', []),
            processing_engine=data.get('processing_engine', 'paddleocr')
        )

@dataclass
class ProcessingLog:
    """Data model for processing logs"""
    id: Optional[str] = None
    claim_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    log_level: LogLevel = LogLevel.INFO
    message: str = ""
    error_details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            'id': self.id,
            'claim_id': self.claim_id,
            'created_at': self.created_at.isoformat(),
            'log_level': self.log_level.value,
            'message': self.message,
            'error_details': self.error_details
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingLog':
        """Create instance from dictionary"""
        return cls(
            id=data.get('id'),
            claim_id=data.get('claim_id'),
            created_at=datetime.fromisoformat(data['created_at'].replace('Z', '+00:00')),
            log_level=LogLevel(data['log_level']),
            message=data['message'],
            error_details=data.get('error_details')
        )

@dataclass
class ClaimDecision:
    """Data model for claim processing decisions"""
    status: ClaimStatus
    confidence: float
    amount: Optional[float] = None
    reasons: List[str] = field(default_factory=list)
    processing_notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'status': self.status.value,
            'confidence': self.confidence,
            'amount': self.amount,
            'reasons': self.reasons,
            'processing_notes': self.processing_notes
        }

@dataclass
class ClaimStats:
    """Data model for claim statistics"""
    total_claims: int = 0
    approved_claims: int = 0
    rejected_claims: int = 0
    review_claims: int = 0
    processing_claims: int = 0
    avg_processing_time_ms: float = 0
    max_processing_time_ms: float = 0
    min_processing_time_ms: float = 0
    avg_confidence_score: float = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_claims': self.total_claims,
            'approved_claims': self.approved_claims,
            'rejected_claims': self.rejected_claims,
            'review_claims': self.review_claims,
            'processing_claims': self.processing_claims,
            'avg_processing_time_ms': self.avg_processing_time_ms,
            'max_processing_time_ms': self.max_processing_time_ms,
            'min_processing_time_ms': self.min_processing_time_ms,
            'avg_confidence_score': self.avg_confidence_score
        }
    
    @property
    def approval_rate(self) -> float:
        """Calculate approval rate percentage"""
        if self.total_claims == 0:
            return 0
        return (self.approved_claims / self.total_claims) * 100
    
    @property
    def rejection_rate(self) -> float:
        """Calculate rejection rate percentage"""
        if self.total_claims == 0:
            return 0
        return (self.rejected_claims / self.total_claims) * 100
    
    @property
    def review_rate(self) -> float:
        """Calculate review rate percentage"""
        if self.total_claims == 0:
            return 0
        return (self.review_claims / self.total_claims) * 100

# Utility functions for data validation and conversion
def validate_claim_amount(amount: Any) -> Optional[float]:
    """Validate and convert claim amount to float"""
    if amount is None:
        return None
    
    try:
        float_amount = float(amount)
        if float_amount < 0:
            raise ValueError("Claim amount cannot be negative")
        if float_amount > 1000000:  # 1 million limit
            raise ValueError("Claim amount exceeds maximum limit")
        return round(float_amount, 2)
    except (ValueError, TypeError):
        raise ValueError("Invalid claim amount format")

def validate_confidence_score(score: Any) -> float:
    """Validate and convert confidence score to float between 0 and 1"""
    if score is None:
        return 0.0
    
    try:
        float_score = float(score)
        if float_score < 0:
            return 0.0
        elif float_score > 1:
            return 1.0
        return round(float_score, 4)
    except (ValueError, TypeError):
        return 0.0

def validate_file_size(size: Any) -> int:
    """Validate file size"""
    try:
        int_size = int(size)
        if int_size < 0:
            raise ValueError("File size cannot be negative")
        if int_size > 50 * 1024 * 1024:  # 50MB limit
            raise ValueError("File size exceeds maximum limit")
        return int_size
    except (ValueError, TypeError):
        raise ValueError("Invalid file size format")

def sanitize_text(text: str, max_length: int = 10000) -> str:
    """Sanitize and truncate text for storage"""
    if not isinstance(text, str):
        text = str(text)
    
    # Remove or replace problematic characters
    sanitized = text.replace('\x00', '')  # Remove null bytes
    sanitized = sanitized.strip()
    
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "... [truncated]"
    
    return sanitized