import os
import time
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from PIL import Image, ImageStat, ImageFilter
import io

class QualityIssue(Enum):
    """Types of document quality issues"""
    BLUR = "blur"
    LOW_RESOLUTION = "low_resolution"
    POOR_CONTRAST = "poor_contrast"
    SKEWED = "skewed"
    PARTIAL_VISIBILITY = "partial_visibility"
    OVEREXPOSED = "overexposed"
    UNDEREXPOSED = "underexposed"
    NOISE = "noise"
    COMPRESSION_ARTIFACTS = "compression_artifacts"
    INCOMPLETE_SCAN = "incomplete_scan"

@dataclass
class QualityScore:
    """Document quality assessment score"""
    overall_score: float  # 0.0 to 1.0
    readability_score: float  # 0.0 to 1.0
    completeness_score: float  # 0.0 to 1.0
    clarity_score: float  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'overall_score': self.overall_score,
            'readability_score': self.readability_score,
            'completeness_score': self.completeness_score,
            'clarity_score': self.clarity_score
        }

@dataclass
class QualityAssessmentResult:
    """Result of document quality assessment"""
    quality_score: QualityScore
    issues_detected: List[QualityIssue]
    recommendations: List[str]
    is_acceptable: bool
    processing_time_ms: int
    technical_metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'quality_score': self.quality_score.to_dict(),
            'issues_detected': [issue.value for issue in self.issues_detected],
            'recommendations': self.recommendations,
            'is_acceptable': self.is_acceptable,
            'processing_time_ms': self.processing_time_ms,
            'technical_metrics': self.technical_metrics
        }

class DocumentQualityAssessor:
    """AI-powered document quality assessment for insurance claims"""
    
    def __init__(self):
        """Initialize quality assessor"""
        self.min_acceptable_score = 0.6
        self.min_resolution = (800, 600)  # Minimum width, height
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.min_contrast_threshold = 20
        self.blur_threshold = 100  # Laplacian variance threshold
        
        # Quality thresholds
        self.thresholds = {
            'excellent': 0.9,
            'good': 0.7,
            'acceptable': 0.6,
            'poor': 0.4,
            'unacceptable': 0.0
        }
    
    def assess_document_quality(self, image_path: str, ocr_confidence: Optional[float] = None) -> QualityAssessmentResult:
        """Assess document quality comprehensively"""
        start_time = time.time()
        
        try:
            # Load and validate image
            if not os.path.exists(image_path):
                return self._create_error_result("Image file not found", start_time)
            
            # Basic file checks
            file_size = os.path.getsize(image_path)
            if file_size > self.max_file_size:
                return self._create_error_result("File too large", start_time)
            
            # Load image with PIL and OpenCV
            pil_image = Image.open(image_path)
            cv_image = cv2.imread(image_path)
            
            if pil_image is None or cv_image is None:
                return self._create_error_result("Cannot load image", start_time)
            
            # Perform various quality assessments
            technical_metrics = self._collect_technical_metrics(pil_image, cv_image, file_size)
            
            # Individual quality assessments
            clarity_assessment = self._assess_clarity(cv_image)
            contrast_assessment = self._assess_contrast(pil_image, cv_image)
            resolution_assessment = self._assess_resolution(pil_image)
            completeness_assessment = self._assess_completeness(cv_image)
            exposure_assessment = self._assess_exposure(pil_image)
            noise_assessment = self._assess_noise(cv_image)
            
            # Combine assessments
            issues_detected = []
            recommendations = []
            
            # Clarity issues
            if clarity_assessment['blur_detected']:
                issues_detected.append(QualityIssue.BLUR)
                recommendations.append("Document appears blurry - ensure camera focus and steady hand")
            
            if clarity_assessment['compression_artifacts']:
                issues_detected.append(QualityIssue.COMPRESSION_ARTIFACTS)
                recommendations.append("Image has compression artifacts - use higher quality settings")
            
            # Contrast issues
            if contrast_assessment['poor_contrast']:
                issues_detected.append(QualityIssue.POOR_CONTRAST)
                recommendations.append("Poor contrast detected - improve lighting or scanner settings")
            
            # Resolution issues
            if resolution_assessment['low_resolution']:
                issues_detected.append(QualityIssue.LOW_RESOLUTION)
                recommendations.append(f"Low resolution - minimum {self.min_resolution[0]}x{self.min_resolution[1]} recommended")
            
            # Completeness issues
            if completeness_assessment['incomplete']:
                issues_detected.append(QualityIssue.INCOMPLETE_SCAN)
                recommendations.append("Document appears incomplete - ensure entire document is captured")
            
            if completeness_assessment['skewed']:
                issues_detected.append(QualityIssue.SKEWED)
                recommendations.append("Document is skewed - align document properly before scanning")
            
            # Exposure issues
            if exposure_assessment['overexposed']:
                issues_detected.append(QualityIssue.OVEREXPOSED)
                recommendations.append("Image overexposed - reduce lighting or camera exposure")
            elif exposure_assessment['underexposed']:
                issues_detected.append(QualityIssue.UNDEREXPOSED)
                recommendations.append("Image underexposed - increase lighting or camera exposure")
            
            # Noise issues
            if noise_assessment['noisy']:
                issues_detected.append(QualityIssue.NOISE)
                recommendations.append("Image noise detected - improve lighting conditions")
            
            # Calculate quality scores
            clarity_score = clarity_assessment['score']
            readability_score = self._calculate_readability_score(
                clarity_score, contrast_assessment['score'], ocr_confidence
            )
            completeness_score = completeness_assessment['score']
            
            # Overall score (weighted average)
            overall_score = (
                clarity_score * 0.4 +
                readability_score * 0.3 +
                completeness_score * 0.2 +
                contrast_assessment['score'] * 0.1
            )
            
            quality_score = QualityScore(
                overall_score=overall_score,
                readability_score=readability_score,
                completeness_score=completeness_score,
                clarity_score=clarity_score
            )
            
            # Determine if acceptable
            is_acceptable = overall_score >= self.min_acceptable_score
            
            if not is_acceptable:
                recommendations.append("Document quality below acceptable threshold - consider retaking photo/scan")
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return QualityAssessmentResult(
                quality_score=quality_score,
                issues_detected=issues_detected,
                recommendations=recommendations,
                is_acceptable=is_acceptable,
                processing_time_ms=processing_time,
                technical_metrics=technical_metrics
            )
            
        except Exception as e:
            return self._create_error_result(f"Quality assessment error: {str(e)}", start_time)
    
    def _collect_technical_metrics(self, pil_image: Image.Image, cv_image: np.ndarray, file_size: int) -> Dict[str, Any]:
        """Collect technical metrics about the image"""
        try:
            return {
                'width': pil_image.width,
                'height': pil_image.height,
                'file_size_bytes': file_size,
                'format': pil_image.format,
                'mode': pil_image.mode,
                'aspect_ratio': pil_image.width / pil_image.height,
                'megapixels': (pil_image.width * pil_image.height) / 1000000,
                'color_channels': len(cv_image.shape) if len(cv_image.shape) > 2 else 1
            }
        except:
            return {}
    
    def _assess_clarity(self, cv_image: np.ndarray) -> Dict[str, Any]:
        """Assess image clarity and detect blur"""
        try:
            # Convert to grayscale
            if len(cv_image.shape) == 3:
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = cv_image
            
            # Calculate Laplacian variance (blur detection)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Detect compression artifacts using gradient analysis
            gradients = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
            gradient_var = np.var(gradients)
            
            # Score based on sharpness
            clarity_score = min(laplacian_var / self.blur_threshold, 1.0)
            
            return {
                'score': clarity_score,
                'laplacian_variance': laplacian_var,
                'blur_detected': laplacian_var < self.blur_threshold,
                'compression_artifacts': gradient_var < 1000,  # Low gradient variance suggests compression
                'sharpness_metric': laplacian_var
            }
        except:
            return {
                'score': 0.5,
                'blur_detected': False,
                'compression_artifacts': False,
                'sharpness_metric': 0
            }
    
    def _assess_contrast(self, pil_image: Image.Image, cv_image: np.ndarray) -> Dict[str, Any]:
        """Assess image contrast"""
        try:
            # PIL-based contrast assessment
            stat = ImageStat.Stat(pil_image.convert('L'))
            contrast_pil = stat.stddev[0]
            
            # OpenCV-based contrast assessment
            if len(cv_image.shape) == 3:
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = cv_image
            
            contrast_cv = gray.std()
            
            # Average the two methods
            contrast_score = min((contrast_pil + contrast_cv) / (2 * 100), 1.0)
            poor_contrast = contrast_score < (self.min_contrast_threshold / 100)
            
            return {
                'score': contrast_score,
                'contrast_std': contrast_cv,
                'poor_contrast': poor_contrast,
                'histogram_spread': np.ptp(gray)  # Peak-to-peak range
            }
        except:
            return {
                'score': 0.5,
                'poor_contrast': False,
                'histogram_spread': 0
            }
    
    def _assess_resolution(self, pil_image: Image.Image) -> Dict[str, Any]:
        """Assess image resolution"""
        try:
            width, height = pil_image.size
            
            # Check if meets minimum resolution
            meets_min_resolution = width >= self.min_resolution[0] and height >= self.min_resolution[1]
            
            # Calculate resolution score
            resolution_score = min(
                (width * height) / (self.min_resolution[0] * self.min_resolution[1]),
                1.0
            )
            
            return {
                'score': resolution_score,
                'width': width,
                'height': height,
                'low_resolution': not meets_min_resolution,
                'pixel_density': width * height
            }
        except:
            return {
                'score': 0.5,
                'low_resolution': True
            }
    
    def _assess_completeness(self, cv_image: np.ndarray) -> Dict[str, Any]:
        """Assess document completeness and orientation"""
        try:
            # Convert to grayscale
            if len(cv_image.shape) == 3:
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = cv_image
            
            # Edge detection to find document boundaries
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze largest contour (presumably the document)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Calculate contour area ratio
                contour_area = cv2.contourArea(largest_contour)
                image_area = gray.shape[0] * gray.shape[1]
                area_ratio = contour_area / image_area
                
                # Detect skew using minimum area rectangle
                rect = cv2.minAreaRect(largest_contour)
                angle = rect[2]
                
                # Normalize angle
                if angle < -45:
                    angle += 90
                elif angle > 45:
                    angle -= 90
                
                skewed = abs(angle) > 5  # More than 5 degrees is considered skewed
                
                # Completeness based on area ratio and edge proximity
                incomplete = area_ratio < 0.3  # Document takes up less than 30% of image
                
                completeness_score = min(area_ratio * 2, 1.0)  # Scale up area ratio
                
                return {
                    'score': completeness_score,
                    'area_ratio': area_ratio,
                    'skew_angle': angle,
                    'incomplete': incomplete,
                    'skewed': skewed,
                    'contour_count': len(contours)
                }
            else:
                return {
                    'score': 0.3,
                    'incomplete': True,
                    'skewed': False,
                    'contour_count': 0
                }
                
        except:
            return {
                'score': 0.5,
                'incomplete': False,
                'skewed': False
            }
    
    def _assess_exposure(self, pil_image: Image.Image) -> Dict[str, Any]:
        """Assess image exposure"""
        try:
            # Convert to grayscale for analysis
            gray_image = pil_image.convert('L')
            
            # Calculate histogram
            histogram = gray_image.histogram()
            
            # Calculate mean brightness
            mean_brightness = sum(i * histogram[i] for i in range(256)) / sum(histogram)
            
            # Detect over/under exposure
            overexposed = mean_brightness > 220  # Very bright
            underexposed = mean_brightness < 50   # Very dark
            
            # Calculate exposure score (best around 128)
            exposure_score = 1.0 - abs(mean_brightness - 128) / 128
            
            return {
                'score': exposure_score,
                'mean_brightness': mean_brightness,
                'overexposed': overexposed,
                'underexposed': underexposed,
                'brightness_histogram_peak': histogram.index(max(histogram))
            }
        except:
            return {
                'score': 0.5,
                'overexposed': False,
                'underexposed': False
            }
    
    def _assess_noise(self, cv_image: np.ndarray) -> Dict[str, Any]:
        """Assess image noise levels"""
        try:
            # Convert to grayscale
            if len(cv_image.shape) == 3:
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = cv_image
            
            # Apply Gaussian blur and calculate difference
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise_estimate = cv2.absdiff(gray, blurred)
            noise_level = np.mean(noise_estimate)
            
            # Score based on noise level
            noise_score = max(0, 1.0 - (noise_level / 50))  # Normalize against threshold
            noisy = noise_level > 20
            
            return {
                'score': noise_score,
                'noise_level': noise_level,
                'noisy': noisy
            }
        except:
            return {
                'score': 0.5,
                'noisy': False
            }
    
    def _calculate_readability_score(self, clarity_score: float, contrast_score: float, 
                                   ocr_confidence: Optional[float] = None) -> float:
        """Calculate overall readability score"""
        # Base score from clarity and contrast
        readability = (clarity_score + contrast_score) / 2
        
        # Incorporate OCR confidence if available
        if ocr_confidence is not None:
            readability = (readability * 0.7) + (ocr_confidence * 0.3)
        
        return min(readability, 1.0)
    
    def _create_error_result(self, error_message: str, start_time: float) -> QualityAssessmentResult:
        """Create error result for failed assessments"""
        processing_time = int((time.time() - start_time) * 1000)
        
        return QualityAssessmentResult(
            quality_score=QualityScore(0.0, 0.0, 0.0, 0.0),
            issues_detected=[],
            recommendations=[f"Quality assessment failed: {error_message}"],
            is_acceptable=False,
            processing_time_ms=processing_time,
            technical_metrics={'error': error_message}
        )
    
    def get_quality_category(self, score: float) -> str:
        """Get quality category from score"""
        if score >= self.thresholds['excellent']:
            return 'excellent'
        elif score >= self.thresholds['good']:
            return 'good'
        elif score >= self.thresholds['acceptable']:
            return 'acceptable'
        elif score >= self.thresholds['poor']:
            return 'poor'
        else:
            return 'unacceptable'
    
    def suggest_improvements(self, issues: List[QualityIssue]) -> List[str]:
        """Suggest specific improvements based on detected issues"""
        suggestions = []
        
        if QualityIssue.BLUR in issues:
            suggestions.extend([
                "Ensure camera is properly focused",
                "Hold device steady or use a tripod",
                "Clean camera lens",
                "Use proper lighting to enable faster shutter speed"
            ])
        
        if QualityIssue.LOW_RESOLUTION in issues:
            suggestions.extend([
                "Use higher resolution camera settings",
                "Move closer to document",
                "Use scanner instead of camera if available"
            ])
        
        if QualityIssue.POOR_CONTRAST in issues:
            suggestions.extend([
                "Improve lighting conditions",
                "Avoid shadows on document",
                "Increase document-background contrast",
                "Adjust camera exposure settings"
            ])
        
        if QualityIssue.SKEWED in issues:
            suggestions.extend([
                "Align document parallel to camera",
                "Use document scanning app with auto-correction",
                "Place document on flat surface"
            ])
        
        if QualityIssue.INCOMPLETE_SCAN in issues:
            suggestions.extend([
                "Ensure entire document is visible",
                "Move camera further back",
                "Use document boundaries as guide"
            ])
        
        if QualityIssue.OVEREXPOSED in issues or QualityIssue.UNDEREXPOSED in issues:
            suggestions.extend([
                "Adjust lighting conditions",
                "Use manual camera exposure control",
                "Avoid direct flash or bright lights"
            ])
        
        return suggestions