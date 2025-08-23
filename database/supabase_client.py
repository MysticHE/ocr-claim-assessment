import os
from datetime import datetime
from typing import Optional, Dict, List, Any
from supabase import create_client, Client
import json

class SupabaseClient:
    """Supabase database client for OCR Claim Assessment system"""
    
    def __init__(self):
        """Initialize Supabase client with environment variables"""
        self.url = os.environ.get('SUPABASE_URL')
        self.service_key = os.environ.get('SUPABASE_SERVICE_KEY')
        
        if not self.url or not self.service_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        
        try:
            self.supabase: Client = create_client(self.url, self.service_key)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Supabase: {str(e)}")
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            # Simple query to test connection
            response = self.supabase.table('claims').select('count', count='exact').execute()
            return True
        except Exception as e:
            print(f"Database connection test failed: {e}")
            return False
    
    # Claims table operations
    def insert_claim(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Insert a new claim record"""
        try:
            # Ensure required fields
            if 'id' not in claim_data:
                raise ValueError("Claim ID is required")
            
            # Convert datetime objects to ISO format if needed
            if 'created_at' not in claim_data:
                claim_data['created_at'] = datetime.utcnow().isoformat()
            
            # Ensure metadata is properly formatted JSON
            if 'metadata' in claim_data and isinstance(claim_data['metadata'], dict):
                claim_data['metadata'] = claim_data['metadata']
            
            response = self.supabase.table('claims').insert(claim_data).execute()
            
            if response.data:
                return {'success': True, 'data': response.data[0]}
            else:
                return {'success': False, 'error': 'No data returned'}
                
        except Exception as e:
            print(f"Error inserting claim: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_claim(self, claim_id: str) -> Optional[Any]:
        """Retrieve a claim by ID"""
        try:
            response = self.supabase.table('claims').select('*').eq('id', claim_id).execute()
            return response
        except Exception as e:
            print(f"Error retrieving claim {claim_id}: {e}")
            return None
    
    def update_claim_status(self, claim_id: str, status: str, confidence: float, 
                           additional_data: Optional[Dict[str, Any]] = None) -> bool:
        """Update claim status and additional information"""
        try:
            update_data = {
                'claim_status': status,
                'confidence_score': confidence,
                'updated_at': datetime.utcnow().isoformat()
            }
            
            # Add additional data if provided
            if additional_data:
                if 'ocr_text' in additional_data:
                    update_data['ocr_text'] = additional_data['ocr_text']
                if 'language_detected' in additional_data:
                    update_data['language_detected'] = additional_data['language_detected']
                if 'claim_amount' in additional_data:
                    update_data['claim_amount'] = additional_data['claim_amount']
                
                # Merge metadata
                if 'metadata' in additional_data or any(k in additional_data for k in ['processing_time_ms', 'decision_reasons']):
                    # Get current metadata
                    current_claim = self.get_claim(claim_id)
                    current_metadata = {}
                    if current_claim and current_claim.data:
                        current_metadata = current_claim.data[0].get('metadata', {}) or {}
                    
                    # Merge with new metadata
                    new_metadata = current_metadata.copy()
                    if 'metadata' in additional_data:
                        new_metadata.update(additional_data['metadata'])
                    if 'processing_time_ms' in additional_data:
                        new_metadata['processing_time_ms'] = additional_data['processing_time_ms']
                    if 'decision_reasons' in additional_data:
                        new_metadata['decision_reasons'] = additional_data['decision_reasons']
                    
                    update_data['metadata'] = new_metadata
            
            response = self.supabase.table('claims').update(update_data).eq('id', claim_id).execute()
            return len(response.data) > 0
            
        except Exception as e:
            print(f"Error updating claim status for {claim_id}: {e}")
            return False
    
    def get_claims_by_status(self, status: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get claims by status"""
        try:
            response = self.supabase.table('claims')\
                .select('*')\
                .eq('claim_status', status)\
                .order('created_at', desc=True)\
                .limit(limit)\
                .execute()
            return response.data or []
        except Exception as e:
            print(f"Error retrieving claims by status {status}: {e}")
            return []
    
    def get_recent_claims(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent claims"""
        try:
            response = self.supabase.table('claims')\
                .select('*')\
                .order('created_at', desc=True)\
                .limit(limit)\
                .execute()
            return response.data or []
        except Exception as e:
            print(f"Error retrieving recent claims: {e}")
            return []
    
    # OCR results table operations
    def insert_ocr_result(self, ocr_data: Dict[str, Any]) -> bool:
        """Insert OCR processing result"""
        try:
            # Ensure required fields
            if 'claim_id' not in ocr_data:
                raise ValueError("Claim ID is required for OCR result")
            
            if 'created_at' not in ocr_data:
                ocr_data['created_at'] = datetime.utcnow().isoformat()
            
            # Ensure bounding_boxes is properly formatted JSON
            if 'bounding_boxes' in ocr_data and isinstance(ocr_data['bounding_boxes'], (list, dict)):
                ocr_data['bounding_boxes'] = ocr_data['bounding_boxes']
            
            response = self.supabase.table('ocr_results').insert(ocr_data).execute()
            return len(response.data) > 0
            
        except Exception as e:
            print(f"Error inserting OCR result: {e}")
            return False
    
    def get_ocr_result(self, claim_id: str) -> Optional[Any]:
        """Get OCR results for a claim"""
        try:
            response = self.supabase.table('ocr_results')\
                .select('*')\
                .eq('claim_id', claim_id)\
                .order('created_at', desc=True)\
                .execute()
            return response
        except Exception as e:
            print(f"Error retrieving OCR result for claim {claim_id}: {e}")
            return None
    
    # Processing logs table operations
    def insert_log(self, claim_id: Optional[str], log_level: str, message: str, 
                   error_details: Optional[Dict[str, Any]] = None) -> bool:
        """Insert a processing log entry"""
        try:
            log_data = {
                'claim_id': claim_id,
                'log_level': log_level,
                'message': message,
                'created_at': datetime.utcnow().isoformat()
            }
            
            if error_details:
                log_data['error_details'] = error_details
            
            response = self.supabase.table('processing_logs').insert(log_data).execute()
            return len(response.data) > 0
            
        except Exception as e:
            print(f"Error inserting log: {e}")
            return False
    
    def get_logs_for_claim(self, claim_id: str) -> List[Dict[str, Any]]:
        """Get all logs for a specific claim"""
        try:
            response = self.supabase.table('processing_logs')\
                .select('*')\
                .eq('claim_id', claim_id)\
                .order('created_at', desc=True)\
                .execute()
            return response.data or []
        except Exception as e:
            print(f"Error retrieving logs for claim {claim_id}: {e}")
            return []
    
    def get_recent_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent processing logs"""
        try:
            response = self.supabase.table('processing_logs')\
                .select('*')\
                .order('created_at', desc=True)\
                .limit(limit)\
                .execute()
            return response.data or []
        except Exception as e:
            print(f"Error retrieving recent logs: {e}")
            return []
    
    # Analytics and reporting methods
    def get_claim_stats(self) -> Dict[str, Any]:
        """Get claim processing statistics"""
        try:
            stats = {}
            
            # Total claims
            total_response = self.supabase.table('claims').select('id', count='exact').execute()
            stats['total_claims'] = total_response.count if total_response.count else 0
            
            # Claims by status
            for status in ['approved', 'rejected', 'review', 'processing']:
                status_response = self.supabase.table('claims')\
                    .select('id', count='exact')\
                    .eq('claim_status', status)\
                    .execute()
                stats[f'{status}_claims'] = status_response.count if status_response.count else 0
            
            # Average processing time (from metadata)
            processing_times = []
            recent_claims = self.get_recent_claims(100)
            for claim in recent_claims:
                if claim.get('metadata') and isinstance(claim['metadata'], dict):
                    proc_time = claim['metadata'].get('processing_time_ms')
                    if proc_time and isinstance(proc_time, (int, float)):
                        processing_times.append(proc_time)
            
            if processing_times:
                stats['avg_processing_time_ms'] = sum(processing_times) / len(processing_times)
                stats['max_processing_time_ms'] = max(processing_times)
                stats['min_processing_time_ms'] = min(processing_times)
            else:
                stats['avg_processing_time_ms'] = 0
                stats['max_processing_time_ms'] = 0
                stats['min_processing_time_ms'] = 0
            
            # Average confidence score
            confidence_scores = []
            for claim in recent_claims:
                if claim.get('confidence_score') and isinstance(claim['confidence_score'], (int, float)):
                    confidence_scores.append(float(claim['confidence_score']))
            
            if confidence_scores:
                stats['avg_confidence_score'] = sum(confidence_scores) / len(confidence_scores)
            else:
                stats['avg_confidence_score'] = 0
            
            return stats
            
        except Exception as e:
            print(f"Error retrieving claim stats: {e}")
            return {
                'total_claims': 0,
                'approved_claims': 0,
                'rejected_claims': 0,
                'review_claims': 0,
                'processing_claims': 0,
                'avg_processing_time_ms': 0,
                'avg_confidence_score': 0
            }
    
    def cleanup_old_claims(self, days_old: int = 90) -> int:
        """Clean up claims older than specified days (for maintenance)"""
        try:
            cutoff_date = datetime.utcnow().replace(microsecond=0)
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days_old)
            
            # Note: This is a destructive operation, use with caution
            response = self.supabase.table('claims')\
                .delete()\
                .lt('created_at', cutoff_date.isoformat())\
                .execute()
            
            return len(response.data) if response.data else 0
            
        except Exception as e:
            print(f"Error cleaning up old claims: {e}")
            return 0