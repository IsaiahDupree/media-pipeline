"""
Idempotency & Retry Management
================================
Idempotency keys, retry policies, and dead-letter queue for Media Factory.
"""

import hashlib
import json
import logging
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class RetryPolicy(str, Enum):
    """Retry policy types."""
    NO_RETRY = "no_retry"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"


class IdempotencyManager:
    """
    Manages idempotency keys and retry logic for pipeline stages.
    
    Idempotency Key Format: {job_id}:{stage_name}:{input_hash}
    """
    
    def __init__(self, storage_backend: Optional[Callable] = None):
        """
        Initialize idempotency manager.
        
        Args:
            storage_backend: Optional storage backend (default: in-memory dict)
        """
        self._storage = storage_backend or {}
        self._results: Dict[str, Dict[str, Any]] = {}
    
    def generate_idempotency_key(
        self,
        job_id: str,
        stage_name: str,
        input_data: Dict[str, Any]
    ) -> str:
        """
        Generate idempotency key from job_id, stage_name, and input hash.
        
        Args:
            job_id: Job identifier
            stage_name: Stage name (e.g., 'tts', 'remotion')
            input_data: Input data for the stage
        
        Returns:
            Idempotency key string
        """
        # Create deterministic hash of input data
        input_json = json.dumps(input_data, sort_keys=True)
        input_hash = hashlib.sha256(input_json.encode()).hexdigest()[:16]
        
        key = f"{job_id}:{stage_name}:{input_hash}"
        return key
    
    def check_idempotency(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        """
        Check if operation with this key was already executed.
        
        Args:
            idempotency_key: Idempotency key
        
        Returns:
            Previous result if exists, None otherwise
        """
        return self._results.get(idempotency_key)
    
    def store_result(
        self,
        idempotency_key: str,
        result: Dict[str, Any],
        ttl_seconds: int = 3600
    ) -> None:
        """
        Store result for idempotency check.
        
        Args:
            idempotency_key: Idempotency key
            result: Result to store
            ttl_seconds: Time-to-live in seconds
        """
        self._results[idempotency_key] = {
            "result": result,
            "created_at": datetime.now(timezone.utc),
            "expires_at": datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        }


class RetryManager:
    """
    Manages retry logic with exponential backoff.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        policy: RetryPolicy = RetryPolicy.EXPONENTIAL_BACKOFF,
        base_delay: float = 1.0,
        max_delay: float = 60.0
    ):
        """
        Initialize retry manager.
        
        Args:
            max_retries: Maximum number of retries
            policy: Retry policy
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
        """
        self.max_retries = max_retries
        self.policy = policy
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for retry attempt.
        
        Args:
            attempt: Retry attempt number (0-indexed)
        
        Returns:
            Delay in seconds
        """
        if self.policy == RetryPolicy.NO_RETRY:
            return 0.0
        elif self.policy == RetryPolicy.EXPONENTIAL_BACKOFF:
            delay = self.base_delay * (2 ** attempt)
            return min(delay, self.max_delay)
        elif self.policy == RetryPolicy.LINEAR_BACKOFF:
            delay = self.base_delay * (attempt + 1)
            return min(delay, self.max_delay)
        elif self.policy == RetryPolicy.FIXED_DELAY:
            return self.base_delay
        else:
            return self.base_delay
    
    async def execute_with_retry(
        self,
        operation: Callable,
        operation_name: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute operation with retry logic.
        
        Args:
            operation: Async operation to execute
            operation_name: Name of operation (for logging)
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Operation result
        
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = await operation(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"✅ {operation_name} succeeded on attempt {attempt + 1}")
                return result
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        f"⚠️  {operation_name} failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    import asyncio
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"❌ {operation_name} failed after {self.max_retries + 1} attempts")
        
        raise last_exception


class DeadLetterQueue:
    """
    Dead Letter Queue for failed operations.
    
    Stores failed operations with reasons and payload snapshots for debugging.
    """
    
    def __init__(self, storage_backend: Optional[Callable] = None):
        """
        Initialize DLQ.
        
        Args:
            storage_backend: Optional storage backend (default: in-memory list)
        """
        self._storage = storage_backend or []
        self._max_size = 1000  # Prevent unbounded growth
    
    def add_failure(
        self,
        job_id: str,
        stage_name: str,
        error: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
        retry_count: int = 0
    ) -> None:
        """
        Add failed operation to DLQ.
        
        Args:
            job_id: Job identifier
            stage_name: Stage name
            error: Error message
            payload: Operation payload snapshot
            correlation_id: Correlation ID
            retry_count: Number of retries attempted
        """
        failure = {
            "job_id": job_id,
            "stage_name": stage_name,
            "error": error,
            "payload": payload,
            "correlation_id": correlation_id,
            "retry_count": retry_count,
            "failed_at": datetime.now(timezone.utc).isoformat()
        }
        
        self._storage.append(failure)
        
        # Prevent unbounded growth
        if len(self._storage) > self._max_size:
            self._storage.pop(0)
        
        logger.error(
            f"❌ DLQ: {stage_name} failed for job {job_id} "
            f"(retries: {retry_count}): {error}"
        )
    
    def get_failures(
        self,
        job_id: Optional[str] = None,
        stage_name: Optional[str] = None,
        limit: int = 100
    ) -> list:
        """
        Get failures from DLQ.
        
        Args:
            job_id: Filter by job ID
            stage_name: Filter by stage name
            limit: Maximum number of results
        
        Returns:
            List of failures
        """
        failures = self._storage
        
        if job_id:
            failures = [f for f in failures if f["job_id"] == job_id]
        
        if stage_name:
            failures = [f for f in failures if f["stage_name"] == stage_name]
        
        return failures[-limit:]
    
    def clear_failures(self, job_id: Optional[str] = None) -> int:
        """
        Clear failures from DLQ.
        
        Args:
            job_id: Clear failures for specific job (None = all)
        
        Returns:
            Number of failures cleared
        """
        if job_id:
            initial_count = len(self._storage)
            self._storage = [f for f in self._storage if f["job_id"] != job_id]
            return initial_count - len(self._storage)
        else:
            count = len(self._storage)
            self._storage.clear()
            return count

