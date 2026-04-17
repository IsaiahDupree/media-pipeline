"""
Queue processor for managing media processing tasks.
Implements task queueing, worker management, and status tracking.
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime, timezone
import uuid


class TaskStatus(str, Enum):
    """Task status enum."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class QueueItem:
    """Represents a task in the processing queue."""
    id: str
    media_id: str
    operation: str
    status: str = TaskStatus.PENDING.value
    priority: int = 0
    retry_count: int = 0
    max_retries: int = 3
    progress: float = 0
    error: Optional[str] = None
    params: Dict[str, Any] = None
    result: Dict[str, Any] = None
    created_at: str = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if self.params is None:
            self.params = {}
        if self.result is None:
            self.result = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class QueueProcessor:
    """Process items from a task queue."""

    def __init__(self, max_queue_size: int = 1000):
        self.queue: List[QueueItem] = []
        self.max_queue_size = max_queue_size
        self.processing = set()  # Track items being processed
        self.completed = {}  # Track completed items
        self.failed = {}  # Track failed items

    def enqueue(self, media_id: str, operation: str, params: Dict[str, Any] = None, priority: int = 0) -> str:
        """Enqueue a task."""
        if len(self.queue) >= self.max_queue_size:
            raise RuntimeError("Queue is full")

        item = QueueItem(
            id=None,  # Will be generated
            media_id=media_id,
            operation=operation,
            priority=priority,
            params=params or {}
        )

        self.queue.append(item)
        # Sort by priority (descending) and creation time
        self.queue.sort(key=lambda x: (-x.priority, x.created_at))

        return item.id

    def dequeue(self) -> Optional[QueueItem]:
        """Get next item from queue."""
        if self.queue:
            item = self.queue.pop(0)
            item.status = TaskStatus.PROCESSING.value
            item.started_at = datetime.now(timezone.utc).isoformat()
            self.processing.add(item.id)
            return item
        return None

    def get_status(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a queue item."""
        # Check in queue
        for item in self.queue:
            if item.id == item_id:
                return {"status": item.status, "progress": item.progress, "position": self.queue.index(item)}

        # Check if processing
        if item_id in self.processing:
            return {"status": TaskStatus.PROCESSING.value, "progress": 0}

        # Check if completed
        if item_id in self.completed:
            return {"status": TaskStatus.COMPLETED.value, "progress": 100}

        # Check if failed
        if item_id in self.failed:
            error_data = self.failed[item_id]
            return {"status": TaskStatus.FAILED.value, "error": error_data.get("error")}

        return None

    def update_progress(self, item_id: str, progress: float) -> bool:
        """Update progress of a processing item."""
        for item in self.queue:
            if item.id == item_id:
                item.progress = min(progress, 100)
                return True
        return False

    def complete_item(self, item_id: str, result: Dict[str, Any] = None) -> bool:
        """Mark item as completed."""
        if item_id in self.processing:
            self.processing.remove(item_id)
            self.completed[item_id] = {
                "result": result or {},
                "completed_at": datetime.now(timezone.utc).isoformat()
            }
            return True
        return False

    def fail_item(self, item_id: str, error: str, retry: bool = True) -> bool:
        """Mark item as failed."""
        if item_id in self.processing:
            self.processing.remove(item_id)

            # Try to find item in queue for retry tracking
            for item in self.queue:
                if item.id == item_id:
                    item.retry_count += 1
                    if retry and item.retry_count < item.max_retries:
                        item.status = TaskStatus.RETRYING.value
                        # Re-enqueue
                        self.queue.append(item)
                        self.queue.sort(key=lambda x: (-x.priority, x.created_at))
                    else:
                        self.failed[item_id] = {
                            "error": error,
                            "retry_count": item.retry_count,
                            "failed_at": datetime.now(timezone.utc).isoformat()
                        }
                    return True

            # Item not in queue, just record failure
            self.failed[item_id] = {
                "error": error,
                "failed_at": datetime.now(timezone.utc).isoformat()
            }
            return True

        return False

    def get_queue_status(self) -> Dict[str, Any]:
        """Get overall queue status."""
        return {
            "queue_size": len(self.queue),
            "processing_count": len(self.processing),
            "completed_count": len(self.completed),
            "failed_count": len(self.failed),
            "pending_items": [item.to_dict() for item in self.queue[:10]]  # Show first 10
        }

    def clear_completed(self, older_than_hours: int = 24) -> int:
        """Clear old completed items."""
        # For now, just clear all
        count = len(self.completed)
        self.completed.clear()
        return count

    def retry_failed(self, item_id: str) -> bool:
        """Retry a failed item."""
        if item_id in self.failed:
            del self.failed[item_id]
            return True
        return False


class DeadLetterQueue:
    """Handle failed jobs that can't be processed."""

    def __init__(self):
        self.items: Dict[str, Dict[str, Any]] = {}

    def add(self, item_id: str, item: QueueItem, error: str) -> None:
        """Add item to dead letter queue."""
        self.items[item_id] = {
            "item": item.to_dict(),
            "error": error,
            "added_at": datetime.now(timezone.utc).isoformat(),
            "retry_count": 0
        }

    def get(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get item from DLQ."""
        return self.items.get(item_id)

    def retry(self, item_id: str) -> Optional[QueueItem]:
        """Retry an item from DLQ."""
        if item_id in self.items:
            data = self.items[item_id]
            data["retry_count"] += 1
            item_data = data["item"]
            return QueueItem(**item_data)
        return None

    def remove(self, item_id: str) -> bool:
        """Remove item from DLQ."""
        if item_id in self.items:
            del self.items[item_id]
            return True
        return False

    def list_items(self) -> List[Dict[str, Any]]:
        """List all items in DLQ."""
        return list(self.items.values())
