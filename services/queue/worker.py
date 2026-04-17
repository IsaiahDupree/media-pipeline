"""
Processing worker for queue tasks.
Pulls tasks from queue and executes them.
"""
from typing import Dict, Any, Callable, Optional
import time
from datetime import datetime, timezone


class ProcessingWorker:
    """Worker that processes queue items."""

    def __init__(self, queue_processor, worker_id: str = "worker-1"):
        self.queue = queue_processor
        self.worker_id = worker_id
        self.running = False
        self.current_item = None
        self.stats = {
            "processed": 0,
            "failed": 0,
            "completed": 0,
            "total_processing_time": 0
        }

    def process_item(self, item_id: str, handler: Callable) -> bool:
        """Process a single item."""
        self.current_item = item_id
        start_time = time.time()

        try:
            # Execute handler
            result = handler(item_id)

            # Mark as completed
            self.queue.complete_item(item_id, result)
            self.stats["completed"] += 1

            return True
        except Exception as e:
            # Mark as failed with retry
            self.queue.fail_item(item_id, str(e), retry=True)
            self.stats["failed"] += 1
            return False
        finally:
            processing_time = time.time() - start_time
            self.stats["total_processing_time"] += processing_time
            self.stats["processed"] += 1
            self.current_item = None

    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        stats = self.stats.copy()
        if self.stats["processed"] > 0:
            stats["avg_processing_time"] = self.stats["total_processing_time"] / self.stats["processed"]
        return stats

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = {
            "processed": 0,
            "failed": 0,
            "completed": 0,
            "total_processing_time": 0
        }


class BatchProcessor:
    """Process multiple items in batches."""

    def __init__(self, queue_processor, batch_size: int = 10):
        self.queue = queue_processor
        self.batch_size = batch_size

    def process_batch(self, handler: Callable) -> Dict[str, Any]:
        """Process a batch of items."""
        results = {
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
            "items": []
        }

        for _ in range(self.batch_size):
            item = self.queue.dequeue()
            if not item:
                break

            try:
                result = handler(item)
                self.queue.complete_item(item.id, result)
                results["succeeded"] += 1
            except Exception as e:
                self.queue.fail_item(item.id, str(e), retry=True)
                results["failed"] += 1

            results["processed"] += 1
            results["items"].append({
                "id": item.id,
                "operation": item.operation,
                "status": "completed" if results["processed"] == results["succeeded"] else "failed"
            })

        return results
