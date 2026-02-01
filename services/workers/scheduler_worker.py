"""
Scheduler Worker
================
Event-driven worker for scheduled post management.

Emits on tick:
    - schedule.due (for posts that are due)
    - scheduler.tick (heartbeat)

Subscribes to:
    - schedule.created (new scheduled post)
    - schedule.updated (schedule modified)
    - schedule.cancelled (schedule cancelled)

This worker replaces the polling-based PostScheduler with
an event-driven approach.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from services.event_bus import EventBus, Event, Topics
from services.workers.base import BaseWorker

logger = logging.getLogger(__name__)


class SchedulerWorker(BaseWorker):
    """
    Worker for managing scheduled posts.
    
    Runs a periodic check for due posts and emits schedule.due events
    for each post that needs to be published.
    
    Usage:
        worker = SchedulerWorker(check_interval=60)
        await worker.start()
        await worker.run_scheduler_loop()  # Starts the periodic check
    """
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        worker_id: Optional[str] = None,
        check_interval: int = 60
    ):
        super().__init__(event_bus, worker_id)
        self.check_interval = check_interval
        self._scheduler_task: Optional[asyncio.Task] = None
        self._check_count = 0
    
    def get_subscriptions(self) -> List[str]:
        """Subscribe to schedule-related events."""
        return [
            Topics.SCHEDULE_CREATED,
            Topics.SCHEDULE_UPDATED,
            Topics.SCHEDULE_CANCELLED,
        ]
    
    async def handle_event(self, event: Event) -> None:
        """Handle schedule-related events."""
        if event.topic == Topics.SCHEDULE_CREATED:
            logger.info(f"[{self.worker_id}] New schedule created: {event.payload.get('post_id')}")
        elif event.topic == Topics.SCHEDULE_UPDATED:
            logger.info(f"[{self.worker_id}] Schedule updated: {event.payload.get('post_id')}")
        elif event.topic == Topics.SCHEDULE_CANCELLED:
            logger.info(f"[{self.worker_id}] Schedule cancelled: {event.payload.get('post_id')}")
    
    async def start(self) -> None:
        """Start the worker and the scheduler loop."""
        await super().start()
        
        # Emit scheduler started event
        await self.emit(
            Topics.SCHEDULER_STARTED,
            {
                "worker_id": self.worker_id,
                "check_interval": self.check_interval
            }
        )
    
    async def stop(self) -> None:
        """Stop the worker and the scheduler loop."""
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None
        
        # Emit scheduler stopped event
        await self.emit(
            Topics.SCHEDULER_STOPPED,
            {
                "worker_id": self.worker_id,
                "total_checks": self._check_count
            }
        )
        
        await super().stop()
    
    async def run_scheduler_loop(self) -> None:
        """
        Run the scheduler loop that checks for due posts.
        
        This runs indefinitely until stop() is called.
        """
        logger.info(f"[{self.worker_id}] 🕐 Scheduler loop started (every {self.check_interval}s)")
        
        while self.is_running:
            try:
                await self._check_for_due_posts()
                self._check_count += 1
                
                # Emit heartbeat
                await self.emit(
                    Topics.SCHEDULER_TICK,
                    {
                        "check_number": self._check_count,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
                
                # Emit system health check every 5 ticks (every 5 minutes with default interval)
                if self._check_count % 5 == 0:
                    await self.emit(
                        Topics.SYSTEM_HEALTH_CHECK,
                        {
                            "worker_id": self.worker_id,
                            "uptime_seconds": self.get_uptime_seconds(),
                            "events_processed": self._events_processed,
                            "scheduler_running": True,
                            "check_count": self._check_count,
                        }
                    )
                
            except Exception as e:
                logger.error(f"[{self.worker_id}] Scheduler check failed: {e}")
            
            await asyncio.sleep(self.check_interval)
    
    def start_scheduler_task(self) -> asyncio.Task:
        """Start the scheduler loop as a background task."""
        if self._scheduler_task is None or self._scheduler_task.done():
            self._scheduler_task = asyncio.create_task(self.run_scheduler_loop())
        return self._scheduler_task
    
    async def _check_for_due_posts(self) -> None:
        """Check database for posts that are due and emit events."""
        try:
            # BUG FIX: _get_due_posts now atomically marks posts as 'processing'
            # So we don't need to call _mark_post_processing again
            due_posts = await self._get_due_posts()
            
            if due_posts:
                logger.info(f"[{self.worker_id}] 📋 Found {len(due_posts)} due post(s)")
                
                for post in due_posts:
                    # Emit schedule.due event for each post
                    # Posts are already marked as 'processing' by _get_due_posts
                    await self.emit(
                        Topics.SCHEDULE_DUE,
                        {
                            "post_id": str(post["id"]),
                            "media_id": post.get("content_id"),
                            "platform": post.get("platform"),
                            "account_id": post.get("account_id"),
                            "scheduled_at": post.get("scheduled_at"),
                            "title": post.get("title")
                        }
                    )
            else:
                logger.debug(f"[{self.worker_id}] No due posts")
                
        except Exception as e:
            logger.error(f"[{self.worker_id}] Error checking for due posts: {e}")
    
    async def _get_due_posts(self) -> List[Dict[str, Any]]:
        """Get posts that are due for publishing (with atomic status update)."""
        try:
            from sqlalchemy import create_engine, text
            import os
            
            DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:54322/postgres")
            engine = create_engine(DATABASE_URL)
            
            with engine.connect() as conn:
                # BUG FIX: Use atomic update with FOR UPDATE SKIP LOCKED pattern
                # This prevents multiple workers from processing the same posts
                result = conn.execute(text("""
                    UPDATE scheduled_posts
                    SET status = 'processing', updated_at = NOW()
                    WHERE id IN (
                        SELECT id
                        FROM scheduled_posts
                        WHERE status = 'scheduled'
                          AND scheduled_time <= NOW()
                        ORDER BY scheduled_time ASC
                        LIMIT 10
                        FOR UPDATE SKIP LOCKED
                    )
                    RETURNING id, clip_id, media_project_id, platform, platform_account_id, scheduled_time, title
                """)).fetchall()
                
                conn.commit()
                
                return [
                    {
                        "id": str(row[0]),
                        "content_id": str(row[1]) if row[1] else (str(row[2]) if row[2] else None),
                        "title": row[6] if len(row) > 6 else None,
                        "platform": row[3],
                        "account_id": str(row[4]) if row[4] else None,
                        "scheduled_at": str(row[5]) if row[5] else None
                    }
                    for row in result
                ]
        except Exception as e:
            logger.error(f"Error fetching due posts: {e}")
            return []
    
    async def _mark_post_processing(self, post_id: str) -> bool:
        """Atomically mark a post as being processed to prevent duplicate events (idempotency)."""
        try:
            from sqlalchemy import create_engine, text
            import os
            
            DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:54322/postgres")
            engine = create_engine(DATABASE_URL)
            
            with engine.connect() as conn:
                # BUG FIX: Atomic update with return value to check if update succeeded
                result = conn.execute(text("""
                    UPDATE scheduled_posts
                    SET status = 'processing', updated_at = NOW()
                    WHERE id = :id AND status = 'scheduled'
                    RETURNING id
                """), {"id": post_id})
                conn.commit()
                
                # Return True if we successfully updated (got the lock)
                return result.rowcount > 0
        except Exception as e:
            logger.error(f"Error marking post as processing: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        stats = super().get_stats()
        stats.update({
            "check_interval": self.check_interval,
            "total_checks": self._check_count,
            "scheduler_running": self._scheduler_task is not None and not self._scheduler_task.done()
        })
        return stats


# Convenience function to create and start worker
async def start_scheduler_worker(
    event_bus: Optional[EventBus] = None,
    check_interval: int = 60,
    start_loop: bool = True
) -> SchedulerWorker:
    """Create and start a scheduler worker."""
    worker = SchedulerWorker(event_bus, check_interval=check_interval)
    await worker.start()
    
    if start_loop:
        worker.start_scheduler_task()
    
    return worker
