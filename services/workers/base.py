"""
Base Worker
===========
Abstract base class for event-driven workers.

Workers subscribe to topics, process events, and emit new events.
Provides standardized logging, error handling, and retry logic.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from uuid import uuid4

from services.event_bus import EventBus, Event, Topics

logger = logging.getLogger(__name__)


class BaseWorker(ABC):
    """
    Base class for event-driven workers.
    
    Subclass this to create workers that:
        - Subscribe to specific topics
        - Process events asynchronously
        - Emit progress and completion events
        - Handle errors with retry logic
    
    Example:
        class AnalysisWorker(BaseWorker):
            def get_subscriptions(self) -> List[str]:
                return [Topics.ANALYSIS_REQUESTED]
            
            async def handle_event(self, event: Event) -> None:
                media_id = event.payload.get("media_id")
                # Do analysis...
                await self.emit(Topics.ANALYSIS_COMPLETED, {"media_id": media_id})
    """
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        worker_id: Optional[str] = None
    ):
        """
        Initialize the worker.

        Args:
            event_bus: EventBus instance (uses singleton if not provided)
            worker_id: Unique worker identifier (auto-generated if not provided)
        """
        self.event_bus = event_bus or EventBus.get_instance()
        self.worker_id = worker_id or f"{self.__class__.__name__}-{uuid4().hex[:8]}"
        self.is_running = False
        self._events_processed = 0
        self._events_failed = 0
        self._started_at: Optional[datetime] = None

        # Sleep mode support (SLEEP-008)
        self._is_paused = False
        self._paused_at: Optional[datetime] = None
        self._total_pause_seconds = 0.0

        # Set up subscriptions
        self._setup_subscriptions()

        # Subscribe to sleep/wake events (SLEEP-008)
        self._setup_sleep_subscriptions()

        logger.info(f"🔧 Worker initialized: {self.worker_id}")
    
    @abstractmethod
    def get_subscriptions(self) -> List[str]:
        """
        Return list of topic patterns this worker subscribes to.
        
        Override this in subclasses to define which events to process.
        
        Returns:
            List of topic patterns (supports wildcards)
        """
        pass
    
    @abstractmethod
    async def handle_event(self, event: Event) -> None:
        """
        Process a received event.
        
        Override this in subclasses to implement event processing logic.
        
        Args:
            event: The received event to process
        
        Raises:
            Exception: Any error will trigger retry logic
        """
        pass
    
    def _setup_subscriptions(self) -> None:
        """Set up event subscriptions based on get_subscriptions()."""
        for topic_pattern in self.get_subscriptions():
            self.event_bus.subscribe(topic_pattern, self._wrapped_handler)
            logger.debug(f"📫 {self.worker_id} subscribed to: {topic_pattern}")

    def _setup_sleep_subscriptions(self) -> None:
        """
        Subscribe to sleep/wake events for automatic pause/resume (SLEEP-008).

        Workers automatically pause when system enters sleep mode and
        resume when system wakes up. This reduces CPU usage during idle periods.
        """
        self.event_bus.subscribe(Topics.SLEEP_ENTERED, self._handle_sleep_entered)
        self.event_bus.subscribe(Topics.SLEEP_WAKE, self._handle_sleep_wake)
        logger.debug(f"💤 {self.worker_id} subscribed to sleep/wake events")
    
    async def _wrapped_handler(self, event: Event) -> None:
        """
        Wrapper that adds logging, metrics, and error handling.

        This is the actual handler registered with the event bus.
        It wraps the user's handle_event method.
        """
        # SLEEP-008: Skip processing if worker is paused
        if self._is_paused:
            logger.debug(
                f"[{self.worker_id}] ⏸️  Skipping event (paused): {event.topic}"
            )
            return

        start_time = time.time()

        logger.info(
            f"[{self.worker_id}] 📥 Received: {event.topic} | "
            f"cid={event.correlation_id[:8]}..."
        )

        try:
            await self.handle_event(event)

            duration = time.time() - start_time
            self._events_processed += 1

            logger.info(
                f"[{self.worker_id}] ✅ Completed: {event.topic} | "
                f"duration={duration:.2f}s"
            )

        except Exception as e:
            duration = time.time() - start_time
            self._events_failed += 1

            logger.error(
                f"[{self.worker_id}] ❌ Failed: {event.topic} | "
                f"error={str(e)} | duration={duration:.2f}s"
            )

            # Re-raise to let event bus handle dead-lettering
            raise
    
    async def emit(
        self,
        topic: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Convenience method to publish events with worker source.
        
        Args:
            topic: Event topic
            payload: Event payload
            correlation_id: Optional correlation ID (uses new UUID if not provided)
        
        Returns:
            Published event ID
        """
        return await self.event_bus.publish(
            topic=topic,
            payload=payload,
            correlation_id=correlation_id,
            source=self.worker_id
        )
    
    async def emit_progress(
        self,
        base_topic: str,
        progress: int,
        step: str,
        correlation_id: str,
        **extra_payload
    ) -> str:
        """
        Emit a progress event for long-running operations.
        
        Args:
            base_topic: Base topic (e.g., "media.analysis")
            progress: Progress percentage (0-100)
            step: Current step name
            correlation_id: Workflow correlation ID
            **extra_payload: Additional payload fields
        
        Returns:
            Published event ID
        """
        return await self.emit(
            topic=f"{base_topic}.progress",
            payload={
                "progress": progress,
                "step": step,
                "worker_id": self.worker_id,
                **extra_payload
            },
            correlation_id=correlation_id
        )
    
    async def start(self) -> None:
        """Start the worker (mark as running)."""
        self.is_running = True
        self._started_at = datetime.now(timezone.utc)
        
        await self.emit(
            Topics.WORKER_STARTED,
            {
                "worker_id": self.worker_id,
                "worker_type": self.__class__.__name__,
                "subscriptions": self.get_subscriptions()
            }
        )
        
        logger.info(f"🚀 Worker started: {self.worker_id}")
    
    async def stop(self) -> None:
        """Stop the worker gracefully."""
        self.is_running = False
        
        await self.emit(
            Topics.WORKER_STOPPED,
            {
                "worker_id": self.worker_id,
                "events_processed": self._events_processed,
                "events_failed": self._events_failed,
                "uptime_seconds": self.get_uptime_seconds()
            }
        )
        
        logger.info(f"🛑 Worker stopped: {self.worker_id}")
    
    def get_uptime_seconds(self) -> float:
        """Get worker uptime in seconds."""
        if self._started_at:
            return (datetime.now(timezone.utc) - self._started_at).total_seconds()
        return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "worker_id": self.worker_id,
            "worker_type": self.__class__.__name__,
            "is_running": self.is_running,
            "is_paused": self._is_paused,
            "events_processed": self._events_processed,
            "events_failed": self._events_failed,
            "uptime_seconds": self.get_uptime_seconds(),
            "total_pause_seconds": self._total_pause_seconds,
            "subscriptions": self.get_subscriptions(),
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "paused_at": self._paused_at.isoformat() if self._paused_at else None,
        }

    async def _handle_sleep_entered(self, event: Event) -> None:
        """
        Handle system entering sleep mode (SLEEP-008).

        Pauses the worker to reduce CPU usage. The worker will stop
        processing new events until the system wakes up.

        Args:
            event: sleep.entered event
        """
        if not self._is_paused:
            self._is_paused = True
            self._paused_at = datetime.now(timezone.utc)

            logger.info(
                f"💤 Worker paused due to sleep mode: {self.worker_id}"
            )

    async def _handle_sleep_wake(self, event: Event) -> None:
        """
        Handle system waking from sleep mode (SLEEP-008).

        Resumes the worker to process events normally.

        Args:
            event: sleep.wake event
        """
        if self._is_paused:
            # Calculate pause duration
            if self._paused_at:
                pause_duration = (datetime.now(timezone.utc) - self._paused_at).total_seconds()
                self._total_pause_seconds += pause_duration

            self._is_paused = False
            self._paused_at = None

            logger.info(
                f"⏰ Worker resumed after sleep mode: {self.worker_id}"
            )
