"""
Event Bus
=========
Central message broker for topic-based pub/sub communication.

Supports:
    - Topic subscriptions with wildcard patterns
    - Async event handlers
    - Event logging and dead-letter queue
    - Singleton pattern for global access
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Callable, Any, Optional, Awaitable
from uuid import uuid4

from .event import Event
from .topics import Topics

logger = logging.getLogger(__name__)


# Type alias for event handlers
EventHandler = Callable[[Event], Awaitable[None]]


class EventBus:
    """
    In-memory event bus for topic-based pub/sub.
    
    Features:
        - Subscribe to topics with wildcard support ("media.*", "*.completed")
        - Async event handlers
        - Event history logging
        - Dead-letter queue for failed events
        - Singleton pattern via get_instance()
    
    Usage:
        bus = EventBus.get_instance()
        
        # Subscribe
        bus.subscribe("media.ingested", my_handler)
        bus.subscribe("publish.*", publish_handler)
        
        # Publish
        event_id = await bus.publish("media.ingested", {"media_id": "abc"})
    """
    
    _instance: Optional["EventBus"] = None
    
    def __init__(self):
        self._subscribers: Dict[str, List[EventHandler]] = defaultdict(list)
        self._event_log: List[Event] = []
        self._dead_letter_queue: List[tuple[Event, Exception]] = []
        self._max_log_size = 1000
        self._source = "event-bus"
        self._is_running = True
        
        logger.info("🚌 EventBus initialized")
    
    @classmethod
    def get_instance(cls) -> "EventBus":
        """Get or create the singleton EventBus instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None
    
    def set_source(self, source: str) -> None:
        """Set the source name for events published from this bus."""
        self._source = source
    
    async def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Publish an event to a topic.
        
        Args:
            topic: Event topic (e.g., "media.ingested")
            payload: Event-specific data
            correlation_id: Optional ID to link related events
            source: Optional source override
            metadata: Optional additional metadata
        
        Returns:
            Event ID for tracking
        """
        event = Event(
            id=str(uuid4()),
            topic=topic,
            timestamp=datetime.now(timezone.utc),
            source=source or self._source,
            correlation_id=correlation_id or str(uuid4()),
            payload=payload,
            metadata=metadata or {}
        )
        
        # Log the event
        self._log_event(event)
        
        # Dispatch to subscribers
        await self._dispatch(event)
        
        return event.id
    
    async def publish_event(self, event: Event) -> str:
        """Publish a pre-constructed Event object."""
        self._log_event(event)
        await self._dispatch(event)
        return event.id
    
    def subscribe(self, topic_pattern: str, handler: EventHandler) -> str:
        """
        Subscribe a handler to a topic pattern.
        
        Supports wildcards:
            - "media.*" - all media events
            - "*.completed" - all completion events
            - "*" - all events
        
        Args:
            topic_pattern: Topic or pattern to subscribe to
            handler: Async function that receives Event
            
        Returns:
            Subscription ID for tracking/unsubscribing
        """
        self._subscribers[topic_pattern].append(handler)
        subscription_id = f"{topic_pattern}:{id(handler)}"
        logger.debug(f"📫 Subscribed to '{topic_pattern}': {handler.__name__}")
        return subscription_id
    
    def unsubscribe(self, topic_pattern: str, handler: EventHandler) -> bool:
        """
        Remove a handler from a topic pattern.
        
        Returns:
            True if handler was found and removed
        """
        if topic_pattern in self._subscribers:
            try:
                self._subscribers[topic_pattern].remove(handler)
                logger.debug(f"📭 Unsubscribed from '{topic_pattern}': {handler.__name__}")
                return True
            except ValueError:
                pass
        return False
    
    async def _dispatch(self, event: Event) -> None:
        """Dispatch event to all matching subscribers."""
        handlers_called = 0
        
        for pattern, handlers in self._subscribers.items():
            if Topics.matches_pattern(pattern, event.topic):
                for handler in handlers:
                    try:
                        await handler(event)
                        handlers_called += 1
                    except Exception as e:
                        logger.error(f"❌ Handler {handler.__name__} failed for {event.topic}: {e}")
                        self._dead_letter_queue.append((event, e))
        
        # Log dispatch result
        log_msg = f"📤 {event.topic} | cid={event.correlation_id[:8]}..."
        if handlers_called > 0:
            logger.info(f"{log_msg} → {handlers_called} handler(s)")
        else:
            logger.debug(f"{log_msg} → no subscribers")
    
    def _log_event(self, event: Event) -> None:
        """Add event to history log with size limit."""
        self._event_log.append(event)
        
        # Trim log if too large
        if len(self._event_log) > self._max_log_size:
            self._event_log = self._event_log[-self._max_log_size:]
    
    def get_recent_events(
        self,
        topic_pattern: Optional[str] = None,
        limit: int = 50,
        correlation_id: Optional[str] = None
    ) -> List[Event]:
        """
        Get recent events from the log.
        
        Args:
            topic_pattern: Optional filter by topic pattern
            limit: Maximum events to return
            correlation_id: Optional filter by correlation ID
        
        Returns:
            List of matching events, newest first
        """
        events = self._event_log[::-1]  # Reverse for newest first
        
        if topic_pattern:
            events = [e for e in events if Topics.matches_pattern(topic_pattern, e.topic)]
        
        if correlation_id:
            events = [e for e in events if e.correlation_id == correlation_id]
        
        return events[:limit]
    
    def get_dead_letter_queue(self, limit: int = 50) -> List[tuple[Event, str]]:
        """Get failed events from the dead-letter queue."""
        return [(e, str(ex)) for e, ex in self._dead_letter_queue[-limit:]]
    
    def clear_dead_letter_queue(self) -> int:
        """Clear the dead-letter queue and return count of cleared items."""
        count = len(self._dead_letter_queue)
        self._dead_letter_queue.clear()
        return count
    
    async def replay_event(self, event_id: str) -> bool:
        """
        Replay an event from the log.
        
        Args:
            event_id: ID of event to replay
        
        Returns:
            True if event was found and replayed
        """
        for event in self._event_log:
            if event.id == event_id:
                replayed_event = event.with_metadata(
                    replayed_at=datetime.now(timezone.utc).isoformat(),
                    original_timestamp=event.timestamp.isoformat()
                )
                await self._dispatch(replayed_event)
                return True
        return False
    
    def get_subscriber_count(self, topic_pattern: Optional[str] = None) -> Dict[str, int]:
        """Get count of subscribers per topic pattern."""
        if topic_pattern:
            return {topic_pattern: len(self._subscribers.get(topic_pattern, []))}
        return {pattern: len(handlers) for pattern, handlers in self._subscribers.items()}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            "is_running": self._is_running,
            "total_events_logged": len(self._event_log),
            "dead_letter_count": len(self._dead_letter_queue),
            "subscriber_patterns": len(self._subscribers),
            "total_subscribers": sum(len(h) for h in self._subscribers.values()),
            "topics_with_subscribers": list(self._subscribers.keys()),
        }
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the event bus."""
        self._is_running = False
        await self.publish(Topics.SYSTEM_SHUTDOWN, {"reason": "graceful_shutdown"})
        logger.info("🛑 EventBus shutdown complete")
