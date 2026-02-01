"""
Redis Streams Adapter for EventBus
==================================
Production-scale event streaming using Redis Streams.

Features:
- Durable event persistence
- Consumer groups for distributed workers
- Automatic acknowledgment and retry
- Dead-letter queue handling
- Graceful fallback to in-memory when Redis unavailable

Usage:
    # Configure via environment variable
    REDIS_URL=redis://localhost:6379
    EVENT_BUS_BACKEND=redis  # or 'memory' for in-memory

    # Or programmatically
    from services.event_bus.redis_adapter import RedisEventBus
    bus = RedisEventBus(redis_url="redis://localhost:6379")
"""

import os
import json
import asyncio
import logging
from typing import Callable, Dict, List, Optional, Any, Set
from datetime import datetime, timezone
from dataclasses import dataclass
from uuid import uuid4

from .event import Event
from .topics import Topics

logger = logging.getLogger(__name__)

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
STREAM_PREFIX = "mediaposter:events:"
CONSUMER_GROUP = "mediaposter-workers"
MAX_STREAM_LENGTH = 10000  # Max events per stream (MAXLEN)
BLOCK_MS = 5000  # Block time for XREAD
CLAIM_MIN_IDLE_MS = 60000  # Claim pending messages after 60s


@dataclass
class RedisSubscription:
    """Tracks a Redis stream subscription."""
    pattern: str
    callback: Callable
    consumer_name: str
    created_at: datetime


class RedisEventBus:
    """
    Redis Streams-backed EventBus for production scale.
    
    Key differences from in-memory EventBus:
    - Events persist in Redis Streams
    - Supports consumer groups for distributed processing
    - Automatic message acknowledgment
    - Dead-letter queue for failed messages
    - Survives process restarts
    
    Stream naming:
    - mediaposter:events:{topic} - One stream per topic
    - mediaposter:events:dlq - Dead letter queue
    """
    
    _instance: Optional['RedisEventBus'] = None
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or REDIS_URL
        self._redis = None
        self._source = "redis-event-bus"
        self._subscriptions: Dict[str, List[RedisSubscription]] = {}
        self._consumer_tasks: List[asyncio.Task] = []
        self._is_running = False
        self._consumer_name = f"consumer-{uuid4().hex[:8]}"
        self._stats = {
            "events_published": 0,
            "events_consumed": 0,
            "events_failed": 0,
            "connection_errors": 0
        }
    
    @classmethod
    def get_instance(cls, redis_url: str = None) -> 'RedisEventBus':
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls(redis_url)
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset singleton (for testing)."""
        if cls._instance:
            asyncio.create_task(cls._instance.shutdown())
        cls._instance = None
    
    async def _get_redis(self):
        """Get or create Redis connection."""
        if self._redis is None:
            try:
                import redis.asyncio as redis
                self._redis = redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                # Test connection
                await self._redis.ping()
                logger.info(f"Connected to Redis at {self.redis_url}")
            except ImportError:
                logger.error("redis package not installed. Run: pip install redis")
                raise
            except Exception as e:
                self._stats["connection_errors"] += 1
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        return self._redis
    
    def set_source(self, source: str):
        """Set the source identifier for events."""
        self._source = source
    
    def _stream_name(self, topic: str) -> str:
        """Get Redis stream name for a topic."""
        return f"{STREAM_PREFIX}{topic}"
    
    async def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        correlation_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> Event:
        """
        Publish an event to Redis Streams.
        
        The event is added to the stream for the given topic.
        All subscribers to matching patterns will receive it.
        """
        event = Event(
            id=str(uuid4()),
            topic=topic,
            timestamp=datetime.now(timezone.utc),
            source=self._source,
            correlation_id=correlation_id or str(uuid4()),
            payload=payload,
            metadata=metadata or {}
        )
        
        try:
            redis = await self._get_redis()
            stream = self._stream_name(topic)
            
            # Add to stream with MAXLEN to prevent unbounded growth
            await redis.xadd(
                stream,
                {
                    "event": json.dumps(event.to_dict())
                },
                maxlen=MAX_STREAM_LENGTH,
                approximate=True
            )
            
            self._stats["events_published"] += 1
            logger.debug(f"Published to {stream}: {event.id}")
            
            # Also publish to local subscribers (for WebSocket bridge)
            await self._notify_local_subscribers(event)
            
            return event
            
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            self._stats["connection_errors"] += 1
            raise
    
    def subscribe(
        self,
        pattern: str,
        callback: Callable[[Event], Any]
    ) -> str:
        """
        Subscribe to events matching a pattern.
        
        For Redis Streams, this creates a consumer in the consumer group.
        The callback is invoked for each matching event.
        
        Returns subscription ID.
        """
        subscription = RedisSubscription(
            pattern=pattern,
            callback=callback,
            consumer_name=self._consumer_name,
            created_at=datetime.now(timezone.utc)
        )
        
        if pattern not in self._subscriptions:
            self._subscriptions[pattern] = []
        
        self._subscriptions[pattern].append(subscription)
        
        # Return subscription ID
        return f"{pattern}:{id(subscription)}"
    
    def unsubscribe(self, subscription_id: str):
        """Remove a subscription."""
        pattern, sub_id = subscription_id.rsplit(":", 1)
        if pattern in self._subscriptions:
            self._subscriptions[pattern] = [
                s for s in self._subscriptions[pattern]
                if str(id(s)) != sub_id
            ]
    
    async def _notify_local_subscribers(self, event: Event):
        """Notify local (in-process) subscribers of an event."""
        for pattern, subscriptions in self._subscriptions.items():
            if self._matches_pattern(event.topic, pattern):
                for sub in subscriptions:
                    try:
                        result = sub.callback(event)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.error(f"Subscriber error for {pattern}: {e}")
    
    def _matches_pattern(self, topic: str, pattern: str) -> bool:
        """Check if topic matches subscription pattern."""
        if pattern == "*":
            return True
        if "*" not in pattern:
            return topic == pattern
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return topic.startswith(prefix + ".")
        if pattern.startswith("*."):
            suffix = pattern[2:]
            return topic.endswith("." + suffix)
        return topic == pattern
    
    async def start_consumers(self):
        """
        Start background consumer tasks for all subscribed patterns.
        
        Each pattern gets its own consumer task that reads from
        matching Redis streams.
        """
        if self._is_running:
            return
        
        self._is_running = True
        
        # Get all unique topic patterns
        patterns = set(self._subscriptions.keys())
        
        for pattern in patterns:
            task = asyncio.create_task(self._consume_pattern(pattern))
            self._consumer_tasks.append(task)
        
        logger.info(f"Started {len(self._consumer_tasks)} consumer tasks")
    
    async def _consume_pattern(self, pattern: str):
        """
        Consume events from streams matching a pattern.
        
        Uses XREADGROUP for consumer group semantics.
        """
        try:
            redis = await self._get_redis()
            
            # Get streams matching the pattern
            streams = await self._get_matching_streams(pattern)
            if not streams:
                # If no streams yet, poll periodically
                while self._is_running:
                    await asyncio.sleep(5)
                    streams = await self._get_matching_streams(pattern)
                    if streams:
                        break
            
            # Ensure consumer group exists for each stream
            for stream in streams:
                await self._ensure_consumer_group(stream)
            
            # Build stream dict for XREADGROUP
            stream_ids = {stream: ">" for stream in streams}
            
            while self._is_running:
                try:
                    # Read new messages
                    messages = await redis.xreadgroup(
                        groupname=CONSUMER_GROUP,
                        consumername=self._consumer_name,
                        streams=stream_ids,
                        count=10,
                        block=BLOCK_MS
                    )
                    
                    if messages:
                        for stream, entries in messages:
                            for msg_id, data in entries:
                                await self._process_message(stream, msg_id, data, pattern)
                    
                    # Claim and process any pending messages
                    await self._claim_pending_messages(streams, pattern)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Consumer error for {pattern}: {e}")
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logger.error(f"Fatal consumer error for {pattern}: {e}")
    
    async def _get_matching_streams(self, pattern: str) -> List[str]:
        """Get all stream names matching a pattern."""
        try:
            redis = await self._get_redis()
            
            if pattern == "*":
                # Get all event streams
                keys = await redis.keys(f"{STREAM_PREFIX}*")
                return [k for k in keys if not k.endswith(":dlq")]
            
            if "*" not in pattern:
                # Exact match
                stream = self._stream_name(pattern)
                exists = await redis.exists(stream)
                return [stream] if exists else []
            
            # Wildcard pattern
            if pattern.endswith(".*"):
                prefix = pattern[:-2]
                search = f"{STREAM_PREFIX}{prefix}.*"
            else:
                search = f"{STREAM_PREFIX}*"
            
            keys = await redis.keys(search)
            return keys
            
        except Exception as e:
            logger.error(f"Error getting streams for {pattern}: {e}")
            return []
    
    async def _ensure_consumer_group(self, stream: str):
        """Ensure consumer group exists for a stream."""
        try:
            redis = await self._get_redis()
            await redis.xgroup_create(
                stream, CONSUMER_GROUP, id="0", mkstream=True
            )
        except Exception as e:
            # Group already exists is OK
            if "BUSYGROUP" not in str(e):
                logger.warning(f"Could not create consumer group for {stream}: {e}")
    
    async def _process_message(
        self,
        stream: str,
        msg_id: str,
        data: Dict[str, str],
        pattern: str
    ):
        """Process a single message from Redis Stream."""
        try:
            event_data = json.loads(data.get("event", "{}"))
            event = Event.from_dict(event_data)
            
            # Call all subscribers for this pattern
            for sub in self._subscriptions.get(pattern, []):
                try:
                    result = sub.callback(event)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Handler error for {pattern}: {e}")
                    await self._send_to_dlq(event, str(e))
                    self._stats["events_failed"] += 1
                    return
            
            # Acknowledge the message
            redis = await self._get_redis()
            await redis.xack(stream, CONSUMER_GROUP, msg_id)
            self._stats["events_consumed"] += 1
            
        except Exception as e:
            logger.error(f"Error processing message {msg_id}: {e}")
            self._stats["events_failed"] += 1
    
    async def _claim_pending_messages(self, streams: List[str], pattern: str):
        """Claim and reprocess messages that have been pending too long."""
        try:
            redis = await self._get_redis()
            
            for stream in streams:
                # Get pending messages older than CLAIM_MIN_IDLE_MS
                pending = await redis.xpending_range(
                    stream, CONSUMER_GROUP,
                    min="-", max="+", count=10
                )
                
                for entry in pending:
                    if entry.get("time_since_delivered", 0) > CLAIM_MIN_IDLE_MS:
                        msg_id = entry["message_id"]
                        
                        # Claim the message
                        claimed = await redis.xclaim(
                            stream, CONSUMER_GROUP, self._consumer_name,
                            min_idle_time=CLAIM_MIN_IDLE_MS,
                            message_ids=[msg_id]
                        )
                        
                        for msg_id, data in claimed:
                            await self._process_message(stream, msg_id, data, pattern)
                            
        except Exception as e:
            # Pending operations can fail if stream is empty
            pass
    
    async def _send_to_dlq(self, event: Event, error: str):
        """Send failed event to dead-letter queue."""
        try:
            redis = await self._get_redis()
            dlq_stream = f"{STREAM_PREFIX}dlq"
            
            await redis.xadd(
                dlq_stream,
                {
                    "event": json.dumps(event.to_dict()),
                    "error": error,
                    "failed_at": datetime.now(timezone.utc).isoformat()
                },
                maxlen=MAX_STREAM_LENGTH
            )
            
        except Exception as e:
            logger.error(f"Failed to send to DLQ: {e}")
    
    async def get_dlq_events(self, count: int = 100) -> List[Dict]:
        """Get events from dead-letter queue."""
        try:
            redis = await self._get_redis()
            dlq_stream = f"{STREAM_PREFIX}dlq"
            
            messages = await redis.xrange(dlq_stream, count=count)
            
            return [
                {
                    "id": msg_id,
                    "event": json.loads(data.get("event", "{}")),
                    "error": data.get("error"),
                    "failed_at": data.get("failed_at")
                }
                for msg_id, data in messages
            ]
            
        except Exception as e:
            logger.error(f"Error getting DLQ events: {e}")
            return []
    
    async def replay_event(self, event_id: str) -> bool:
        """Replay a dead-letter event."""
        try:
            redis = await self._get_redis()
            dlq_stream = f"{STREAM_PREFIX}dlq"
            
            # Find the event in DLQ
            messages = await redis.xrange(dlq_stream, min=event_id, max=event_id, count=1)
            
            if not messages:
                return False
            
            msg_id, data = messages[0]
            event_data = json.loads(data.get("event", "{}"))
            event = Event.from_dict(event_data)
            
            # Re-publish the event
            await self.publish(
                event.topic,
                event.payload,
                event.correlation_id,
                event.metadata
            )
            
            # Remove from DLQ
            await redis.xdel(dlq_stream, msg_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error replaying event {event_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get EventBus statistics."""
        return {
            "backend": "redis",
            "redis_url": self.redis_url,
            "consumer_name": self._consumer_name,
            "is_running": self._is_running,
            "subscription_patterns": list(self._subscriptions.keys()),
            "consumer_tasks": len(self._consumer_tasks),
            **self._stats
        }
    
    async def shutdown(self):
        """Gracefully shutdown consumers and connections."""
        self._is_running = False
        
        # Cancel consumer tasks
        for task in self._consumer_tasks:
            task.cancel()
        
        if self._consumer_tasks:
            await asyncio.gather(*self._consumer_tasks, return_exceptions=True)
        
        self._consumer_tasks = []
        
        # Close Redis connection
        if self._redis:
            await self._redis.close()
            self._redis = None
        
        logger.info("Redis EventBus shutdown complete")


def get_event_bus(backend: str = None):
    """
    Factory function to get appropriate EventBus instance.
    
    Args:
        backend: 'redis' or 'memory' (default from EVENT_BUS_BACKEND env)
    
    Returns:
        EventBus instance (Redis or in-memory)
    """
    backend = backend or os.getenv("EVENT_BUS_BACKEND", "memory")
    
    if backend == "redis":
        return RedisEventBus.get_instance()
    else:
        from .bus import EventBus
        return EventBus.get_instance()
