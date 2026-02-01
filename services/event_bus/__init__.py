"""
Event Bus Module
================
Pub/sub event system for service communication.

Supports two backends:
- In-memory (default): Fast, single-process
- Redis Streams: Durable, distributed, production-scale

Usage:
    from services.event_bus import EventBus, Event, Topics
    
    # In-memory (default)
    bus = EventBus.get_instance()
    await bus.publish(Topics.MEDIA_INGESTED, {"media_id": "123"})
    
    # Redis Streams (set EVENT_BUS_BACKEND=redis)
    from services.event_bus import get_event_bus
    bus = get_event_bus()  # Auto-selects based on config
"""

from .event import Event
from .topics import Topics
from .bus import EventBus
from .redis_adapter import RedisEventBus, get_event_bus

__all__ = ['Event', 'Topics', 'EventBus', 'RedisEventBus', 'get_event_bus']
