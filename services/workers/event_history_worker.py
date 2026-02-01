"""
Event History Worker
====================
Persists all events to the database for debugging, replay, and analytics.

Subscribes to:
    - * (all events)

This worker runs silently in the background, persisting events without
affecting the event bus performance.
"""

import asyncio
import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from uuid import UUID


class UUIDEncoder(json.JSONEncoder):
    """JSON encoder that handles UUID objects."""
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

from services.event_bus import EventBus, Event, Topics
from services.workers.base import BaseWorker
from database.connection import async_session_maker
from sqlalchemy import text

logger = logging.getLogger(__name__)


class EventHistoryWorker(BaseWorker):
    """
    Worker that persists all events to the database.
    
    This enables:
    - Event replay for debugging
    - Event querying and analytics
    - Audit trail
    - Recovery from failures
    
    Usage:
        worker = EventHistoryWorker()
        await worker.start()
        
        # All events will be automatically persisted
    """
    
    # Batch size for bulk inserts
    BATCH_SIZE = 50
    _event_buffer: List[Event] = []
    _buffer_lock = asyncio.Lock()
    
    def get_subscriptions(self) -> List[str]:
        """Subscribe to all events."""
        return ["*"]  # Wildcard subscribes to everything
    
    async def handle_event(self, event: Event) -> None:
        """Persist event to database."""
        # Add to buffer for batch processing
        async with self._buffer_lock:
            self._event_buffer.append(event)
            
            # Flush if buffer is full
            if len(self._event_buffer) >= self.BATCH_SIZE:
                await self._flush_buffer()
    
    async def _flush_buffer(self) -> None:
        """Flush buffered events to database."""
        if not self._event_buffer:
            return
        
        if not async_session_maker:
            logger.warning("Database not initialized, skipping event persistence")
            self._event_buffer.clear()
            return
        
        events_to_persist = self._event_buffer.copy()
        self._event_buffer.clear()
        
        try:
            async with async_session_maker() as session:
                # Bulk insert events
                values = []
                for event in events_to_persist:
                    values.append({
                        "event_id": event.id,
                        "topic": event.topic,
                        "source": event.source,
                        "correlation_id": event.correlation_id,
                        "payload": json.dumps(event.payload, cls=UUIDEncoder),
                        "metadata": json.dumps(event.metadata, cls=UUIDEncoder),
                        "timestamp": event.timestamp
                    })
                
                if values:
                    # Insert events one by one (asyncpg doesn't support bulk execute with list of dicts)
                    stmt = text("""
                        INSERT INTO event_history 
                        (event_id, topic, source, correlation_id, payload, metadata, timestamp)
                        VALUES 
                        (:event_id, :topic, :source, :correlation_id, 
                         CAST(:payload AS jsonb), CAST(:metadata AS jsonb), :timestamp)
                        ON CONFLICT DO NOTHING
                    """)
                    
                    for value in values:
                        await session.execute(stmt, value)
                    await session.commit()
                    
                    logger.debug(f"Persisted {len(values)} events to database")
                    
        except Exception as e:
            # Check if it's a missing table error - don't spam logs
            error_str = str(e).lower()
            if "does not exist" in error_str or "relation" in error_str:
                logger.warning(f"Event history table not available: {e}. Events will be buffered but not persisted.")
            else:
                logger.error(f"Failed to persist events: {e}", exc_info=True)
            # Re-add events to buffer for retry (but limit buffer size)
            async with self._buffer_lock:
                self._event_buffer = events_to_persist[:self.BATCH_SIZE] + self._event_buffer
    
    async def start(self) -> None:
        """Start the worker and set up periodic flush."""
        await super().start()
        
        # Start periodic flush task (every 5 seconds)
        asyncio.create_task(self._periodic_flush())
        
        logger.info(f"✅ {self.worker_id} started - persisting all events to database")
    
    async def _periodic_flush(self) -> None:
        """Periodically flush buffer even if not full."""
        while self.is_running:
            await asyncio.sleep(5)  # Flush every 5 seconds
            if self._event_buffer:
                await self._flush_buffer()
    
    async def stop(self) -> None:
        """Stop the worker and flush remaining events."""
        # Flush any remaining events
        if self._event_buffer:
            await self._flush_buffer()
        
        await super().stop()
        logger.info(f"✅ {self.worker_id} stopped - flushed remaining events")

