"""
Event Model
===========
Standardized event structure for all pub/sub communication.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from uuid import uuid4
import json


@dataclass
class Event:
    """
    Represents an event in the pub/sub system.
    
    Attributes:
        id: Unique identifier for deduplication
        topic: Event topic (e.g., "media.analysis.completed")
        timestamp: When the event was created
        source: Service that created the event
        correlation_id: Links related events in a workflow
        payload: Event-specific data
        metadata: Tracing, retry info, etc.
    """
    topic: str
    payload: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "unknown"
    correlation_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "id": self.id,
            "topic": self.topic,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "correlation_id": self.correlation_id,
            "payload": self.payload,
            "metadata": self.metadata,
        }
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        return cls(
            id=data.get("id", str(uuid4())),
            topic=data["topic"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else data.get("timestamp", datetime.now(timezone.utc)),
            source=data.get("source", "unknown"),
            correlation_id=data.get("correlation_id", str(uuid4())),
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {}),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "Event":
        """Create event from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    @property
    def event_id(self) -> str:
        """Alias for id to match test expectations."""
        return self.id

    def with_metadata(self, **kwargs) -> "Event":
        """Return a new event with additional metadata."""
        new_metadata = {**self.metadata, **kwargs}
        return Event(
            id=self.id,
            topic=self.topic,
            timestamp=self.timestamp,
            source=self.source,
            correlation_id=self.correlation_id,
            payload=self.payload,
            metadata=new_metadata,
        )

    def __repr__(self) -> str:
        return f"Event(topic={self.topic}, id={self.id[:8]}..., cid={self.correlation_id[:8]}...)"
