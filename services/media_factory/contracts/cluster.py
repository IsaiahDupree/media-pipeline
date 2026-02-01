"""
Cluster Data Contract
=====================
Schema for clustered trends (merged duplicates across platforms).
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

from .trend_card import TrendCardSchema


class ClusterSchema(BaseModel):
    """
    Cluster - Grouped trends with semantic similarity.
    
    Multiple trends from different platforms that represent the same concept.
    """
    
    # Identity
    cluster_id: str = Field(..., description="Unique identifier for the cluster")
    name: str = Field(..., description="Cluster name (usually from most common trend)")
    
    # Trends
    trends: List[TrendCardSchema] = Field(default_factory=list, description="Trends in this cluster")
    
    # Summary
    what_changed: Optional[str] = Field(None, description="What changed in this trend cluster")
    why_people_care: Optional[str] = Field(None, description="Why people care about this trend")
    what_debate: Optional[str] = Field(None, description="What debate exists around this trend")
    
    # Aggregated Metrics
    total_views: int = Field(default=0, description="Total views across all trends")
    avg_velocity: float = Field(default=0.0, description="Average velocity across trends")
    avg_intent_score: float = Field(default=0.0, ge=0.0, le=20.0, description="Average intent score (0-20)")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When cluster was created")
    
    class Config:
        json_schema_extra = {
            "example": {
                "cluster_id": "cluster_abc123",
                "name": "productivityhacks",
                "trends": [
                    {
                        "trend_id": "hashtag_123",
                        "trend_type": "hashtag",
                        "trend_name": "productivityhacks",
                        "platform": "instagram"
                    }
                ],
                "what_changed": "New productivity methods trending",
                "why_people_care": "People want to optimize their time",
                "total_views": 1000000,
                "avg_velocity": 125.5,
                "avg_intent_score": 15.2,
                "created_at": "2024-12-26T10:00:00Z"
            }
        }

