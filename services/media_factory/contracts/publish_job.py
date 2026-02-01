"""
Publish Job Data Contract
=========================
Schema for multi-platform publishing jobs.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class PlatformConfigSchema(BaseModel):
    """Platform-specific publishing configuration."""
    platform: str = Field(..., description="Platform: 'youtube_shorts', 'tiktok', 'instagram_reels'")
    title: Optional[str] = Field(None, description="Platform-specific title")
    description: Optional[str] = Field(None, description="Platform-specific description")
    hashtags: List[str] = Field(default_factory=list, description="Platform-specific hashtags")
    scheduled_for: Optional[datetime] = Field(None, description="Scheduled publish time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Platform-specific metadata")


class PublishJobSchema(BaseModel):
    """
    Publish Job - Multi-platform publishing job specification.
    
    This is the input to Publishing Service for platform distribution.
    """
    
    # Identity
    job_id: str = Field(..., description="Unique publish job identifier")
    correlation_id: str = Field(..., description="Correlation ID for tracking")
    
    # Media
    video_path: str = Field(..., description="Path to video file to publish")
    thumbnail_path: Optional[str] = Field(None, description="Path to thumbnail image")
    
    # Platforms
    platforms: List[PlatformConfigSchema] = Field(..., description="Platform configurations")
    
    # Metadata
    pipeline_id: Optional[str] = Field(None, description="Parent pipeline ID")
    brief_id: Optional[str] = Field(None, description="Source brief ID")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When job was created")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "publish_job_123",
                "correlation_id": "corr_abc456",
                "video_path": "/data/remotion_outputs/final_video.mp4",
                "thumbnail_path": "/data/thumbnails/thumbnail.jpg",
                "platforms": [
                    {
                        "platform": "youtube_shorts",
                        "title": "The 5-Minute Productivity Hack",
                        "description": "Stop doing this...",
                        "hashtags": ["productivity", "hacks"],
                        "scheduled_for": "2024-12-26T15:00:00Z"
                    },
                    {
                        "platform": "tiktok",
                        "title": "Productivity hack that works",
                        "hashtags": ["productivity", "hacks", "tips"],
                        "scheduled_for": "2024-12-26T15:00:00Z"
                    }
                ],
                "pipeline_id": "pipeline_xyz789",
                "brief_id": "brief_xyz789"
            }
        }

