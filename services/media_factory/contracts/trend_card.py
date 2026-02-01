"""
TrendCard Data Contract
=======================
Schema for trend input cards from social media platforms.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class TrendCardSchema(BaseModel):
    """
    TrendCard - Raw trend input from social platforms.
    
    This is the entry point for the Media Factory pipeline.
    """
    
    # Identity
    trend_id: str = Field(..., description="Unique identifier for the trend")
    trend_type: str = Field(..., description="Type: 'hashtag', 'sound', 'topic', 'cluster'")
    trend_name: str = Field(..., description="Display name of the trend")
    platform: str = Field(..., description="Source platform: 'instagram', 'tiktok', 'youtube', etc.")
    
    # Categorization
    niche_tags: List[str] = Field(default_factory=list, description="Niche categories")
    
    # Velocity Signals (0.0-1.0 or absolute values)
    views_growth: float = Field(default=0.0, description="Views/hour growth rate")
    likes_per_min: float = Field(default=0.0, description="Likes per minute")
    shares_save_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Share/save rate (0.0-1.0)")
    comment_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Comment rate (0.0-1.0)")
    
    # Context
    what_people_achieve: Optional[str] = Field(None, description="What people are trying to achieve")
    what_people_stuck_on: Optional[str] = Field(None, description="What people are stuck on")
    
    # Evidence
    top_comments: List[str] = Field(default_factory=list, description="Top comments from posts")
    repeated_questions: List[str] = Field(default_factory=list, description="Repeated questions")
    creator_hooks: List[str] = Field(default_factory=list, description="Creator hooks used")
    
    # Format
    format: Optional[str] = Field(None, description="Content format: 'talking_head', 'screen_record', 'meme_edit', 'explainer', 'listicle'")
    
    # Metadata
    collected_at: datetime = Field(default_factory=datetime.utcnow, description="When trend was collected")
    
    class Config:
        json_schema_extra = {
            "example": {
                "trend_id": "hashtag_123",
                "trend_type": "hashtag",
                "trend_name": "productivityhacks",
                "platform": "instagram",
                "niche_tags": ["productivity", "business"],
                "views_growth": 150.5,
                "likes_per_min": 25.3,
                "shares_save_rate": 0.12,
                "comment_rate": 0.08,
                "what_people_achieve": "Increase daily productivity",
                "what_people_stuck_on": "Time management",
                "top_comments": ["how do i start?", "what tools?", "template?"],
                "repeated_questions": ["how to", "best app"],
                "creator_hooks": ["Stop doing this", "The 5-minute hack"],
                "format": "explainer",
                "collected_at": "2024-12-26T10:00:00Z"
            }
        }

