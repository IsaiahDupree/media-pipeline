"""
Content Brief Data Contract
===========================
Schema for enhanced content briefs with scoring.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from .cluster import ClusterSchema


class BriefScoreSchema(BaseModel):
    """Brief score breakdown (0-100)."""
    total: float = Field(..., ge=0.0, le=100.0, description="Total score (0-100)")
    velocity: float = Field(default=0.0, ge=0.0, le=25.0, description="Velocity score (0-25)")
    intent: float = Field(default=0.0, ge=0.0, le=20.0, description="Intent score (0-20)")
    product_fit: float = Field(default=0.0, ge=0.0, le=25.0, description="Product fit score (0-25)")
    differentiation: float = Field(default=0.0, ge=0.0, le=15.0, description="Differentiation score (0-15)")
    production_feasibility: float = Field(default=0.0, ge=0.0, le=15.0, description="Feasibility score (0-15)")


class BriefAngleSchema(BaseModel):
    """Content angle for a brief."""
    angle_id: str = Field(..., description="Unique angle identifier")
    audience_role: str = Field(..., description="Target audience: 'creator', 'ecom_owner', 'dev', etc.")
    intent: str = Field(..., description="User intent: 'learn', 'compare', 'buy', etc.")
    stakes: str = Field(..., description="Stakes: 'time', 'money', 'reputation', etc.")
    format: str = Field(..., description="Format: 'myth_bust', 'teardown', 'tutorial', etc.")
    promise: str = Field(..., description="Main promise of the angle")
    unique_lens: str = Field(..., description="What makes this angle unique")
    convergence_pattern: str = Field(..., description="Convergence pattern: 'Problem × Tool', etc.")


class ContentBriefSchema(BaseModel):
    """
    Content Brief - Production-ready brief with scoring.
    
    This is the output of the Enhanced Brief Service and input to Script Generator.
    """
    
    # Identity
    brief_id: str = Field(..., description="Unique brief identifier")
    status: str = Field(default="draft", description="Status: 'draft', 'scored', 'approved', 'in_production', 'completed', 'rejected'")
    
    # Source
    cluster: Optional[ClusterSchema] = Field(None, description="Source trend cluster")
    angle: Optional[BriefAngleSchema] = Field(None, description="Selected content angle")
    
    # Scoring
    score: Optional[BriefScoreSchema] = Field(None, description="Brief score breakdown")
    worth_covering: bool = Field(default=False, description="Whether brief meets threshold")
    
    # Content
    title: Optional[str] = Field(None, description="Video title")
    hook: Optional[str] = Field(None, description="Opening hook")
    promise: Optional[str] = Field(None, description="Main promise")
    unique_lens: Optional[str] = Field(None, description="Unique lens/angle")
    
    # Video Spec
    format: str = Field(default="shorts", description="Format: 'shorts', 'reels', 'tiktok', 'longform'")
    length_sec: int = Field(default=45, ge=1, description="Target length in seconds")
    hook_sec: float = Field(default=1.2, ge=0.0, description="Hook duration in seconds")
    pattern_interrupt_sec: float = Field(default=4.0, ge=0.0, description="Pattern interrupt interval in seconds")
    
    # CTA
    cta: Optional[Dict[str, Any]] = Field(None, description="Call-to-action configuration")
    
    # Metadata
    niche: Optional[str] = Field(None, description="Target niche")
    platform: Optional[str] = Field(None, description="Target platform")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="When brief was generated")
    expires_at: Optional[datetime] = Field(None, description="When brief expires")
    
    class Config:
        json_schema_extra = {
            "example": {
                "brief_id": "brief_xyz789",
                "status": "scored",
                "score": {
                    "total": 78.5,
                    "velocity": 20.0,
                    "intent": 18.0,
                    "product_fit": 22.0,
                    "differentiation": 12.0,
                    "production_feasibility": 6.5
                },
                "worth_covering": True,
                "title": "The 5-Minute Productivity Hack That Changed Everything",
                "hook": "Stop doing this (here's the framework that works)",
                "format": "shorts",
                "length_sec": 45,
                "generated_at": "2024-12-26T10:00:00Z"
            }
        }

