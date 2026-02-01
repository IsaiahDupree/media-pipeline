"""
Render Job Data Contract
========================
Schema for Remotion render jobs.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

from .timeline import TimelineSchema


class RenderJobSchema(BaseModel):
    """
    Render Job - Remotion rendering job specification.
    
    This is the input to Remotion Service for video rendering.
    """
    
    # Identity
    job_id: str = Field(..., description="Unique render job identifier")
    correlation_id: str = Field(..., description="Correlation ID for tracking")
    
    # Composition
    composition: str = Field(default="MainComposition", description="Remotion composition name")
    timeline: TimelineSchema = Field(..., description="Timeline specification")
    props: Optional[Dict[str, Any]] = Field(None, description="Composition props")
    
    # Output
    output_path: Optional[str] = Field(None, description="Output file path")
    output_format: str = Field(default="mp4", description="Output format: 'mp4', 'mov', etc.")
    resolution: str = Field(default="1080x1920", description="Output resolution")
    fps: int = Field(default=30, ge=1, description="Output FPS")
    
    # Metadata
    pipeline_id: Optional[str] = Field(None, description="Parent pipeline ID")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When job was created")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "render_job_123",
                "correlation_id": "corr_abc456",
                "composition": "MainComposition",
                "timeline": {
                    "fps": 30,
                    "resolution": "1080x1920",
                    "duration": 45.0,
                    "layers": [],
                    "audio": []
                },
                "output_format": "mp4",
                "resolution": "1080x1920",
                "fps": 30,
                "pipeline_id": "pipeline_xyz789"
            }
        }

