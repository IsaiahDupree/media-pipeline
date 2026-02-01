"""
Matting Service Models
======================
Data models for video matting requests, responses, and job status.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Any, List
from uuid import UUID, uuid4


class MattingModel(str, Enum):
    """Supported matting models."""
    RVM = "rvm"  # Robust Video Matting (recommended)
    MEDIAPIPE = "mediapipe"  # MediaPipe Selfie Segmentation (fast, lightweight)
    BACKGROUND_MATTING_V2 = "background_matting_v2"  # With clean plate
    REMBG = "rembg"  # Simple batch processing
    SAM2 = "sam2"  # Advanced segmentation (future)


class MattingOperation(str, Enum):
    """Matting operations."""
    EXTRACT_PERSON = "extract_person"  # Extract person from video
    EXTRACT_OBJECT = "extract_object"  # Extract specific object
    REMOVE_BACKGROUND = "remove_background"  # Remove background
    COMPOSITE = "composite"  # Composite into target video


@dataclass
class MattingConfig:
    """Configuration for matting operation."""
    operation: MattingOperation = MattingOperation.EXTRACT_PERSON
    target_description: Optional[str] = None  # For SAM 2: "person in center"
    clean_background_plate: Optional[str] = None  # For BackgroundMattingV2
    preserve_alpha: bool = True  # Output alpha channel
    downsample_ratio: float = 0.25  # For RVM: 0.25 = faster, 1.0 = best quality
    device: str = "auto"  # "auto", "cuda", "cpu"
    model_variant: Optional[str] = None  # e.g., "mobilenetv3" or "resnet50" for RVM


@dataclass
class MattingRequest:
    """Video matting request."""
    source_video: str  # Path to source video
    target_video: Optional[str] = None  # Optional: for compositing
    model: MattingModel = MattingModel.RVM
    config: Optional[MattingConfig] = None
    output_path: Optional[str] = None  # Auto-generated if not provided
    correlation_id: Optional[str] = None
    job_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate job_id and correlation_id if not provided."""
        if self.job_id is None:
            self.job_id = str(uuid4())
        if self.correlation_id is None:
            self.correlation_id = str(uuid4())
        if self.config is None:
            self.config = MattingConfig()


@dataclass
class MattingResponse:
    """Video matting response."""
    job_id: str
    success: bool
    output_path: Optional[str] = None
    mask_path: Optional[str] = None  # Alpha channel mask
    processing_time: Optional[float] = None
    model_used: str = ""
    frames_processed: Optional[int] = None
    error: Optional[str] = None
    correlation_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MattingJobStatus:
    """Matting job status tracking."""
    job_id: str
    status: str  # "pending", "processing", "segmenting", "extracting", "compositing", "completed", "failed"
    progress: float = 0.0  # 0.0-1.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    response: Optional[MattingResponse] = None
    correlation_id: Optional[str] = None

