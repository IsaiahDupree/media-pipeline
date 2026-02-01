"""
Timeline Data Contract
======================
Schema for timeline.json (Remotion input).
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class LayerSchema(BaseModel):
    """Layer in the Remotion composition."""
    id: str = Field(..., description="Layer identifier")
    type: str = Field(..., description="Layer type: 'video', 'image', 'text', 'audio'")
    source: Optional[str] = Field(None, description="Source path/URL")
    source_type: Optional[str] = Field(None, description="Source type: 'local', 'url', 'tts', 'mediaposter', 'matting'")
    position: Dict[str, Any] = Field(default_factory=lambda: {"x": 0, "y": 0, "width": 1080, "height": 1920}, description="Position and dimensions")
    start: float = Field(default=0.0, ge=0.0, description="Start time in seconds")
    end: Optional[float] = Field(None, ge=0.0, description="End time in seconds")
    opacity: float = Field(default=1.0, ge=0.0, le=1.0, description="Opacity (0.0-1.0)")
    style: Optional[Dict[str, Any]] = Field(None, description="Style properties")
    animation: Optional[str] = Field(None, description="Animation type")
    content: Optional[str] = Field(None, description="Content (for text layers)")


class AudioTrackSchema(BaseModel):
    """Audio track in the composition."""
    id: str = Field(..., description="Audio track identifier")
    source: str = Field(..., description="Audio source path/URL")
    source_type: str = Field(..., description="Source type: 'local', 'url', 'tts', 'music'")
    start: float = Field(default=0.0, ge=0.0, description="Start time in seconds")
    volume: float = Field(default=1.0, ge=0.0, le=1.0, description="Volume (0.0-1.0)")
    ducking: Optional[Dict[str, Any]] = Field(None, description="Ducking configuration")


class CaptionConfigSchema(BaseModel):
    """Caption configuration."""
    enabled: bool = Field(default=True, description="Whether captions are enabled")
    style: str = Field(default="burned_in", description="Style: 'burned_in', 'overlay', 'none'")
    source: Optional[str] = Field(None, description="Path to word_timestamps.json")
    emphasis_words: bool = Field(default=True, description="Whether to emphasize words")
    position: str = Field(default="bottom", description="Position: 'top', 'bottom', 'center'")


class TimelineSchema(BaseModel):
    """
    Timeline - Complete timeline.json for Remotion composition.
    
    This is the output of Remotion Composer and input to Remotion CLI.
    """
    
    # Configuration
    fps: int = Field(default=30, ge=1, description="Frames per second")
    resolution: str = Field(default="1080x1920", description="Resolution (width x height)")
    duration: float = Field(..., ge=0.0, description="Total duration in seconds")
    
    # Layers
    layers: List[LayerSchema] = Field(default_factory=list, description="Visual layers")
    
    # Audio
    audio: List[AudioTrackSchema] = Field(default_factory=list, description="Audio tracks")
    
    # Captions
    captions: Optional[CaptionConfigSchema] = Field(None, description="Caption configuration")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "fps": 30,
                "resolution": "1080x1920",
                "duration": 45.0,
                "layers": [
                    {
                        "id": "layer_001",
                        "type": "video",
                        "source": "/data/broll/tech_lifestyle.mp4",
                        "source_type": "local",
                        "position": {"x": 0, "y": 0, "width": 1080, "height": 1920},
                        "start": 0.0,
                        "end": 45.0,
                        "opacity": 1.0
                    }
                ],
                "audio": [
                    {
                        "id": "audio_001",
                        "source": "/data/tts_outputs/voice.wav",
                        "source_type": "tts",
                        "start": 0.0,
                        "volume": 1.0
                    },
                    {
                        "id": "audio_002",
                        "source": "/data/music/trending_track.mp3",
                        "source_type": "music",
                        "start": 0.0,
                        "volume": 0.3,
                        "ducking": {"duck_under": "audio_001", "duck_db": -6}
                    }
                ],
                "captions": {
                    "enabled": True,
                    "style": "burned_in",
                    "source": "/data/tts_outputs/word_timestamps.json",
                    "emphasis_words": True,
                    "position": "bottom"
                }
            }
        }

