"""
Script Data Contract
====================
Schema for script.json (Stage A output, TTS input).
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ScriptBeatSchema(BaseModel):
    """
    Script Beat - A segment in the script with timing and intent.
    """
    id: str = Field(..., description="Unique beat identifier (e.g., 'seg_001')")
    t: str = Field(..., description="Time range (e.g., '0-2', '2-12')")
    text: str = Field(..., description="Script text for this beat")
    intent: str = Field(..., description="Intent: 'hook', 'problem', 'solution', 'proof', 'cta', 'example'")
    on_screen: List[str] = Field(default_factory=list, description="Keywords to show on screen")
    visual_style: Optional[str] = Field(None, description="Visual style: 'big_text_punch_in', 'diagram', 'meme', etc.")
    emphasis_words: List[str] = Field(default_factory=list, description="Words to emphasize")


class ScriptSchema(BaseModel):
    """
    Script - Complete script.json for Media Factory pipeline.
    
    This is the output of Script Generator (Stage A) and input to TTS Service (Stage B).
    """
    
    # Identity
    brief_id: str = Field(..., description="Source brief identifier")
    title: str = Field(..., description="Video title")
    hook: str = Field(..., description="Opening hook text")
    
    # Segments
    segments: List[ScriptBeatSchema] = Field(default_factory=list, description="Script beats with timing")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "brief_id": "brief_xyz789",
                "title": "The 5-Minute Productivity Hack",
                "hook": "Stop doing this (here's the framework that works)",
                "segments": [
                    {
                        "id": "seg_001",
                        "t": "0-2",
                        "text": "Stop doing this (here's the framework that works)",
                        "intent": "hook",
                        "on_screen": ["Stop", "Framework"],
                        "visual_style": "big_text_punch_in",
                        "emphasis_words": ["stop", "framework", "works"]
                    },
                    {
                        "id": "seg_002",
                        "t": "2-12",
                        "text": "Most people approach productivity without considering time constraints.",
                        "intent": "problem",
                        "on_screen": ["Problem", "Time"],
                        "visual_style": "diagram",
                        "emphasis_words": ["most", "without", "time"]
                    }
                ],
                "metadata": {
                    "total_duration_sec": 45.0,
                    "word_count": 250,
                    "estimated_tts_duration": 100.0,
                    "format": "shorts",
                    "platform": "multi"
                }
            }
        }

