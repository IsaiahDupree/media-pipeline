"""
SFX Library Types

Pydantic models for SFX manifest, audio events, and context packs.
"""

from typing import Optional, Literal, Union
from pydantic import BaseModel, Field
from datetime import datetime


class SfxLicense(BaseModel):
    """License information for an SFX item."""
    source: Optional[str] = None
    requires_attribution: bool = False
    attribution_text: Optional[str] = None
    url: Optional[str] = None


class SfxItem(BaseModel):
    """A single sound effect in the library."""
    id: str = Field(..., min_length=1, description="Stable ID for AI reference")
    file: str = Field(..., min_length=1, description="Relative path to audio file")
    tags: list[str] = Field(default_factory=list, description="Searchable tags")
    description: str = Field(default="", description="Human-readable description")
    intensity: Optional[int] = Field(None, ge=1, le=10, description="Intensity 1-10")
    category: Optional[str] = Field(None, description="ui, transition, impact, etc.")
    license: Optional[SfxLicense] = None
    duration_ms: Optional[int] = Field(None, description="Duration in milliseconds")
    created_at: Optional[datetime] = None


class SfxManifest(BaseModel):
    """The complete SFX library manifest."""
    version: str = Field(..., min_length=1)
    items: list[SfxItem] = Field(default_factory=list)
    
    def get_by_id(self, sfx_id: str) -> Optional[SfxItem]:
        """Get an SFX item by ID."""
        for item in self.items:
            if item.id == sfx_id:
                return item
        return None
    
    def get_id_set(self) -> set[str]:
        """Get set of all valid IDs."""
        return {item.id for item in self.items}


class SfxAudioEvent(BaseModel):
    """An SFX event on the timeline."""
    type: Literal["sfx"] = "sfx"
    sfx_id: str = Field(..., alias="sfxId", min_length=1)
    frame: int = Field(..., ge=0)
    volume: float = Field(default=1.0, ge=0, le=2.0)
    
    class Config:
        populate_by_name = True


class MusicAudioEvent(BaseModel):
    """A music event on the timeline."""
    type: Literal["music"] = "music"
    src: str = Field(..., min_length=1)
    frame: int = Field(..., ge=0)
    volume: float = Field(default=0.25, ge=0, le=2.0)


class VoiceoverAudioEvent(BaseModel):
    """A voiceover event on the timeline."""
    type: Literal["voiceover"] = "voiceover"
    src: str = Field(..., min_length=1)
    frame: int = Field(..., ge=0)
    volume: float = Field(default=1.0, ge=0, le=2.0)


AudioEvent = Union[SfxAudioEvent, MusicAudioEvent, VoiceoverAudioEvent]


class AudioEvents(BaseModel):
    """Timeline of audio events."""
    fps: int = Field(..., ge=1, le=240)
    events: list[AudioEvent] = Field(default_factory=list)
    
    def get_sfx_events(self) -> list[SfxAudioEvent]:
        """Get only SFX events."""
        return [e for e in self.events if isinstance(e, SfxAudioEvent)]
    
    def get_music_events(self) -> list[MusicAudioEvent]:
        """Get only music events."""
        return [e for e in self.events if isinstance(e, MusicAudioEvent)]
    
    def get_voiceover_events(self) -> list[VoiceoverAudioEvent]:
        """Get only voiceover events."""
        return [e for e in self.events if isinstance(e, VoiceoverAudioEvent)]


class SfxContextItem(BaseModel):
    """Compact SFX item for LLM context."""
    id: str
    tags: list[str]
    desc: str
    intensity: Optional[int] = None
    category: Optional[str] = None


class SfxContextPack(BaseModel):
    """Token-efficient context pack for LLM prompts."""
    version: str
    rules: list[str]
    sfx_index: list[SfxContextItem] = Field(alias="sfxIndex")
    
    class Config:
        populate_by_name = True


class FixedEvent(BaseModel):
    """Record of an auto-fixed event."""
    from_id: str = Field(alias="from")
    to_id: str = Field(alias="to")
    frame: int
    reason: str
    
    class Config:
        populate_by_name = True


class RejectedEvent(BaseModel):
    """Record of a rejected event."""
    sfx_id: str = Field(alias="sfxId")
    frame: int
    reason: str
    
    class Config:
        populate_by_name = True


class FixReport(BaseModel):
    """Report of auto-fix operations."""
    fixed: list[FixedEvent] = Field(default_factory=list)
    rejected: list[RejectedEvent] = Field(default_factory=list)
    
    @property
    def has_fixes(self) -> bool:
        return len(self.fixed) > 0
    
    @property
    def has_rejections(self) -> bool:
        return len(self.rejected) > 0


class Beat(BaseModel):
    """A narrative beat for SFX selection."""
    beat_id: str = Field(alias="beatId")
    frame: int
    text: str
    action: Optional[Literal["hook", "reveal", "transition", "punchline", "cta", "explain"]] = None
    
    class Config:
        populate_by_name = True


class QATimelineIssue(BaseModel):
    """A QA issue found in the timeline."""
    code: str
    level: Literal["error", "warn"]
    message: str
    frame: Optional[int] = None
    beat_id: Optional[str] = None


class QATimelineReport(BaseModel):
    """QA report for audio timeline."""
    passed: bool
    issues: list[QATimelineIssue] = Field(default_factory=list)
    stats: dict = Field(default_factory=dict)
