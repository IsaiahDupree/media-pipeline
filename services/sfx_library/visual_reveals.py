"""
Visual Reveals System

Tracks when visual elements appear on screen (keywords, bullets, code, charts, etc.)
Used by the macro placement policy to add appropriate SFX at reveal moments.
"""

import json
from typing import Optional, Literal
from pathlib import Path
from pydantic import BaseModel, Field


RevealKind = Literal["keyword", "bullet", "code", "chart", "cta", "error", "success", "image", "transition"]


class VisualReveal(BaseModel):
    """A visual reveal event."""
    t: float = Field(ge=0, description="Time in seconds")
    kind: RevealKind
    key: Optional[str] = Field(None, description="Label like 'Supabase', 'Deploy OK'")
    
    class Config:
        populate_by_name = True


class VisualRevealsFile(BaseModel):
    """Collection of visual reveals."""
    version: str = "1.0.0"
    reveals: list[VisualReveal] = Field(default_factory=list)
    
    def get_reveals_at(self, t: float, tolerance: float = 0.1) -> list[VisualReveal]:
        """Get reveals at a specific time within tolerance."""
        return [r for r in self.reveals if abs(r.t - t) <= tolerance]
    
    def get_reveals_in_range(self, start: float, end: float) -> list[VisualReveal]:
        """Get reveals in a time range."""
        return [r for r in self.reveals if start <= r.t <= end]
    
    def get_by_kind(self, kind: RevealKind) -> list[VisualReveal]:
        """Get all reveals of a specific kind."""
        return [r for r in self.reveals if r.kind == kind]


def load_visual_reveals(path: str) -> VisualRevealsFile:
    """Load visual reveals from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return VisualRevealsFile.model_validate(data)


def save_visual_reveals(reveals: VisualRevealsFile, path: str) -> str:
    """Save visual reveals to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(reveals.model_dump(by_alias=True), f, indent=2)
    return path


def beats_to_visual_reveals(
    beats: list[dict],
    fps: int = 30,
) -> VisualRevealsFile:
    """
    Extract visual reveals from beats.
    
    Looks for beat properties like:
    - on_screen.headline/bullet/label → keyword reveal
    - type == "CTA" → cta reveal
    - action == "error" or "success" → error/success reveal
    
    Args:
        beats: List of beat dicts
        fps: Frames per second
        
    Returns:
        VisualRevealsFile
    """
    reveals = []
    
    for beat in beats:
        frame = beat.get("frame", 0)
        t = frame / fps
        
        on_screen = beat.get("on_screen") or beat.get("onScreen") or {}
        beat_type = beat.get("type", "")
        action = beat.get("action", "")
        
        # Keyword reveals from on_screen content
        if on_screen.get("headline"):
            reveals.append(VisualReveal(
                t=t,
                kind="keyword",
                key=on_screen["headline"][:50],
            ))
        elif on_screen.get("bullet"):
            reveals.append(VisualReveal(
                t=t,
                kind="bullet",
                key=on_screen["bullet"][:50],
            ))
        elif on_screen.get("label"):
            reveals.append(VisualReveal(
                t=t,
                kind="keyword",
                key=on_screen["label"][:30],
            ))
        
        # CTA reveals
        if beat_type == "CTA":
            reveals.append(VisualReveal(
                t=t,
                kind="cta",
                key=on_screen.get("headline") or "CTA",
            ))
        
        # Error/success reveals
        if action == "error" or "error" in beat.get("narration", "").lower():
            reveals.append(VisualReveal(
                t=t,
                kind="error",
                key=beat.get("narration", "")[:30],
            ))
        elif action == "success" or "success" in beat.get("narration", "").lower():
            reveals.append(VisualReveal(
                t=t,
                kind="success",
                key=beat.get("narration", "")[:30],
            ))
    
    # Sort by time
    reveals.sort(key=lambda r: r.t)
    
    return VisualRevealsFile(reveals=reveals)


def story_ir_to_visual_reveals(
    ir: dict,
    fps: Optional[int] = None,
) -> VisualRevealsFile:
    """
    Extract visual reveals from a Story IR.
    
    Args:
        ir: Story IR dict
        fps: Override FPS (uses ir.meta.fps if not provided)
        
    Returns:
        VisualRevealsFile
    """
    fps = fps or ir.get("meta", {}).get("fps", 30)
    beats = ir.get("beats", [])
    
    reveals = []
    cursor_frames = 0
    
    for beat in beats:
        duration_s = beat.get("duration_s") or beat.get("durationS", 3)
        beat_frames = round(duration_s * fps)
        t = cursor_frames / fps
        
        on_screen = beat.get("on_screen") or beat.get("onScreen") or {}
        beat_type = beat.get("type", "")
        
        # Keyword reveals
        if on_screen.get("headline"):
            reveals.append(VisualReveal(
                t=t + 0.2,  # Slight delay for animation
                kind="keyword",
                key=on_screen["headline"][:50],
            ))
        elif on_screen.get("bullet"):
            reveals.append(VisualReveal(
                t=t + 0.2,
                kind="bullet",
                key=on_screen["bullet"][:50],
            ))
        
        # Beat type reveals
        if beat_type == "CTA":
            reveals.append(VisualReveal(
                t=t + 0.3,
                kind="cta",
                key="CTA",
            ))
        elif beat_type == "PROOF":
            reveals.append(VisualReveal(
                t=t + 0.1,
                kind="success",
                key="proof",
            ))
        
        cursor_frames += beat_frames
    
    reveals.sort(key=lambda r: r.t)
    return VisualRevealsFile(reveals=reveals)


def get_reveal_macro_mapping() -> dict[RevealKind, str]:
    """
    Get default macro mappings for reveal kinds.
    
    Returns:
        Dict mapping RevealKind to macroId
    """
    return {
        "keyword": "text_ping",
        "bullet": "text_ping",
        "code": "glitch_cut",
        "chart": "reveal_riser",
        "cta": "cta_sparkle",
        "error": "warning_buzz_soft",
        "success": "success_ding",
        "image": "impact_soft",
        "transition": "transition_fast",
    }
