"""
SFX Cue Sheet

Generates and processes SFX cue sheets for audio mixing.
Used to create a mixed audio bus for Motion Canvas rendering.
"""

import json
from typing import Optional
from pathlib import Path
from pydantic import BaseModel, Field

from .types import SfxManifest, AudioEvents


class SfxCue(BaseModel):
    """A single SFX cue."""
    id: str
    start_sec: float = Field(alias="startSec")
    gain_db: float = Field(default=-8, alias="gainDb")
    pan: float = Field(default=0, ge=-1, le=1)
    
    class Config:
        populate_by_name = True


class CueSheet(BaseModel):
    """A cue sheet for audio mixing."""
    base_audio: str = Field(alias="baseAudio")
    sample_rate: int = Field(default=48000, alias="sampleRate")
    cues: list[SfxCue] = Field(default_factory=list)
    
    class Config:
        populate_by_name = True


def audio_events_to_cue_sheet(
    events: AudioEvents,
    base_audio_path: str,
    sfx_root_dir: str,
    manifest: SfxManifest,
    default_gain_db: float = -8,
) -> CueSheet:
    """
    Convert AudioEvents to a CueSheet for mixing.
    
    Args:
        events: Audio events with SFX
        base_audio_path: Path to voiceover/music base audio
        sfx_root_dir: Root directory for SFX files
        manifest: SFX manifest for file lookup
        default_gain_db: Default gain for SFX
        
    Returns:
        CueSheet for mixing
    """
    fps = events.fps
    sfx_events = events.get_sfx_events()
    
    cues = []
    for ev in sfx_events:
        item = manifest.get_by_id(ev.sfx_id)
        if not item:
            continue
        
        # Convert frame to seconds
        start_sec = ev.frame / fps
        
        # Volume to dB (rough conversion)
        gain_db = default_gain_db
        if ev.volume != 1.0:
            import math
            if ev.volume > 0:
                gain_db = 20 * math.log10(ev.volume) + default_gain_db
        
        cues.append(SfxCue(
            id=ev.sfx_id,
            start_sec=start_sec,
            gain_db=gain_db,
        ))
    
    return CueSheet(
        base_audio=base_audio_path,
        cues=cues,
    )


def beats_to_cue_sheet(
    beats: list[dict],
    base_audio_path: str,
    fps: int = 30,
    default_gain_db: float = -8,
) -> CueSheet:
    """
    Convert beats with SFX hints to a CueSheet.
    
    Args:
        beats: List of beat dicts with 'frame' and optional 'sfx' list
        base_audio_path: Path to base audio
        fps: Frames per second
        default_gain_db: Default gain
        
    Returns:
        CueSheet
    """
    cues = []
    
    for beat in beats:
        frame = beat.get("frame", 0)
        sfx_list = beat.get("sfx", []) or beat.get("audio", {}).get("sfx", [])
        
        if not sfx_list:
            continue
        
        start_sec = frame / fps
        
        for sfx_id in sfx_list:
            if isinstance(sfx_id, dict):
                sfx_id = sfx_id.get("id", sfx_id.get("sfxId"))
            
            cues.append(SfxCue(
                id=sfx_id,
                start_sec=start_sec,
                gain_db=default_gain_db,
            ))
    
    return CueSheet(
        base_audio=base_audio_path,
        cues=cues,
    )


def save_cue_sheet(cue_sheet: CueSheet, output_path: str) -> str:
    """
    Save a cue sheet to JSON file.
    
    Args:
        cue_sheet: The cue sheet
        output_path: Path to save
        
    Returns:
        Path to saved file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(cue_sheet.model_dump(by_alias=True), f, indent=2)
    
    return output_path


def load_cue_sheet(path: str) -> CueSheet:
    """
    Load a cue sheet from JSON file.
    
    Args:
        path: Path to cue sheet JSON
        
    Returns:
        CueSheet
    """
    with open(path, "r") as f:
        data = json.load(f)
    return CueSheet.model_validate(data)


def resolve_cue_paths(
    cue_sheet: CueSheet,
    manifest: SfxManifest,
    sfx_root_dir: str,
) -> list[dict]:
    """
    Resolve SFX IDs to file paths.
    
    Args:
        cue_sheet: The cue sheet
        manifest: SFX manifest
        sfx_root_dir: Root directory for SFX files
        
    Returns:
        List of dicts with 'path', 'startSec', 'gainDb'
    """
    resolved = []
    
    for cue in cue_sheet.cues:
        item = manifest.get_by_id(cue.id)
        if not item:
            continue
        
        path = str(Path(sfx_root_dir) / item.file)
        
        resolved.append({
            "path": path,
            "startSec": cue.start_sec,
            "gainDb": cue.gain_db,
            "pan": cue.pan,
        })
    
    return resolved


def validate_cue_sheet(
    cue_sheet: CueSheet,
    manifest: SfxManifest,
) -> list[str]:
    """
    Validate a cue sheet against a manifest.
    
    Args:
        cue_sheet: The cue sheet
        manifest: SFX manifest
        
    Returns:
        List of validation error messages
    """
    errors = []
    valid_ids = manifest.get_id_set()
    
    for cue in cue_sheet.cues:
        if cue.id not in valid_ids:
            errors.append(f"Unknown SFX ID: {cue.id}")
        
        if cue.start_sec < 0:
            errors.append(f"Negative start time for {cue.id}: {cue.start_sec}")
        
        if cue.gain_db > 0:
            errors.append(f"Positive gain (clipping risk) for {cue.id}: {cue.gain_db}")
    
    return errors


def merge_cue_sheets(sheets: list[CueSheet]) -> CueSheet:
    """
    Merge multiple cue sheets into one.
    
    Args:
        sheets: List of cue sheets
        
    Returns:
        Merged CueSheet
    """
    if not sheets:
        raise ValueError("No cue sheets to merge")
    
    all_cues = []
    for sheet in sheets:
        all_cues.extend(sheet.cues)
    
    # Sort by start time
    all_cues.sort(key=lambda c: c.start_sec)
    
    return CueSheet(
        base_audio=sheets[0].base_audio,
        sample_rate=sheets[0].sample_rate,
        cues=all_cues,
    )


def get_cue_sheet_stats(cue_sheet: CueSheet) -> dict:
    """
    Get statistics about a cue sheet.
    
    Args:
        cue_sheet: The cue sheet
        
    Returns:
        Dict with stats
    """
    if not cue_sheet.cues:
        return {
            "total_cues": 0,
            "duration_sec": 0,
            "first_cue_sec": None,
            "last_cue_sec": None,
            "unique_sfx": 0,
        }
    
    start_times = [c.start_sec for c in cue_sheet.cues]
    unique_ids = {c.id for c in cue_sheet.cues}
    
    return {
        "total_cues": len(cue_sheet.cues),
        "duration_sec": max(start_times) - min(start_times),
        "first_cue_sec": min(start_times),
        "last_cue_sec": max(start_times),
        "unique_sfx": len(unique_ids),
    }
