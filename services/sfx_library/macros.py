"""
SFX Macros

Intent-based SFX selection system. Instead of AI picking individual SFX files,
it picks macros like "soft_emphasis", "transition_fast", "reveal" and the system
expands them to actual SFX IDs with intensity-based gain.
"""

import json
import math
from typing import Optional
from pathlib import Path
from pydantic import BaseModel, Field


class MacroCandidate(BaseModel):
    """A candidate SFX for a macro."""
    sfx_id: str = Field(alias="sfxId")
    weight: float = Field(default=1.0, ge=0, le=1)
    
    class Config:
        populate_by_name = True


class SfxMacro(BaseModel):
    """An SFX macro mapping intent to SFX candidates."""
    macro_id: str = Field(alias="macroId")
    description: str = ""
    default_gain_db: float = Field(default=-12, alias="defaultGainDb")
    gain_db_range: tuple[float, float] = Field(default=(-18, -6), alias="gainDbRange")
    candidates: list[MacroCandidate]
    
    class Config:
        populate_by_name = True


class SfxMacros(BaseModel):
    """Collection of SFX macros."""
    version: str = "1.0.0"
    macros: list[SfxMacro]
    
    def get_by_id(self, macro_id: str) -> Optional[SfxMacro]:
        """Get macro by ID."""
        for m in self.macros:
            if m.macro_id == macro_id:
                return m
        return None
    
    def get_all_ids(self) -> set[str]:
        """Get all macro IDs."""
        return {m.macro_id for m in self.macros}
    
    class Config:
        populate_by_name = True


class MacroCue(BaseModel):
    """A cue using a macro instead of direct SFX ID."""
    t: float = Field(ge=0, description="Time in seconds")
    macro_id: str = Field(alias="macroId")
    intensity: float = Field(default=0.5, ge=0, le=1)
    
    class Config:
        populate_by_name = True


class MacroCueSheet(BaseModel):
    """A cue sheet using macros."""
    version: str = "1.0.0"
    sample_rate: int = Field(default=48000, alias="sampleRate")
    cues: list[MacroCue] = Field(default_factory=list)
    
    class Config:
        populate_by_name = True


class ExpandedCue(BaseModel):
    """An expanded cue with actual SFX ID."""
    t: float
    id: str
    gain_db: float = Field(alias="gainDb")
    
    class Config:
        populate_by_name = True


# Default macros library
DEFAULT_MACROS = SfxMacros(
    version="1.0.0",
    macros=[
        SfxMacro(
            macro_id="soft_emphasis",
            description="Small accent on a key word, bullet, or highlight",
            default_gain_db=-16,
            gain_db_range=(-22, -10),
            candidates=[
                MacroCandidate(sfx_id="pop_ui_02", weight=0.55),
                MacroCandidate(sfx_id="ui_click_07", weight=0.30),
                MacroCandidate(sfx_id="tick_soft_01", weight=0.15),
            ],
        ),
        SfxMacro(
            macro_id="transition_fast",
            description="Quick section switch, zoom, swipe, topic change",
            default_gain_db=-12,
            gain_db_range=(-18, -6),
            candidates=[
                MacroCandidate(sfx_id="whoosh_short_02", weight=0.70),
                MacroCandidate(sfx_id="whoosh_fast_01", weight=0.30),
            ],
        ),
        SfxMacro(
            macro_id="reveal",
            description="Moment of surprise, 'here's the trick', a key reveal",
            default_gain_db=-12,
            gain_db_range=(-18, -6),
            candidates=[
                MacroCandidate(sfx_id="riser_short_01", weight=0.55),
                MacroCandidate(sfx_id="sparkle_01", weight=0.45),
            ],
        ),
        SfxMacro(
            macro_id="impact",
            description="Strong emphasis, punch, statement landing",
            default_gain_db=-10,
            gain_db_range=(-16, -4),
            candidates=[
                MacroCandidate(sfx_id="impact_soft_01", weight=0.60),
                MacroCandidate(sfx_id="thud_01", weight=0.40),
            ],
        ),
        SfxMacro(
            macro_id="success",
            description="Achievement, completion, positive outcome",
            default_gain_db=-14,
            gain_db_range=(-20, -8),
            candidates=[
                MacroCandidate(sfx_id="chime_success_01", weight=0.65),
                MacroCandidate(sfx_id="ding_positive_01", weight=0.35),
            ],
        ),
        SfxMacro(
            macro_id="tension",
            description="Building suspense, anticipation",
            default_gain_db=-14,
            gain_db_range=(-20, -8),
            candidates=[
                MacroCandidate(sfx_id="tension_build_01", weight=0.70),
                MacroCandidate(sfx_id="suspense_low_01", weight=0.30),
            ],
        ),
    ],
)


def clamp(n: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, n))


def weighted_pick(candidates: list[MacroCandidate], seed: float) -> MacroCandidate:
    """
    Deterministic weighted pick using seed.
    
    Args:
        candidates: List of candidates with weights
        seed: Seed value for deterministic selection
        
    Returns:
        Selected candidate
    """
    total = sum(c.weight for c in candidates) or 1
    r = (abs(math.sin(seed)) % 1) * total
    
    acc = 0
    for c in candidates:
        acc += c.weight
        if r <= acc:
            return c
    
    return candidates[-1]


def expand_macro_cue(
    cue: MacroCue,
    macros: SfxMacros,
    valid_sfx_ids: Optional[set[str]] = None,
) -> ExpandedCue:
    """
    Expand a single macro cue to an actual SFX cue.
    
    Args:
        cue: Macro cue
        macros: Macros library
        valid_sfx_ids: Optional set of valid SFX IDs for validation
        
    Returns:
        ExpandedCue with actual SFX ID
        
    Raises:
        ValueError: If macro or SFX ID is invalid
    """
    macro = macros.get_by_id(cue.macro_id)
    if not macro:
        raise ValueError(f"Unknown macroId: {cue.macro_id}")
    
    intensity = clamp(cue.intensity, 0, 1)
    
    # Deterministic pick based on time and macro
    seed = cue.t * 997 + len(cue.macro_id) * 13
    candidate = weighted_pick(macro.candidates, seed)
    
    if valid_sfx_ids and candidate.sfx_id not in valid_sfx_ids:
        raise ValueError(
            f'Macro "{macro.macro_id}" candidate references unknown sfxId: {candidate.sfx_id}'
        )
    
    # Gain interpolation: quieter at low intensity, louder at high
    min_db, max_db = macro.gain_db_range
    gain_db = min_db + (max_db - min_db) * intensity
    
    return ExpandedCue(
        t=cue.t,
        id=candidate.sfx_id,
        gain_db=gain_db,
    )


def expand_macro_cue_sheet(
    sheet: MacroCueSheet,
    macros: SfxMacros,
    valid_sfx_ids: Optional[set[str]] = None,
) -> list[ExpandedCue]:
    """
    Expand a macro cue sheet to actual SFX cues.
    
    Args:
        sheet: Macro cue sheet
        macros: Macros library
        valid_sfx_ids: Optional set of valid SFX IDs
        
    Returns:
        List of expanded cues
    """
    expanded = []
    
    for cue in sheet.cues:
        expanded.append(expand_macro_cue(cue, macros, valid_sfx_ids))
    
    return expanded


def load_macros(path: str) -> SfxMacros:
    """Load macros from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return SfxMacros.model_validate(data)


def save_macros(macros: SfxMacros, path: str) -> str:
    """Save macros to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(macros.model_dump(by_alias=True), f, indent=2)
    return path


def load_macro_cue_sheet(path: str) -> MacroCueSheet:
    """Load macro cue sheet from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return MacroCueSheet.model_validate(data)


def save_macro_cue_sheet(sheet: MacroCueSheet, path: str) -> str:
    """Save macro cue sheet to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(sheet.model_dump(by_alias=True), f, indent=2)
    return path


def validate_macro_cue_sheet(
    sheet: MacroCueSheet,
    macros: SfxMacros,
) -> list[str]:
    """
    Validate a macro cue sheet.
    
    Args:
        sheet: Macro cue sheet
        macros: Macros library
        
    Returns:
        List of validation error messages
    """
    errors = []
    valid_macro_ids = macros.get_all_ids()
    
    for i, cue in enumerate(sheet.cues):
        if cue.macro_id not in valid_macro_ids:
            errors.append(f"Cue {i}: unknown macroId '{cue.macro_id}'")
        
        if cue.t < 0:
            errors.append(f"Cue {i}: negative time {cue.t}")
        
        if cue.intensity < 0 or cue.intensity > 1:
            errors.append(f"Cue {i}: intensity {cue.intensity} out of range [0, 1]")
    
    return errors


def get_macro_context_for_ai(macros: SfxMacros) -> str:
    """
    Generate a context string for AI to understand available macros.
    
    Args:
        macros: Macros library
        
    Returns:
        Human-readable context string
    """
    lines = ["Available SFX Macros:"]
    
    for m in macros.macros:
        lines.append(f"\n- {m.macro_id}: {m.description}")
        lines.append(f"  Intensity range: 0 (quiet/subtle) to 1 (loud/dramatic)")
    
    lines.append("\nOutput format:")
    lines.append('{"macroId": "<macro>", "intensity": 0.0-1.0}')
    
    return "\n".join(lines)


def beats_to_macro_cues(
    beats: list[dict],
    fps: int = 30,
    default_intensity: float = 0.5,
) -> MacroCueSheet:
    """
    Convert beats with macro hints to a MacroCueSheet.
    
    Expects beats with 'frame' and 'sfx_macro' or 'macro' field.
    
    Args:
        beats: List of beat dicts
        fps: Frames per second
        default_intensity: Default intensity if not specified
        
    Returns:
        MacroCueSheet
    """
    cues = []
    
    for beat in beats:
        frame = beat.get("frame", 0)
        t = frame / fps
        
        # Look for macro hints
        macro_hints = (
            beat.get("sfx_macro") or 
            beat.get("macro") or 
            beat.get("audio", {}).get("macro")
        )
        
        if not macro_hints:
            continue
        
        if isinstance(macro_hints, str):
            macro_hints = [{"macroId": macro_hints}]
        elif isinstance(macro_hints, dict):
            macro_hints = [macro_hints]
        
        for hint in macro_hints:
            if isinstance(hint, str):
                macro_id = hint
                intensity = default_intensity
            else:
                macro_id = hint.get("macroId") or hint.get("macro_id")
                intensity = hint.get("intensity", default_intensity)
            
            if macro_id:
                cues.append(MacroCue(
                    t=t,
                    macro_id=macro_id,
                    intensity=intensity,
                ))
    
    return MacroCueSheet(cues=cues)
