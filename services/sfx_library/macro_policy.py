"""
Macro Placement Policy

Automatically places SFX macro cues based on beats and visual reveals.
Supports hybrid format (explainer + dev vlog rhythms).
"""

from typing import Optional
from pydantic import BaseModel, Field

from .macros import MacroCue, MacroCueSheet
from .visual_reveals import VisualReveal, VisualRevealsFile, get_reveal_macro_mapping


class PolicyConfig(BaseModel):
    """Configuration for macro placement policy."""
    # Spacing rules
    min_gap_sec: float = Field(default=0.35, alias="minGapSec")
    min_gap_by_macro_sec: dict[str, float] = Field(
        default_factory=lambda: {
            "transition_fast": 0.20,
            "text_ping": 0.50,
            "impact_soft": 0.80,
        },
        alias="minGapByMacroSec"
    )
    
    # Limits
    max_cues_per_beat: int = Field(default=1, alias="maxCuesPerBeat")
    allow_double_on_transition: bool = Field(default=True, alias="allowDoubleOnTransition")
    
    # Auto transition detection
    transition_gap_sec: float = Field(default=2.5, alias="transitionGapSec")
    low_similarity_threshold: float = Field(default=0.25, alias="lowSimilarityThreshold")
    
    class Config:
        populate_by_name = True


class BeatSec(BaseModel):
    """Beat with timing in seconds."""
    beat_id: str = Field(alias="beatId")
    t: float
    text: str
    action: Optional[str] = None
    beat_type: Optional[str] = Field(None, alias="beatType")
    
    class Config:
        populate_by_name = True


DEFAULT_POLICY = PolicyConfig()


def tokenize(s: str) -> set[str]:
    """Tokenize text for similarity comparison."""
    return set(
        s.lower()
        .replace("/", " ")
        .replace("-", " ")
        .replace("_", " ")
        .split()
    )


def similarity(a: str, b: str) -> float:
    """Calculate Jaccard similarity between two strings."""
    A = tokenize(a)
    B = tokenize(b)
    if not A or not B:
        return 0
    intersection = len(A & B)
    union = len(A | B)
    return intersection / union if union else 0


def intensity_from_action(action: Optional[str]) -> float:
    """Get intensity based on beat action."""
    if action == "hook":
        return 0.75
    if action == "transition":
        return 0.70
    if action == "punchline":
        return 0.80
    if action == "cta":
        return 0.70
    if action == "reveal":
        return 0.75
    if action == "error":
        return 0.65
    if action == "success":
        return 0.70
    return 0.55


def macro_for_action(action: Optional[str]) -> Optional[str]:
    """Get macro ID for a beat action."""
    mapping = {
        "hook": "impact_soft",
        "transition": "transition_fast",
        "punchline": "micro_joke_hit",
        "cta": "cta_sparkle",
        "reveal": "reveal_riser",
        "error": "warning_buzz_soft",
        "success": "success_ding",
    }
    return mapping.get(action) if action else None


def macro_for_beat_type(beat_type: Optional[str]) -> Optional[str]:
    """Get macro ID for a beat type."""
    mapping = {
        "HOOK": "impact_soft",
        "CTA": "cta_sparkle",
        "PROOF": "reveal_riser",
    }
    return mapping.get(beat_type) if beat_type else None


def plan_macro_cues_hybrid(
    beats: list[BeatSec],
    reveals: Optional[VisualRevealsFile] = None,
    config: Optional[PolicyConfig] = None,
) -> MacroCueSheet:
    """
    Plan macro cues for hybrid format (explainer + dev vlog).
    
    Args:
        beats: List of beats with timing
        reveals: Optional visual reveals
        config: Policy configuration
        
    Returns:
        MacroCueSheet with planned cues
    """
    cfg = config or DEFAULT_POLICY
    reveal_macro_map = get_reveal_macro_mapping()
    
    cues: list[MacroCue] = []
    last_cue_t: dict[str, float] = {}  # Track last cue time per macro
    
    def can_place_cue(t: float, macro_id: str) -> bool:
        """Check if we can place a cue at this time."""
        # Global minimum gap
        for prev_t in last_cue_t.values():
            if abs(t - prev_t) < cfg.min_gap_sec:
                return False
        
        # Per-macro cooldown
        cooldown = cfg.min_gap_by_macro_sec.get(macro_id, cfg.min_gap_sec)
        prev = last_cue_t.get(macro_id, -999)
        return (t - prev) >= cooldown
    
    def add_cue(t: float, macro_id: str, intensity: float, reason: str = ""):
        """Add a cue if spacing allows."""
        if can_place_cue(t, macro_id):
            cues.append(MacroCue(t=t, macro_id=macro_id, intensity=intensity))
            last_cue_t[macro_id] = t
    
    # 1) Add cues from beats
    for i, beat in enumerate(beats):
        cues_this_beat = 0
        max_cues = cfg.max_cues_per_beat
        
        # Check for transitions (gap or content change)
        if i > 0:
            prev = beats[i - 1]
            gap = beat.t - (prev.t + 1)  # Assume ~1s per beat minimum
            
            is_transition = (
                beat.action == "transition" or
                gap >= cfg.transition_gap_sec or
                similarity(prev.text, beat.text) < cfg.low_similarity_threshold
            )
            
            if is_transition:
                add_cue(beat.t, "transition_fast", 0.65, "auto_transition")
                cues_this_beat += 1
                if cfg.allow_double_on_transition:
                    max_cues = 2
        
        # Add cue from action
        if cues_this_beat < max_cues:
            action_macro = macro_for_action(beat.action)
            if action_macro:
                add_cue(beat.t, action_macro, intensity_from_action(beat.action), f"action:{beat.action}")
                cues_this_beat += 1
        
        # Add cue from beat type
        if cues_this_beat < max_cues:
            type_macro = macro_for_beat_type(beat.beat_type)
            if type_macro and type_macro not in [c.macro_id for c in cues if abs(c.t - beat.t) < 0.1]:
                add_cue(beat.t + 0.1, type_macro, 0.60, f"type:{beat.beat_type}")
                cues_this_beat += 1
    
    # 2) Add cues from visual reveals
    if reveals:
        for reveal in reveals.reveals:
            macro_id = reveal_macro_map.get(reveal.kind)
            if not macro_id:
                continue
            
            # Check if we already have a cue near this time
            existing_near = any(abs(c.t - reveal.t) < 0.2 for c in cues)
            if existing_near:
                continue
            
            intensity = 0.55 if reveal.kind in ["keyword", "bullet"] else 0.65
            add_cue(reveal.t, macro_id, intensity, f"reveal:{reveal.kind}")
    
    # Sort by time
    cues.sort(key=lambda c: c.t)
    
    return MacroCueSheet(cues=cues)


def thin_macro_cues(
    sheet: MacroCueSheet,
    min_gap_sec: float = 0.35,
    max_total: Optional[int] = None,
) -> MacroCueSheet:
    """
    Thin out macro cues to prevent overwhelming audio.
    
    Args:
        sheet: Input cue sheet
        min_gap_sec: Minimum gap between any cues
        max_total: Maximum total cues
        
    Returns:
        Thinned MacroCueSheet
    """
    if not sheet.cues:
        return sheet
    
    # Sort by time
    sorted_cues = sorted(sheet.cues, key=lambda c: c.t)
    
    # Remove cues that are too close
    kept: list[MacroCue] = []
    for cue in sorted_cues:
        if not kept or (cue.t - kept[-1].t) >= min_gap_sec:
            kept.append(cue)
    
    # Limit total if specified
    if max_total and len(kept) > max_total:
        # Keep highest intensity cues
        kept.sort(key=lambda c: -c.intensity)
        kept = kept[:max_total]
        kept.sort(key=lambda c: c.t)
    
    return MacroCueSheet(
        version=sheet.version,
        sample_rate=sheet.sample_rate,
        cues=kept,
    )


def merge_ai_cues_with_policy(
    policy_cues: MacroCueSheet,
    ai_cues: MacroCueSheet,
    min_gap_sec: float = 0.35,
) -> MacroCueSheet:
    """
    Merge AI-generated cues with policy-generated cues.
    
    Policy cues take precedence (they're deterministic).
    AI cues fill gaps.
    
    Args:
        policy_cues: Cues from placement policy
        ai_cues: Cues from AI
        min_gap_sec: Minimum gap between cues
        
    Returns:
        Merged MacroCueSheet
    """
    # Start with policy cues
    merged = list(policy_cues.cues)
    policy_times = {c.t for c in policy_cues.cues}
    
    # Add AI cues that don't conflict
    for ai_cue in ai_cues.cues:
        # Check if too close to any existing cue
        too_close = any(abs(ai_cue.t - t) < min_gap_sec for t in policy_times)
        if not too_close:
            merged.append(ai_cue)
            policy_times.add(ai_cue.t)
    
    # Sort by time
    merged.sort(key=lambda c: c.t)
    
    return MacroCueSheet(
        version=policy_cues.version,
        sample_rate=policy_cues.sample_rate,
        cues=merged,
    )


def beats_frames_to_seconds(
    beats: list[dict],
    fps: int,
) -> list[BeatSec]:
    """
    Convert beats with frame timing to seconds.
    
    Args:
        beats: List of beat dicts with 'frame' key
        fps: Frames per second
        
    Returns:
        List of BeatSec
    """
    return [
        BeatSec(
            beat_id=b.get("beatId") or b.get("beat_id") or b.get("id", str(i)),
            t=round(b.get("frame", 0) / fps, 3),
            text=b.get("text") or b.get("narration", ""),
            action=b.get("action"),
            beat_type=b.get("type") or b.get("beatType"),
        )
        for i, b in enumerate(beats)
    ]
