"""
Beat Extractor

Extracts narrative beats from script text with frame timing
based on word count and speech rate (WPM).
"""

import re
from typing import Optional, Literal
from pydantic import BaseModel, Field


BeatAction = Literal["hook", "reveal", "transition", "punchline", "cta", "explain"]


class ExtractedBeat(BaseModel):
    """A beat extracted from script text."""
    beat_id: str = Field(alias="beatId")
    frame: int
    text: str
    action: Optional[BeatAction] = None
    word_count: int = Field(alias="wordCount")
    duration_frames: int = Field(alias="durationFrames")
    
    class Config:
        populate_by_name = True


class BeatExtractionResult(BaseModel):
    """Result of beat extraction."""
    beats: list[ExtractedBeat]
    estimated_total_frames: int = Field(alias="estimatedTotalFrames")
    fps: int
    wpm: int
    
    class Config:
        populate_by_name = True


def slug(text: str) -> str:
    """Create a URL-safe slug from text."""
    s = text.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"^_+|_+$", "", s)
    return s[:40]


def detect_action(text: str) -> Optional[BeatAction]:
    """
    Detect the action type of a beat based on text content.
    
    Args:
        text: The beat text
        
    Returns:
        Detected action type or None
    """
    s = text.lower()
    
    # CTA patterns
    if re.search(r"(subscribe|follow|download|join|link in bio|waitlist|comment|save this|try it)", s):
        return "cta"
    
    # Hook patterns
    if re.search(r"(but here's the thing|plot twist|you won't believe|here's why|the truth|most people|secret|nobody)", s):
        return "hook"
    
    # Transition patterns
    if re.search(r"(so next|now let's|moving on|meanwhile|then|next up|switch to|let's move)", s):
        return "transition"
    
    # Punchline patterns
    if re.search(r"(boom|gotcha|and that's it|that's the trick|mic drop|done|that's all)", s):
        return "punchline"
    
    # Explain patterns
    if re.search(r"(here's how|step|first|second|third|do this|use this|the way to|you need to)", s):
        return "explain"
    
    return "reveal"


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def extract_beats_from_script(
    script: str,
    fps: int,
    wpm: int = 165,
    start_frame: int = 0,
    min_beat_words: int = 6,
) -> BeatExtractionResult:
    """
    Extract narrative beats from a script with frame timing.
    
    Args:
        script: The script text
        fps: Frames per second
        wpm: Words per minute speaking rate (default 165)
        start_frame: Starting frame number (default 0)
        min_beat_words: Minimum words per beat (default 6)
        
    Returns:
        BeatExtractionResult with beats and timing
    """
    words_per_second = wpm / 60
    frames_per_word = fps / words_per_second
    
    # Split into raw lines
    raw_lines = [line.strip() for line in script.split("\n") if line.strip()]
    
    # Process lines - handle labeled lines and split long sentences
    chunks: list[str] = []
    
    for line in raw_lines:
        # Check for explicit labels like "Hook:", "Step 1:", "CTA:"
        has_label = re.match(r"^[A-Za-z ]{2,12}:\s+", line)
        if has_label:
            # Remove the label, keep the content
            content = re.sub(r"^[A-Za-z ]{2,12}:\s+", "", line).strip()
            if content:
                chunks.append(content)
            continue
        
        # Split long lines into sentence-ish beats
        parts = re.split(r"(?<=[.!?])\s+", line)
        for part in parts:
            if part.strip():
                chunks.append(part.strip())
    
    # Merge tiny chunks to avoid micro-beats
    merged: list[str] = []
    buffer: list[str] = []
    buffer_words = 0
    
    for chunk in chunks:
        word_count = count_words(chunk)
        
        if buffer_words < min_beat_words:
            buffer.append(chunk)
            buffer_words += word_count
            continue
        
        # Flush buffer
        merged.append(" ".join(buffer).strip())
        buffer = [chunk]
        buffer_words = word_count
    
    # Don't forget remaining buffer
    if buffer:
        merged.append(" ".join(buffer).strip())
    
    # Assign frames based on word count
    beats: list[ExtractedBeat] = []
    frame_cursor = start_frame
    
    for i, text in enumerate(merged):
        wc = count_words(text)
        duration_frames = max(1, round(wc * frames_per_word))
        action = detect_action(text)
        beat_id = f"{i + 1}_{slug(text)}"
        
        beats.append(ExtractedBeat(
            beat_id=beat_id,
            frame=frame_cursor,
            text=text,
            action=action,
            word_count=wc,
            duration_frames=duration_frames,
        ))
        
        frame_cursor += duration_frames
    
    return BeatExtractionResult(
        beats=beats,
        estimated_total_frames=frame_cursor,
        fps=fps,
        wpm=wpm,
    )


def extract_beats_with_markers(
    script: str,
    fps: int,
    wpm: int = 165,
) -> BeatExtractionResult:
    """
    Extract beats from a script with explicit markers.
    
    Expects format like:
    [HOOK] This is the hook text.
    [STEP] Step one explanation.
    [CTA] Save this and try it!
    
    Args:
        script: Script with [MARKER] format
        fps: Frames per second
        wpm: Words per minute
        
    Returns:
        BeatExtractionResult
    """
    words_per_second = wpm / 60
    frames_per_word = fps / words_per_second
    
    # Find all [MARKER] text patterns
    pattern = r"\[([A-Z_]+)\]\s*(.+?)(?=\[|$)"
    matches = re.findall(pattern, script, re.DOTALL)
    
    beats: list[ExtractedBeat] = []
    frame_cursor = 0
    
    marker_to_action: dict[str, BeatAction] = {
        "HOOK": "hook",
        "CTA": "cta",
        "STEP": "explain",
        "TRANSITION": "transition",
        "PUNCHLINE": "punchline",
        "REVEAL": "reveal",
        "EXPLAIN": "explain",
    }
    
    for i, (marker, text) in enumerate(matches):
        text = text.strip()
        if not text:
            continue
        
        wc = count_words(text)
        duration_frames = max(1, round(wc * frames_per_word))
        action = marker_to_action.get(marker.upper(), detect_action(text))
        beat_id = f"{i + 1}_{marker.lower()}_{slug(text)}"
        
        beats.append(ExtractedBeat(
            beat_id=beat_id,
            frame=frame_cursor,
            text=text,
            action=action,
            word_count=wc,
            duration_frames=duration_frames,
        ))
        
        frame_cursor += duration_frames
    
    return BeatExtractionResult(
        beats=beats,
        estimated_total_frames=frame_cursor,
        fps=fps,
        wpm=wpm,
    )


def adjust_beats_to_duration(
    result: BeatExtractionResult,
    target_duration_frames: int,
) -> BeatExtractionResult:
    """
    Adjust beat durations to fit a target duration.
    
    Args:
        result: Original extraction result
        target_duration_frames: Target total frames
        
    Returns:
        Adjusted BeatExtractionResult
    """
    if not result.beats:
        return result
    
    current_total = result.estimated_total_frames
    if current_total == 0:
        return result
    
    scale_factor = target_duration_frames / current_total
    
    new_beats: list[ExtractedBeat] = []
    frame_cursor = 0
    
    for beat in result.beats:
        new_duration = max(1, round(beat.duration_frames * scale_factor))
        
        new_beats.append(ExtractedBeat(
            beat_id=beat.beat_id,
            frame=frame_cursor,
            text=beat.text,
            action=beat.action,
            word_count=beat.word_count,
            duration_frames=new_duration,
        ))
        
        frame_cursor += new_duration
    
    return BeatExtractionResult(
        beats=new_beats,
        estimated_total_frames=frame_cursor,
        fps=result.fps,
        wpm=result.wpm,
    )


def beats_to_sfx_input(result: BeatExtractionResult) -> list[dict]:
    """
    Convert extracted beats to SFX library input format.
    
    Args:
        result: Beat extraction result
        
    Returns:
        List of beat dicts for SFX selection
    """
    return [
        {
            "beatId": beat.beat_id,
            "frame": beat.frame,
            "text": beat.text,
            "action": beat.action,
        }
        for beat in result.beats
    ]
