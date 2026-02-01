"""
Audio Event Utilities

Merge, clamp, snap, and thin audio events for timeline management.
"""

from typing import Optional
from .types import AudioEvents, SfxAudioEvent, SfxManifest


def merge_audio_events(
    base: AudioEvents,
    sfx_only: AudioEvents,
) -> AudioEvents:
    """
    Merge base audio events (voiceover/music) with SFX events.
    
    Args:
        base: Base events (typically voiceover + music)
        sfx_only: SFX-only events
        
    Returns:
        Merged AudioEvents sorted by frame
        
    Raises:
        ValueError: If FPS doesn't match
    """
    if base.fps != sfx_only.fps:
        raise ValueError(f"FPS mismatch: base={base.fps} sfx={sfx_only.fps}")
    
    # Merge and sort
    merged = [*base.events, *sfx_only.events]
    
    # Sort by frame, then by type priority (voiceover, music, sfx)
    def sort_key(ev):
        type_rank = {"voiceover": 0, "music": 1, "sfx": 2}
        return (ev.frame, type_rank.get(ev.type, 3))
    
    merged.sort(key=sort_key)
    
    return AudioEvents(fps=base.fps, events=merged)


def clamp_events_to_duration(
    events: AudioEvents,
    duration_in_frames: int,
    drop_if_beyond: bool = True,
) -> AudioEvents:
    """
    Clamp events to fit within a duration.
    
    Args:
        events: Audio events
        duration_in_frames: Max duration in frames
        drop_if_beyond: If True, drop events beyond duration; else clamp to last frame
        
    Returns:
        Clamped AudioEvents
    """
    clamped = []
    
    for ev in events.events:
        if ev.frame < 0:
            # Clamp negative to 0
            if isinstance(ev, SfxAudioEvent):
                clamped.append(SfxAudioEvent(
                    type="sfx",
                    sfx_id=ev.sfx_id,
                    frame=0,
                    volume=ev.volume,
                ))
            else:
                clamped.append(ev)
        elif ev.frame >= duration_in_frames:
            if drop_if_beyond:
                continue  # Drop
            else:
                # Clamp to last valid frame
                if isinstance(ev, SfxAudioEvent):
                    clamped.append(SfxAudioEvent(
                        type="sfx",
                        sfx_id=ev.sfx_id,
                        frame=max(0, duration_in_frames - 1),
                        volume=ev.volume,
                    ))
                else:
                    clamped.append(ev)
        else:
            clamped.append(ev)
    
    return AudioEvents(fps=events.fps, events=clamped)


def snap_sfx_to_beats(
    events: AudioEvents,
    beats: list[dict],
    max_snap_frames: int = 6,
    only_for_actions: Optional[list[str]] = None,
) -> AudioEvents:
    """
    Snap SFX events to nearest beat frame.
    
    Args:
        events: Audio events
        beats: List of beat dicts with 'frame' key
        max_snap_frames: Maximum frames to snap (default 6)
        only_for_actions: If set, only snap for these action types
        
    Returns:
        AudioEvents with snapped SFX
    """
    if not beats:
        return events
    
    # Sort beats by frame
    sorted_beats = sorted(beats, key=lambda b: b.get("frame", 0))
    beat_frames = [b.get("frame", 0) for b in sorted_beats]
    
    def find_nearest_beat(frame: int) -> tuple[int, int]:
        """Find nearest beat frame and distance."""
        best_frame = beat_frames[0]
        best_dist = abs(frame - best_frame)
        
        for bf in beat_frames:
            dist = abs(bf - frame)
            if dist < best_dist:
                best_dist = dist
                best_frame = bf
        
        return best_frame, best_dist
    
    snapped = []
    
    for ev in events.events:
        if not isinstance(ev, SfxAudioEvent):
            snapped.append(ev)
            continue
        
        nearest, dist = find_nearest_beat(ev.frame)
        
        if dist <= max_snap_frames:
            snapped.append(SfxAudioEvent(
                type="sfx",
                sfx_id=ev.sfx_id,
                frame=nearest,
                volume=ev.volume,
            ))
        else:
            snapped.append(ev)
    
    return AudioEvents(fps=events.fps, events=snapped)


def thin_sfx_events(
    events: AudioEvents,
    manifest: SfxManifest,
    min_seconds_between_sfx: float = 0.35,
    allow_burst_for_transitions: bool = True,
    max_sfx_per_second: int = 2,
) -> AudioEvents:
    """
    Thin out SFX events to prevent spam.
    
    Args:
        events: Audio events
        manifest: SFX manifest (for category lookup)
        min_seconds_between_sfx: Minimum gap between SFX
        allow_burst_for_transitions: Allow tighter spacing for transitions
        max_sfx_per_second: Hard ceiling per second
        
    Returns:
        Thinned AudioEvents
    """
    fps = events.fps
    min_frames = max(1, round(min_seconds_between_sfx * fps))
    
    # Build ID lookup
    id_to_item = {item.id: item for item in manifest.items}
    
    # Sort events by frame
    sorted_events = sorted(events.events, key=lambda e: e.frame)
    
    out = []
    last_sfx_frame = float("-inf")
    sfx_count_by_second: dict[int, int] = {}
    
    for ev in sorted_events:
        if not isinstance(ev, SfxAudioEvent):
            out.append(ev)
            continue
        
        item = id_to_item.get(ev.sfx_id)
        category = (item.category or "").lower() if item else ""
        tags = [t.lower() for t in (item.tags or [])] if item else []
        is_transition = category == "transition" or "transition" in tags
        
        second_bucket = ev.frame // fps
        count_this_second = sfx_count_by_second.get(second_bucket, 0)
        
        # Hard ceiling
        if count_this_second >= max_sfx_per_second:
            continue
        
        # Soft spacing rule
        too_close = ev.frame - last_sfx_frame < min_frames
        if too_close and not (allow_burst_for_transitions and is_transition):
            continue
        
        # De-dupe: same sfxId within 0.2s
        dedupe_frames = max(1, round(0.2 * fps))
        recent_same = any(
            isinstance(x, SfxAudioEvent) and 
            x.sfx_id == ev.sfx_id and 
            abs(x.frame - ev.frame) <= dedupe_frames
            for x in out[-6:]
        )
        if recent_same:
            continue
        
        out.append(ev)
        last_sfx_frame = ev.frame
        sfx_count_by_second[second_bucket] = count_this_second + 1
    
    return AudioEvents(fps=fps, events=out)


def finalize_audio_events(
    base: AudioEvents,
    sfx_only: AudioEvents,
    duration_in_frames: int,
    manifest: Optional[SfxManifest] = None,
    beats: Optional[list[dict]] = None,
    snap: bool = True,
    thin: bool = True,
) -> AudioEvents:
    """
    Finalize audio events: merge, snap, thin, and clamp.
    
    Args:
        base: Base events (voiceover/music)
        sfx_only: SFX events
        duration_in_frames: Total duration
        manifest: SFX manifest (for thinning)
        beats: Beat list (for snapping)
        snap: Whether to snap SFX to beats
        thin: Whether to thin SFX events
        
    Returns:
        Finalized AudioEvents
    """
    # Merge
    merged = merge_audio_events(base, sfx_only)
    
    # Snap
    if snap and beats:
        merged = snap_sfx_to_beats(merged, beats, max_snap_frames=6)
    
    # Thin
    if thin and manifest:
        merged = thin_sfx_events(
            merged,
            manifest,
            min_seconds_between_sfx=0.35,
            max_sfx_per_second=2,
        )
    
    # Clamp
    merged = clamp_events_to_duration(merged, duration_in_frames, drop_if_beyond=True)
    
    return merged


def get_sfx_density_stats(events: AudioEvents) -> dict:
    """
    Get statistics about SFX density.
    
    Args:
        events: Audio events
        
    Returns:
        Dict with density stats
    """
    sfx_events = events.get_sfx_events()
    
    if not sfx_events:
        return {
            "total_sfx": 0,
            "duration_seconds": 0,
            "sfx_per_second": 0,
            "max_per_second": 0,
            "busiest_second": None,
        }
    
    fps = events.fps
    sfx_frames = [e.frame for e in sfx_events]
    
    max_frame = max(sfx_frames)
    duration_seconds = max_frame / fps
    
    # Count per second
    count_by_second: dict[int, int] = {}
    for frame in sfx_frames:
        second = frame // fps
        count_by_second[second] = count_by_second.get(second, 0) + 1
    
    max_per_second = max(count_by_second.values()) if count_by_second else 0
    busiest_second = max(count_by_second, key=count_by_second.get) if count_by_second else None
    
    return {
        "total_sfx": len(sfx_events),
        "duration_seconds": duration_seconds,
        "sfx_per_second": len(sfx_events) / max(duration_seconds, 0.1),
        "max_per_second": max_per_second,
        "busiest_second": busiest_second,
    }
