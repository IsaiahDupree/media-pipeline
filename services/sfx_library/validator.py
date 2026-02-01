"""
SFX Event Validator

Validates audio events against the manifest and optionally auto-fixes
hallucinated IDs using the autofix module.
"""

from typing import Optional

from .types import (
    SfxManifest,
    AudioEvents,
    SfxAudioEvent,
    FixReport,
    FixedEvent,
    RejectedEvent,
    QATimelineReport,
    QATimelineIssue,
)
from .autofix import best_sfx_match


def validate_audio_events(
    events: AudioEvents,
    manifest: SfxManifest,
) -> list[str]:
    """
    Validate that all SFX IDs in events exist in the manifest.
    
    Args:
        events: The audio events to validate
        manifest: The SFX manifest
        
    Returns:
        List of invalid SFX IDs
    """
    valid_ids = manifest.get_id_set()
    invalid = []
    
    for event in events.events:
        if isinstance(event, SfxAudioEvent):
            if event.sfx_id not in valid_ids:
                invalid.append(event.sfx_id)
    
    return invalid


def validate_and_fix_events(
    events: AudioEvents,
    manifest: SfxManifest,
    allow_auto_fix: bool = True,
    min_match_score: int = 1,
) -> tuple[AudioEvents, FixReport]:
    """
    Validate audio events and optionally auto-fix invalid SFX IDs.
    
    Args:
        events: The audio events to validate
        manifest: The SFX manifest
        allow_auto_fix: Whether to attempt auto-fixing invalid IDs
        min_match_score: Minimum score for auto-fix matches
        
    Returns:
        Tuple of (cleaned AudioEvents, FixReport)
    """
    valid_ids = manifest.get_id_set()
    report = FixReport(fixed=[], rejected=[])
    cleaned_events = []
    
    for event in events.events:
        # Non-SFX events pass through unchanged
        if not isinstance(event, SfxAudioEvent):
            cleaned_events.append(event)
            continue
        
        # Valid SFX ID - keep as-is
        if event.sfx_id in valid_ids:
            cleaned_events.append(event)
            continue
        
        # Invalid ID - try to fix or reject
        if not allow_auto_fix:
            report.rejected.append(RejectedEvent(
                sfx_id=event.sfx_id,
                frame=event.frame,
                reason="Unknown sfxId"
            ))
            continue
        
        # Attempt auto-fix
        match = best_sfx_match(
            manifest=manifest,
            requested_id_or_hint=event.sfx_id,
            min_score=min_match_score,
        )
        
        if not match:
            report.rejected.append(RejectedEvent(
                sfx_id=event.sfx_id,
                frame=event.frame,
                reason="No reasonable match found"
            ))
            continue
        
        # Auto-fix successful
        report.fixed.append(FixedEvent(
            from_id=event.sfx_id,
            to_id=match["id"],
            frame=event.frame,
            reason=f"Auto-mapped by tag/desc similarity (score={match['score']})"
        ))
        
        # Create fixed event
        fixed_event = SfxAudioEvent(
            type="sfx",
            sfx_id=match["id"],
            frame=event.frame,
            volume=event.volume,
        )
        cleaned_events.append(fixed_event)
    
    cleaned = AudioEvents(fps=events.fps, events=cleaned_events)
    return cleaned, report


def run_qa_gate(
    events: AudioEvents,
    manifest: SfxManifest,
    max_sfx_per_5_seconds: int = 8,
    min_gap_frames: int = 5,
    max_total_sfx: int = 50,
) -> QATimelineReport:
    """
    Run QA checks on audio events timeline.
    
    Checks:
    - Invalid SFX IDs
    - SFX density (max per 5 seconds)
    - Minimum gap between SFX
    - Total SFX count
    
    Args:
        events: The audio events
        manifest: The SFX manifest
        max_sfx_per_5_seconds: Maximum SFX in any 5-second window
        min_gap_frames: Minimum frames between SFX
        max_total_sfx: Maximum total SFX events
        
    Returns:
        QATimelineReport with pass/fail status and issues
    """
    issues: list[QATimelineIssue] = []
    valid_ids = manifest.get_id_set()
    
    sfx_events = events.get_sfx_events()
    sfx_frames = sorted([e.frame for e in sfx_events])
    
    # Check: Invalid IDs
    for event in sfx_events:
        if event.sfx_id not in valid_ids:
            issues.append(QATimelineIssue(
                code="INVALID_SFX_ID",
                level="error",
                message=f"Unknown sfxId: {event.sfx_id}",
                frame=event.frame,
            ))
    
    # Check: Total count
    if len(sfx_events) > max_total_sfx:
        issues.append(QATimelineIssue(
            code="TOO_MANY_SFX",
            level="warn",
            message=f"Total SFX count ({len(sfx_events)}) exceeds recommended max ({max_total_sfx})",
        ))
    
    # Check: Density (SFX per 5-second window)
    if sfx_frames:
        window_frames = 5 * events.fps  # 5 seconds in frames
        for start_frame in range(0, max(sfx_frames) + 1, events.fps):
            end_frame = start_frame + window_frames
            count = sum(1 for f in sfx_frames if start_frame <= f < end_frame)
            if count > max_sfx_per_5_seconds:
                issues.append(QATimelineIssue(
                    code="SFX_DENSITY_HIGH",
                    level="warn",
                    message=f"Dense SFX zone: {count} effects in frames {start_frame}-{end_frame}",
                    frame=start_frame,
                ))
    
    # Check: Minimum gap
    for i in range(1, len(sfx_frames)):
        gap = sfx_frames[i] - sfx_frames[i - 1]
        if gap < min_gap_frames:
            issues.append(QATimelineIssue(
                code="SFX_GAP_TOO_SMALL",
                level="warn",
                message=f"SFX too close: {gap} frames apart (min: {min_gap_frames})",
                frame=sfx_frames[i],
            ))
    
    # Determine pass/fail
    has_errors = any(i.level == "error" for i in issues)
    
    # Stats
    stats = {
        "total_sfx": len(sfx_events),
        "total_music": len(events.get_music_events()),
        "total_voiceover": len(events.get_voiceover_events()),
        "fps": events.fps,
        "duration_frames": max(sfx_frames) if sfx_frames else 0,
    }
    
    return QATimelineReport(
        passed=not has_errors,
        issues=issues,
        stats=stats,
    )


def apply_anti_spam_filter(
    events: AudioEvents,
    max_per_window: int = 3,
    window_frames: int = 150,  # 5 seconds at 30fps
) -> AudioEvents:
    """
    Remove excess SFX that exceed density limits.
    
    Keeps the first N SFX per window, removes the rest.
    
    Args:
        events: The audio events
        max_per_window: Maximum SFX per window
        window_frames: Window size in frames
        
    Returns:
        Filtered AudioEvents
    """
    sfx_events = events.get_sfx_events()
    non_sfx_events = [e for e in events.events if not isinstance(e, SfxAudioEvent)]
    
    # Sort SFX by frame
    sfx_events_sorted = sorted(sfx_events, key=lambda e: e.frame)
    
    # Track which SFX to keep
    kept_sfx = []
    
    for sfx in sfx_events_sorted:
        # Count how many SFX are in this window already
        window_start = max(0, sfx.frame - window_frames // 2)
        window_end = sfx.frame + window_frames // 2
        
        count_in_window = sum(
            1 for k in kept_sfx
            if window_start <= k.frame <= window_end
        )
        
        if count_in_window < max_per_window:
            kept_sfx.append(sfx)
    
    # Combine and return
    all_events = non_sfx_events + kept_sfx
    all_events.sort(key=lambda e: e.frame)
    
    return AudioEvents(fps=events.fps, events=all_events)
