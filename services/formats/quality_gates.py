"""
Quality Gates
Validates render props and artifacts against quality profiles
"""
import logging
from typing import Dict, Any, List, Optional

from .schema import (
    GateRule,
    GateLevel,
    GateResult,
    QualityGateResult,
    RenderProps,
    VideoConfig,
)
from .compiler import get_path

logger = logging.getLogger(__name__)


async def run_quality_gates(
    phase: str,  # "pre" or "post"
    format_def: Dict,
    quality_profile: Dict,
    render_props: Dict,
    video_config: Dict,
    artifacts: Dict = None
) -> QualityGateResult:
    """
    Run quality gates against render props and artifacts.
    
    Args:
        phase: "pre" (before render) or "post" (after render)
        format_def: The format definition
        quality_profile: Quality profile with gates
        render_props: Compiled render props
        video_config: Video configuration (fps, dimensions, duration)
        artifacts: Generated artifacts (voice, video, etc.)
    
    Returns:
        QualityGateResult with ok status and individual gate results
    """
    artifacts = artifacts or {}
    results: List[GateResult] = []
    
    # Combine gates from quality profile and format-specific gates
    profile_gates = quality_profile.get("gates") or quality_profile.get("gates_json", [])
    format_gates = format_def.get("gates", [])
    all_gates = profile_gates + format_gates
    
    for gate in all_gates:
        result = eval_gate(gate, render_props, video_config, artifacts)
        results.append(result)
        
        # Stop on first failure
        if not result.ok and result.level == GateLevel.FAIL:
            return QualityGateResult(ok=False, results=results)
    
    return QualityGateResult(ok=True, results=results)


def eval_gate(
    gate: Dict,
    render_props: Dict,
    video_config: Dict,
    artifacts: Dict
) -> GateResult:
    """Evaluate a single quality gate."""
    gate_id = gate.get("id", "unknown")
    gate_type = gate.get("type")
    level = GateLevel(gate.get("level", "warn"))
    config = gate.get("config", {})
    
    try:
        if gate_type == "required_fields":
            return gate_required_fields(gate_id, level, config, render_props)
        
        elif gate_type == "duration":
            return gate_duration(gate_id, level, config, video_config)
        
        elif gate_type == "captions":
            return gate_captions(gate_id, level, config, render_props)
        
        elif gate_type == "audio":
            return gate_audio_presence(gate_id, level, config, artifacts)
        
        elif gate_type == "visual":
            return gate_visual_density(gate_id, level, config, render_props)
        
        elif gate_type == "publish":
            # Placeholder for publish-time gates
            return GateResult(
                gate_id=gate_id,
                level=level,
                ok=True,
                message="publish gate placeholder"
            )
        
        else:
            return GateResult(
                gate_id=gate_id,
                level=level,
                ok=True,
                message=f"unknown gate type '{gate_type}' treated as pass"
            )
    
    except Exception as e:
        logger.error(f"Gate {gate_id} error: {e}")
        return GateResult(
            gate_id=gate_id,
            level=level,
            ok=False,
            message=f"gate error: {str(e)}"
        )


def gate_required_fields(
    gate_id: str,
    level: GateLevel,
    config: Dict,
    render_props: Dict
) -> GateResult:
    """Check that required fields are present."""
    required_paths = config.get("paths", [])
    
    if not required_paths:
        return GateResult(
            gate_id=gate_id,
            level=level,
            ok=True,
            message="no required paths configured"
        )
    
    missing = [path for path in required_paths if get_path(render_props, path) is None]
    ok = len(missing) == 0
    
    return GateResult(
        gate_id=gate_id,
        level=level,
        ok=ok,
        message="required fields present" if ok else f"missing: {', '.join(missing)}"
    )


def gate_duration(
    gate_id: str,
    level: GateLevel,
    config: Dict,
    video_config: Dict
) -> GateResult:
    """Check video duration is within limits."""
    max_sec = float(config.get("maxSec", config.get("max_sec", 60)))
    
    fps = video_config.get("fps", 30)
    duration_frames = video_config.get("duration_in_frames", video_config.get("durationInFrames", 0))
    duration_sec = duration_frames / fps if fps > 0 else 0
    
    ok = duration_sec <= max_sec
    
    return GateResult(
        gate_id=gate_id,
        level=level,
        ok=ok,
        message=f"duration {duration_sec:.2f}s ok" if ok else f"duration {duration_sec:.2f}s > {max_sec}s"
    )


def gate_captions(
    gate_id: str,
    level: GateLevel,
    config: Dict,
    render_props: Dict
) -> GateResult:
    """Check caption lengths are within limits."""
    max_chars = int(config.get("maxCharsPerLine", config.get("max_chars_per_line", 44)))
    
    segments = render_props.get("script", {}).get("segments", [])
    too_long = [s for s in segments if len(s.get("text", "")) > max_chars]
    
    ok = len(too_long) == 0
    
    if ok:
        return GateResult(gate_id=gate_id, level=level, ok=True, message="caption lengths ok")
    
    segment_ids = [s.get("id", "?") for s in too_long[:5]]
    suffix = "…" if len(too_long) > 5 else ""
    
    return GateResult(
        gate_id=gate_id,
        level=level,
        ok=False,
        message=f"segments too long: {', '.join(segment_ids)}{suffix}"
    )


def gate_audio_presence(
    gate_id: str,
    level: GateLevel,
    config: Dict,
    artifacts: Dict
) -> GateResult:
    """Check that required audio artifacts are present."""
    require_voice = config.get("requireVoice", config.get("require_voice", True))
    
    voice_url = (
        artifacts.get("voiceUrl") or 
        artifacts.get("voice_url") or 
        artifacts.get("voice", {}).get("url")
    )
    
    ok = not require_voice or bool(voice_url)
    
    return GateResult(
        gate_id=gate_id,
        level=level,
        ok=ok,
        message="voice present" if ok else "missing voice artifact"
    )


def gate_visual_density(
    gate_id: str,
    level: GateLevel,
    config: Dict,
    render_props: Dict
) -> GateResult:
    """Check that on-screen text density is within limits."""
    max_words = int(config.get("maxOnScreenWords", config.get("max_on_screen_words", 12)))
    
    segments = render_props.get("script", {}).get("segments", [])
    
    def word_count(seg):
        on_screen = seg.get("on_screen") or seg.get("onScreen", [])
        text = " ".join(on_screen).strip()
        return len(text.split()) if text else 0
    
    offenders = [s for s in segments if word_count(s) > max_words]
    ok = len(offenders) == 0
    
    if ok:
        return GateResult(gate_id=gate_id, level=level, ok=True, message="visual density ok")
    
    segment_ids = [s.get("id", "?") for s in offenders[:5]]
    suffix = "…" if len(offenders) > 5 else ""
    
    return GateResult(
        gate_id=gate_id,
        level=level,
        ok=False,
        message=f"too many on-screen words in: {', '.join(segment_ids)}{suffix}"
    )


def gate_hook_timing(
    gate_id: str,
    level: GateLevel,
    config: Dict,
    render_props: Dict
) -> GateResult:
    """Check that hook appears early enough."""
    hook_must_appear_by = float(config.get("hookMustAppearBySec", config.get("hook_must_appear_by_sec", 1.5)))
    
    segments = render_props.get("script", {}).get("segments", [])
    hook_segments = [s for s in segments if s.get("intent") == "hook"]
    
    if not hook_segments:
        return GateResult(
            gate_id=gate_id,
            level=level,
            ok=False,
            message="no hook segment found"
        )
    
    first_hook = hook_segments[0]
    hook_start = first_hook.get("t_start_sec") or first_hook.get("tStartSec", 0)
    
    ok = hook_start <= hook_must_appear_by
    
    return GateResult(
        gate_id=gate_id,
        level=level,
        ok=ok,
        message=f"hook at {hook_start:.2f}s ok" if ok else f"hook at {hook_start:.2f}s > {hook_must_appear_by}s"
    )
