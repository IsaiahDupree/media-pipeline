"""
SFX Library Service

AI-addressable sound effects library with manifest validation,
context pack generation for LLMs, and auto-fix capabilities.
"""

from .types import (
    SfxItem,
    SfxManifest,
    AudioEvent,
    AudioEvents,
    SfxContextPack,
    FixReport,
    Beat,
    QATimelineReport,
)
from .manifest import (
    load_manifest,
    save_manifest,
    get_sfx_by_id,
    search_sfx_by_tags,
    get_categories,
    get_all_tags,
)
from .validator import (
    validate_audio_events,
    validate_and_fix_events,
    run_qa_gate,
    apply_anti_spam_filter,
)
from .context_pack import (
    build_sfx_context_pack,
    build_filtered_context_pack,
    make_sfx_selection_prompt,
    get_context_pack_stats,
)
from .autofix import (
    best_sfx_match,
    tokenize_text,
    suggest_sfx_for_action,
)
from .beat_extractor import (
    extract_beats_from_script,
    extract_beats_with_markers,
    ExtractedBeat,
    BeatExtractionResult,
)
from .audio_utils import (
    merge_audio_events,
    clamp_events_to_duration,
    snap_sfx_to_beats,
    thin_sfx_events,
    finalize_audio_events,
    get_sfx_density_stats,
)
from .cue_sheet import (
    CueSheet,
    SfxCue,
    audio_events_to_cue_sheet,
    beats_to_cue_sheet,
    save_cue_sheet,
    load_cue_sheet,
    validate_cue_sheet,
)
from .audio_mixer import (
    mix_audio_bus,
    mix_audio_bus_sync,
    mix_tracks,
    get_audio_duration,
    normalize_audio,
)
from .macros import (
    SfxMacro,
    SfxMacros,
    MacroCue,
    MacroCueSheet,
    ExpandedCue,
    DEFAULT_MACROS,
    expand_macro_cue_sheet,
    load_macros,
    save_macros,
    validate_macro_cue_sheet,
    get_macro_context_for_ai,
    beats_to_macro_cues,
)
from .visual_reveals import (
    VisualReveal,
    VisualRevealsFile,
    load_visual_reveals,
    save_visual_reveals,
    beats_to_visual_reveals,
    story_ir_to_visual_reveals,
    get_reveal_macro_mapping,
)
from .macro_policy import (
    PolicyConfig,
    BeatSec,
    plan_macro_cues_hybrid,
    thin_macro_cues,
    merge_ai_cues_with_policy,
    beats_frames_to_seconds,
)

__all__ = [
    # Types
    "SfxItem",
    "SfxManifest", 
    "AudioEvent",
    "AudioEvents",
    "SfxContextPack",
    "FixReport",
    "Beat",
    "QATimelineReport",
    "ExtractedBeat",
    "BeatExtractionResult",
    # Manifest
    "load_manifest",
    "save_manifest",
    "get_sfx_by_id",
    "search_sfx_by_tags",
    "get_categories",
    "get_all_tags",
    # Validator
    "validate_audio_events",
    "validate_and_fix_events",
    "run_qa_gate",
    "apply_anti_spam_filter",
    # Context Pack
    "build_sfx_context_pack",
    "build_filtered_context_pack",
    "make_sfx_selection_prompt",
    "get_context_pack_stats",
    # Autofix
    "best_sfx_match",
    "tokenize_text",
    "suggest_sfx_for_action",
    # Beat Extractor
    "extract_beats_from_script",
    "extract_beats_with_markers",
    # Audio Utils
    "merge_audio_events",
    "clamp_events_to_duration",
    "snap_sfx_to_beats",
    "thin_sfx_events",
    "finalize_audio_events",
    "get_sfx_density_stats",
    # Cue Sheet
    "CueSheet",
    "SfxCue",
    "audio_events_to_cue_sheet",
    "beats_to_cue_sheet",
    "save_cue_sheet",
    "load_cue_sheet",
    "validate_cue_sheet",
    # Audio Mixer
    "mix_audio_bus",
    "mix_audio_bus_sync",
    "mix_tracks",
    "get_audio_duration",
    "normalize_audio",
    # Macros
    "SfxMacro",
    "SfxMacros",
    "MacroCue",
    "MacroCueSheet",
    "ExpandedCue",
    "DEFAULT_MACROS",
    "expand_macro_cue_sheet",
    "load_macros",
    "save_macros",
    "validate_macro_cue_sheet",
    "get_macro_context_for_ai",
    "beats_to_macro_cues",
    # Visual Reveals
    "VisualReveal",
    "VisualRevealsFile",
    "load_visual_reveals",
    "save_visual_reveals",
    "beats_to_visual_reveals",
    "story_ir_to_visual_reveals",
    "get_reveal_macro_mapping",
    # Macro Policy
    "PolicyConfig",
    "BeatSec",
    "plan_macro_cues_hybrid",
    "thin_macro_cues",
    "merge_ai_cues_with_policy",
    "beats_frames_to_seconds",
]
