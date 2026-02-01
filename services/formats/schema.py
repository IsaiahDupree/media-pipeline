"""
Formats System Schema
Pydantic models for the parameterized video format system
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum
import uuid


# =============================================================================
# ENUMS
# =============================================================================

class FormatStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    FAILED = "failed"
    SUCCEEDED = "succeeded"
    PUBLISHED = "published"


class TriggerType(str, Enum):
    MANUAL = "manual"
    SCHEDULE = "schedule"
    WEBHOOK = "webhook"
    EVENT = "event"


class GateLevel(str, Enum):
    WARN = "warn"
    FAIL = "fail"


class GateType(str, Enum):
    REQUIRED_FIELDS = "required_fields"
    DURATION = "duration"
    CAPTIONS = "captions"
    AUDIO = "audio"
    VISUAL = "visual"
    PUBLISH = "publish"


# =============================================================================
# PROVIDER CONFIGS
# =============================================================================

class TTSProvider(BaseModel):
    provider: Literal["huggingface", "elevenlabs", "coqui", "openai"] = "huggingface"
    model: Optional[str] = None
    voice_id: Optional[str] = None


class MusicProvider(BaseModel):
    provider: Literal["suno", "soundcloud", "library", "local"] = "library"
    mood: Optional[str] = None
    genre: Optional[str] = None


class VisualsProvider(BaseModel):
    provider: Literal["local", "rapidapi", "mediaposter", "pexels"] = "local"
    library_id: Optional[str] = None


class MattingProvider(BaseModel):
    provider: Literal["rvm", "mediapipe"] = "rvm"


class ProviderConfig(BaseModel):
    tts: TTSProvider = Field(default_factory=TTSProvider)
    music: MusicProvider = Field(default_factory=MusicProvider)
    visuals: VisualsProvider = Field(default_factory=VisualsProvider)
    matting: Optional[MattingProvider] = None


# =============================================================================
# VARIANT SETS
# =============================================================================

class VariantSet(BaseModel):
    id: str
    label: str
    width: int = 1080
    height: int = 1920
    max_duration_sec: int = 60
    overrides: Optional[Dict[str, Any]] = None


# =============================================================================
# DATA SOURCES
# =============================================================================

class SupabaseQuerySource(BaseModel):
    id: str
    type: Literal["supabase_query"] = "supabase_query"
    query_name: str
    params: Optional[Dict[str, Any]] = None


class HttpApiSource(BaseModel):
    id: str
    type: Literal["http_api"] = "http_api"
    url: str
    method: Literal["GET", "POST"] = "GET"
    headers: Optional[Dict[str, str]] = None
    body_template: Optional[Dict[str, Any]] = None


class RssSource(BaseModel):
    id: str
    type: Literal["rss"] = "rss"
    url: str


class LocalLibrarySource(BaseModel):
    id: str
    type: Literal["local_library"] = "local_library"
    library_id: str
    filter: Optional[Dict[str, Any]] = None


DataSource = SupabaseQuerySource | HttpApiSource | RssSource | LocalLibrarySource


# =============================================================================
# BINDINGS & TRANSFORMS
# =============================================================================

class PickTransform(BaseModel):
    type: Literal["pick"] = "pick"
    path: str


class MapTransform(BaseModel):
    type: Literal["map"] = "map"
    map_template: Dict[str, Any]


class TemplateTransform(BaseModel):
    type: Literal["template"] = "template"
    template: str


class CoerceTransform(BaseModel):
    type: Literal["coerce"] = "coerce"
    to: Literal["string", "number", "boolean", "json"]


class DefaultTransform(BaseModel):
    type: Literal["default"] = "default"
    value: Any


Transform = PickTransform | MapTransform | TemplateTransform | CoerceTransform | DefaultTransform


class Binding(BaseModel):
    target: str  # dot path in render props
    from_path: str = Field(alias="from")  # dot path in resolved inputs
    transform: Optional[Transform] = None
    required: bool = False

    class Config:
        populate_by_name = True


# =============================================================================
# QUALITY GATES
# =============================================================================

class GateRule(BaseModel):
    id: str
    type: GateType
    level: GateLevel = GateLevel.WARN
    config: Dict[str, Any] = Field(default_factory=dict)


class QualityProfile(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    gates: List[GateRule] = Field(default_factory=list)
    is_default: bool = False


# =============================================================================
# COMPOSITION CONFIG
# =============================================================================

class CompositionConfig(BaseModel):
    remotion_composition_id: str
    fps: int = 30
    width: int = 1080
    height: int = 1920
    default_duration_sec: int = 55
    variant_sets: Optional[List[VariantSet]] = None


# =============================================================================
# FORMAT DEFAULTS
# =============================================================================

class FormatDefaults(BaseModel):
    params: Dict[str, Any] = Field(default_factory=dict)
    providers: ProviderConfig = Field(default_factory=ProviderConfig)
    quality_profile_id: str = "qp_shortform_v1"


# =============================================================================
# FORMAT DEFINITION (The heart of the system)
# =============================================================================

class FormatDefinition(BaseModel):
    """
    The complete definition of a video format.
    This is stored in formats.definition_json.
    """
    id: str
    name: str
    description: Optional[str] = None
    version: str = "1.0.0"
    status: FormatStatus = FormatStatus.DRAFT
    
    # Remotion composition config
    composition: CompositionConfig
    
    # Defaults for this format
    defaults: FormatDefaults = Field(default_factory=FormatDefaults)
    
    # Where data comes from
    data_sources: List[DataSource] = Field(default_factory=list)
    
    # How data maps to render props
    bindings: List[Binding] = Field(default_factory=list)
    
    # Additional quality gates specific to this format
    gates: Optional[List[GateRule]] = None


# =============================================================================
# RENDER PROPS (What Remotion receives)
# =============================================================================

class ScriptSegment(BaseModel):
    id: str
    t_start_sec: Optional[float] = None
    t_end_sec: Optional[float] = None
    text: str
    on_screen: Optional[List[str]] = None
    intent: Optional[Literal["hook", "explain", "joke", "cta"]] = None


class Script(BaseModel):
    title: Optional[str] = None
    hook: Optional[str] = None
    segments: List[ScriptSegment] = Field(default_factory=list)


class DuckingConfig(BaseModel):
    enabled: bool = True
    music_gain_db: float = -10
    attack_ms: int = 80
    release_ms: int = 180
    fallback_to_segments: bool = True


class AudioConfig(BaseModel):
    voice_url: Optional[str] = None
    music_url: Optional[str] = None
    ducking: Optional[DuckingConfig] = None


class MemeItem(BaseModel):
    id: str
    url: str
    tags: Optional[List[str]] = None


class BrollItem(BaseModel):
    id: str
    url: str


class UGCConfig(BaseModel):
    cutout_url: Optional[str] = None
    original_url: Optional[str] = None
    placement: Literal["bottom_left", "bottom_right"] = "bottom_left"


class VisualsConfig(BaseModel):
    memes: Optional[List[MemeItem]] = None
    broll: Optional[List[BrollItem]] = None
    ugc: Optional[UGCConfig] = None


class StyleConfig(BaseModel):
    caption_style: Literal["bold_pop", "clean_subs"] = "bold_pop"
    hook_intensity: float = 0.85
    pattern_interrupt_sec: float = 4.0
    zoom_punch: bool = True


class RenderMeta(BaseModel):
    run_id: str
    format_id: str
    variant_id: Optional[str] = None


class RenderProps(BaseModel):
    """
    The final props passed to Remotion for rendering.
    """
    topic: str = "Untitled"
    script: Script = Field(default_factory=Script)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    visuals: VisualsConfig = Field(default_factory=VisualsConfig)
    style: StyleConfig = Field(default_factory=StyleConfig)
    meta: RenderMeta


# =============================================================================
# FORMAT RUN MODELS
# =============================================================================

class FormatRunCreate(BaseModel):
    format_id: str
    params: Dict[str, Any] = Field(default_factory=dict)
    trigger_type: TriggerType = TriggerType.MANUAL
    triggered_by: Optional[str] = None
    variant_id: Optional[str] = None


class FormatRun(BaseModel):
    id: str
    format_id: str
    status: RunStatus
    trigger_type: TriggerType
    triggered_by: Optional[str] = None
    params_json: Dict[str, Any] = Field(default_factory=dict)
    resolved_inputs_json: Optional[Dict[str, Any]] = None
    render_props_json: Optional[Dict[str, Any]] = None
    variant_id: Optional[str] = None
    error_json: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class RunArtifact(BaseModel):
    id: str
    run_id: str
    kind: Literal["voice", "music", "timeline", "captions", "video", "thumbnail", "logs"]
    url: str
    file_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    duration_sec: Optional[float] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


# =============================================================================
# COMPILE RESULT
# =============================================================================

class VideoConfig(BaseModel):
    fps: int
    width: int
    height: int
    duration_in_frames: int


class CompileResult(BaseModel):
    resolved_inputs: Dict[str, Any]
    render_props: RenderProps
    video_config: VideoConfig


# =============================================================================
# GATE RESULT
# =============================================================================

class GateResult(BaseModel):
    gate_id: str
    level: GateLevel
    ok: bool
    message: str


class QualityGateResult(BaseModel):
    ok: bool
    results: List[GateResult]
