"""
RemotionRenderSpec Service
Builds Remotion composition specs from deep audit data for video generation.
"""
import os
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List, Literal
from dataclasses import dataclass, field, asdict
from loguru import logger

from sqlalchemy import create_engine, text


BeatRole = Literal["hook", "problem", "solution", "proof", "cta", "transition", "intro", "outro", "other"]
TimelineLayerType = Literal["background_video", "broll", "image", "text_overlay", "captions", "sfx", "music", "shape", "logo", "transition"]


@dataclass
class CaptionSegment:
    start_sec: float
    end_sec: float
    text: str
    emphasis: List[str] = field(default_factory=list)


@dataclass
class Beat:
    beat_id: str
    start_sec: float
    end_sec: float
    role: BeatRole
    summary: str
    emotion: Optional[str] = None


@dataclass
class TimelineEvent:
    start_sec: float
    end_sec: float
    type: TimelineLayerType
    src: Optional[str] = None
    text: Optional[str] = None
    preset: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioSpec:
    narration_url: Optional[str] = None
    music_url: Optional[str] = None
    ducking: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CaptionsSpec:
    style_id: str
    segments: List[CaptionSegment] = field(default_factory=list)


@dataclass
class ExportSpec:
    format: str = "mp4"
    crf: int = 18
    audio_codec: str = "aac"


@dataclass
class RemotionRenderSpecV1:
    schema: str = "remotion_render_spec_v1"
    composition_id: str = "ShortFormV1"
    fps: int = 30
    width: int = 1080
    height: int = 1920
    duration_in_frames: int = 0
    
    audio: Optional[AudioSpec] = None
    captions: Optional[CaptionsSpec] = None
    beats: List[Beat] = field(default_factory=list)
    timeline: List[TimelineEvent] = field(default_factory=list)
    export: ExportSpec = field(default_factory=ExportSpec)


class RemotionSpecService:
    """
    Service for building Remotion render specs from deep audit data.
    """
    
    # Composition presets
    COMPOSITIONS = {
        "ShortFormV1": {
            "fps": 30,
            "width": 1080,
            "height": 1920,
            "aspect": "9:16"
        },
        "ShortFormHD": {
            "fps": 60,
            "width": 1080,
            "height": 1920,
            "aspect": "9:16"
        },
        "LongFormV1": {
            "fps": 30,
            "width": 1920,
            "height": 1080,
            "aspect": "16:9"
        },
        "SquareV1": {
            "fps": 30,
            "width": 1080,
            "height": 1080,
            "aspect": "1:1"
        }
    }
    
    # Caption style presets
    CAPTION_STYLES = {
        "CaptionStyleA": {
            "font": "Inter",
            "size": 48,
            "color": "#FFFFFF",
            "stroke": "#000000",
            "stroke_width": 2,
            "position": "bottom",
            "animation": "pop"
        },
        "CaptionStyleB": {
            "font": "Montserrat",
            "size": 56,
            "color": "#FFFFFF",
            "background": "rgba(0,0,0,0.7)",
            "position": "center",
            "animation": "fade"
        },
        "KaraokeSyle": {
            "font": "Inter",
            "size": 52,
            "color": "#FFFFFF",
            "highlight_color": "#FFD700",
            "position": "bottom",
            "animation": "highlight"
        }
    }
    
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv("DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:54322/postgres")
        self.engine = create_engine(self.db_url)
    
    def words_to_caption_segments(
        self, 
        words: List[Dict[str, Any]], 
        max_words_per_segment: int = 6
    ) -> List[CaptionSegment]:
        """
        Convert word-level timestamps to caption segments.
        
        Args:
            words: List of word objects with {w, start, end, conf?}
            max_words_per_segment: Maximum words per caption segment
        
        Returns:
            List of CaptionSegment objects
        """
        if not words:
            return []
        
        segments = []
        buffer = []
        
        def flush():
            nonlocal buffer
            if not buffer:
                return
            
            segment = CaptionSegment(
                start_sec=buffer[0].get("start", 0),
                end_sec=buffer[-1].get("end", buffer[-1].get("start", 0) + 0.5),
                text=" ".join(w.get("w", w.get("word", "")) for w in buffer)
            )
            segments.append(segment)
            buffer = []
        
        for word in words:
            buffer.append(word)
            if len(buffer) >= max_words_per_segment:
                flush()
        
        flush()  # Flush remaining words
        return segments
    
    def scene_structure_to_beats(
        self, 
        scene_structure: List[Dict[str, Any]]
    ) -> List[Beat]:
        """
        Convert scene structure from video analysis to Beat objects.
        """
        beats = []
        for i, scene in enumerate(scene_structure):
            beat = Beat(
                beat_id=scene.get("beat_id", f"b{i+1}"),
                start_sec=scene.get("start_sec", 0),
                end_sec=scene.get("end_sec", 0),
                role=scene.get("role", "other"),
                summary=scene.get("summary", ""),
                emotion=scene.get("emotion")
            )
            beats.append(beat)
        return beats
    
    def build_from_deep_audit(
        self,
        composition_id: str,
        duration_sec: float,
        deep_audit_data: Dict[str, Any],
        fps: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        narration_url: Optional[str] = None,
        music_url: Optional[str] = None,
        caption_style_id: str = "CaptionStyleA"
    ) -> RemotionRenderSpecV1:
        """
        Build Remotion render spec from deep audit data.
        
        Args:
            composition_id: Remotion composition ID (e.g., 'ShortFormV1')
            duration_sec: Video duration in seconds
            deep_audit_data: Deep audit data JSONB blob
            fps: Frames per second (uses composition default if not specified)
            width: Video width (uses composition default if not specified)
            height: Video height (uses composition default if not specified)
            narration_url: URL to narration audio
            music_url: URL to background music
            caption_style_id: Caption style preset ID
        
        Returns:
            Complete RemotionRenderSpecV1 object
        """
        # Get composition preset
        comp_preset = self.COMPOSITIONS.get(composition_id, self.COMPOSITIONS["ShortFormV1"])
        
        fps = fps or comp_preset["fps"]
        width = width or comp_preset["width"]
        height = height or comp_preset["height"]
        duration_in_frames = int(duration_sec * fps)
        
        # Extract transcript words for captions
        words = []
        transcript_data = deep_audit_data.get("transcript", {})
        if isinstance(transcript_data, dict):
            words = transcript_data.get("words", [])
        elif deep_audit_data.get("transcription_data"):
            words = deep_audit_data["transcription_data"].get("words", [])
        
        # Build captions from words
        captions = None
        if words:
            segments = self.words_to_caption_segments(words, max_words_per_segment=6)
            if segments:
                captions = CaptionsSpec(
                    style_id=caption_style_id,
                    segments=segments
                )
        
        # Build beats from scene structure
        beats = []
        scene_structure = deep_audit_data.get("scene_structure", [])
        if not scene_structure:
            scene_structure = deep_audit_data.get("beat_sheet", [])
        if scene_structure:
            beats = self.scene_structure_to_beats(scene_structure)
        
        # Build base timeline
        source_video_url = deep_audit_data.get("source_video_url", deep_audit_data.get("source_url"))
        timeline = self._build_timeline(
            duration_sec=duration_sec,
            source_video_url=source_video_url,
            has_captions=captions is not None,
            caption_style_id=caption_style_id,
            music_url=music_url
        )
        
        # Build audio spec
        audio = None
        if narration_url or music_url:
            audio = AudioSpec(
                narration_url=narration_url,
                music_url=music_url,
                ducking=[{"start_sec": 0, "end_sec": duration_sec, "amount_db": 10}] if narration_url and music_url else []
            )
        
        return RemotionRenderSpecV1(
            composition_id=composition_id,
            fps=fps,
            width=width,
            height=height,
            duration_in_frames=duration_in_frames,
            audio=audio,
            captions=captions,
            beats=beats,
            timeline=timeline,
            export=ExportSpec()
        )
    
    def _build_timeline(
        self,
        duration_sec: float,
        source_video_url: Optional[str] = None,
        has_captions: bool = False,
        caption_style_id: str = "CaptionStyleA",
        music_url: Optional[str] = None
    ) -> List[TimelineEvent]:
        """Build base timeline events"""
        timeline = []
        
        # Background video layer
        if source_video_url:
            timeline.append(TimelineEvent(
                start_sec=0,
                end_sec=duration_sec,
                type="background_video",
                src=source_video_url
            ))
        
        # Captions layer
        if has_captions:
            timeline.append(TimelineEvent(
                start_sec=0,
                end_sec=duration_sec,
                type="captions",
                preset=caption_style_id
            ))
        
        # Music layer
        if music_url:
            timeline.append(TimelineEvent(
                start_sec=0,
                end_sec=duration_sec,
                type="music",
                src=music_url,
                params={"gain_db": -12}
            ))
        
        return timeline
    
    def build_from_video_analysis(
        self,
        video_id: str,
        composition_id: str = "ShortFormV1",
        narration_url: Optional[str] = None,
        music_url: Optional[str] = None,
        caption_style_id: str = "CaptionStyleA"
    ) -> Optional[RemotionRenderSpecV1]:
        """
        Build Remotion spec from existing video analysis in database.
        
        Args:
            video_id: UUID of the video
            composition_id: Remotion composition ID
            narration_url: URL to narration audio
            music_url: URL to background music
            caption_style_id: Caption style preset
        
        Returns:
            RemotionRenderSpecV1 or None if video not found
        """
        try:
            with self.engine.connect() as conn:
                # Get video analysis
                result = conn.execute(text("""
                    SELECT 
                        va.transcription_data,
                        va.transcription_duration_sec,
                        va.scene_structure,
                        va.detected_hook,
                        va.topics,
                        va.tone,
                        va.music_suggestion,
                        v.duration,
                        v.file_path
                    FROM video_analysis va
                    JOIN videos v ON v.id = va.video_id
                    WHERE va.video_id = :video_id
                """), {"video_id": video_id})
                
                row = result.fetchone()
                if not row:
                    logger.warning(f"No analysis found for video {video_id}")
                    return None
                
                # Build deep audit data from analysis
                deep_audit_data = {
                    "transcription_data": row[0] or {},
                    "scene_structure": row[2] or [],
                    "detected_hook": row[3],
                    "topics": row[4] or [],
                    "tone": row[5],
                    "music_suggestion": row[6],
                    "source_video_url": row[8]  # file_path
                }
                
                duration_sec = float(row[1] or row[7] or 30)
                
                return self.build_from_deep_audit(
                    composition_id=composition_id,
                    duration_sec=duration_sec,
                    deep_audit_data=deep_audit_data,
                    narration_url=narration_url,
                    music_url=music_url,
                    caption_style_id=caption_style_id
                )
                
        except Exception as e:
            logger.error(f"Failed to build Remotion spec from video analysis: {e}")
            return None
    
    async def save_render_spec(
        self,
        spec: RemotionRenderSpecV1,
        asset_id: str,
        audit_id: Optional[str] = None
    ) -> str:
        """
        Save render spec to database.
        
        Returns:
            render_spec_id
        """
        render_spec_id = str(uuid.uuid4())
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO remotion_render_spec (
                        render_spec_id, asset_id, audit_id,
                        composition_id, fps, width, height, duration_in_frames,
                        spec, status
                    ) VALUES (
                        :render_spec_id, :asset_id, :audit_id,
                        :composition_id, :fps, :width, :height, :duration_in_frames,
                        :spec, 'draft'
                    )
                """), {
                    "render_spec_id": render_spec_id,
                    "asset_id": asset_id,
                    "audit_id": audit_id,
                    "composition_id": spec.composition_id,
                    "fps": spec.fps,
                    "width": spec.width,
                    "height": spec.height,
                    "duration_in_frames": spec.duration_in_frames,
                    "spec": json.dumps(asdict(spec))
                })
                conn.commit()
                logger.info(f"Saved render spec {render_spec_id}")
                
        except Exception as e:
            logger.error(f"Failed to save render spec: {e}")
            raise
        
        return render_spec_id
    
    def to_remotion_input_props(self, spec: RemotionRenderSpecV1) -> Dict[str, Any]:
        """
        Convert spec to Remotion inputProps format.
        This is what you pass to renderMedia() in Remotion.
        """
        return asdict(spec)
    
    def get_composition_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get available composition presets"""
        return self.COMPOSITIONS.copy()
    
    def get_caption_styles(self) -> Dict[str, Dict[str, Any]]:
        """Get available caption style presets"""
        return self.CAPTION_STYLES.copy()
