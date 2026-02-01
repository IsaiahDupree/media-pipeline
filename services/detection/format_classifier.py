"""
Format Classifier Service
==========================
Classifies videos into content format categories based on analysis data.
Identifies candidates for b-roll + text overlays, talking head, etc.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import Video, VideoAnalysis

logger = logging.getLogger(__name__)


class VideoFormat(str, Enum):
    """Video format classifications"""
    # === UGC (Human-Created) Formats ===
    TALKING_HEAD = "talking_head"           # Person talking directly to camera
    VLOG = "vlog"                           # Vlog real-time style recording
    GREEN_SCREEN = "green_screen"           # Green screen background replacement
    LIPSYNC = "lipsync"                     # Lipsyncing to audio/music
    
    # === B-Roll Formats (Can be UGC or AI) ===
    BROLL_TEXT_CANDIDATE = "broll_text"     # B-roll suitable for text overlay (person not talking)
    PURE_BROLL = "pure_broll"               # No person, no speech - pure b-roll footage
    
    # === Audio-Based Formats ===
    VOICEOVER = "voiceover"                 # Voice but no visible person (VO style)
    MUSIC_ONLY = "music_only"               # Just background music, no speech
    SILENT = "silent"                       # No audio at all
    
    # === AI-Generated Formats ===
    ANIMATED = "animated"                   # AI-generated animated content
    AI_AVATAR = "ai_avatar"                 # AI avatar/digital human
    
    # === Static/Image Formats ===
    CAROUSEL = "carousel"                   # Multi-image carousel post
    STATIC_POST = "static_post"             # Single static image post
    STORY_IMAGE = "story_image"             # Story format image (9:16)
    
    # === Short-Form Video Formats ===
    STORY_VIDEO = "story_video"             # Story format video (9:16, <60s)
    REEL = "reel"                           # Instagram Reel / TikTok style
    
    # === Other ===
    MIXED = "mixed"                         # Mixed content type
    UNKNOWN = "unknown"                     # Not enough data to classify


# Format categorization for filtering
FORMAT_CATEGORIES = {
    "ugc": [
        VideoFormat.TALKING_HEAD,
        VideoFormat.VLOG,
        VideoFormat.GREEN_SCREEN,
        VideoFormat.LIPSYNC,
        VideoFormat.BROLL_TEXT_CANDIDATE,
        VideoFormat.PURE_BROLL,
        VideoFormat.VOICEOVER,
        VideoFormat.CAROUSEL,
        VideoFormat.STATIC_POST,
        VideoFormat.STORY_IMAGE,
        VideoFormat.STORY_VIDEO,
        VideoFormat.REEL,
    ],
    "ai_generated": [
        VideoFormat.ANIMATED,
        VideoFormat.AI_AVATAR,
        VideoFormat.BROLL_TEXT_CANDIDATE,  # Can be AI-enhanced
        VideoFormat.PURE_BROLL,            # Can be AI-generated
    ],
    "video": [
        VideoFormat.TALKING_HEAD,
        VideoFormat.VLOG,
        VideoFormat.GREEN_SCREEN,
        VideoFormat.LIPSYNC,
        VideoFormat.BROLL_TEXT_CANDIDATE,
        VideoFormat.PURE_BROLL,
        VideoFormat.VOICEOVER,
        VideoFormat.ANIMATED,
        VideoFormat.AI_AVATAR,
        VideoFormat.STORY_VIDEO,
        VideoFormat.REEL,
    ],
    "image": [
        VideoFormat.CAROUSEL,
        VideoFormat.STATIC_POST,
        VideoFormat.STORY_IMAGE,
    ],
}


@dataclass
class FormatClassification:
    """Result of format classification"""
    format: VideoFormat
    confidence: float
    reasons: List[str] = field(default_factory=list)
    
    # Detection flags
    has_person: bool = False
    person_is_talking: bool = False
    has_speech: bool = False
    has_music: bool = False
    has_captions: bool = False
    has_transcript: bool = False
    
    # Metrics
    face_presence_pct: float = 0.0
    speech_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format.value,
            "confidence": round(self.confidence, 2),
            "reasons": self.reasons,
            "has_person": self.has_person,
            "person_is_talking": self.person_is_talking,
            "has_speech": self.has_speech,
            "has_music": self.has_music,
            "has_captions": self.has_captions,
            "has_transcript": self.has_transcript,
            "face_presence_pct": round(self.face_presence_pct, 1),
            "speech_ratio": round(self.speech_ratio, 2),
        }


class FormatClassifier:
    """Classifies videos into content format categories"""
    
    def __init__(self, db: Optional[AsyncSession] = None):
        self.db = db
    
    def classify_from_analysis(
        self,
        has_face: bool = False,
        face_presence_pct: float = 0.0,
        has_speech: bool = False,
        speech_ratio: float = 0.0,
        has_music: bool = False,
        has_transcript: bool = False,
        transcript_length: int = 0,
        has_captions: bool = False,
    ) -> FormatClassification:
        """
        Classify video format based on analysis data.
        
        Args:
            has_face: Whether faces were detected in video
            face_presence_pct: Percentage of frames with faces (0-100)
            has_speech: Whether speech audio was detected
            speech_ratio: Ratio of speech to total audio (0-1)
            has_music: Whether background music was detected
            has_transcript: Whether a transcript exists
            transcript_length: Length of transcript in characters
            has_captions: Whether embedded captions exist
        
        Returns:
            FormatClassification with format type and metadata
        """
        reasons = []
        confidence = 0.5
        
        # Determine if person is talking (has face + has speech)
        person_is_talking = has_face and has_speech and speech_ratio > 0.3
        
        # Significant face presence (>30% of frames)
        significant_face = face_presence_pct > 30
        
        # Has meaningful transcript (>50 chars typically means actual speech)
        meaningful_transcript = has_transcript and transcript_length > 50
        
        # Classification logic
        if significant_face and person_is_talking:
            # TALKING HEAD: Person visible and speaking
            video_format = VideoFormat.TALKING_HEAD
            confidence = min(0.9, 0.5 + (face_presence_pct / 200) + (speech_ratio * 0.3))
            reasons.append(f"Face visible in {face_presence_pct:.0f}% of frames")
            reasons.append(f"Speech detected ({speech_ratio:.0%} of audio)")
            if meaningful_transcript:
                reasons.append("Has transcript content")
                
        elif significant_face and not has_speech:
            # B-ROLL + TEXT CANDIDATE: Person visible but NOT talking
            video_format = VideoFormat.BROLL_TEXT_CANDIDATE
            confidence = min(0.85, 0.5 + (face_presence_pct / 200))
            reasons.append(f"Person visible ({face_presence_pct:.0f}% of frames)")
            reasons.append("No speech detected - ideal for text overlay")
            if not has_captions:
                reasons.append("No existing captions")
                confidence += 0.1
                
        elif not has_face and not has_speech:
            # PURE B-ROLL: No person, no speech
            if has_music:
                video_format = VideoFormat.MUSIC_ONLY
                confidence = 0.8
                reasons.append("Music background only")
                reasons.append("No person visible")
                reasons.append("Perfect for text overlay")
            elif speech_ratio < 0.1:
                video_format = VideoFormat.PURE_BROLL
                confidence = 0.85
                reasons.append("No person detected")
                reasons.append("No speech detected")
                reasons.append("Ideal for adding text overlays")
            else:
                video_format = VideoFormat.SILENT
                confidence = 0.7
                reasons.append("Silent footage")
                
        elif not has_face and has_speech:
            # VOICEOVER: Voice but no visible person
            video_format = VideoFormat.VOICEOVER
            confidence = 0.75
            reasons.append("Speech detected but no visible person")
            reasons.append("Likely voiceover content")
            
        else:
            video_format = VideoFormat.MIXED
            confidence = 0.5
            reasons.append("Mixed content type")
        
        # Reduce confidence if missing key data
        if face_presence_pct == 0 and not has_face:
            confidence *= 0.8
            reasons.append("Limited visual analysis data")
        
        return FormatClassification(
            format=video_format,
            confidence=confidence,
            reasons=reasons,
            has_person=has_face or significant_face,
            person_is_talking=person_is_talking,
            has_speech=has_speech,
            has_music=has_music,
            has_captions=has_captions,
            has_transcript=has_transcript,
            face_presence_pct=face_presence_pct,
            speech_ratio=speech_ratio,
        )
    
    async def classify_video(self, media_id: str) -> FormatClassification:
        """Classify a video by its media_id using stored analysis data."""
        if not self.db:
            raise ValueError("Database session required for video classification")
        
        # Fetch video and analysis data
        query = select(Video).where(Video.id == media_id)
        result = await self.db.execute(query)
        video = result.scalar_one_or_none()
        
        if not video:
            return FormatClassification(
                format=VideoFormat.UNKNOWN,
                confidence=0,
                reasons=["Video not found"]
            )
        
        # Get analysis data
        analysis_query = select(VideoAnalysis).where(VideoAnalysis.video_id == media_id)
        analysis_result = await self.db.execute(analysis_query)
        analysis = analysis_result.scalar_one_or_none()
        
        # Extract relevant fields from analysis (transcript is in video_analysis, not video)
        transcript = ""
        visual_analysis = {}
        audio_analysis_data = {}
        
        if analysis:
            transcript = analysis.transcript or ""
            visual_analysis = analysis.visual_analysis or {}
            audio_analysis_data = analysis.audio_analysis or {}
        
        has_transcript = bool(transcript and len(transcript) > 10)
        transcript_length = len(transcript) if transcript else 0
        
        # Get audio analysis from analysis record or visual_analysis JSON
        audio_analysis = audio_analysis_data or visual_analysis.get('audio_analysis', {})
        
        has_speech = audio_analysis.get('has_speech', False) if isinstance(audio_analysis, dict) else False
        has_music = audio_analysis.get('has_music', False) if isinstance(audio_analysis, dict) else False
        speech_ratio = audio_analysis.get('speech_ratio', 0.0) if isinstance(audio_analysis, dict) else 0.0
        
        # Get face detection from analysis or visual_analysis
        has_face = False
        face_presence_pct = 0.0
        
        if analysis:
            # Check frame analysis for face detection
            frame_data = analysis.frame_analyses or []
            if frame_data:
                face_frames = sum(1 for f in frame_data if f.get('has_face', False))
                face_presence_pct = (face_frames / len(frame_data)) * 100
                has_face = face_presence_pct > 10
        
        # Also check visual_analysis for people_detected
        people_detected = visual_analysis.get('people_detected', [])
        if people_detected and not has_face:
            has_face = True
            face_presence_pct = max(face_presence_pct, 50)  # Assume 50% if detected but no frame data
        
        # Check for embedded captions (would be in visual analysis or separate field)
        has_captions = visual_analysis.get('has_embedded_captions', False)
        
        return self.classify_from_analysis(
            has_face=has_face,
            face_presence_pct=face_presence_pct,
            has_speech=has_speech,
            speech_ratio=speech_ratio,
            has_music=has_music,
            has_transcript=has_transcript,
            transcript_length=transcript_length,
            has_captions=has_captions,
        )
    
    async def find_broll_text_candidates(
        self,
        limit: int = 50,
        include_pure_broll: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find videos that are good candidates for b-roll + text overlay.
        
        Returns videos where:
        - Person is visible but NOT talking, OR
        - No person and no speech (pure b-roll)
        - No existing captions
        """
        if not self.db:
            raise ValueError("Database session required")
        
        # Query videos with analysis data
        query = select(Video).where(
            Video.duration_sec > 0  # Only videos
        ).order_by(Video.created_at.desc()).limit(limit * 3)  # Fetch more to filter
        
        result = await self.db.execute(query)
        videos = result.scalars().all()
        
        candidates = []
        
        for video in videos:
            classification = await self.classify_video(str(video.id))
            
            # Check if it's a b-roll candidate
            is_candidate = False
            if classification.format == VideoFormat.BROLL_TEXT_CANDIDATE:
                is_candidate = True
            elif include_pure_broll and classification.format in [
                VideoFormat.PURE_BROLL, 
                VideoFormat.MUSIC_ONLY,
                VideoFormat.SILENT
            ]:
                is_candidate = True
            
            if is_candidate and not classification.has_captions:
                candidates.append({
                    "media_id": str(video.id),
                    "filename": video.file_name,
                    "duration_sec": video.duration_sec,
                    "classification": classification.to_dict(),
                    "thumbnail_path": video.thumbnail_path,
                })
                
                if len(candidates) >= limit:
                    break
        
        return candidates


# Singleton instance
_classifier: Optional[FormatClassifier] = None


def get_classifier(db: Optional[AsyncSession] = None) -> FormatClassifier:
    """Get or create format classifier instance"""
    global _classifier
    if _classifier is None or db is not None:
        _classifier = FormatClassifier(db)
    return _classifier
