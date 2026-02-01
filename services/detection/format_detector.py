"""
Format Detector Service

Comprehensive content format detection that classifies videos into categories:

PRIMARY FORMATS:
- talking_head: Person speaking directly to camera (vlog, tutorial, commentary)
- interview: Two or more people in conversation
- broll_scenic: B-roll footage of landscapes, environments, no people
- broll_action: B-roll footage with movement/action, may have people not speaking
- broll_people: B-roll with people present but not speaking to camera
- animated: Animation, motion graphics, screen recordings with graphics
- screen_recording: Screen capture, software demo, gameplay
- slideshow: Static images with transitions
- music_video: Music-focused content with visuals
- montage: Quick cuts of multiple clips, often with music
- documentary: Narrated footage, voiceover style
- reaction: Person reacting to other content (split screen or PiP)
- tutorial_hands: Hands-on tutorial (cooking, crafts, unboxing)
- live_event: Concert, sports, live performance footage
- meme_content: Meme-style edits, text overlays, viral format

SECONDARY ATTRIBUTES:
- has_captions: Burned-in captions/subtitles visible
- has_music: Background music present
- has_voiceover: Off-camera narration
- is_vertical: 9:16 aspect ratio (TikTok/Reels/Shorts)
- is_horizontal: 16:9 aspect ratio (YouTube)
- production_quality: low, medium, high, professional
"""
import os
from typing import Dict, List, Optional, Tuple
from loguru import logger
from dataclasses import dataclass, field
from enum import Enum


class ContentFormat(Enum):
    """Primary content format types"""
    TALKING_HEAD = "talking_head"
    INTERVIEW = "interview"
    BROLL_SCENIC = "broll_scenic"
    BROLL_ACTION = "broll_action"
    BROLL_PEOPLE = "broll_people"
    ANIMATED = "animated"
    SCREEN_RECORDING = "screen_recording"
    SLIDESHOW = "slideshow"
    MUSIC_VIDEO = "music_video"
    MONTAGE = "montage"
    DOCUMENTARY = "documentary"
    REACTION = "reaction"
    TUTORIAL_HANDS = "tutorial_hands"
    LIVE_EVENT = "live_event"
    MEME_CONTENT = "meme_content"
    UNKNOWN = "unknown"


class ProductionQuality(Enum):
    """Production quality levels"""
    LOW = "low"           # Phone footage, poor lighting
    MEDIUM = "medium"     # Decent quality, basic editing
    HIGH = "high"         # Good production, proper editing
    PROFESSIONAL = "professional"  # Studio quality


@dataclass
class FormatAnalysis:
    """Complete format analysis result"""
    primary_format: ContentFormat
    confidence: float  # 0.0 to 1.0
    secondary_formats: List[ContentFormat] = field(default_factory=list)
    
    # Attributes
    has_speech: bool = False
    has_music: bool = False
    has_voiceover: bool = False
    has_captions: bool = False
    has_text_overlay: bool = False
    has_people: bool = False
    people_speaking: bool = False
    people_count_estimate: int = 0
    
    # Technical
    is_vertical: bool = False
    is_horizontal: bool = True
    duration_category: str = "short"  # short (<60s), medium (60-180s), long (>180s)
    production_quality: ProductionQuality = ProductionQuality.MEDIUM
    
    # Reasoning
    reasons: List[str] = field(default_factory=list)
    
    # Platform suitability
    best_platforms: List[str] = field(default_factory=list)
    suggested_use: str = "standalone"  # standalone, overlay, cutaway, primary


class FormatDetector:
    """
    Comprehensive format detection using transcript and visual analysis.
    
    Uses existing video_analysis data to classify content format.
    """
    
    # Keywords for detection
    TALKING_HEAD_VISUAL_KEYWORDS = [
        "person speaking", "talking", "speaking to camera", "face", 
        "presenter", "host", "vlogger", "direct address", "close-up face",
        "looking at camera", "eye contact"
    ]
    
    INTERVIEW_KEYWORDS = [
        "interview", "conversation", "two people", "multiple people",
        "discussion", "podcast", "dialogue"
    ]
    
    ANIMATED_KEYWORDS = [
        "animation", "animated", "cartoon", "motion graphics", "graphics",
        "illustrated", "drawn", "render", "3d", "cgi", "digital art"
    ]
    
    SCREEN_RECORDING_KEYWORDS = [
        "screen", "computer", "software", "interface", "cursor",
        "desktop", "browser", "app", "application", "code", "terminal"
    ]
    
    SCENIC_KEYWORDS = [
        "landscape", "nature", "sky", "water", "ocean", "mountain",
        "forest", "city", "street", "building", "outdoor", "scenery",
        "aerial", "drone", "sunset", "sunrise", "beach", "park"
    ]
    
    ACTION_KEYWORDS = [
        "movement", "action", "walking", "running", "driving", "moving",
        "activity", "sports", "exercise", "dancing", "performing"
    ]
    
    LIVE_EVENT_KEYWORDS = [
        "concert", "festival", "crowd", "stage", "performance", "live",
        "audience", "venue", "event", "show"
    ]
    
    TUTORIAL_KEYWORDS = [
        "hands", "demonstration", "showing", "step by step", "how to",
        "tutorial", "making", "creating", "cooking", "crafting", "unboxing"
    ]
    
    MEME_KEYWORDS = [
        "meme", "viral", "trend", "duet", "stitch", "reaction",
        "funny", "comedy", "joke", "parody"
    ]
    
    def __init__(self):
        logger.info("[FormatDetector] Initialized")
    
    def detect_format(
        self,
        transcript: Optional[str],
        visual_analysis: Optional[Dict],
        transcription_data: Optional[Dict] = None,
        duration_sec: Optional[float] = None,
        topics: Optional[List[str]] = None,
        tone: Optional[str] = None,
        existing_broll_analysis: Optional[Dict] = None
    ) -> FormatAnalysis:
        """
        Detect the content format of a video.
        
        Args:
            transcript: Video transcript text
            visual_analysis: Visual analysis dict with visual_summary, etc.
            transcription_data: Detailed transcription metadata
            duration_sec: Video duration in seconds
            topics: List of detected topics
            tone: Detected tone
            existing_broll_analysis: Existing B-roll detection if available
            
        Returns:
            FormatAnalysis with complete classification
        """
        reasons = []
        format_scores: Dict[ContentFormat, float] = {f: 0.0 for f in ContentFormat}
        
        # Initialize attributes
        has_speech = False
        has_music = False
        has_voiceover = False
        has_captions = False
        has_text_overlay = False
        has_people = False
        people_speaking = False
        people_count = 0
        
        # === TRANSCRIPT ANALYSIS ===
        transcript_clean = (transcript or "").strip().lower()
        word_count = len(transcript_clean.split()) if transcript_clean else 0
        
        if word_count >= 30:
            has_speech = True
            # Significant speech suggests talking head or documentary
            format_scores[ContentFormat.TALKING_HEAD] += 0.3
            format_scores[ContentFormat.DOCUMENTARY] += 0.2
            reasons.append(f"Significant speech detected ({word_count} words)")
        elif word_count >= 10:
            has_speech = True
            # Some speech - could be various formats
            format_scores[ContentFormat.TALKING_HEAD] += 0.1
            format_scores[ContentFormat.MONTAGE] += 0.1
        else:
            # Minimal/no speech - likely B-roll or music video
            format_scores[ContentFormat.BROLL_SCENIC] += 0.2
            format_scores[ContentFormat.BROLL_ACTION] += 0.2
            format_scores[ContentFormat.MUSIC_VIDEO] += 0.2
            reasons.append("Minimal/no speech detected")
        
        # Check transcription metadata
        if transcription_data:
            wpm = float(transcription_data.get("words_per_minute", 0) or 0)
            silence_ratio = float(transcription_data.get("silence_ratio", 0) or 0)
            
            if wpm > 120:
                # Fast speech - energetic content
                format_scores[ContentFormat.TALKING_HEAD] += 0.2
                reasons.append(f"Fast speech pace ({wpm:.0f} wpm)")
            elif wpm > 0 and wpm < 60:
                # Slow/deliberate speech - documentary or tutorial
                format_scores[ContentFormat.DOCUMENTARY] += 0.1
                format_scores[ContentFormat.TUTORIAL_HANDS] += 0.1
            
            if silence_ratio > 0.5:
                # High silence - B-roll or music content
                format_scores[ContentFormat.BROLL_SCENIC] += 0.2
                format_scores[ContentFormat.MUSIC_VIDEO] += 0.1
        
        # === VISUAL ANALYSIS ===
        if visual_analysis:
            visual_summary = (visual_analysis.get("visual_summary", "") or "").lower()
            
            # Check for people
            people_keywords = ["person", "people", "man", "woman", "face", "someone"]
            has_people = any(kw in visual_summary for kw in people_keywords)
            
            # Check for speaking
            speaking_keywords = ["speaking", "talking", "narrating", "presenting"]
            people_speaking = any(kw in visual_summary for kw in speaking_keywords)
            
            # Estimate people count
            if "two people" in visual_summary or "multiple" in visual_summary:
                people_count = 2
            elif has_people:
                people_count = 1
            
            # Check for text/captions
            text_keywords = ["text", "caption", "subtitle", "title", "overlay"]
            has_text_overlay = any(kw in visual_summary for kw in text_keywords)
            
            # === KEYWORD MATCHING ===
            
            # Talking head
            if any(kw in visual_summary for kw in self.TALKING_HEAD_VISUAL_KEYWORDS):
                format_scores[ContentFormat.TALKING_HEAD] += 0.4
                reasons.append("Visual shows person speaking to camera")
            
            # Interview
            if any(kw in visual_summary for kw in self.INTERVIEW_KEYWORDS):
                format_scores[ContentFormat.INTERVIEW] += 0.5
                reasons.append("Interview/conversation detected")
            
            # Animated
            if any(kw in visual_summary for kw in self.ANIMATED_KEYWORDS):
                format_scores[ContentFormat.ANIMATED] += 0.6
                reasons.append("Animation/graphics detected")
            
            # Screen recording
            if any(kw in visual_summary for kw in self.SCREEN_RECORDING_KEYWORDS):
                format_scores[ContentFormat.SCREEN_RECORDING] += 0.5
                reasons.append("Screen recording detected")
            
            # Scenic B-roll
            if any(kw in visual_summary for kw in self.SCENIC_KEYWORDS):
                format_scores[ContentFormat.BROLL_SCENIC] += 0.3
                if not has_speech:
                    format_scores[ContentFormat.BROLL_SCENIC] += 0.3
                reasons.append("Scenic/environmental content")
            
            # Action B-roll
            if any(kw in visual_summary for kw in self.ACTION_KEYWORDS):
                format_scores[ContentFormat.BROLL_ACTION] += 0.3
                if not has_speech:
                    format_scores[ContentFormat.BROLL_ACTION] += 0.2
                reasons.append("Action/movement content")
            
            # Live event
            if any(kw in visual_summary for kw in self.LIVE_EVENT_KEYWORDS):
                format_scores[ContentFormat.LIVE_EVENT] += 0.5
                reasons.append("Live event footage")
            
            # Tutorial
            if any(kw in visual_summary for kw in self.TUTORIAL_KEYWORDS):
                format_scores[ContentFormat.TUTORIAL_HANDS] += 0.4
                reasons.append("Tutorial/hands-on content")
            
            # Meme content
            if any(kw in visual_summary for kw in self.MEME_KEYWORDS):
                format_scores[ContentFormat.MEME_CONTENT] += 0.4
                reasons.append("Meme/viral content style")
            
            # People but not speaking = B-roll with people
            if has_people and not people_speaking and not has_speech:
                format_scores[ContentFormat.BROLL_PEOPLE] += 0.4
                reasons.append("People present but not speaking")
        
        # === USE EXISTING B-ROLL ANALYSIS ===
        if existing_broll_analysis:
            if existing_broll_analysis.get("is_broll"):
                visual_type = existing_broll_analysis.get("broll_visual_type", "")
                if visual_type == "scenic":
                    format_scores[ContentFormat.BROLL_SCENIC] += 0.3
                elif visual_type == "action":
                    format_scores[ContentFormat.BROLL_ACTION] += 0.3
                elif visual_type == "people":
                    format_scores[ContentFormat.BROLL_PEOPLE] += 0.3
        
        # === TOPIC ANALYSIS ===
        if topics:
            topic_str = " ".join(topics).lower()
            
            if any(kw in topic_str for kw in ["music", "song", "audio"]):
                format_scores[ContentFormat.MUSIC_VIDEO] += 0.2
                has_music = True
            
            if any(kw in topic_str for kw in ["tutorial", "how to", "guide"]):
                format_scores[ContentFormat.TUTORIAL_HANDS] += 0.2
            
            if any(kw in topic_str for kw in ["vlog", "day in", "routine"]):
                format_scores[ContentFormat.TALKING_HEAD] += 0.2
        
        # === DURATION ANALYSIS ===
        duration_category = "short"
        if duration_sec:
            dur = float(duration_sec)
            if dur < 60:
                duration_category = "short"
                format_scores[ContentFormat.MEME_CONTENT] += 0.1
            elif dur < 180:
                duration_category = "medium"
            else:
                duration_category = "long"
                format_scores[ContentFormat.DOCUMENTARY] += 0.1
                format_scores[ContentFormat.INTERVIEW] += 0.1
        
        # === DETERMINE PRIMARY FORMAT ===
        # Find highest scoring format
        sorted_formats = sorted(format_scores.items(), key=lambda x: x[1], reverse=True)
        primary_format = sorted_formats[0][0]
        confidence = min(sorted_formats[0][1], 1.0)
        
        # If no clear winner, use heuristics
        if confidence < 0.2:
            if has_speech and has_people:
                primary_format = ContentFormat.TALKING_HEAD
                confidence = 0.4
            elif not has_speech and has_people:
                primary_format = ContentFormat.BROLL_PEOPLE
                confidence = 0.4
            elif not has_speech:
                primary_format = ContentFormat.BROLL_SCENIC
                confidence = 0.3
            else:
                primary_format = ContentFormat.UNKNOWN
                confidence = 0.1
        
        # Get secondary formats (scores > 0.2 and not primary)
        secondary_formats = [
            f for f, score in sorted_formats[1:5]
            if score >= 0.2 and f != primary_format
        ]
        
        # === DETERMINE ATTRIBUTES ===
        is_vertical = False  # Would need aspect ratio from video metadata
        
        # Production quality estimation
        production_quality = ProductionQuality.MEDIUM
        if duration_sec and float(duration_sec) > 300:
            production_quality = ProductionQuality.HIGH
        
        # Best platforms
        best_platforms = self._get_best_platforms(primary_format, duration_category)
        
        # Suggested use
        suggested_use = self._get_suggested_use(primary_format)
        
        result = FormatAnalysis(
            primary_format=primary_format,
            confidence=confidence,
            secondary_formats=secondary_formats,
            has_speech=has_speech,
            has_music=has_music,
            has_voiceover=has_voiceover,
            has_captions=has_captions,
            has_text_overlay=has_text_overlay,
            has_people=has_people,
            people_speaking=people_speaking,
            people_count_estimate=people_count,
            is_vertical=is_vertical,
            duration_category=duration_category,
            production_quality=production_quality,
            reasons=reasons,
            best_platforms=best_platforms,
            suggested_use=suggested_use
        )
        
        logger.info(f"[FormatDetector] Detected: {primary_format.value} (confidence: {confidence:.2f})")
        
        return result
    
    def _get_best_platforms(self, format_type: ContentFormat, duration: str) -> List[str]:
        """Determine best platforms for content format"""
        platform_map = {
            ContentFormat.TALKING_HEAD: ["youtube", "tiktok", "instagram"],
            ContentFormat.INTERVIEW: ["youtube", "spotify"],
            ContentFormat.BROLL_SCENIC: ["instagram", "tiktok"],
            ContentFormat.BROLL_ACTION: ["tiktok", "instagram", "youtube"],
            ContentFormat.BROLL_PEOPLE: ["tiktok", "instagram"],
            ContentFormat.ANIMATED: ["youtube", "tiktok", "instagram"],
            ContentFormat.SCREEN_RECORDING: ["youtube", "tiktok"],
            ContentFormat.MUSIC_VIDEO: ["tiktok", "instagram", "youtube"],
            ContentFormat.MONTAGE: ["tiktok", "instagram"],
            ContentFormat.DOCUMENTARY: ["youtube"],
            ContentFormat.TUTORIAL_HANDS: ["youtube", "tiktok"],
            ContentFormat.LIVE_EVENT: ["instagram", "tiktok", "youtube"],
            ContentFormat.MEME_CONTENT: ["tiktok", "twitter", "instagram"],
            ContentFormat.REACTION: ["youtube", "tiktok"],
        }
        
        platforms = platform_map.get(format_type, ["tiktok", "instagram"])
        
        # Adjust based on duration
        if duration == "long" and "youtube" not in platforms:
            platforms.insert(0, "youtube")
        if duration == "short":
            if "tiktok" not in platforms:
                platforms.insert(0, "tiktok")
        
        return platforms[:4]
    
    def _get_suggested_use(self, format_type: ContentFormat) -> str:
        """Determine suggested use for content"""
        use_map = {
            ContentFormat.TALKING_HEAD: "primary",
            ContentFormat.INTERVIEW: "primary",
            ContentFormat.BROLL_SCENIC: "overlay",
            ContentFormat.BROLL_ACTION: "cutaway",
            ContentFormat.BROLL_PEOPLE: "cutaway",
            ContentFormat.ANIMATED: "standalone",
            ContentFormat.SCREEN_RECORDING: "primary",
            ContentFormat.MUSIC_VIDEO: "standalone",
            ContentFormat.MONTAGE: "standalone",
            ContentFormat.DOCUMENTARY: "primary",
            ContentFormat.TUTORIAL_HANDS: "primary",
            ContentFormat.LIVE_EVENT: "standalone",
            ContentFormat.MEME_CONTENT: "standalone",
        }
        return use_map.get(format_type, "standalone")
    
    def detect_from_db_record(self, video_analysis: Dict) -> FormatAnalysis:
        """Detect format from a video_analysis database record"""
        return self.detect_format(
            transcript=video_analysis.get("transcript"),
            visual_analysis=video_analysis.get("visual_analysis"),
            transcription_data=video_analysis.get("transcription_data"),
            duration_sec=video_analysis.get("transcription_duration_sec"),
            topics=video_analysis.get("topics"),
            tone=video_analysis.get("tone"),
            existing_broll_analysis={
                "is_broll": video_analysis.get("is_broll"),
                "broll_visual_type": video_analysis.get("broll_visual_type"),
            } if video_analysis.get("is_broll") is not None else None
        )

    def route_video_by_orientation(
        self,
        width: int,
        height: int,
        format_analysis: Optional[FormatAnalysis] = None
    ) -> List[str]:
        """
        Route videos to appropriate platforms based on orientation (VID-001).

        Routing rules:
        - 9:16 (vertical) → TikTok, Instagram Reels, YouTube Shorts
        - 16:9 (horizontal) → YouTube, LinkedIn
        - Square (1:1) → Instagram Feed, LinkedIn, Twitter/X

        Args:
            width: Video width in pixels
            height: Video height in pixels
            format_analysis: Optional format analysis for additional routing context

        Returns:
            List of recommended platform identifiers
        """
        aspect_ratio = width / height if height > 0 else 1.0
        platforms = []

        # Vertical video (9:16, approximately 0.5625)
        if 0.4 < aspect_ratio < 0.7:
            platforms = ["tiktok", "instagram_reels", "youtube_shorts"]
            logger.info(f"[VideoRouter] Vertical video ({width}x{height}) → {platforms}")

        # Horizontal video (16:9, approximately 1.777)
        elif 1.5 < aspect_ratio < 2.0:
            platforms = ["youtube", "linkedin", "facebook"]
            logger.info(f"[VideoRouter] Horizontal video ({width}x{height}) → {platforms}")

        # Square video (1:1)
        elif 0.9 < aspect_ratio < 1.1:
            platforms = ["instagram_feed", "linkedin", "twitter", "facebook"]
            logger.info(f"[VideoRouter] Square video ({width}x{height}) → {platforms}")

        # Ultra-wide (21:9 and wider)
        elif aspect_ratio >= 2.0:
            platforms = ["youtube", "twitter"]
            logger.info(f"[VideoRouter] Ultra-wide video ({width}x{height}) → {platforms}")

        # Other aspect ratios
        else:
            platforms = ["youtube", "instagram_feed"]
            logger.info(f"[VideoRouter] Non-standard video ({width}x{height}, {aspect_ratio:.2f}) → {platforms}")

        # If format analysis is provided, refine recommendations
        if format_analysis:
            # Short-form content strongly prefers vertical platforms
            if format_analysis.duration_category == "short" and "tiktok" not in platforms:
                platforms.insert(0, "tiktok")

            # Long-form content prefers YouTube
            if format_analysis.duration_category == "long" and "youtube" not in platforms:
                platforms.insert(0, "youtube")

            # Professional quality content goes to YouTube
            if format_analysis.production_quality == ProductionQuality.PROFESSIONAL:
                if "youtube" not in platforms:
                    platforms.insert(0, "youtube")

        return platforms

    def get_video_orientation_category(self, width: int, height: int) -> str:
        """
        Get human-readable orientation category for video.

        Returns:
            "vertical", "horizontal", "square", or "ultra-wide"
        """
        aspect_ratio = width / height if height > 0 else 1.0

        if 0.4 < aspect_ratio < 0.7:
            return "vertical"
        elif 1.5 < aspect_ratio < 2.0:
            return "horizontal"
        elif 0.9 < aspect_ratio < 1.1:
            return "square"
        elif aspect_ratio >= 2.0:
            return "ultra-wide"
        else:
            return "non-standard"
