"""
Clip Editor & Configuration Service
Manages clip configurations, platform variants, and metadata
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import uuid

from database.models import VideoClip, ClipPost, AnalyzedVideo
from services.thumbnail_generator import ThumbnailGenerator, PLATFORM_DIMENSIONS

logger = logging.getLogger(__name__)


class ClipEditor:
    """
    Service for creating and managing video clip configurations
    
   Features:
    - Create and update clips with timing and metadata
    - Configure overlays, captions, and thumbnails
    - Generate platform-specific variants
    - Validate clip timing against video bounds
    """
    
    def __init__(self, db: Session):
        """Initialize clip editor"""
        self.db = db
        self.thumbnail_gen = ThumbnailGenerator()
    
    def create_clip(
        self,
        video_id: str,
        user_id: str,
        start_time: float,
        end_time: float,
        title: Optional[str] = None,
        description: Optional[str] = None,
        clip_type: str = "custom",
        overlay_config: Optional[Dict] = None,
        caption_config: Optional[Dict] = None,
        thumbnail_config: Optional[Dict] = None,
        segment_ids: Optional[List[str]] = None,
        ai_suggested: bool = False,
        ai_score: Optional[float] = None,
        ai_reasoning: Optional[str] = None
    ) -> VideoClip:
        """
        Create a new video clip
        
        Args:
            video_id: UUID of source video
            user_id: UUID of user creating clip
            start_time: Clip start time in seconds
            end_time: Clip end time in seconds
            title: Optional clip title
            description: Optional description
            clip_type: Type of clip (highlight, full, custom, ai_generated)
            overlay_config: Text overlay configuration
            caption_config: Caption styling configuration
            thumbnail_config: Thumbnail selection configuration
            segment_ids: Associated segment UUIDs
            ai_suggested: Whether clip was AI-generated
            ai_score: AI quality score (0-1)
            ai_reasoning: Why AI suggested this clip
            
        Returns:
            Created VideoClip object
            
        Raises:
            ValueError: If clip timing is invalid
        """
        # Validate video exists and get its duration
        video = self.db.query(AnalyzedVideo).filter(AnalyzedVideo.id == uuid.UUID(str(video_id))).first()
        if not video:
            raise ValueError(f"Video {video_id} not found")
        
        # Validate timing
        if end_time <= start_time:
            raise ValueError("End time must be greater than start time")
        
        if start_time < 0:
            raise ValueError("Start time cannot be negative")
        
        # Validate clip timing
        self._validate_clip_timing(start_time, end_time, video.duration_seconds)
        
        # Convert string UUIDs to UUID objects
        video_uuid = uuid.UUID(str(video_id))
        user_uuid = uuid.UUID(str(user_id))
        
        # Convert segment_ids if provided
        segment_uuids = [uuid.UUID(str(sid)) for sid in segment_ids] if segment_ids else []
        
        # Create clip
        clip = VideoClip(
            id=uuid.uuid4(),
            video_id=video_uuid,
            user_id=user_uuid,
            start_time=start_time,
            end_time=end_time,
            title=title or f"Clip {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            description=description,
            clip_type=clip_type,
            overlay_config=overlay_config or {},
            caption_config=caption_config or {},
            thumbnail_config=thumbnail_config or {},
            segment_ids=segment_ids or [],
            ai_suggested=ai_suggested,
            ai_score=ai_score,
            ai_reasoning=ai_reasoning,
            status="draft"
        )
        
        try:
            self.db.add(clip)
            self.db.commit()
            self.db.refresh(clip)
            
            logger.info(f"Created clip {clip.id} for video {video_id} ({start_time}s-{end_time}s)")
            
            return clip
            
        except IntegrityError as e:
            self.db.rollback()
            logger.error(f"Error creating clip: {e}")
            raise ValueError("Failed to create clip")
    
    def update_clip_config(
        self,
        clip_id: str,
        overlay_config: Optional[Dict] = None,
        caption_config: Optional[Dict] = None,
        thumbnail_config: Optional[Dict] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[str] = None
    ) -> VideoClip:
        """
        Update clip configuration
        
        Args:
            clip_id: UUID of clip to update
            overlay_config: Updated overlay configuration
            caption_config: Updated caption configuration
            thumbnail_config: Updated thumbnail configuration
            title: Updated title
            description: Updated description
            status: Updated status
            
        Returns:
            Updated VideoClip object
        """
        clip = self.db.query(VideoClip).filter(VideoClip.id == uuid.UUID(str(clip_id))).first()
        if not clip:
            raise ValueError(f"Clip {clip_id} not found")
        
        # Update fields
        if overlay_config is not None:
            clip.overlay_config = overlay_config
        if caption_config is not None:
            clip.caption_config = caption_config
        if thumbnail_config is not None:
            clip.thumbnail_config = thumbnail_config
        if title is not None:
            clip.title = title
        if description is not None:
            clip.description = description
        if status is not None:
            clip.status = status
        
        try:
            self.db.commit()
            self.db.refresh(clip)
            
            logger.info(f"Updated clip {clip_id} configuration")
            
            return clip
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating clip {clip_id}: {e}")
            raise
    
    def generate_platform_variants(
        self,
        clip_id: str,
        platforms: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate platform-specific clip configurations
        
        Creates optimized configs for each platform considering:
        - Aspect ratio requirements
        - Optimal duration ranges
        - Caption placement
        - Thumbnail specifications
        
        Args:
            clip_id: UUID of clip
            platforms: List of platform keys or None for all
            
        Returns:
            Dict mapping platform to variant config
        """
        clip = self.db.query(VideoClip).filter(VideoClip.id == uuid.UUID(str(clip_id))).first()
        if not clip:
            raise ValueError(f"Clip {clip_id} not found")
        
        duration = clip.end_time - clip.start_time
        
        # Default to all platforms if none specified
        if platforms is None:
            platforms = list(PLATFORM_DIMENSIONS.keys())
        
        variants = {}
        
        for platform in platforms:
            if platform not in PLATFORM_DIMENSIONS:
                logger.warning(f"Unknown platform: {platform}")
                continue
            
            dims = PLATFORM_DIMENSIONS[platform]
            
            # Platform-specific configuration
            variant = {
                "platform": platform,
                "aspect_ratio": dims.aspect_ratio,
                "orientation": dims.orientation,
                "dimensions": {
                    "width": dims.width,
                    "height": dims.height
                },
                "recommended_duration": self._get_optimal_duration(platform, duration),
                "caption_config": self._get_platform_caption_config(platform, dims.orientation),
                "overlay_config": self._get_platform_overlay_config(platform, dims.orientation),
                "thumbnail_specs": {
                    "width": dims.width,
                    "height": dims.height,
                    "format": "JPEG",
                    "quality": 95
                }
            }
            
            variants[platform] = variant
        
        # Store variants in clip
        clip.platform_variants = variants
        clip.target_platforms = platforms
        
        try:
            self.db.commit()
            self.db.refresh(clip)
            
            logger.info(f"Generated {len(variants)} platform variants for clip {clip_id}")
            
            return variants
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error generating platform variants: {e}")
            raise
    
    def get_clip_preview_data(self, clip_id: str) -> Dict[str, Any]:
        """
        Get clip preview data including frames, metadata, and configs
        
        Args:
            clip_id: UUID of clip
            
        Returns:
            Dict with preview information
        """
        clip = self.db.query(VideoClip).filter(VideoClip.id == uuid.UUID(str(clip_id))).first()
        if not clip:
            raise ValueError(f"Clip {clip_id} not found")
        
        video = self.db.query(AnalyzedVideo).filter(AnalyzedVideo.id == clip.video_id).first()
        
        return {
            "clip_id": str(clip.id),
            "video_id": str(clip.video_id),
            "video_title": video.content_item.title if video and video.content_item else "Unknown",
            "start_time": clip.start_time,
            "end_time": clip.end_time,
            "duration": clip.end_time - clip.start_time,
            "title": clip.title,
            "description": clip.description,
            "clip_type": clip.clip_type,
            "status": clip.status,
            "overlay_config": clip.overlay_config,
            "caption_config": clip.caption_config,
            "thumbnail_config": clip.thumbnail_config,
            "platform_variants": clip.platform_variants or {},
            "target_platforms": clip.target_platforms or [],
            "ai_suggested": clip.ai_suggested,
            "ai_score": clip.ai_score,
            "ai_reasoning": clip.ai_reasoning,
            "created_at": clip.created_at.isoformat() if clip.created_at else None,
            "updated_at": clip.updated_at.isoformat() if clip.updated_at else None
        }
    
    def get_video_clips(
        self,
        video_id: str,
        status: Optional[str] = None
    ) -> List[VideoClip]:
        """
        Get all clips for a video
        
        Args:
            video_id: UUID of video
            status: Optional status filter (draft, ready, published, archived)
            
        Returns:
            List of VideoClip objects
        """
        query = self.db.query(VideoClip).filter(VideoClip.video_id == uuid.UUID(str(video_id)))
        
        if status:
            query = query.filter(VideoClip.status == status)
        
        clips = query.order_by(VideoClip.created_at.desc()).all()
        
        return clips
    
    def delete_clip(self, clip_id: str) -> bool:
        """
        Delete a clip
        
        Args:
            clip_id: UUID of clip to delete
            
        Returns:
            True if deleted, False if not found
        """
        clip = self.db.query(VideoClip).filter(VideoClip.id == uuid.UUID(str(clip_id))).first()
        
        if not clip:
            return False
        
        try:
            self.db.delete(clip)
            self.db.commit()
            
            logger.info(f"Deleted clip {clip_id}")
            
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting clip {clip_id}: {e}")
            raise
    
    def _validate_clip_timing(
        self,
        start_time: float,
        end_time: float,
        video_duration: Optional[float]
    ):
        """Validate clip timing constraints"""
        if start_time < 0:
            raise ValueError("Start time must be >= 0")
        
        if end_time <= start_time:
            raise ValueError("End time must be greater than start time")
        
        if video_duration and end_time > video_duration:
            raise ValueError(f"End time ({end_time}s) exceeds video duration ({video_duration}s)")
        
        # Minimum clip duration: 5 seconds
        if (end_time - start_time) < 5:
            raise ValueError("Clip duration must be at least 5 seconds")
    
    def _get_optimal_duration(self, platform: str, current_duration: float) -> Dict[str, Any]:
        """Get optimal duration info for platform"""
        ranges = {
            "tiktok": (15, 60),
            "instagram_reel": (15, 90),
            "youtube_short": (15, 60),
            "youtube": (60, 180),
            "linkedin": (30, 90),
            "facebook": (30, 120),
            "twitter": (20, 140),
            "snapchat": (10, 60),
            "pinterest": (15, 60),
            "threads": (15, 90)
        }
        
        min_dur, max_dur = ranges.get(platform, (15, 180))
        
        return {
            "min": min_dur,
            "max": max_dur,
            "current": current_duration,
            "optimal": current_duration >= min_dur and current_duration <= max_dur,
            "recommendation": self._get_duration_recommendation(current_duration, min_dur, max_dur)
        }
    
    def _get_duration_recommendation(
        self,
        current: float,
        min_dur: float,
        max_dur: float
    ) -> str:
        """Get duration recommendation text"""
        if current < min_dur:
            return f"Clip is shorter than recommended minimum ({min_dur}s). Consider lengthening."
        elif current > max_dur:
            return f"Clip is longer than recommended maximum ({max_dur}s). Consider shortening."
        else:
            return "Duration is optimal for this platform"
    
    def _get_platform_caption_config(self, platform: str, orientation: str) -> Dict[str, Any]:
        """Get platform-specific caption configuration"""
        base_config = {
            "enabled": True,
            "style": "bold",
            "font_size": "large" if orientation == "portrait" else "medium",
            "color": "#FFFFFF",
            "background": "rgba(0, 0, 0, 0.7)",
            "animation": "fade_in"
        }
        
        # Platform-specific adjustments
        if platform in ["tiktok", "instagram_reel", "youtube_short"]:
            base_config["position"] = "center"
            base_config["word_by_word"] = True  # Trendy caption style
        elif platform == "youtube":
            base_config["position"] = "bottom"
            base_config["word_by_word"] = False
        else:
            base_config["position"] = "bottom"
        
        return base_config
    
    def _get_platform_overlay_config(self, platform: str, orientation: str) -> Dict[str, Any]:
        """Get platform-specific overlay configuration"""
        base_config = {
            "enabled": False,
            "text": "",
            "position": "bottom_third" if orientation == "portrait" else "bottom_center",
            "font_size": 48 if orientation == "portrait" else 36,
            "color": "#FFFFFF",
            "stroke_color": "#000000",
            "stroke_width": 3
        }
        
        return base_config
