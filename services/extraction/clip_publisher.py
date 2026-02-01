"""
Clip Publisher Service
Connects clips to platform posts for multi-platform publishing
"""
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from sqlalchemy.orm import Session
import uuid

from database.models import VideoClip, ClipPost, PlatformPost
from services.multi_platform_publisher import MultiPlatformPublisher
from services.thumbnail_generator import ThumbnailGenerator

logger = logging.getLogger(__name__)


class ClipPublisher:
    """
    Service for publishing clips to multiple platforms
    
    Features:
    - Create platform posts from clips
    - Generate clip-specific thumbnails
    - Map clip configs to post metadata
    - Track clip-post relationships
    """
    
    def __init__(self, db: Session):
        """Initialize clip publisher"""
        self.db = db
        self.publisher = MultiPlatformPublisher(db)
        self.thumbnail_gen = ThumbnailGenerator()
    
    async def publish_clip(
        self,
        clip_id: str,
        platforms: List[str],
        schedule_time: Optional[datetime] = None,
        custom_configs: Optional[Dict[str, Dict]] = None
    ) -> List[PlatformPost]:
        """
        Publish a clip to multiple platforms
        
        Args:
            clip_id: UUID of clip to publish
            platforms: List of platform keys to publish to
            schedule_time: Optional scheduled publish time
            custom_configs: Optional platform-specific config overrides
            
        Returns:
            List of created PlatformPost objects
        """
        clip = self.db.query(VideoClip).filter(VideoClip.id == uuid.UUID(str(clip_id))).first()
        if not clip:
            raise ValueError(f"Clip {clip_id} not found")
        
        if not platforms:
            raise ValueError("At least one platform must be specified")
        
        # Ensure clip has platform variants
        if not clip.platform_variants:
            logger.warning(f"Clip {clip_id} has no platform variants, using defaults")
        
        # Create posts for each platform
        created_posts = []
        
        for platform in platforms:
            # Get platform-specific config
            variant_config = clip.platform_variants.get(platform, {}) if clip.platform_variants else {}
            custom_config = custom_configs.get(platform, {}) if custom_configs else {}
            
            # Merge configs (custom overrides variant)
            config = {**variant_config, **custom_config}
            
            try:
                # Create platform post
                post = await self._create_platform_post(
                    clip=clip,
                    platform=platform,
                    config=config,
                    schedule_time=schedule_time
                )
                
                # Create clip-post relationship
                clip_post = ClipPost(
                    id=uuid.uuid4(),
                    clip_id=clip.id,
                    post_id=post.id,
                    platform=platform,
                    platform_config=config
                )
                
                self.db.add(clip_post)
                created_posts.append(post)
                
                logger.info(f"Created post for clip {clip_id} on {platform}")
                
            except Exception as e:
                logger.error(f"Error creating post for {platform}: {e}")
                continue
        
        # Update clip status
        if created_posts:
            clip.status = "published" if not schedule_time else "ready"
        
        try:
            self.db.commit()
            
            logger.info(f"Published clip {clip_id} to {len(created_posts)} platforms")
            
            return created_posts
        
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error committing clip publication: {e}")
            raise
    
    async def _create_platform_post(
        self,
        clip: VideoClip,
        platform: str,
        config: Dict,
        schedule_time: Optional[datetime]
    ) -> PlatformPost:
        """Create a PlatformPost from clip configuration"""
        #  Generate thumbnail for this clip/platform combo
        thumbnail_url = None
        if clip.thumbnail_config:
            try:
                # Implementation would extract frame at thumbnail_config["frame_time"]
                # and generate thumbnail for platform
                pass
            except Exception as e:
                logger.warning(f"Error generating thumbnail: {e}")
        
        # Create ContentItem first (root of the content hierarchy)
        from database.models import ContentItem, ContentVariant
        
        content = ContentItem(
            id=uuid.uuid4(),
            type="video",
            title=clip.title or f"Clip {clip.id}",
            description=clip.description,
            created_at=datetime.now()
        )
        self.db.add(content)
        self.db.flush()  # Ensure content.id is available
        
        # Create ContentVariant (platform-specific version)
        variant = ContentVariant(
            id=uuid.uuid4(),
            content_id=content.id,
            platform=platform,
            variant_type="video",
            status="draft"
        )
        self.db.add(variant)
        self.db.flush()  # Ensure variant.id is available
        
        # Create post
        post = PlatformPost(
            id=uuid.uuid4(),
            content_variant_id=variant.id,
            clip_id=clip.id,
            platform=platform,
            platform_post_id=f"clip_{clip.id}_{platform}_{datetime.now().timestamp()}",
            title=clip.title,
            caption=clip.description,
            thumbnail_url=thumbnail_url,
            status="scheduled" if schedule_time else "draft",
            scheduled_for=schedule_time,
            created_at=datetime.now()
        )
        
        self.db.add(post)
        
        return post
    
    def get_clip_posts(
        self,
        clip_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all platform posts for a clip
        
        Args:
            clip_id: UUID of clip
            
        Returns:
            List of post data with platform info
        """
        clip_posts = self.db.query(ClipPost).filter(
            ClipPost.clip_id == clip_id
        ).all()
        
        results = []
        for cp in clip_posts:
            post = self.db.query(PlatformPost).filter(
                PlatformPost.id == cp.post_id
            ).first()
            
            if post:
                results.append({
                    "clip_post_id": str(cp.id),
                    "platform": cp.platform,
                    "post_id": str(post.id),
                    "platform_post_id": post.platform_post_id,
                    "status": post.status,
                    "scheduled_for": post.scheduled_for.isoformat() if post.scheduled_for else None,
                    "published_at": post.published_at.isoformat() if post.published_at else None,
                    "platform_url": post.platform_url
                })
        
        return results
    
    def get_clip_performance(
        self,
        clip_id: str
    ) -> Dict[str, Any]:
        """
        Get aggregated performance metrics for a clip across all platforms
        
        Args:
            clip_id: UUID of clip
            
        Returns:
            Dict with performance metrics
        """
        clip_posts = self.db.query(ClipPost).filter(
            ClipPost.clip_id == clip_id
        ).all()
        
        if not clip_posts:
            return {
                "clip_id": clip_id,
                "total_posts": 0,
                "platforms": [],
                "total_views": 0,
                "total_likes": 0,
                "total_shares": 0,
                "total_comments": 0,
                "avg_engagement_rate": 0,
                "platform_breakdown": {}
            }
        
        # Aggregate metrics (would query from platform_checkbacks in real implementation)
        platform_breakdown = {}
        total_views = 0
        total_likes = 0
        total_shares = 0
        total_comments = 0
        
        for cp in clip_posts:
            # Placeholder: In real implementation, query latest checkback metrics
            platform_breakdown[cp.platform] = {
                "post_id": str(cp.post_id),
                "views": 0,
                "likes": 0,
                "shares": 0,
                "comments": 0,
                "engagement_rate": 0
            }
        
        avg_engagement = 0
        if total_views > 0:
            avg_engagement = ((total_likes + total_shares + total_comments) / total_views) * 100
        
        return {
            "clip_id": clip_id,
            "total_posts": len(clip_posts),
            "platforms": [cp.platform for cp in clip_posts],
            "total_views": total_views,
            "total_likes": total_likes,
            "total_shares": total_shares,
            "total_comments": total_comments,
             "avg_engagement_rate": round(avg_engagement, 2),
            "platform_breakdown": platform_breakdown
        }
