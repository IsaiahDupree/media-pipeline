"""
Base classes for swappable social media API providers
Allows easy switching between different RapidAPI providers
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Platform(Enum):
    """Supported social media platforms"""
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    LINKEDIN = "linkedin"
    PINTEREST = "pinterest"
    THREADS = "threads"
    BLUESKY = "bluesky"


@dataclass
class ProfileData:
    """Standardized profile data structure"""
    username: str
    full_name: str
    bio: str
    profile_pic_url: str
    followers_count: int
    following_count: int
    posts_count: int
    is_verified: bool
    is_business: bool
    platform: Platform
    raw_data: Dict[str, Any]  # Original provider response


@dataclass
class PostData:
    """Standardized post/video data structure"""
    post_id: str
    url: str
    caption: str
    media_type: str  # 'video', 'image', 'carousel'
    thumbnail_url: Optional[str]
    media_url: Optional[str]
    likes_count: int
    comments_count: int
    views_count: int
    shares_count: int
    posted_at: datetime
    is_video: bool
    duration: Optional[float]
    platform: Platform
    raw_data: Dict[str, Any]


@dataclass
class AnalyticsData:
    """Standardized analytics data"""
    profile: ProfileData
    posts: List[PostData]
    total_likes: int
    total_comments: int
    total_views: int
    total_shares: int
    engagement_rate: float
    avg_likes_per_post: float
    avg_comments_per_post: float
    best_performing_post: Optional[PostData]
    top_hashtags: List[str]
    platform: Platform


@dataclass
class ProviderMetrics:
    """Provider performance metrics"""
    name: str
    latency_ms: float
    success: bool
    error: Optional[str] = None
    timestamp: datetime = None


class ProviderInterface(ABC):
    """
    Abstract base class for social media API providers
    All providers must implement this interface
    """
    
    def __init__(
self,
        api_key: str,
        base_url: str,
        name: str,
        platform: Platform
    ):
        """
        Initialize provider
        
        Args:
            api_key: RapidAPI key
            base_url: Provider base URL
            name: Provider name
            platform: Platform enum
        """
        self.api_key = api_key
        self.base_url = base_url
        self.name = name
        self.platform = platform
        self.headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": base_url.replace("https://", "").replace("http://", "")
        }
    
    @abstractmethod
    async def get_profile(self, username: str) -> Optional[ProfileData]:
        """
        Get profile data for a user
        
        Args:
            username: Username to fetch
            
        Returns:
            ProfileData object or None if failed
        """
        pass
    
    @abstractmethod
    async def get_posts(
        self,
        username: str,
        limit: int = 20,
        cursor: Optional[str] = None
    ) -> List[PostData]:
        """
        Get posts/videos for a user
        
        Args:
            username: Username to fetch posts for
            limit: Number of posts to fetch
            cursor: Pagination cursor
            
        Returns:
            List of PostData objects
        """
        pass
    
    @abstractmethod
    async def get_post_details(self, post_id: str) -> Optional[PostData]:
        """
        Get detailed information for a specific post
        
        Args:
            post_id: Post/video ID
            
        Returns:
            PostData object or None if failed
        """
        pass
    
    @abstractmethod
    async def search_users(
        self,
        query: str,
        limit: int = 20
    ) -> List[ProfileData]:
        """
        Search for users by query
        
        Args:
            query: Search query
            limit: Number of results
            
        Returns:
            List of ProfileData objects
        """
        pass
    
    @abstractmethod
    async def search_content(
        self,
        query: str,
        limit: int = 20
    ) -> List[PostData]:
        """
        Search for posts/videos by query
        
        Args:
            query: Search query
            limit: Number of results
            
        Returns:
            List of PostData objects
        """
        pass
    
    async def get_analytics(
        self,
        username: str,
        posts_limit: int = 50
    ) -> Optional[AnalyticsData]:
        """
        Get complete analytics for a profile
        
        Default implementation combines profile + posts
        Providers can override for optimized implementation
        
        Args:
            username: Username to analyze
            posts_limit: Number of posts to analyze
            
        Returns:
            AnalyticsData object or None if failed
        """
        try:
            profile = await self.get_profile(username)
            if not profile:
                return None
            
            posts = await self.get_posts(username, posts_limit)
            if not posts:
                return AnalyticsData(
                    profile=profile,
                    posts=[],
                    total_likes=0,
                    total_comments=0,
                    total_views=0,
                    total_shares=0,
                    engagement_rate=0.0,
                    avg_likes_per_post=0.0,
                    avg_comments_per_post=0.0,
                    best_performing_post=None,
                    top_hashtags=[],
                    platform=self.platform
                )
            
            # Calculate analytics
            total_likes = sum(p.likes_count for p in posts)
            total_comments = sum(p.comments_count for p in posts)
            total_views = sum(p.views_count for p in posts)
            total_shares = sum(p.shares_count for p in posts)
            
            avg_likes = total_likes / len(posts)
            avg_comments = total_comments / len(posts)
            
            # Engagement rate
            total_engagement = total_likes + total_comments
            engagement_rate = (total_engagement / (len(posts) * profile.followers_count)) * 100 if profile.followers_count > 0 else 0
            
            # Best post
            best_post = max(posts, key=lambda p: p.likes_count + p.comments_count)
            
            # Extract hashtags
            all_hashtags = []
            for post in posts:
                hashtags = [word for word in post.caption.split() if word.startswith('#')]
                all_hashtags.extend(hashtags)
            
            from collections import Counter
            top_hashtags = [tag for tag, count in Counter(all_hashtags).most_common(10)]
            
            return AnalyticsData(
                profile=profile,
                posts=posts,
                total_likes=total_likes,
                total_comments=total_comments,
                total_views=total_views,
                total_shares=total_shares,
                engagement_rate=round(engagement_rate, 2),
                avg_likes_per_post=round(avg_likes, 2),
                avg_comments_per_post=round(avg_comments, 2),
                best_performing_post=best_post,
                top_hashtags=top_hashtags,
                platform=self.platform
            )
            
        except Exception as e:
            logger.error(f"Error calculating analytics: {e}")
            return None
    
    async def health_check(self) -> bool:
        """
        Check if provider is healthy and accessible
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Override in subclass for specific health check
            return True
        except Exception as e:
            logger.error(f"Health check failed for {self.name}: {e}")
            return False
    
    def get_cost_per_request(self) -> float:
        """
        Get estimated cost per request
        Override in subclass with actual pricing
        
        Returns:
            Cost in USD
        """
        return 0.0


class ProviderConfig:
    """Configuration for provider instances"""
    
    def __init__(
        self,
        provider_class: type,
        api_key: str,
        base_url: str,
        name: str,
        platform: Platform,
        priority: int = 1,
        enabled: bool = True
    ):
        self.provider_class = provider_class
        self.api_key = api_key
        self.base_url = base_url
        self.name = name
        self.platform = platform
        self.priority = priority
        self.enabled = enabled
    
    def create_instance(self) -> ProviderInterface:
        """Create an instance of the provider"""
        return self.provider_class(
            api_key=self.api_key,
            base_url=self.base_url,
            name=self.name,
            platform=self.platform
        )
