"""
TikTok Provider Implementations
Multiple RapidAPI providers for TikTok with swappable interface
"""
import logging
import httpx
from typing import List, Optional
from datetime import datetime

from .provider_base import (
    ProviderInterface,
    Platform,
    ProfileData,
    PostData
)

logger = logging.getLogger(__name__)


class TikTokFeatureSummaryProvider(ProviderInterface):
    """
    TikTok Video Feature Summary Provider
    API: liuzhaolong765481/tiktok-video-feature-summary
    
    Features:
    - HD videos without watermark
    - Comprehensive user data
    - 23 endpoints
    - Best performance metrics
    """
    
    def __init__(self, api_key: str, base_url: str, name: str, platform: Platform):
        super().__init__(api_key, base_url, name, platform)
        self.rapidapi_host = "tiktok-video-feature-summary.p.rapidapi.com"
        self.headers["X-RapidAPI-Host"] = self.rapidapi_host
    
    async def get_profile(self, username: str) -> Optional[ProfileData]:
        """Get TikTok user profile"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/user/info",
                    headers=self.headers,
                    params={"unique_id": username}
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get("code") != 0:
                    logger.error(f"API error: {data.get('msg')}")
                    return None
                
                user_data = data.get("data", {}).get("user", {})
                stats = data.get("data", {}).get("stats", {})
                
                return ProfileData(
                    username=user_data.get("unique_id", username),
                    full_name=user_data.get("nickname", ""),
                    bio=user_data.get("signature", ""),
                    profile_pic_url=user_data.get("avatar_larger", ""),
                    followers_count=stats.get("follower_count", 0),
                    following_count=stats.get("following_count", 0),
                    posts_count=stats.get("video_count", 0),
                    is_verified=user_data.get("verified", False),
                    is_business=user_data.get("commerce_user_info", {}).get("is_ecommerce", False),
                    platform=Platform.TIKTOK,
                    raw_data=data
                )
                
        except Exception as e:
            logger.error(f"Error fetching TikTok profile: {e}")
            return None
    
    async def get_posts(
        self,
        username: str,
        limit: int = 20,
        cursor: Optional[str] = None
    ) -> List[PostData]:
        """Get user's TikTok videos"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                params = {
                    "unique_id": username,
                    "count": min(limit, 35)  # Max per request
                }
                if cursor:
                    params["cursor"] = cursor
                
                response = await client.get(
                    f"{self.base_url}/user/posts",
                    headers=self.headers,
                    params=params
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get("code") != 0:
                    logger.error(f"API error: {data.get('msg')}")
                    return []
                
                videos = data.get("data", {}).get("videos", [])
                posts = []
                
                for video in videos:
                    post = self._parse_video(video)
                    if post:
                        posts.append(post)
                
                return posts
                
        except Exception as e:
            logger.error(f"Error fetching TikTok posts: {e}")
            return []
    
    async def get_post_details(self, post_id: str) -> Optional[PostData]:
        """Get detailed video info"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/",
                    headers=self.headers,
                    params={"url": f"https://www.tiktok.com/@user/video/{post_id}"}
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get("code") != 0:
                    return None
                
                video = data.get("data", {})
                return self._parse_video(video)
                
        except Exception as e:
            logger.error(f"Error fetching post details: {e}")
            return None
    
    async def search_users(self, query: str, limit: int = 20) -> List[ProfileData]:
        """Search TikTok users"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/user/search",
                    headers=self.headers,
                    params={"keywords": query, "count": limit}
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get("code") != 0:
                    return []
                
                users = data.get("data", {}).get("user_list", [])
                profiles = []
                
                for user_data in users:
                    user_info = user_data.get("user_info", {})
                    profiles.append(ProfileData(
                        username=user_info.get("unique_id", ""),
                        full_name=user_info.get("nickname", ""),
                        bio=user_info.get("signature", ""),
                        profile_pic_url=user_info.get("avatar_larger", ""),
                        followers_count=user_info.get("follower_count", 0),
                        following_count=user_info.get("following_count", 0),
                        posts_count=user_info.get("video_count", 0),
                        is_verified=user_info.get("verified", False),
                        is_business=False,
                        platform=Platform.TIKTOK,
                        raw_data=user_data
                    ))
                
                return profiles
                
        except Exception as e:
            logger.error(f"Error searching users: {e}")
            return []
    
    async def search_content(self, query: str, limit: int = 20) -> List[PostData]:
        """Search TikTok videos"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/feed/search",
                    headers=self.headers,
                    params={"keywords": query, "count": limit}
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get("code") != 0:
                    return []
                
                videos = data.get("data", {}).get("videos", [])
                return [self._parse_video(v) for v in videos if self._parse_video(v)]
                
        except Exception as e:
            logger.error(f"Error searching content: {e}")
            return []
    
    def _parse_video(self, video_data: dict) -> Optional[PostData]:
        """Parse video data to PostData"""
        try:
            # This API returns stats directly at video level, not nested
            # Check both locations for backwards compatibility
            stats = video_data.get("stats", {})
            
            # Get author info
            author = video_data.get("author", {})
            author_username = author.get("unique_id", "") if isinstance(author, dict) else ""
            
            return PostData(
                post_id=video_data.get("video_id", ""),
                url=f"https://www.tiktok.com/@{author_username}/video/{video_data.get('video_id', '')}",
                caption=video_data.get("title", "") or video_data.get("desc", ""),
                media_type="video",
                thumbnail_url=video_data.get("cover", ""),
                media_url=video_data.get("play", ""),
                # Try stats object first, then direct fields
                likes_count=stats.get("digg_count", video_data.get("digg_count", 0)),
                comments_count=stats.get("comment_count", video_data.get("comment_count", 0)),
                views_count=stats.get("play_count", video_data.get("play_count", 0)),
                shares_count=stats.get("share_count", video_data.get("share_count", 0)),
                posted_at=datetime.fromtimestamp(video_data.get("create_time", 0)),
                is_video=True,
                duration=video_data.get("duration", 0),
                platform=Platform.TIKTOK,
                raw_data=video_data
            )
        except Exception as e:
            logger.warning(f"Failed to parse video: {e}")
            return None
    
    def get_cost_per_request(self) -> float:
        """Cost per request (ULTRA tier)"""
        return 0.0000109  # $32.80 / 3M requests


class TikTokScraper7Provider(ProviderInterface):
    """
    TikTok Scraper7 Provider (Existing fallback)
    API: tikwm-tikwm-default/tiktok-scraper7
    
    Features:
    - Original quality videos
    - Fast and stable
    - Good fallback option
    """
    
    def __init__(self, api_key: str, base_url: str, name: str, platform: Platform):
        super().__init__(api_key, base_url, name, platform)
        self.rapidapi_host = "tiktok-scraper7.p.rapidapi.com"
        self.headers["X-RapidAPI-Host"] = self.rapidapi_host
    
    async def get_profile(self, username: str) -> Optional[ProfileData]:
        """Get TikTok user info"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/user/info",
                    headers=self.headers,
                    params={"unique_id": username}
                )
                response.raise_for_status()
                data = response.json()
                
                if not data.get("data"):
                    return None
                
                user_data = data["data"]["user"]
                stats = data["data"]["stats"]
                
                return ProfileData(
                    username=user_data.get("uniqueId", username),
                    full_name=user_data.get("nickname", ""),
                    bio=user_data.get("signature", ""),
                    profile_pic_url=user_data.get("avatarLarger", ""),
                    followers_count=stats.get("followerCount", 0),
                    following_count=stats.get("followingCount", 0),
                    posts_count=stats.get("videoCount", 0),
                    is_verified=user_data.get("verified", False),
                    is_business=False,
                    platform=Platform.TIKTOK,
                    raw_data=data
                )
                
        except Exception as e:
            logger.error(f"Error fetching profile (Scraper7): {e}")
            return None
    
    async def get_posts(
        self,
        username: str,
        limit: int = 20,
        cursor: Optional[str] = None
    ) -> List[PostData]:
        """Get user videos"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                params = {"unique_id": username, "count": limit}
                if cursor:
                    params["cursor"] = cursor
                
                response = await client.get(
                    f"{self.base_url}/user/posts",
                    headers=self.headers,
                    params=params
                )
                response.raise_for_status()
                data = response.json()
                
                videos = data.get("data", {}).get("videos", [])
                return [self._parse_video(v) for v in videos if self._parse_video(v)]
                
        except Exception as e:
            logger.error(f"Error fetching posts (Scraper7): {e}")
            return []
    
    async def get_post_details(self, post_id: str) -> Optional[PostData]:
        """Get video details"""
        # Implementation similar to Feature Summary
        return None
    
    async def search_users(self, query: str, limit: int = 20) -> List[ProfileData]:
        """Search users"""
        return []  # Not implemented in this provider
    
    async def search_content(self, query: str, limit: int = 20) -> List[PostData]:
        """Search videos"""
        return []  # Not implemented in this provider
    
    def _parse_video(self, video_data: dict) -> Optional[PostData]:
        """Parse video to PostData"""
        try:
            return PostData(
                post_id=video_data.get("id", ""),
                url=video_data.get("video_url", ""),
                caption=video_data.get("desc", ""),
                media_type="video",
                thumbnail_url=video_data.get("cover", ""),
                media_url=video_data.get("play", ""),
                likes_count=video_data.get("diggCount", 0),
                comments_count=video_data.get("commentCount", 0),
                views_count=video_data.get("playCount", 0),
                shares_count=video_data.get("shareCount", 0),
                posted_at=datetime.fromtimestamp(video_data.get("createTime", 0)),
                is_video=True,
                duration=video_data.get("duration", 0),
                platform=Platform.TIKTOK,
                raw_data=video_data
            )
        except Exception as e:
            logger.warning(f"Failed to parse video (Scraper7): {e}")
            return None
