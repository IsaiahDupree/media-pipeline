"""
Instagram Provider Implementations
Multiple RapidAPI providers for Instagram with swappable interface
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


class InstagramStatisticsProvider(ProviderInterface):
    """
    Instagram Statistics API Provider (PRIMARY)
    API: artemlipko/instagram-statistics-api
    
    Features:
    - Universal multi-platform API
    - Advanced analytics and demographics
    - Historical data
    - Influencer metrics
    - Fake follower detection
    """
    
    def __init__(self, api_key: str, base_url: str, name: str, platform: Platform):
        super().__init__(api_key, base_url, name, platform)
        self.rapidapi_host = "instagram-statistics-api.p.rapidapi.com"
        self.headers["X-RapidAPI-Host"] = self.rapidapi_host
    
    async def get_profile(self, username: str) -> Optional[ProfileData]:
        """Get Instagram profile with analytics"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/user/info",
                    headers=self.headers,
                    params={"username": username}
                )
                response.raise_for_status()
                data = response.json()
                
                if not data.get("success"):
                    logger.error(f"API error: {data.get('message')}")
                    return None
                
                user_data = data.get("data", {})
                
                return ProfileData(
                    username=user_data.get("username", username),
                    full_name=user_data.get("full_name", ""),
                    bio=user_data.get("biography", ""),
                    profile_pic_url=user_data.get("profile_pic_url_hd", ""),
                    followers_count=user_data.get("follower_count", 0),
                    following_count=user_data.get("following_count", 0),
                    posts_count=user_data.get("media_count", 0),
                    is_verified=user_data.get("is_verified", False),
                    is_business=user_data.get("is_business_account", False),
                    platform=Platform.INSTAGRAM,
                    raw_data=data
                )
                
        except Exception as e:
            logger.error(f"Error fetching Instagram profile: {e}")
            return None
    
    async def get_posts(
        self,
        username: str,
        limit: int = 20,
        cursor: Optional[str] = None
    ) -> List[PostData]:
        """Get user's Instagram posts"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                params = {"username": username, "count": limit}
                if cursor:
                    params["end_cursor"] = cursor
                
                response = await client.get(
                    f"{self.base_url}/user/media",
                    headers=self.headers,
                    params=params
                )
                response.raise_for_status()
                data = response.json()
                
                if not data.get("success"):
                    return []
                
                items = data.get("data", {}).get("items", [])
                return [self._parse_post(item) for item in items if self._parse_post(item)]
                
        except Exception as e:
            logger.error(f"Error fetching Instagram posts: {e}")
            return []
    
    async def get_post_details(self, post_id: str) -> Optional[PostData]:
        """Get detailed post info"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/media/info",
                    headers=self.headers,
                    params={"media_id": post_id}
                )
                response.raise_for_status()
                data = response.json()
                
                if not data.get("success"):
                    return None
                
                return self._parse_post(data.get("data", {}))
                
        except Exception as e:
            logger.error(f"Error fetching post details: {e}")
            return None
    
    async def search_users(self, query: str, limit: int = 20) -> List[ProfileData]:
        """Search Instagram users"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/user/search",
                    headers=self.headers,
                    params={"query": query, "count": limit}
                )
                response.raise_for_status()
                data = response.json()
                
                if not data.get("success"):
                    return []
                
                users = data.get("data", {}).get("users", [])
                profiles = []
                
                for user in users:
                    profiles.append(ProfileData(
                        username=user.get("username", ""),
                        full_name=user.get("full_name", ""),
                        bio=user.get("biography", ""),
                        profile_pic_url=user.get("profile_pic_url", ""),
                        followers_count=user.get("follower_count", 0),
                        following_count=user.get("following_count", 0),
                        posts_count=user.get("media_count", 0),
                        is_verified=user.get("is_verified", False),
                        is_business=user.get("is_business", False),
                        platform=Platform.INSTAGRAM,
                        raw_data=user
                    ))
                
                return profiles
                
        except Exception as e:
            logger.error(f"Error searching users: {e}")
            return []
    
    async def search_content(self, query: str, limit: int = 20) -> List[PostData]:
        """Search Instagram hashtags/content"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/hashtag/media",
                    headers=self.headers,
                    params={"hashtag": query.lstrip("#"), "count": limit}
                )
                response.raise_for_status()
                data = response.json()
                
                if not data.get("success"):
                    return []
                
                items = data.get("data", {}).get("items", [])
                return [self._parse_post(item) for item in items if self._parse_post(item)]
                
        except Exception as e:
            logger.error(f"Error searching content: {e}")
            return []
    
    def _parse_post(self, post_data: dict) -> Optional[PostData]:
        """Parse post data to PostData"""
        try:
            media_type = "video" if post_data.get("is_video", False) else "image"
            if post_data.get("product_type") == "carousel_container":
                media_type = "carousel"
            
            return PostData(
                post_id=post_data.get("id", ""),
                url=f"https://www.instagram.com/p/{post_data.get('code', '')}/",
                caption=post_data.get("caption", {}).get("text", "") if post_data.get("caption") else "",
                media_type=media_type,
                thumbnail_url=post_data.get("thumbnail_url") or post_data.get("display_url"),
                media_url=post_data.get("video_url") if media_type == "video" else post_data.get("display_url"),
                likes_count=post_data.get("like_count", 0),
                comments_count=post_data.get("comment_count", 0),
                views_count=post_data.get("video_view_count", 0) if media_type == "video" else 0,
                shares_count=0,  # Not available
                posted_at=datetime.fromtimestamp(post_data.get("taken_at_timestamp", 0)),
                is_video=media_type == "video",
                duration=post_data.get("video_duration") if media_type == "video" else None,
                platform=Platform.INSTAGRAM,
                raw_data=post_data
            )
        except Exception as e:
            logger.warning(f"Failed to parse post: {e}")
            return None


class InstagramPremiumProvider(ProviderInterface):
    """
    Instagram Premium API 2023 (FALLBACK)
    API: NikitusLLP/instagram-premium-api-2023 (HikerAPI reseller)
    
    Features:
    - Basic Instagram scraping
    - Stories and highlights
    - Reels and media
    - Good for high-volume requests
    """
    
    def __init__(self, api_key: str, base_url: str, name: str, platform: Platform):
        super().__init__(api_key, base_url, name, platform)
        self.rapidapi_host = "instagram-premium-api-2023.p.rapidapi.com"
        self.headers["X-RapidAPI-Host"] = self.rapidapi_host
    
    async def get_profile(self, username: str) -> Optional[ProfileData]:
        """Get Instagram user info"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/user_info",
                    headers=self.headers,
                    params={"username": username}
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get("status") != "ok":
                    return None
                
                user_data = data.get("data", {})
                
                return ProfileData(
                    username=user_data.get("username", username),
                    full_name=user_data.get("full_name", ""),
                    bio=user_data.get("biography", ""),
                    profile_pic_url=user_data.get("profile_pic_url_hd", ""),
                    followers_count=user_data.get("edge_followed_by", {}).get("count", 0),
                    following_count=user_data.get("edge_follow", {}).get("count", 0),
                    posts_count=user_data.get("edge_owner_to_timeline_media", {}).get("count", 0),
                    is_verified=user_data.get("is_verified", False),
                    is_business=user_data.get("is_business_account", False),
                    platform=Platform.INSTAGRAM,
                    raw_data=data
                )
                
        except Exception as e:
            logger.error(f"Error fetching profile (Premium): {e}")
            return None
    
    async def get_posts(
        self,
        username: str,
        limit: int = 20,
        cursor: Optional[str] = None
    ) -> List[PostData]:
        """Get user posts"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                params = {"username": username, "first": limit}
                if cursor:
                    params["after"] = cursor
                
                response = await client.get(
                    f"{self.base_url}/user_posts",
                    headers=self.headers,
                    params=params
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get("status") != "ok":
                    return []
                
                edges = data.get("data", {}).get("edges", [])
                return [self._parse_post(edge.get("node", {})) for edge in edges if self._parse_post(edge.get("node", {}))]
                
        except Exception as e:
            logger.error(f"Error fetching posts (Premium): {e}")
            return []
    
    async def get_post_details(self, post_id: str) -> Optional[PostData]:
        """Get post details"""
        return None  # Implement if needed
    
    async def search_users(self, query: str, limit: int = 20) -> List[ProfileData]:
        """Search users"""
        return []  # Not implemented in this provider
    
    async def search_content(self, query: str, limit: int = 20) -> List[PostData]:
        """Search content"""
        return []  # Not implemented in this provider
    
    def _parse_post(self, post_data: dict) -> Optional[PostData]:
        """Parse post to PostData"""
        try:
            is_video = post_data.get("is_video", False)
            media_type = "video" if is_video else "image"
            
            return PostData(
                post_id=post_data.get("id", ""),
                url=f"https://www.instagram.com/p/{post_data.get('shortcode', '')}/",
                caption=post_data.get("edge_media_to_caption", {}).get("edges", [{}])[0].get("node", {}).get("text", ""),
                media_type=media_type,
                thumbnail_url=post_data.get("thumbnail_src") or post_data.get("display_url"),
                media_url=post_data.get("video_url") if is_video else post_data.get("display_url"),
                likes_count=post_data.get("edge_liked_by", {}).get("count", 0),
                comments_count=post_data.get("edge_media_to_comment", {}).get("count", 0),
                views_count=post_data.get("video_view_count", 0) if is_video else 0,
                shares_count=0,
                posted_at=datetime.fromtimestamp(post_data.get("taken_at_timestamp", 0)),
                is_video=is_video,
                duration=post_data.get("video_duration") if is_video else None,
                platform=Platform.INSTAGRAM,
                raw_data=post_data
            )
        except Exception as e:
            logger.warning(f"Failed to parse post (Premium): {e}")
            return None
