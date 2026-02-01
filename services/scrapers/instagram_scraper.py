"""
Instagram Analytics Scraper
Pulls analytics data from Instagram accounts using multiple methods
"""
import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import httpx
import asyncio
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class InstagramPost:
    """Instagram post data structure"""
    post_id: str
    shortcode: str
    url: str
    caption: str
    media_type: str  # 'image', 'video', 'carousel'
    thumbnail_url: Optional[str]
    video_url: Optional[str]
    likes_count: int
    comments_count: int
    views_count: int
    posted_at: datetime
    is_video: bool
    duration: Optional[float]  # For videos


@dataclass
class InstagramProfile:
    """Instagram profile/account data"""
    username: str
    full_name: str
    bio: str
    profile_pic_url: str
    followers_count: int
    following_count: int
    posts_count: int
    is_verified: bool
    is_business: bool


@dataclass
class InstagramAnalytics:
    """Aggregated Instagram analytics"""
    profile: InstagramProfile
    posts: List[InstagramPost]
    total_likes: int
    total_comments: int
    total_views: int
    engagement_rate: float
    avg_likes_per_post: float
    avg_comments_per_post: float
    best_performing_post: Optional[InstagramPost]
    recent_growth: Dict[str, int]  # follower growth over time
    top_hashtags: List[str]


class InstagramScraper:
    """
    Instagram Analytics Scraper
    
    Methods:
    1. RapidAPI Instagram Scraper (Primary)
    2. Instagram Graph API (For business accounts)
    3. Instaloader (Fallback - requires login)
    """
    
    def __init__(
        self,
        rapidapi_key: Optional[str] = None,
        instagram_access_token: Optional[str] = None
    ):
        """
        Initialize Instagram scraper
        
        Args:
            rapidapi_key: RapidAPI key for Instagram scraper
            instagram_access_token: Instagram Graph API access token
        """
        self.rapidapi_key = rapidapi_key or os.getenv("RAPIDAPI_KEY")
        self.instagram_token = instagram_access_token or os.getenv("INSTAGRAM_ACCESS_TOKEN")
        
        # RapidAPI configuration
        self.rapidapi_base = "https://instagram-scraper-api2.p.rapidapi.com/v1"
        self.rapidapi_headers = {
            "X-RapidAPI-Key": self.rapidapi_key or "",
            "X-RapidAPI-Host": "instagram-scraper-api2.p.rapidapi.com"
        }
        
        # Instagram Graph API configuration
        self.graph_api_base = "https://graph.instagram.com"
        
        if not self.rapidapi_key:
            logger.warning("RAPIDAPI_KEY not configured - RapidAPI method disabled")
    
    async def get_profile_analytics(
        self,
        username: str,
        posts_limit: int = 50
    ) -> Optional[InstagramAnalytics]:
        """
        Get comprehensive analytics for an Instagram profile
        
        Args:
            username: Instagram username (without @)
            posts_limit: Number of recent posts to analyze
            
        Returns:
            InstagramAnalytics object with complete data
        """
        try:
            # Try RapidAPI first (most reliable)
            if self.rapidapi_key:
                return await self._scrape_via_rapidapi(username, posts_limit)
            
            # Fall back to Graph API if available
            elif self.instagram_token:
                return await self._scrape_via_graph_api(username, posts_limit)
            
            else:
                logger.error("No Instagram scraping method available")
                return None
                
        except Exception as e:
            logger.error(f"Error getting Instagram analytics for @{username}: {e}")
            return None
    
    async def _scrape_via_rapidapi(
        self,
        username: str,
        posts_limit: int
    ) -> Optional[InstagramAnalytics]:
        """
        Scrape Instagram using RapidAPI Instagram Scraper
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get profile info
                profile_url = f"{self.rapidapi_base}/info"
                profile_params = {"username_or_id_or_url": username}
                
                logger.info(f"Fetching Instagram profile: @{username}")
                profile_response = await client.get(
                    profile_url,
                    headers=self.rapidapi_headers,
                    params=profile_params
                )
                profile_response.raise_for_status()
                profile_data = profile_response.json()
                
                # Get recent posts
                posts_url = f"{self.rapidapi_base}/posts"
                posts_params = {
                    "username_or_id_or_url": username,
                    "count": posts_limit
                }
                
                logger.info(f"Fetching {posts_limit} recent posts")
                posts_response = await client.get(
                    posts_url,
                    headers=self.rapidapi_headers,
                    params=posts_params
                )
                posts_response.raise_for_status()
                posts_data = posts_response.json()
                
                # Parse data
                profile = self._parse_profile_rapidapi(profile_data)
                posts = self._parse_posts_rapidapi(posts_data)
                
                # Calculate analytics
                analytics = self._calculate_analytics(profile, posts)
                
                logger.info(f"✅ Scraped @{username}: {len(posts)} posts, {profile.followers_count} followers")
                return analytics
                
        except httpx.HTTPStatusError as e:
            logger.error(f"RapidAPI HTTP error: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"RapidAPI scraping error: {e}")
            return None
    
    async def _scrape_via_graph_api(
        self,
        username: str,
        posts_limit: int
    ) -> Optional[InstagramAnalytics]:
        """
        Scrape Instagram using official Graph API (Business accounts only)
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get user ID first
                user_url = f"{self.graph_api_base}/me"
                user_params = {
                    "fields": "id,username,name,biography,profile_picture_url,followers_count,follows_count,media_count",
                    "access_token": self.instagram_token
                }
                
                user_response = await client.get(user_url, params=user_params)
                user_response.raise_for_status()
                user_data = user_response.json()
                
                # Get media
                media_url = f"{self.graph_api_base}/{user_data['id']}/media"
                media_params = {
                    "fields": "id,caption,media_type,media_url,permalink,thumbnail_url,timestamp,like_count,comments_count,insights.metric(engagement,impressions,reach,saved,video_views)",
                    "limit": posts_limit,
                    "access_token": self.instagram_token
                }
                
                media_response = await client.get(media_url, params=media_params)
                media_response.raise_for_status()
                media_data = media_response.json()
                
                # Parse data
                profile = self._parse_profile_graph_api(user_data)
                posts = self._parse_posts_graph_api(media_data)
                
                # Calculate analytics
                analytics = self._calculate_analytics(profile, posts)
                
                logger.info(f"✅ Scraped @{username} via Graph API")
                return analytics
                
        except Exception as e:
            logger.error(f"Graph API scraping error: {e}")
            return None
    
    def _parse_profile_rapidapi(self, data: Dict) -> InstagramProfile:
        """Parse profile data from RapidAPI response"""
        user_data = data.get("data", {})
        
        return InstagramProfile(
            username=user_data.get("username", ""),
            full_name=user_data.get("full_name", ""),
            bio=user_data.get("biography", ""),
            profile_pic_url=user_data.get("profile_pic_url", ""),
            followers_count=user_data.get("follower_count", 0),
            following_count=user_data.get("following_count", 0),
            posts_count=user_data.get("media_count", 0),
            is_verified=user_data.get("is_verified", False),
            is_business=user_data.get("is_business_account", False)
        )
    
    def _parse_posts_rapidapi(self, data: Dict) -> List[InstagramPost]:
        """Parse posts data from RapidAPI response"""
        posts = []
        items = data.get("data", {}).get("items", [])
        
        for item in items:
            try:
                post = InstagramPost(
                    post_id=item.get("id", ""),
                    shortcode=item.get("code", ""),
                    url=f"https://www.instagram.com/p/{item.get('code', '')}/",
                    caption=item.get("caption", {}).get("text", ""),
                    media_type=item.get("media_type", "unknown"),
                    thumbnail_url=item.get("thumbnail_url"),
                    video_url=item.get("video_url") if item.get("media_type") == "video" else None,
                    likes_count=item.get("like_count", 0),
                    comments_count=item.get("comment_count", 0),
                    views_count=item.get("view_count", 0) if item.get("media_type") == "video" else 0,
                    posted_at=datetime.fromtimestamp(item.get("taken_at", 0)),
                    is_video=item.get("media_type") == "video",
                    duration=item.get("video_duration") if item.get("media_type") == "video" else None
                )
                posts.append(post)
            except Exception as e:
                logger.warning(f"Failed to parse post: {e}")
                continue
        
        return posts
    
    def _parse_profile_graph_api(self, data: Dict) -> InstagramProfile:
        """Parse profile data from Graph API response"""
        return InstagramProfile(
            username=data.get("username", ""),
            full_name=data.get("name", ""),
            bio=data.get("biography", ""),
            profile_pic_url=data.get("profile_picture_url", ""),
            followers_count=data.get("followers_count", 0),
            following_count=data.get("follows_count", 0),
            posts_count=data.get("media_count", 0),
            is_verified=False,  # Not available in basic Graph API
            is_business=True  # Assumed since using Graph API
        )
    
    def _parse_posts_graph_api(self, data: Dict) -> List[InstagramPost]:
        """Parse posts data from Graph API response"""
        posts = []
        items = data.get("data", [])
        
        for item in items:
            try:
                insights = item.get("insights", {}).get("data", [])
                views = next((i["values"][0]["value"] for i in insights if i["name"] == "video_views"), 0)
                
                post = InstagramPost(
                    post_id=item.get("id", ""),
                    shortcode=item.get("permalink", "").split("/p/")[1].rstrip("/") if "/p/" in item.get("permalink", "") else "",
                    url=item.get("permalink", ""),
                    caption=item.get("caption", ""),
                    media_type=item.get("media_type", "").lower(),
                    thumbnail_url=item.get("thumbnail_url"),
                    video_url=item.get("media_url") if item.get("media_type") == "VIDEO" else None,
                    likes_count=item.get("like_count", 0),
                    comments_count=item.get("comments_count", 0),
                    views_count=views,
                    posted_at=datetime.fromisoformat(item.get("timestamp", "").replace("Z", "+00:00")),
                    is_video=item.get("media_type") == "VIDEO",
                    duration=None  # Not available in basic response
                )
                posts.append(post)
            except Exception as e:
                logger.warning(f"Failed to parse post: {e}")
                continue
        
        return posts
    
    def _calculate_analytics(
        self,
        profile: InstagramProfile,
        posts: List[InstagramPost]
    ) -> InstagramAnalytics:
        """Calculate aggregated analytics from profile and posts"""
        
        if not posts:
            return InstagramAnalytics(
                profile=profile,
                posts=[],
                total_likes=0,
                total_comments=0,
                total_views=0,
                engagement_rate=0.0,
                avg_likes_per_post=0.0,
                avg_comments_per_post=0.0,
                best_performing_post=None,
                recent_growth={},
                top_hashtags=[]
            )
        
        # Calculate totals
        total_likes = sum(p.likes_count for p in posts)
        total_comments = sum(p.comments_count for p in posts)
        total_views = sum(p.views_count for p in posts)
        
        # Calculate averages
        avg_likes = total_likes / len(posts)
        avg_comments = total_comments / len(posts)
        
        # Calculate engagement rate
        total_engagement = total_likes + total_comments
        engagement_rate = (total_engagement / (len(posts) * profile.followers_count)) * 100 if profile.followers_count > 0 else 0
        
        # Find best post
        best_post = max(posts, key=lambda p: p.likes_count + p.comments_count)
        
        # Extract top hashtags
        all_hashtags = []
        for post in posts:
            hashtags = [word for word in post.caption.split() if word.startswith('#')]
            all_hashtags.extend(hashtags)
        
        from collections import Counter
        top_hashtags = [tag for tag, count in Counter(all_hashtags).most_common(10)]
        
        return InstagramAnalytics(
            profile=profile,
            posts=posts,
            total_likes=total_likes,
            total_comments=total_comments,
            total_views=total_views,
            engagement_rate=round(engagement_rate, 2),
            avg_likes_per_post=round(avg_likes, 2),
            avg_comments_per_post=round(avg_comments, 2),
            best_performing_post=best_post,
            recent_growth={},  # Would need historical data
            top_hashtags=top_hashtags
        )
    
    async def get_post_analytics(self, post_url: str) -> Optional[InstagramPost]:
        """
        Get analytics for a specific Instagram post
        
        Args:
            post_url: Full Instagram post URL
            
        Returns:
            InstagramPost object with metrics
        """
        try:
            # Extract shortcode from URL
            shortcode = post_url.split("/p/")[1].rstrip("/") if "/p/" in post_url else None
            if not shortcode:
                logger.error("Invalid Instagram post URL")
                return None
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                url = f"{self.rapidapi_base}/post/info"
                params = {"code_or_id_or_url": post_url}
                
                response = await client.get(
                    url,
                    headers=self.rapidapi_headers,
                    params=params
                )
                response.raise_for_status()
                data = response.json()
                
                # Parse single post
                post_data = data.get("data", {})
                post = InstagramPost(
                    post_id=post_data.get("id", ""),
                    shortcode=shortcode,
                    url=post_url,
                    caption=post_data.get("caption", {}).get("text", ""),
                    media_type=post_data.get("media_type", "unknown"),
                    thumbnail_url=post_data.get("thumbnail_url"),
                    video_url=post_data.get("video_url"),
                    likes_count=post_data.get("like_count", 0),
                    comments_count=post_data.get("comment_count", 0),
                    views_count=post_data.get("view_count", 0),
                    posted_at=datetime.fromtimestamp(post_data.get("taken_at", 0)),
                    is_video=post_data.get("media_type") == "video",
                    duration=post_data.get("video_duration")
                )
                
                logger.info(f"✅ Fetched post analytics: {post.likes_count} likes, {post.comments_count} comments")
                return post
                
        except Exception as e:
            logger.error(f"Error fetching post analytics: {e}")
            return None
    
    async def search_hashtag(self, hashtag: str, limit: int = 20) -> List[InstagramPost]:
        """
        Search Instagram posts by hashtag
        
        Args:
            hashtag: Hashtag to search (without #)
            limit: Number of posts to retrieve
            
        Returns:
            List of InstagramPost objects
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                url = f"{self.rapidapi_base}/hashtag"
                params = {
                    "hashtag": hashtag.lstrip("#"),
                    "count": limit
                }
                
                response = await client.get(
                    url,
                    headers=self.rapidapi_headers,
                    params=params
                )
                response.raise_for_status()
                data = response.json()
                
                posts = self._parse_posts_rapidapi(data)
                logger.info(f"✅ Found {len(posts)} posts for #{hashtag}")
                return posts
                
        except Exception as e:
            logger.error(f"Error searching hashtag: {e}")
            return []
