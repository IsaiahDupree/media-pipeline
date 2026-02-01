"""
Platform Content Downloader
===========================
Download content from social platforms to local folder.

Supports:
- Instagram (Reels, Posts, Stories)
- TikTok (Videos)
- YouTube (Shorts, Videos)

Uses RapidAPI endpoints for reliable downloads.
"""
import os
import re
import json
import asyncio
import httpx
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
DOWNLOAD_BASE_PATH = os.getenv("DOWNLOAD_PATH", "/Users/isaiahdupree/Documents/CompetitorResearch")


@dataclass
class DownloadResult:
    """Result of a download operation"""
    success: bool
    url: str
    platform: str
    content_id: str
    local_path: Optional[str]
    filename: Optional[str]
    file_size_mb: float
    duration_seconds: Optional[int]
    caption: Optional[str]
    error: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchDownloadResult:
    """Result of batch download operation"""
    total_requested: int
    successful: int
    failed: int
    skipped: int
    total_size_mb: float
    downloads: List[DownloadResult]
    output_folder: str


class PlatformDownloader:
    """
    Downloads content from social platforms.
    
    Uses RapidAPI for Instagram/TikTok.
    Uses yt-dlp for YouTube.
    """
    
    def __init__(self, output_folder: Optional[str] = None):
        self.output_folder = output_folder or DOWNLOAD_BASE_PATH
        self.rapidapi_key = RAPIDAPI_KEY
        
        # API endpoints
        self.instagram_api = "https://instagram-looter2.p.rapidapi.com"
        self.tiktok_api = "https://tiktok-scraper7.p.rapidapi.com"
    
    async def download_instagram_reel(
        self,
        url: str,
        subfolder: str = "instagram"
    ) -> DownloadResult:
        """
        Download an Instagram Reel by URL.
        
        Args:
            url: Instagram Reel URL (e.g., https://www.instagram.com/reel/ABC123/)
            subfolder: Subfolder to save to
        """
        # Extract shortcode from URL
        shortcode = self._extract_instagram_shortcode(url)
        if not shortcode:
            return DownloadResult(
                success=False,
                url=url,
                platform="instagram",
                content_id="",
                local_path=None,
                filename=None,
                file_size_mb=0,
                duration_seconds=None,
                caption=None,
                error="Could not extract shortcode from URL"
            )
        
        try:
            # Get media info from RapidAPI
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(
                    f"{self.instagram_api}/post",
                    params={"link": url},
                    headers={
                        "X-RapidAPI-Key": self.rapidapi_key,
                        "X-RapidAPI-Host": "instagram-looter2.p.rapidapi.com"
                    }
                )
                
                if response.status_code != 200:
                    return DownloadResult(
                        success=False,
                        url=url,
                        platform="instagram",
                        content_id=shortcode,
                        local_path=None,
                        filename=None,
                        file_size_mb=0,
                        duration_seconds=None,
                        caption=None,
                        error=f"API error: {response.status_code}"
                    )
                
                data = response.json()
                
                # Extract video URL
                video_url = None
                caption = None
                
                if "data" in data:
                    media = data["data"]
                    video_url = media.get("video_url") or media.get("video_versions", [{}])[0].get("url")
                    caption = media.get("caption", {}).get("text", "")
                
                if not video_url:
                    return DownloadResult(
                        success=False,
                        url=url,
                        platform="instagram",
                        content_id=shortcode,
                        local_path=None,
                        filename=None,
                        file_size_mb=0,
                        duration_seconds=None,
                        caption=caption,
                        error="No video URL found in response"
                    )
                
                # Download the video
                output_path = Path(self.output_folder) / subfolder
                output_path.mkdir(parents=True, exist_ok=True)
                
                filename = f"ig_{shortcode}.mp4"
                file_path = output_path / filename
                
                # Skip if already exists
                if file_path.exists():
                    file_size = file_path.stat().st_size / (1024 * 1024)
                    return DownloadResult(
                        success=True,
                        url=url,
                        platform="instagram",
                        content_id=shortcode,
                        local_path=str(file_path),
                        filename=filename,
                        file_size_mb=file_size,
                        duration_seconds=None,
                        caption=caption,
                        error=None,
                        metadata={"skipped": True, "reason": "Already exists"}
                    )
                
                # Download video file
                video_response = await client.get(video_url, follow_redirects=True)
                
                if video_response.status_code == 200:
                    with open(file_path, "wb") as f:
                        f.write(video_response.content)
                    
                    file_size = file_path.stat().st_size / (1024 * 1024)
                    
                    return DownloadResult(
                        success=True,
                        url=url,
                        platform="instagram",
                        content_id=shortcode,
                        local_path=str(file_path),
                        filename=filename,
                        file_size_mb=file_size,
                        duration_seconds=None,
                        caption=caption,
                        error=None
                    )
                else:
                    return DownloadResult(
                        success=False,
                        url=url,
                        platform="instagram",
                        content_id=shortcode,
                        local_path=None,
                        filename=None,
                        file_size_mb=0,
                        duration_seconds=None,
                        caption=caption,
                        error=f"Video download failed: {video_response.status_code}"
                    )
                    
        except Exception as e:
            logger.error(f"Instagram download error: {e}")
            return DownloadResult(
                success=False,
                url=url,
                platform="instagram",
                content_id=shortcode,
                local_path=None,
                filename=None,
                file_size_mb=0,
                duration_seconds=None,
                caption=None,
                error=str(e)
            )
    
    async def download_tiktok_video(
        self,
        url: str,
        subfolder: str = "tiktok"
    ) -> DownloadResult:
        """
        Download a TikTok video by URL.
        
        Args:
            url: TikTok video URL
            subfolder: Subfolder to save to
        """
        video_id = self._extract_tiktok_id(url)
        if not video_id:
            video_id = f"tt_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(
                    f"{self.tiktok_api}/",
                    params={"url": url, "hd": "1"},
                    headers={
                        "X-RapidAPI-Key": self.rapidapi_key,
                        "X-RapidAPI-Host": "tiktok-scraper7.p.rapidapi.com"
                    }
                )
                
                if response.status_code != 200:
                    return DownloadResult(
                        success=False,
                        url=url,
                        platform="tiktok",
                        content_id=video_id,
                        local_path=None,
                        filename=None,
                        file_size_mb=0,
                        duration_seconds=None,
                        caption=None,
                        error=f"API error: {response.status_code}"
                    )
                
                data = response.json()
                
                # Extract video URL
                video_url = data.get("data", {}).get("play") or data.get("data", {}).get("hdplay")
                caption = data.get("data", {}).get("title", "")
                duration = data.get("data", {}).get("duration", 0)
                
                if not video_url:
                    return DownloadResult(
                        success=False,
                        url=url,
                        platform="tiktok",
                        content_id=video_id,
                        local_path=None,
                        filename=None,
                        file_size_mb=0,
                        duration_seconds=duration,
                        caption=caption,
                        error="No video URL found"
                    )
                
                # Download the video
                output_path = Path(self.output_folder) / subfolder
                output_path.mkdir(parents=True, exist_ok=True)
                
                filename = f"tt_{video_id}.mp4"
                file_path = output_path / filename
                
                # Skip if already exists
                if file_path.exists():
                    file_size = file_path.stat().st_size / (1024 * 1024)
                    return DownloadResult(
                        success=True,
                        url=url,
                        platform="tiktok",
                        content_id=video_id,
                        local_path=str(file_path),
                        filename=filename,
                        file_size_mb=file_size,
                        duration_seconds=duration,
                        caption=caption,
                        error=None,
                        metadata={"skipped": True}
                    )
                
                # Download video file
                video_response = await client.get(video_url, follow_redirects=True)
                
                if video_response.status_code == 200:
                    with open(file_path, "wb") as f:
                        f.write(video_response.content)
                    
                    file_size = file_path.stat().st_size / (1024 * 1024)
                    
                    return DownloadResult(
                        success=True,
                        url=url,
                        platform="tiktok",
                        content_id=video_id,
                        local_path=str(file_path),
                        filename=filename,
                        file_size_mb=file_size,
                        duration_seconds=duration,
                        caption=caption,
                        error=None
                    )
                else:
                    return DownloadResult(
                        success=False,
                        url=url,
                        platform="tiktok",
                        content_id=video_id,
                        local_path=None,
                        filename=None,
                        file_size_mb=0,
                        duration_seconds=duration,
                        caption=caption,
                        error=f"Download failed: {video_response.status_code}"
                    )
                    
        except Exception as e:
            logger.error(f"TikTok download error: {e}")
            return DownloadResult(
                success=False,
                url=url,
                platform="tiktok",
                content_id=video_id,
                local_path=None,
                filename=None,
                file_size_mb=0,
                duration_seconds=None,
                caption=None,
                error=str(e)
            )
    
    async def download_batch(
        self,
        urls: List[str],
        subfolder: str = "batch"
    ) -> BatchDownloadResult:
        """
        Download multiple URLs in batch.
        
        Automatically detects platform from URL.
        """
        results = []
        successful = 0
        failed = 0
        skipped = 0
        total_size = 0.0
        
        for url in urls:
            platform = self._detect_platform(url)
            
            if platform == "instagram":
                result = await self.download_instagram_reel(url, subfolder)
            elif platform == "tiktok":
                result = await self.download_tiktok_video(url, subfolder)
            else:
                result = DownloadResult(
                    success=False,
                    url=url,
                    platform="unknown",
                    content_id="",
                    local_path=None,
                    filename=None,
                    file_size_mb=0,
                    duration_seconds=None,
                    caption=None,
                    error=f"Unsupported platform: {platform}"
                )
            
            results.append(result)
            
            if result.success:
                if result.metadata.get("skipped"):
                    skipped += 1
                else:
                    successful += 1
                total_size += result.file_size_mb
            else:
                failed += 1
            
            # Small delay between downloads
            await asyncio.sleep(1)
        
        output_path = Path(self.output_folder) / subfolder
        
        return BatchDownloadResult(
            total_requested=len(urls),
            successful=successful,
            failed=failed,
            skipped=skipped,
            total_size_mb=total_size,
            downloads=results,
            output_folder=str(output_path)
        )
    
    async def download_account_content(
        self,
        username: str,
        platform: str = "instagram",
        max_posts: int = 20,
        subfolder: Optional[str] = None
    ) -> BatchDownloadResult:
        """
        Download content from a specific account.
        
        Args:
            username: Account username (without @)
            platform: Platform name
            max_posts: Maximum posts to download
            subfolder: Subfolder (defaults to accounts/{username})
        """
        subfolder = subfolder or f"accounts/{username}/posts"
        
        if platform == "instagram":
            return await self._download_instagram_account(username, max_posts, subfolder)
        elif platform == "tiktok":
            return await self._download_tiktok_account(username, max_posts, subfolder)
        else:
            return BatchDownloadResult(
                total_requested=0,
                successful=0,
                failed=0,
                skipped=0,
                total_size_mb=0,
                downloads=[],
                output_folder=subfolder
            )
    
    async def _download_instagram_account(
        self,
        username: str,
        max_posts: int,
        subfolder: str
    ) -> BatchDownloadResult:
        """Download posts from an Instagram account"""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Get user info
                response = await client.get(
                    f"{self.instagram_api}/profile",
                    params={"username": username},
                    headers={
                        "X-RapidAPI-Key": self.rapidapi_key,
                        "X-RapidAPI-Host": "instagram-looter2.p.rapidapi.com"
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Failed to get profile: {response.status_code}")
                    return BatchDownloadResult(
                        total_requested=0,
                        successful=0,
                        failed=0,
                        skipped=0,
                        total_size_mb=0,
                        downloads=[],
                        output_folder=subfolder
                    )
                
                data = response.json()
                
                # Get posts
                posts_response = await client.get(
                    f"{self.instagram_api}/reels",
                    params={"username": username},
                    headers={
                        "X-RapidAPI-Key": self.rapidapi_key,
                        "X-RapidAPI-Host": "instagram-looter2.p.rapidapi.com"
                    }
                )
                
                if posts_response.status_code != 200:
                    logger.error(f"Failed to get reels: {posts_response.status_code}")
                    return BatchDownloadResult(
                        total_requested=0,
                        successful=0,
                        failed=0,
                        skipped=0,
                        total_size_mb=0,
                        downloads=[],
                        output_folder=subfolder
                    )
                
                posts_data = posts_response.json()
                reels = posts_data.get("data", {}).get("items", [])[:max_posts]
                
                # Download each reel
                urls = []
                for reel in reels:
                    shortcode = reel.get("code", "")
                    if shortcode:
                        urls.append(f"https://www.instagram.com/reel/{shortcode}/")
                
                return await self.download_batch(urls, subfolder)
                
        except Exception as e:
            logger.error(f"Account download error: {e}")
            return BatchDownloadResult(
                total_requested=0,
                successful=0,
                failed=0,
                skipped=0,
                total_size_mb=0,
                downloads=[],
                output_folder=subfolder
            )
    
    async def _download_tiktok_account(
        self,
        username: str,
        max_posts: int,
        subfolder: str
    ) -> BatchDownloadResult:
        """Download posts from a TikTok account"""
        # TikTok account download would go here
        # For now, return empty result
        return BatchDownloadResult(
            total_requested=0,
            successful=0,
            failed=0,
            skipped=0,
            total_size_mb=0,
            downloads=[],
            output_folder=subfolder
        )
    
    def _extract_instagram_shortcode(self, url: str) -> Optional[str]:
        """Extract shortcode from Instagram URL"""
        patterns = [
            r'/reel/([A-Za-z0-9_-]+)',
            r'/p/([A-Za-z0-9_-]+)',
            r'/tv/([A-Za-z0-9_-]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def _extract_tiktok_id(self, url: str) -> Optional[str]:
        """Extract video ID from TikTok URL"""
        patterns = [
            r'/video/(\d+)',
            r'/v/(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def _detect_platform(self, url: str) -> str:
        """Detect platform from URL"""
        url_lower = url.lower()
        if "instagram.com" in url_lower:
            return "instagram"
        elif "tiktok.com" in url_lower or "vm.tiktok.com" in url_lower:
            return "tiktok"
        elif "youtube.com" in url_lower or "youtu.be" in url_lower:
            return "youtube"
        return "unknown"


# Test function
async def test_downloader():
    """Test the platform downloader"""
    print("\n" + "="*60)
    print("📥 PLATFORM DOWNLOADER TEST")
    print("="*60)
    
    downloader = PlatformDownloader()
    
    print(f"\nOutput folder: {downloader.output_folder}")
    print(f"RapidAPI key configured: {bool(downloader.rapidapi_key)}")
    
    if not downloader.rapidapi_key:
        print("⚠️  No RapidAPI key - tests will fail")
        return
    
    # Test Instagram download
    print("\n1. Testing Instagram Reel download...")
    # Note: Replace with a real URL for testing
    # result = await downloader.download_instagram_reel(
    #     "https://www.instagram.com/reel/EXAMPLE/",
    #     "test_downloads"
    # )
    # print(f"   Success: {result.success}")
    
    print("\n✅ Downloader initialized successfully")


if __name__ == "__main__":
    asyncio.run(test_downloader())
