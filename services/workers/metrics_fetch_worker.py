"""
Metrics Fetch Worker
====================
Event-driven worker that auto-fetches platform metrics after content is published.

Subscribes to:
    - publish.completed (content successfully posted)

Emits:
    - metrics.fetch.started
    - metrics.fetch.completed
    - metrics.updated
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from services.event_bus import EventBus, Event, Topics
from services.workers.base import BaseWorker

logger = logging.getLogger(__name__)


class MetricsFetchWorker(BaseWorker):
    """
    Worker that automatically fetches platform metrics after content is published.
    
    Waits a configurable delay before fetching to allow platforms to process the post
    and generate initial engagement metrics.
    
    Usage:
        worker = MetricsFetchWorker()
        await worker.start()
        
        # Metrics will auto-fetch when publish.completed events are emitted
    """
    
    # Delay before fetching metrics (in seconds)
    # Different platforms need different wait times
    PLATFORM_DELAYS = {
        "tiktok": 300,      # 5 minutes - TikTok processes quickly
        "instagram": 300,   # 5 minutes
        "youtube": 600,     # 10 minutes - YouTube needs more time
        "twitter": 180,     # 3 minutes - Twitter is fast
        "threads": 300,     # 5 minutes
        "linkedin": 600,    # 10 minutes
        "facebook": 300,    # 5 minutes
        "pinterest": 600,   # 10 minutes
        "bluesky": 180,     # 3 minutes
    }
    
    DEFAULT_DELAY = 300  # 5 minutes default
    
    def __init__(self, event_bus: Optional[EventBus] = None, worker_id: Optional[str] = None):
        super().__init__(event_bus, worker_id)
        self._pending_fetches: Dict[str, asyncio.Task] = {}
    
    def get_subscriptions(self) -> List[str]:
        """Subscribe to publish completion events."""
        return [
            Topics.PUBLISH_COMPLETED,
        ]
    
    async def handle_event(self, event: Event) -> None:
        """Schedule metrics fetch after publish completion."""
        platform_url = event.payload.get("platform_url")
        media_id = event.payload.get("media_id")
        platform = event.payload.get("platform", "").lower()
        
        if not platform_url:
            logger.warning(f"[{self.worker_id}] No platform_url in publish.completed event, skipping metrics fetch")
            return
        
        # Get platform-specific delay
        delay = self.PLATFORM_DELAYS.get(platform, self.DEFAULT_DELAY)
        
        logger.info(f"[{self.worker_id}] Scheduling metrics fetch for {media_id} on {platform} in {delay}s")
        
        # Schedule the fetch
        task = asyncio.create_task(
            self._delayed_fetch_metrics(
                platform_url=platform_url,
                media_id=media_id,
                platform=platform,
                delay=delay,
                correlation_id=event.correlation_id
            )
        )
        
        # Track pending fetches
        fetch_key = f"{media_id}_{platform}"
        self._pending_fetches[fetch_key] = task
    
    async def _delayed_fetch_metrics(
        self,
        platform_url: str,
        media_id: str,
        platform: str,
        delay: int,
        correlation_id: str
    ) -> None:
        """Wait for delay then fetch metrics from platform."""
        try:
            # Wait for platform to process the post
            logger.info(f"[{self.worker_id}] Waiting {delay}s before fetching metrics for {media_id}")
            await asyncio.sleep(delay)
            
            # Emit fetch started event
            await self.emit(
                Topics.METRICS_FETCH_STARTED,
                {
                    "media_id": media_id,
                    "platform": platform,
                    "platform_url": platform_url,
                },
                correlation_id
            )
            
            # Fetch metrics based on platform
            metrics = await self._fetch_platform_metrics(platform, platform_url, media_id)
            
            if metrics:
                # Emit metrics updated event
                await self.emit(
                    Topics.METRICS_UPDATED,
                    {
                        "media_id": media_id,
                        "platform": platform,
                        "platform_url": platform_url,
                        "metrics": metrics,
                        "fetched_at": datetime.now(timezone.utc).isoformat()
                    },
                    correlation_id
                )
                
                # Store metrics in database
                await self._store_metrics(media_id, platform, platform_url, metrics)
                
                logger.info(f"[{self.worker_id}] Fetched metrics for {media_id} on {platform}: {metrics}")
            
            # Emit completion
            await self.emit(
                Topics.METRICS_FETCH_COMPLETED,
                {
                    "media_id": media_id,
                    "platform": platform,
                    "success": bool(metrics),
                    "metrics": metrics
                },
                correlation_id
            )
            
        except asyncio.CancelledError:
            logger.info(f"[{self.worker_id}] Metrics fetch cancelled for {media_id}")
        except Exception as e:
            logger.error(f"[{self.worker_id}] Failed to fetch metrics for {media_id}: {e}")
        finally:
            # Clean up tracking
            fetch_key = f"{media_id}_{platform}"
            self._pending_fetches.pop(fetch_key, None)
    
    async def _fetch_platform_metrics(
        self,
        platform: str,
        platform_url: str,
        media_id: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch metrics from the appropriate platform API."""
        try:
            if platform == "tiktok":
                return await self._fetch_tiktok_metrics(platform_url)
            elif platform == "instagram":
                return await self._fetch_instagram_metrics(platform_url)
            elif platform == "youtube":
                return await self._fetch_youtube_metrics(platform_url)
            else:
                logger.warning(f"[{self.worker_id}] No metrics fetcher for platform: {platform}")
                return None
        except Exception as e:
            logger.error(f"[{self.worker_id}] Error fetching {platform} metrics: {e}")
            return None
    
    async def _fetch_tiktok_metrics(self, platform_url: str) -> Optional[Dict[str, Any]]:
        """Fetch TikTok video metrics via RapidAPI."""
        import os
        import httpx
        
        rapidapi_key = os.getenv("RAPIDAPI_KEY")
        if not rapidapi_key:
            logger.warning("[MetricsFetch] No RAPIDAPI_KEY configured")
            return None
        
        try:
            # Extract video ID from URL
            # TikTok URLs: https://www.tiktok.com/@username/video/1234567890
            video_id = platform_url.split("/video/")[-1].split("?")[0] if "/video/" in platform_url else None
            
            if not video_id:
                logger.warning(f"[MetricsFetch] Could not extract TikTok video ID from: {platform_url}")
                return None
            
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    "https://tiktok-scraper7.p.rapidapi.com/video/info",
                    params={"video_id": video_id},
                    headers={
                        "X-RapidAPI-Key": rapidapi_key,
                        "X-RapidAPI-Host": "tiktok-scraper7.p.rapidapi.com"
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    video_data = data.get("data", {})
                    return {
                        "views": video_data.get("play_count", 0),
                        "likes": video_data.get("digg_count", 0),
                        "comments": video_data.get("comment_count", 0),
                        "shares": video_data.get("share_count", 0),
                        "saves": video_data.get("collect_count", 0),
                    }
                else:
                    logger.warning(f"[MetricsFetch] TikTok API returned {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"[MetricsFetch] TikTok metrics error: {e}")
            return None
    
    async def _fetch_instagram_metrics(self, platform_url: str) -> Optional[Dict[str, Any]]:
        """Fetch Instagram post metrics via RapidAPI."""
        import os
        import httpx
        
        rapidapi_key = os.getenv("RAPIDAPI_KEY")
        if not rapidapi_key:
            logger.warning("[MetricsFetch] No RAPIDAPI_KEY configured")
            return None
        
        try:
            # Extract shortcode from URL
            # Instagram URLs: https://www.instagram.com/reel/ABC123/ or /p/ABC123/
            shortcode = None
            for pattern in ["/reel/", "/p/"]:
                if pattern in platform_url:
                    shortcode = platform_url.split(pattern)[-1].split("/")[0].split("?")[0]
                    break
            
            if not shortcode:
                logger.warning(f"[MetricsFetch] Could not extract Instagram shortcode from: {platform_url}")
                return None
            
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    "https://instagram-looter2.p.rapidapi.com/post",
                    params={"link": platform_url},
                    headers={
                        "X-RapidAPI-Key": rapidapi_key,
                        "X-RapidAPI-Host": "instagram-looter2.p.rapidapi.com"
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "likes": data.get("edge_liked_by", {}).get("count", 0),
                        "comments": data.get("edge_media_to_comment", {}).get("count", 0),
                        "views": data.get("video_view_count", 0),
                    }
                else:
                    logger.warning(f"[MetricsFetch] Instagram API returned {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"[MetricsFetch] Instagram metrics error: {e}")
            return None
    
    async def _fetch_youtube_metrics(self, platform_url: str) -> Optional[Dict[str, Any]]:
        """Fetch YouTube video metrics."""
        # YouTube requires OAuth or API key - implementation depends on setup
        logger.info(f"[MetricsFetch] YouTube metrics fetching not yet implemented for: {platform_url}")
        return None
    
    async def _store_metrics(
        self,
        media_id: str,
        platform: str,
        platform_url: str,
        metrics: Dict[str, Any]
    ) -> None:
        """Store fetched metrics in the database."""
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=10) as client:
                # Store via posted-content API
                await client.post(
                    "http://localhost:5555/api/posted-content/metrics/update",
                    json={
                        "media_id": media_id,
                        "platform": platform,
                        "platform_url": platform_url,
                        "views": metrics.get("views", 0),
                        "likes": metrics.get("likes", 0),
                        "comments": metrics.get("comments", 0),
                        "shares": metrics.get("shares", 0),
                        "saves": metrics.get("saves", 0),
                    }
                )
                logger.info(f"[MetricsFetch] Stored metrics for {media_id} on {platform}")
                
        except Exception as e:
            logger.warning(f"[MetricsFetch] Failed to store metrics: {e}")
    
    async def stop(self) -> None:
        """Cancel pending fetches and stop worker."""
        # Cancel all pending fetch tasks
        for task in self._pending_fetches.values():
            task.cancel()
        
        self._pending_fetches.clear()
        await super().stop()
