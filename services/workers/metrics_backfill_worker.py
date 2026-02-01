"""
Metrics Backfill Worker
========================
Event-driven worker that backfills historical metrics for posted content.

Subscribes to:
    - metrics.backfill.requested (manual trigger)
    - system.startup (optional daily backfill)

Emits:
    - metrics.backfill.started
    - metrics.backfill.progress
    - metrics.backfill.completed
    - metrics.updated (per-post)
"""

import asyncio
import logging
import os
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta

from services.event_bus import EventBus, Event, Topics
from services.workers.base import BaseWorker

logger = logging.getLogger(__name__)


class MetricsBackfillWorker(BaseWorker):
    """
    Worker that backfills metrics for historical posted content.
    
    Can be triggered manually or scheduled to run periodically.
    Fetches metrics from TikTok and Instagram via RapidAPI.
    
    Usage:
        worker = MetricsBackfillWorker()
        await worker.start()
        
        # Trigger via event
        await event_bus.publish("metrics.backfill.requested", {
            "platform": "tiktok",  # optional filter
            "days_back": 7,        # how far back to go
            "limit": 50            # max posts to update
        })
    """
    
    # Rate limiting
    REQUESTS_PER_MINUTE = 10
    
    def __init__(self, event_bus: Optional[EventBus] = None, worker_id: Optional[str] = None):
        super().__init__(event_bus, worker_id)
        self._is_running = False
    
    def get_subscriptions(self) -> List[str]:
        """Subscribe to backfill request events."""
        return [
            Topics.METRICS_FETCH_REQUESTED,
        ]
    
    async def handle_event(self, event: Event) -> None:
        """Process backfill request."""
        if self._is_running:
            logger.warning(f"[{self.worker_id}] Backfill already in progress, skipping")
            return
        
        platform = event.payload.get("platform")  # Optional filter
        days_back = event.payload.get("days_back", 7)
        limit = event.payload.get("limit", 50)
        
        await self.run_backfill(
            platform=platform,
            days_back=days_back,
            limit=limit,
            correlation_id=event.correlation_id
        )
    
    async def run_backfill(
        self,
        platform: Optional[str] = None,
        days_back: int = 7,
        limit: int = 50,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run metrics backfill for posted content."""
        self._is_running = True
        correlation_id = correlation_id or f"backfill-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        results = {
            "updated": 0,
            "failed": 0,
            "skipped": 0,
            "platforms": {}
        }
        
        try:
            # Emit started event
            await self.emit(
                Topics.METRICS_FETCH_STARTED,
                {
                    "platform": platform,
                    "days_back": days_back,
                    "limit": limit,
                    "type": "backfill"
                },
                correlation_id
            )
            
            logger.info(f"[{self.worker_id}] Starting backfill: platform={platform}, days={days_back}, limit={limit}")
            
            # Get posts to update
            posts = await self._get_posts_to_update(platform, days_back, limit)
            total = len(posts)
            
            logger.info(f"[{self.worker_id}] Found {total} posts to update")
            
            for i, post in enumerate(posts):
                try:
                    post_platform = post.get("platform", "").lower()
                    platform_url = post.get("platform_url")
                    post_id = post.get("id")
                    
                    if not platform_url:
                        results["skipped"] += 1
                        continue
                    
                    # Fetch metrics
                    metrics = await self._fetch_metrics(post_platform, platform_url)
                    
                    if metrics:
                        # Update database
                        await self._update_post_metrics(post_id, metrics)
                        
                        # Emit metrics updated event
                        await self.emit(
                            Topics.METRICS_UPDATED,
                            {
                                "post_id": post_id,
                                "platform": post_platform,
                                "metrics": metrics,
                                "type": "backfill"
                            },
                            correlation_id
                        )
                        
                        results["updated"] += 1
                        results["platforms"][post_platform] = results["platforms"].get(post_platform, 0) + 1
                    else:
                        results["skipped"] += 1
                    
                    # Rate limiting
                    await asyncio.sleep(60 / self.REQUESTS_PER_MINUTE)
                    
                    # Progress update every 10 posts
                    if (i + 1) % 10 == 0:
                        logger.info(f"[{self.worker_id}] Progress: {i+1}/{total} posts processed")
                        
                except Exception as e:
                    logger.warning(f"[{self.worker_id}] Failed to update post {post.get('id')}: {e}")
                    results["failed"] += 1
            
            # Emit completed event
            await self.emit(
                Topics.METRICS_FETCH_COMPLETED,
                {
                    "updated": results["updated"],
                    "failed": results["failed"],
                    "skipped": results["skipped"],
                    "platforms": results["platforms"],
                    "type": "backfill"
                },
                correlation_id
            )
            
            logger.info(f"[{self.worker_id}] Backfill complete: {results}")
            
        except Exception as e:
            logger.error(f"[{self.worker_id}] Backfill failed: {e}")
        finally:
            self._is_running = False
        
        return results
    
    async def _get_posts_to_update(
        self,
        platform: Optional[str],
        days_back: int,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Get posted content that needs metrics update."""
        import httpx
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                params = {
                    "days_back": days_back,
                    "limit": limit,
                    "needs_update": "true"
                }
                if platform:
                    params["platform"] = platform
                
                response = await client.get(
                    "http://localhost:5555/api/posted-content/",
                    params=params
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("posts", data) if isinstance(data, dict) else data
                else:
                    logger.warning(f"[{self.worker_id}] Failed to get posts: {response.status_code}")
                    return []
                    
        except Exception as e:
            logger.error(f"[{self.worker_id}] Error fetching posts: {e}")
            return []
    
    async def _fetch_metrics(self, platform: str, platform_url: str) -> Optional[Dict[str, Any]]:
        """Fetch metrics from platform API."""
        import httpx
        
        rapidapi_key = os.getenv("RAPIDAPI_KEY")
        if not rapidapi_key:
            return None
        
        try:
            if platform == "tiktok":
                return await self._fetch_tiktok_metrics(platform_url, rapidapi_key)
            elif platform == "instagram":
                return await self._fetch_instagram_metrics(platform_url, rapidapi_key)
            else:
                return None
        except Exception as e:
            logger.warning(f"[{self.worker_id}] Metrics fetch failed: {e}")
            return None
    
    async def _fetch_tiktok_metrics(self, url: str, api_key: str) -> Optional[Dict[str, Any]]:
        """Fetch TikTok video metrics."""
        import httpx
        
        # Extract video ID
        video_id = url.split("/video/")[-1].split("?")[0] if "/video/" in url else None
        if not video_id:
            return None
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                "https://tiktok-scraper7.p.rapidapi.com/video/info",
                params={"video_id": video_id},
                headers={
                    "X-RapidAPI-Key": api_key,
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
            return None
    
    async def _fetch_instagram_metrics(self, url: str, api_key: str) -> Optional[Dict[str, Any]]:
        """Fetch Instagram post metrics."""
        import httpx
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                "https://instagram-looter2.p.rapidapi.com/post",
                params={"link": url},
                headers={
                    "X-RapidAPI-Key": api_key,
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
            return None
    
    async def _update_post_metrics(self, post_id: str, metrics: Dict[str, Any]) -> None:
        """Update post metrics in database."""
        import httpx
        
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.patch(
                    f"http://localhost:5555/api/posted-content/{post_id}/metrics",
                    json={
                        "views": metrics.get("views", 0),
                        "likes": metrics.get("likes", 0),
                        "comments": metrics.get("comments", 0),
                        "shares": metrics.get("shares", 0),
                        "saves": metrics.get("saves", 0),
                        "last_metrics_update": datetime.now(timezone.utc).isoformat()
                    }
                )
        except Exception as e:
            logger.warning(f"[{self.worker_id}] Failed to update metrics in DB: {e}")
