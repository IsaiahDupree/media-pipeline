"""
Cleanup Worker
==============
Event-driven worker that cleans up orphaned resources when media is deleted.

Subscribes to:
    - media.deleted (content removed from library)

Actions:
    - Delete orphaned thumbnails
    - Cancel pending scheduled posts
    - Remove from analytics tracking
"""

import asyncio
import logging
import os
from typing import List, Optional
from pathlib import Path

from services.event_bus import EventBus, Event, Topics
from services.workers.base import BaseWorker

logger = logging.getLogger(__name__)


class CleanupWorker(BaseWorker):
    """
    Worker that cleans up orphaned resources when media is deleted.
    
    Handles:
        - Thumbnail file deletion
        - Cancelling scheduled posts for deleted media
        - Removing analytics tracking entries
    
    Usage:
        worker = CleanupWorker()
        await worker.start()
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None, worker_id: Optional[str] = None):
        super().__init__(event_bus, worker_id)
    
    def get_subscriptions(self) -> List[str]:
        """Subscribe to media deletion events."""
        return [
            Topics.MEDIA_DELETED,
        ]
    
    async def handle_event(self, event: Event) -> None:
        """Process cleanup for deleted media."""
        media_id = event.payload.get("media_id")
        thumbnail_path = event.payload.get("thumbnail_path")
        filename = event.payload.get("filename")
        
        logger.info(f"[{self.worker_id}] Processing cleanup for deleted media: {media_id} ({filename})")
        
        # Run cleanup tasks in parallel
        await asyncio.gather(
            self._cleanup_thumbnail(thumbnail_path, media_id),
            self._cancel_scheduled_posts(media_id),
            self._cleanup_analytics(media_id),
            return_exceptions=True
        )
        
        logger.info(f"[{self.worker_id}] Cleanup completed for: {media_id}")
    
    async def _cleanup_thumbnail(self, thumbnail_path: Optional[str], media_id: str) -> None:
        """Delete the thumbnail file if it exists."""
        if not thumbnail_path:
            logger.debug(f"[{self.worker_id}] No thumbnail to clean up for {media_id}")
            return
        
        try:
            path = Path(thumbnail_path)
            if path.exists():
                os.remove(thumbnail_path)
                logger.info(f"[{self.worker_id}] Deleted thumbnail: {thumbnail_path}")
            else:
                logger.debug(f"[{self.worker_id}] Thumbnail already gone: {thumbnail_path}")
        except Exception as e:
            logger.warning(f"[{self.worker_id}] Failed to delete thumbnail {thumbnail_path}: {e}")
    
    async def _cancel_scheduled_posts(self, media_id: str) -> None:
        """Cancel any pending scheduled posts for this media."""
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=10) as client:
                # Get scheduled posts for this media
                response = await client.get(
                    f"http://localhost:5555/api/schedule/by-media/{media_id}"
                )
                
                if response.status_code == 200:
                    posts = response.json()
                    cancelled_count = 0
                    
                    for post in posts:
                        post_id = post.get("id")
                        status = post.get("status")
                        
                        # Only cancel pending/scheduled posts
                        if status in ["scheduled", "pending"]:
                            cancel_response = await client.delete(
                                f"http://localhost:5555/api/schedule/{post_id}"
                            )
                            if cancel_response.status_code == 200:
                                cancelled_count += 1
                                logger.info(f"[{self.worker_id}] Cancelled scheduled post: {post_id}")
                    
                    if cancelled_count > 0:
                        logger.info(f"[{self.worker_id}] Cancelled {cancelled_count} scheduled posts for {media_id}")
                elif response.status_code == 404:
                    logger.debug(f"[{self.worker_id}] No scheduled posts found for {media_id}")
                    
        except Exception as e:
            logger.warning(f"[{self.worker_id}] Failed to cancel scheduled posts for {media_id}: {e}")
    
    async def _cleanup_analytics(self, media_id: str) -> None:
        """Remove analytics tracking entries for this media."""
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=10) as client:
                # Mark posted content as orphaned or delete tracking
                response = await client.post(
                    f"http://localhost:5555/api/posted-content/orphan/{media_id}"
                )
                
                if response.status_code == 200:
                    logger.info(f"[{self.worker_id}] Marked analytics as orphaned for {media_id}")
                elif response.status_code == 404:
                    logger.debug(f"[{self.worker_id}] No analytics tracking for {media_id}")
                    
        except Exception as e:
            # This is expected if the endpoint doesn't exist yet
            logger.debug(f"[{self.worker_id}] Analytics cleanup skipped for {media_id}: {e}")
