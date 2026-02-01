"""
Thumbnail Generation Worker
===========================
Event-driven worker that automatically generates thumbnails when media is ingested.

Subscribes to:
    - media.ingested (new video/image added to library)

Emits:
    - media.thumbnail.ready (thumbnail generated successfully)
    - media.thumbnail.failed (thumbnail generation failed)
"""

import asyncio
import uuid
from loguru import logger
from pathlib import Path
from typing import List, Dict, Any, Optional

from services.event_bus import EventBus, Event, Topics
from services.workers.base import BaseWorker
from services.thumbnail_service import generate_thumbnail
from database.connection import async_session_maker
from database.models import Video
from sqlalchemy import select, update


class ThumbnailGenerationWorker(BaseWorker):
    """
    Worker that automatically generates thumbnails when media is ingested.
    
    Generates thumbnails in the background for videos and images,
    then updates the database and emits completion events.
    
    Usage:
        worker = ThumbnailGenerationWorker()
        await worker.start()
        
        # Thumbnails will auto-generate when media.ingested events are emitted
    """
    
    def get_subscriptions(self) -> List[str]:
        """Subscribe to media ingestion events."""
        return [Topics.MEDIA_INGESTED]
    
    async def handle_event(self, event: Event) -> None:
        """Generate thumbnail for ingested media."""
        payload = event.payload
        media_id = payload.get("media_id")
        file_path = payload.get("file_path")
        media_type = payload.get("media_type", "video")
        
        if not media_id or not file_path:
            logger.warning(f"Missing media_id or file_path in event: {payload}")
            return
        
        logger.info(f"🎬 Generating thumbnail for media {media_id} ({file_path})")
        
        try:
            # Check if file exists
            path = Path(file_path)
            if not path.exists():
                logger.error(f"File not found: {file_path}")
                await self._emit_failed(media_id, f"File not found: {file_path}", event.correlation_id)
                return
            
            # Generate thumbnail (medium size by default)
            # This runs in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            thumbnail_path = await loop.run_in_executor(
                None,
                generate_thumbnail,
                str(path),
                "medium"
            )
            
            if not thumbnail_path or not Path(thumbnail_path).exists():
                logger.error(f"Thumbnail generation failed for {media_id}")
                await self._emit_failed(media_id, "Thumbnail generation returned no path", event.correlation_id)
                return
            
            # Update database with thumbnail path
            await self._update_database(media_id, thumbnail_path)
            
            # Emit success event
            await self.emit(
                Topics.MEDIA_THUMBNAIL_READY,
                {
                    "media_id": media_id,
                    "thumbnail_path": thumbnail_path,
                    "file_path": file_path,
                    "media_type": media_type
                },
                correlation_id=event.correlation_id
            )
            
            logger.success(f"✅ Thumbnail generated for {media_id}: {thumbnail_path}")
            
        except Exception as e:
            logger.error(f"Error generating thumbnail for {media_id}: {e}", exc_info=True)
            await self._emit_failed(media_id, str(e), event.correlation_id)
    
    async def _update_database(self, media_id: str, thumbnail_path: str) -> None:
        """Update video record with thumbnail path."""
        if not async_session_maker:
            logger.warning("Database not initialized, skipping thumbnail path update")
            return
        
        try:
            async with async_session_maker() as session:
                video_uuid = None
                try:
                    video_uuid = uuid.UUID(media_id)
                except ValueError:
                    logger.error(f"Invalid media_id format: {media_id}")
                    return
                
                # Update video record
                stmt = (
                    update(Video)
                    .where(Video.id == video_uuid)
                    .values(thumbnail_path=thumbnail_path)
                )
                await session.execute(stmt)
                await session.commit()
                
                logger.debug(f"Updated database with thumbnail path for {media_id}")
                
        except Exception as e:
            logger.error(f"Failed to update database with thumbnail path: {e}", exc_info=True)
            # Don't fail the whole operation if DB update fails
    
    async def _emit_failed(self, media_id: str, error: str, correlation_id: Optional[str] = None) -> None:
        """Emit thumbnail generation failed event."""
        await self.emit(
            "media.thumbnail.failed",  # Custom topic for failures
            {
                "media_id": media_id,
                "error": error
            },
            correlation_id=correlation_id
        )

