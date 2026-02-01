"""
Sora Worker
============
Event-driven worker for Sora AI video generation.

Subscribes to:
    - sora.usage.check.requested
    - sora.video.requested
    - sora.batch.requested

Emits:
    - sora.usage.checked
    - sora.usage.low
    - sora.video.started
    - sora.video.completed
    - sora.video.downloaded
    - sora.poll.started
    - sora.poll.tick
    - sora.poll.stopped
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from services.event_bus import EventBus, Event, Topics
from services.workers.base import BaseWorker

logger = logging.getLogger(__name__)


class SoraWorker(BaseWorker):
    """
    Worker for Sora AI video generation via Safari automation.
    
    Handles:
        - Usage checking (regular and on-demand)
        - Video generation requests
        - Batch video generation
        - Polling for completion and auto-download
    
    Usage:
        worker = SoraWorker()
        await worker.start()
        
        # Request usage check via event
        await bus.publish(Topics.SORA_USAGE_CHECK_REQUESTED, {})
        
        # Request video generation
        await bus.publish(Topics.SORA_VIDEO_REQUESTED, {
            "prompt": "...",
            "character": "isaiahdupree"
        })
    """
    
    # Timeouts (seconds)
    POLL_INTERVAL = 60
    MAX_POLL_DURATION = 900  # 15 minutes
    LOW_USAGE_THRESHOLD = 5  # Emit warning when below this
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        worker_id: Optional[str] = None
    ):
        super().__init__(event_bus, worker_id)
        self._poll_task: Optional[asyncio.Task] = None
        self._is_polling = False
        self._known_video_ids: set = set()
    
    def get_subscriptions(self) -> List[str]:
        """Subscribe to Sora-related events."""
        return [
            Topics.SORA_USAGE_CHECK_REQUESTED,
            Topics.SORA_VIDEO_REQUESTED,
            Topics.SORA_BATCH_REQUESTED,
        ]
    
    async def handle_event(self, event: Event) -> None:
        """Handle Sora-related events."""
        if event.topic == Topics.SORA_USAGE_CHECK_REQUESTED:
            await self._handle_usage_check(event)
        elif event.topic == Topics.SORA_VIDEO_REQUESTED:
            await self._handle_video_request(event)
        elif event.topic == Topics.SORA_BATCH_REQUESTED:
            await self._handle_batch_request(event)
    
    async def _handle_usage_check(self, event: Event) -> None:
        """Check Sora usage and emit result."""
        logger.info(f"[{self.worker_id}] Checking Sora usage...")
        
        try:
            from automation.sora_full_automation import SoraFullAutomation
            
            sora = SoraFullAutomation()
            usage = sora.get_usage()
            
            # Emit usage checked event
            await self.emit(
                Topics.SORA_USAGE_CHECKED,
                {
                    "video_gens_left": usage.get("video_gens_left", 0),
                    "free_count": usage.get("free_count", 0),
                    "paid_count": usage.get("paid_count", 0),
                    "reset_date": usage.get("reset_date", ""),
                    "checked_at": datetime.now(timezone.utc).isoformat()
                },
                correlation_id=event.correlation_id
            )
            
            # Emit low usage warning if needed
            gens_left = usage.get("video_gens_left", 0)
            if gens_left < self.LOW_USAGE_THRESHOLD:
                await self.emit(
                    Topics.SORA_USAGE_LOW,
                    {
                        "video_gens_left": gens_left,
                        "threshold": self.LOW_USAGE_THRESHOLD
                    },
                    correlation_id=event.correlation_id
                )
                
        except Exception as e:
            logger.error(f"[{self.worker_id}] Usage check failed: {e}")
    
    async def _handle_video_request(self, event: Event) -> None:
        """Handle single video generation request."""
        payload = event.payload
        prompt = payload.get("prompt")
        character = payload.get("character", "isaiahdupree")
        duration = payload.get("duration", 15)
        aspect_ratio = payload.get("aspect_ratio", "Portrait")
        
        if not prompt:
            logger.error(f"[{self.worker_id}] Video request missing prompt")
            return
        
        logger.info(f"[{self.worker_id}] Generating video: {prompt[:40]}...")
        
        try:
            from automation.sora_full_automation import SoraFullAutomation
            
            sora = SoraFullAutomation()
            
            # Emit started event
            await self.emit(
                Topics.SORA_VIDEO_STARTED,
                {
                    "prompt": prompt,
                    "character": character,
                    "duration": duration
                },
                correlation_id=event.correlation_id
            )
            
            # Generate video
            success = await sora.generate_video(
                prompt=prompt,
                character=character,
                duration=duration,
                aspect_ratio=aspect_ratio,
                wait_for_slot=False
            )
            
            if success:
                logger.info(f"[{self.worker_id}] Video generation started")
                
                # Start polling for completion if not already polling
                if not self._is_polling:
                    await self._start_polling(event.correlation_id)
            else:
                await self.emit(
                    Topics.SORA_VIDEO_FAILED,
                    {"prompt": prompt, "error": "Failed to start generation"},
                    correlation_id=event.correlation_id
                )
                
        except Exception as e:
            logger.error(f"[{self.worker_id}] Video generation failed: {e}")
            await self.emit(
                Topics.SORA_VIDEO_FAILED,
                {"prompt": prompt, "error": str(e)},
                correlation_id=event.correlation_id
            )
    
    async def _handle_batch_request(self, event: Event) -> None:
        """
        Handle batch video generation request (ARCH-002).

        Supports two modes:
        1. Multi-part (theme + num_parts): Generates coordinated 3-part series
        2. Custom prompts: Generates each prompt individually

        Payload:
            - theme: Overall theme (for multi-part mode)
            - num_parts: Number of parts (default 3)
            - prompts: List of custom prompts (alternative to theme)
            - character: Sora @character
            - stitch: Whether to stitch parts together
            - remove_watermark: Whether to remove watermarks
            - pipeline_id: Optional pipeline correlation ID
        """
        payload = event.payload
        theme = payload.get("theme")
        num_parts = payload.get("num_parts", 3)
        prompts = payload.get("prompts", [])
        character = payload.get("character")
        stitch = payload.get("stitch", True)
        remove_watermark = payload.get("remove_watermark", True)
        pipeline_id = payload.get("pipeline_id", event.correlation_id)

        # Validate input
        if not theme and not prompts:
            logger.error(f"[{self.worker_id}] Batch request missing theme or prompts")
            await self.emit(
                Topics.SORA_BATCH_COMPLETED,
                {
                    "pipeline_id": pipeline_id,
                    "status": "failed",
                    "error": "Missing theme or prompts"
                },
                correlation_id=event.correlation_id
            )
            return

        logger.info(
            f"[{self.worker_id}] Starting batch: "
            f"theme='{theme}', parts={num_parts if theme else len(prompts)}"
        )

        await self.emit(
            Topics.SORA_BATCH_STARTED,
            {
                "pipeline_id": pipeline_id,
                "theme": theme,
                "num_parts": num_parts if theme else len(prompts),
                "character": character
            },
            correlation_id=event.correlation_id
        )

        try:
            # Use SoraPipeline for coordinated multi-part generation
            from automation.sora.pipeline import SoraPipeline

            pipeline = SoraPipeline()

            if theme:
                # Multi-part mode (ARCH-002)
                logger.info(f"[{self.worker_id}] Generating {num_parts}-part series...")

                result = await pipeline.generate_multi_part(
                    theme=theme,
                    num_parts=num_parts,
                    character=character,
                    part_prompts=prompts if prompts else None,
                    auto_stitch=stitch,
                    auto_analyze=True,
                    remove_watermarks=remove_watermark,
                    pipeline_id=pipeline_id  # ARCH-002: Pass pipeline_id for orchestrator integration
                )

                # Emit batch completed
                await self.emit(
                    Topics.SORA_BATCH_COMPLETED,
                    {
                        "pipeline_id": pipeline_id,
                        "job_id": result.get("id"),
                        "status": result.get("status"),
                        "theme": theme,
                        "num_parts": num_parts,
                        "video_path": result.get("stitched_video") or result.get("video_path"),
                        "parts": result.get("parts", []),
                        "analysis": result.get("analysis"),
                        "generation_time": result.get("total_generation_time"),
                        "prompts": result.get("prompts", [])
                    },
                    correlation_id=event.correlation_id
                )

                logger.success(
                    f"[{self.worker_id}] Batch complete: {result.get('status')} - "
                    f"{result.get('stitched_video') or result.get('video_path')}"
                )

            else:
                # Custom prompts mode (legacy)
                logger.info(f"[{self.worker_id}] Generating {len(prompts)} custom videos...")

                prompt_configs = [
                    {"prompt": p, "character": character}
                    for p in prompts
                ]

                result = await pipeline.generate_batch(
                    prompts=prompt_configs,
                    stitch_output=stitch,
                    add_captions=False,
                    schedule_to=None
                )

                # Emit batch completed
                await self.emit(
                    Topics.SORA_BATCH_COMPLETED,
                    {
                        "pipeline_id": pipeline_id,
                        "batch_id": result.get("id"),
                        "status": result.get("status"),
                        "completed": result.get("completed"),
                        "failed": result.get("failed"),
                        "video_path": result.get("stitched_video"),
                        "jobs": result.get("jobs", [])
                    },
                    correlation_id=event.correlation_id
                )

                logger.success(
                    f"[{self.worker_id}] Batch complete: "
                    f"{result.get('completed')}/{result.get('total_prompts')} succeeded"
                )

        except Exception as e:
            logger.error(f"[{self.worker_id}] Batch generation failed: {e}")
            # ARCH-002: Emit both FAILED and COMPLETED events for orchestrator compatibility
            await self.emit(
                Topics.SORA_BATCH_FAILED,
                {
                    "pipeline_id": pipeline_id,
                    "theme": theme,
                    "error": str(e)
                },
                correlation_id=event.correlation_id
            )
            await self.emit(
                Topics.SORA_BATCH_COMPLETED,
                {
                    "pipeline_id": pipeline_id,
                    "status": "failed",
                    "error": str(e)
                },
                correlation_id=event.correlation_id
            )
    
    async def _start_polling(self, correlation_id: str = None) -> None:
        """Start polling for video completion."""
        if self._is_polling:
            return
        
        self._is_polling = True
        
        await self.emit(
            Topics.SORA_POLL_STARTED,
            {"poll_interval": self.POLL_INTERVAL, "max_duration": self.MAX_POLL_DURATION},
            correlation_id=correlation_id
        )
        
        async def poll_loop():
            from automation.sora_full_automation import SoraFullAutomation
            
            sora = SoraFullAutomation()
            start_time = datetime.now(timezone.utc)
            tick_count = 0
            
            # Get initial known videos
            try:
                initial_videos = sora.get_completed_videos()
                self._known_video_ids = {v.get('id', '') for v in initial_videos}
            except:
                self._known_video_ids = set()
            
            while self._is_polling:
                tick_count += 1
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                
                # Check timeout
                if elapsed > self.MAX_POLL_DURATION:
                    logger.info(f"[{self.worker_id}] Polling timeout reached")
                    break
                
                try:
                    # Navigate to drafts and check for new videos
                    sora.navigate_to_drafts()
                    await asyncio.sleep(2)
                    
                    current_videos = sora.get_completed_videos()
                    queue_count = sora.get_queue_count()
                    
                    # Emit poll tick
                    await self.emit(
                        Topics.SORA_POLL_TICK,
                        {
                            "tick": tick_count,
                            "queue_count": queue_count,
                            "elapsed_seconds": int(elapsed)
                        },
                        correlation_id=correlation_id
                    )
                    
                    # Check for new completed videos
                    for video in current_videos:
                        vid = video.get('id', '')
                        if vid and vid not in self._known_video_ids:
                            logger.info(f"[{self.worker_id}] New video completed: {vid}")
                            self._known_video_ids.add(vid)
                            
                            # Emit completed event
                            await self.emit(
                                Topics.SORA_VIDEO_COMPLETED,
                                {"video_id": vid, "video_info": video},
                                correlation_id=correlation_id
                            )
                            
                            # Download video
                            path = sora.download_video(video_id=vid)
                            if path:
                                await self.emit(
                                    Topics.SORA_VIDEO_DOWNLOADED,
                                    {"video_id": vid, "local_path": path},
                                    correlation_id=correlation_id
                                )
                    
                    # Stop if queue is empty
                    if queue_count == 0:
                        logger.info(f"[{self.worker_id}] Queue empty, stopping poll")
                        break
                        
                except Exception as e:
                    logger.error(f"[{self.worker_id}] Poll error: {e}")
                
                await asyncio.sleep(self.POLL_INTERVAL)
            
            self._is_polling = False
            await self.emit(
                Topics.SORA_POLL_STOPPED,
                {"ticks": tick_count, "duration_seconds": int(elapsed)},
                correlation_id=correlation_id
            )
        
        self._poll_task = asyncio.create_task(poll_loop())
    
    def stop_polling(self) -> None:
        """Stop the polling task."""
        self._is_polling = False
        if self._poll_task:
            self._poll_task.cancel()
            self._poll_task = None
    
    async def stop(self) -> None:
        """Stop the worker and any polling."""
        self.stop_polling()
        await super().stop()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        stats = super().get_stats()
        stats.update({
            "is_polling": self._is_polling,
            "known_video_count": len(self._known_video_ids),
            "poll_interval": self.POLL_INTERVAL,
            "max_poll_duration": self.MAX_POLL_DURATION
        })
        return stats


# Singleton instance
_worker: Optional[SoraWorker] = None

def get_sora_worker(event_bus: Optional[EventBus] = None) -> SoraWorker:
    """Get singleton SoraWorker instance."""
    global _worker
    if _worker is None:
        _worker = SoraWorker(event_bus)
    return _worker
