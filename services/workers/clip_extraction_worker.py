"""
Clip Extraction Worker
======================
Event-driven worker for extracting short-form clips from long-form videos.

Subscribes to:
    - clip.extraction.requested

Emits:
    - clip.extraction.started
    - clip.extraction.progress
    - clip.extraction.transcript (transcript completed)
    - clip.extraction.segments (AI segments identified)
    - clip.extraction.rendering (rendering started)
    - clip.extraction.clip_done (single clip completed)
    - clip.extraction.completed
    - clip.extraction.failed

Based on SupoClip architecture:
    1. Transcribe with AssemblyAI (word-level timing)
    2. AI analysis to find compelling segments
    3. Smart crop with face detection (9:16)
    4. Render clips with subtitles
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from services.event_bus import EventBus, Event, Topics
from services.workers.base import BaseWorker

logger = logging.getLogger(__name__)


class ClipExtractionWorker(BaseWorker):
    """
    Worker for processing clip extraction requests.
    
    Pipeline steps:
        1. Validate source video (0-5%)
        2. Transcribe with AssemblyAI (5-30%)
        3. AI segment identification (30-50%)
        4. Render clips with smart crop (50-95%)
        5. Save to database (95-100%)
    
    Usage:
        worker = ClipExtractionWorker()
        await worker.start()
        
        # Trigger extraction via event:
        await event_bus.publish(Topics.CLIP_EXTRACTION_REQUESTED, {
            "video_path": "/path/to/video.mp4",
            "media_id": "optional-uuid",
            "output_dir": "/path/to/output",
            "options": {
                "min_clip_duration": 10,
                "max_clip_duration": 60,
                "max_clips": 7,
                "add_subtitles": True,
                "font_size": 24,
                "font_color": "#FFFFFF"
            }
        })
    """
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        worker_id: Optional[str] = None,
        default_output_dir: Optional[str] = None
    ):
        super().__init__(event_bus, worker_id)
        self._extraction_service = None  # Lazy load
        self.default_output_dir = default_output_dir or os.getenv(
            "CLIP_OUTPUT_DIR",
            "./clips"
        )
    
    def get_subscriptions(self) -> List[str]:
        """Subscribe to clip extraction events."""
        return [
            Topics.CLIP_EXTRACTION_REQUESTED,
        ]
    
    async def handle_event(self, event: Event) -> None:
        """Process clip extraction events."""
        await self._run_extraction_pipeline(event.payload, event.correlation_id)
    
    async def _run_extraction_pipeline(
        self,
        payload: Dict[str, Any],
        correlation_id: str
    ) -> Dict[str, Any]:
        """
        Run the full clip extraction pipeline with progress events.
        
        Steps:
            1. Validate (0-5%)
            2. Transcribe (5-30%)
            3. Identify segments (30-50%)
            4. Render clips (50-95%)
            5. Finalize (95-100%)
        """
        video_path = payload.get("video_path")
        media_id = payload.get("media_id") or str(uuid4())
        output_dir = payload.get("output_dir") or self.default_output_dir
        options = payload.get("options", {})
        
        job_id = str(uuid4())
        
        try:
            # Emit started event
            await self.emit(
                Topics.CLIP_EXTRACTION_STARTED,
                {
                    "job_id": job_id,
                    "media_id": media_id,
                    "video_path": video_path,
                    "step": "initializing"
                },
                correlation_id
            )
            
            # Step 1: Validate
            await self.emit_progress(
                "clip.extraction", 5, "validating",
                correlation_id, job_id=job_id, media_id=media_id
            )
            
            if not video_path or not Path(video_path).exists():
                raise FileNotFoundError(f"Video not found: {video_path}")
            
            # Get the extraction service
            service = self._get_extraction_service(options)
            
            # Create progress callback that emits events
            async def progress_callback(pct: int, step: str):
                await self._emit_step_progress(
                    pct, step, job_id, media_id, correlation_id
                )
            
            # Run extraction (this handles steps 2-4)
            result = await service.extract_clips(
                video_path=video_path,
                output_dir=Path(output_dir),
                progress_callback=lambda pct, step: asyncio.create_task(
                    progress_callback(pct, step)
                ),
                min_clip_duration=options.get("min_clip_duration", 10),
                max_clip_duration=options.get("max_clip_duration", 60),
                max_clips=options.get("max_clips", 7)
            )
            
            if not result.success:
                raise Exception(result.error or "Extraction failed")
            
            # Step 5: Save to database
            await self.emit_progress(
                "clip.extraction", 95, "saving",
                correlation_id, job_id=job_id, media_id=media_id
            )
            
            await self._save_clips_to_database(media_id, result)
            
            # Emit completion
            await self.emit_progress(
                "clip.extraction", 100, "complete",
                correlation_id, job_id=job_id, media_id=media_id
            )
            
            completion_payload = {
                "job_id": job_id,
                "media_id": media_id,
                "clips_count": len(result.clips),
                "clips": [
                    {
                        "clip_id": c.clip_id,
                        "filename": c.filename,
                        "path": c.path,
                        "start_time": c.start_time,
                        "end_time": c.end_time,
                        "duration": c.duration,
                        "relevance_score": c.relevance_score
                    }
                    for c in result.clips
                ],
                "total_duration": result.total_duration,
                "processing_time": result.processing_time,
                "key_topics": result.key_topics,
                "completed_at": datetime.now(timezone.utc).isoformat()
            }
            
            await self.emit(
                Topics.CLIP_EXTRACTION_COMPLETED,
                completion_payload,
                correlation_id
            )
            
            return {"success": True, **completion_payload}
            
        except Exception as e:
            logger.error(f"[{self.worker_id}] Extraction failed for {media_id}: {e}")
            
            await self.emit(
                Topics.CLIP_EXTRACTION_FAILED,
                {
                    "job_id": job_id,
                    "media_id": media_id,
                    "video_path": video_path,
                    "error": str(e),
                    "failed_at": datetime.now(timezone.utc).isoformat()
                },
                correlation_id
            )
            raise
    
    async def _emit_step_progress(
        self,
        pct: int,
        step: str,
        job_id: str,
        media_id: str,
        correlation_id: str
    ):
        """Emit detailed progress events based on step name."""
        # Map step names to specific topics
        if "transcript" in step.lower():
            if "complete" in step.lower():
                await self.emit(
                    Topics.CLIP_TRANSCRIPT_COMPLETED,
                    {"job_id": job_id, "media_id": media_id},
                    correlation_id
                )
        elif "segment" in step.lower() or "found" in step.lower():
            await self.emit(
                Topics.CLIP_SEGMENTS_IDENTIFIED,
                {"job_id": job_id, "media_id": media_id},
                correlation_id
            )
        elif "render" in step.lower():
            if "clip" in step.lower() and "/" in step:
                # Single clip done
                await self.emit(
                    Topics.CLIP_SINGLE_COMPLETED,
                    {"job_id": job_id, "media_id": media_id, "step": step},
                    correlation_id
                )
            else:
                await self.emit(
                    Topics.CLIP_RENDERING_STARTED,
                    {"job_id": job_id, "media_id": media_id},
                    correlation_id
                )
        
        # Always emit general progress
        await self.emit_progress(
            "clip.extraction", pct, step,
            correlation_id, job_id=job_id, media_id=media_id
        )
    
    async def _save_clips_to_database(self, media_id: str, result) -> None:
        """Save extracted clips to the database."""
        try:
            from sqlalchemy import create_engine, text
            
            DATABASE_URL = os.getenv(
                "DATABASE_URL",
                "postgresql://postgres:postgres@localhost:54322/postgres"
            )
            engine = create_engine(DATABASE_URL)
            
            with engine.connect() as conn:
                for clip in result.clips:
                    conn.execute(
                        text("""
                            INSERT INTO video_clips (
                                id, source_video_id, start_time_sec, end_time_sec,
                                clip_path, clip_type, metadata, created_at
                            ) VALUES (
                                :id, :source_id, :start_sec, :end_sec,
                                :path, 'auto_extracted', :metadata, NOW()
                            )
                            ON CONFLICT (id) DO UPDATE SET
                                clip_path = EXCLUDED.clip_path,
                                metadata = EXCLUDED.metadata
                        """),
                        {
                            "id": clip.clip_id,
                            "source_id": media_id,
                            "start_sec": self._parse_timestamp(clip.start_time),
                            "end_sec": self._parse_timestamp(clip.end_time),
                            "path": clip.path,
                            "metadata": {
                                "text": clip.text,
                                "relevance_score": clip.relevance_score,
                                "reasoning": clip.reasoning,
                                "filename": clip.filename
                            }
                        }
                    )
                conn.commit()
                
            logger.info(f"Saved {len(result.clips)} clips to database for {media_id}")
            
        except Exception as e:
            logger.warning(f"Could not save clips to database: {e}")
    
    def _parse_timestamp(self, timestamp: str) -> float:
        """Parse MM:SS to seconds."""
        try:
            parts = timestamp.strip().split(':')
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            return float(timestamp)
        except Exception:
            return 0.0
    
    def _get_extraction_service(self, options: Dict[str, Any]):
        """Lazy load the extraction service with options."""
        if self._extraction_service is None:
            try:
                from services.clip_extraction_service import ClipExtractionService
                self._extraction_service = ClipExtractionService(
                    font_family=options.get("font_family", "Arial"),
                    font_size=options.get("font_size", 24),
                    font_color=options.get("font_color", "#FFFFFF")
                )
            except Exception as e:
                logger.error(f"Could not load ClipExtractionService: {e}")
                raise
        return self._extraction_service


async def start_clip_extraction_worker(
    event_bus: Optional[EventBus] = None
) -> ClipExtractionWorker:
    """Create and start a clip extraction worker."""
    worker = ClipExtractionWorker(event_bus)
    await worker.start()
    return worker
