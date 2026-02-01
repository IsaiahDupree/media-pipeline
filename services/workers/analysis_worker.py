"""
Analysis Worker
================
Event-driven worker for video analysis pipeline.

Subscribes to:
    - media.ingested (auto-analyze new media)
    - media.analysis.requested (manual analysis request)

Emits:
    - media.analysis.started
    - media.analysis.progress
    - media.analysis.step.completed
    - media.analysis.completed
    - media.analysis.failed
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from services.event_bus import EventBus, Event, Topics
from services.workers.base import BaseWorker

logger = logging.getLogger(__name__)


class AnalysisWorker(BaseWorker):
    """
    Worker for processing video analysis requests.
    
    Pipeline steps:
        1. Transcript extraction
        2. Visual analysis
        3. AI analysis (captions, hashtags, platform content)
    
    Usage:
        worker = AnalysisWorker()
        await worker.start()
        
        # Worker will automatically process events from:
        # - media.ingested
        # - media.analysis.requested
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None, worker_id: Optional[str] = None):
        super().__init__(event_bus, worker_id)
        self._video_analyzer = None  # Lazy load
    
    def get_subscriptions(self) -> List[str]:
        """Subscribe to analysis-related events."""
        return [
            Topics.ANALYSIS_REQUESTED,
            # Topics.MEDIA_INGESTED,  # Uncomment to auto-analyze on ingest
        ]
    
    async def handle_event(self, event: Event) -> None:
        """Process analysis events."""
        media_id = event.payload.get("media_id")
        
        if not media_id:
            logger.warning(f"[{self.worker_id}] No media_id in event payload")
            return
        
        # BUG FIX: Idempotency check - verify analysis not already in progress or completed
        analysis_status = await self._check_analysis_status(media_id)
        if analysis_status == "in_progress":
            logger.info(f"[{self.worker_id}] Analysis already in progress for {media_id}, skipping")
            return
        elif analysis_status == "completed":
            logger.info(f"[{self.worker_id}] Analysis already completed for {media_id}, skipping")
            return
        
        # BUG FIX: File verification before starting analysis
        file_check = await self._verify_media_file(media_id)
        if not file_check.get("valid"):
            error = file_check.get("error", "File verification failed")
            logger.error(f"[{self.worker_id}] File verification failed for {media_id}: {error}")
            await self.emit(
                Topics.ANALYSIS_FAILED,
                {
                    "media_id": media_id,
                    "error": error,
                    "file_verification_failed": True,
                    "failed_at": datetime.now(timezone.utc).isoformat()
                },
                event.correlation_id
            )
            return
        
        # Mark analysis as in progress atomically
        if not await self._mark_analysis_in_progress(media_id):
            logger.warning(f"[{self.worker_id}] Could not mark analysis as in_progress for {media_id} (may be locked)")
            return
        
        # Run the analysis pipeline
        await self._run_analysis_pipeline(media_id, event.correlation_id)
    
    async def _run_analysis_pipeline(self, media_id: str, correlation_id: str) -> Dict[str, Any]:
        """
        Run the full analysis pipeline with progress events.
        
        Steps:
            1. Transcript (0-33%)
            2. Visual analysis (33-66%)
            3. AI analysis (66-100%)
        """
        try:
            # Emit started event
            await self.emit(
                Topics.ANALYSIS_STARTED,
                {"media_id": media_id, "step": "initializing"},
                correlation_id
            )
            
            # Step 1: Transcript
            await self.emit_progress("media.analysis", 5, "transcript", correlation_id, media_id=media_id)
            transcript = await self._run_transcript(media_id)
            await self.emit(
                Topics.TRANSCRIPT_COMPLETED,
                {"media_id": media_id, "transcript_length": len(transcript) if transcript else 0},
                correlation_id
            )
            await self.emit_progress("media.analysis", 33, "transcript_complete", correlation_id, media_id=media_id)
            
            # Step 2: Visual Analysis
            await self.emit_progress("media.analysis", 40, "visual", correlation_id, media_id=media_id)
            visual_data = await self._run_visual_analysis(media_id)
            await self.emit(
                Topics.VISUAL_COMPLETED,
                {"media_id": media_id, "frames_analyzed": visual_data.get("frame_count", 0) if visual_data else 0},
                correlation_id
            )
            await self.emit_progress("media.analysis", 66, "visual_complete", correlation_id, media_id=media_id)
            
            # Step 3: AI Analysis
            await self.emit_progress("media.analysis", 75, "ai_analysis", correlation_id, media_id=media_id)
            analysis_result = await self._run_ai_analysis(media_id, transcript, visual_data)
            await self.emit(
                Topics.AI_ANALYSIS_COMPLETED,
                {"media_id": media_id, "platforms": list(analysis_result.keys()) if analysis_result else []},
                correlation_id
            )
            await self.emit_progress("media.analysis", 100, "complete", correlation_id, media_id=media_id)
            
            # Emit completion
            await self.emit(
                Topics.ANALYSIS_COMPLETED,
                {
                    "media_id": media_id,
                    "has_transcript": bool(transcript),
                    "has_visual": bool(visual_data),
                    "platforms_analyzed": list(analysis_result.keys()) if analysis_result else [],
                    "completed_at": datetime.now(timezone.utc).isoformat()
                },
                correlation_id
            )
            
            return {
                "success": True,
                "media_id": media_id,
                "transcript": transcript,
                "visual": visual_data,
                "analysis": analysis_result
            }
            
        except Exception as e:
            logger.error(f"[{self.worker_id}] Analysis failed for {media_id}: {e}")
            
            await self.emit(
                Topics.ANALYSIS_FAILED,
                {
                    "media_id": media_id,
                    "error": str(e),
                    "failed_at": datetime.now(timezone.utc).isoformat()
                },
                correlation_id
            )
            raise
    
    async def _run_transcript(self, media_id: str) -> Optional[str]:
        """Extract transcript from video."""
        try:
            analyzer = self._get_video_analyzer()
            if analyzer:
                # Get video path from database
                video_path = await self._get_video_path(media_id)
                if video_path:
                    result = await asyncio.to_thread(
                        analyzer.extract_transcript, video_path
                    )
                    return result
            return None
        except Exception as e:
            logger.warning(f"Transcript extraction failed: {e}")
            return None
    
    async def _run_visual_analysis(self, media_id: str) -> Optional[Dict[str, Any]]:
        """Run visual analysis on video frames."""
        try:
            analyzer = self._get_video_analyzer()
            if analyzer:
                video_path = await self._get_video_path(media_id)
                if video_path:
                    result = await asyncio.to_thread(
                        analyzer.analyze_frames, video_path
                    )
                    return result
            return None
        except Exception as e:
            logger.warning(f"Visual analysis failed: {e}")
            return None
    
    async def _run_ai_analysis(
        self,
        media_id: str,
        transcript: Optional[str],
        visual_data: Optional[Dict]
    ) -> Optional[Dict[str, Any]]:
        """Run AI analysis to generate captions and hashtags."""
        try:
            analyzer = self._get_video_analyzer()
            if analyzer:
                result = await asyncio.to_thread(
                    analyzer.generate_platform_content,
                    media_id,
                    transcript,
                    visual_data
                )
                return result
            return None
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
            return None
    
    def _get_video_analyzer(self):
        """Lazy load the video analyzer."""
        if self._video_analyzer is None:
            try:
                from services.video_analyzer import VideoAnalyzer
                self._video_analyzer = VideoAnalyzer()
            except Exception as e:
                logger.warning(f"Could not load VideoAnalyzer: {e}")
        return self._video_analyzer
    
    async def _get_video_path(self, media_id: str) -> Optional[str]:
        """Get video file path from database."""
        try:
            from sqlalchemy import create_engine, text
            import os
            
            DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:54322/postgres")
            engine = create_engine(DATABASE_URL)
            
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT source_uri FROM videos WHERE id = :id"),
                    {"id": media_id}
                ).fetchone()
                
                if result and result[0]:
                    return result[0]
            return None
        except Exception as e:
            logger.warning(f"Could not get video path: {e}")
            return None
    
    async def _check_analysis_status(self, media_id: str) -> Optional[str]:
        """Check if analysis is already in progress or completed (idempotency check)."""
        try:
            from sqlalchemy import create_engine, text
            import os
            
            DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:54322/postgres")
            engine = create_engine(DATABASE_URL)
            
            with engine.connect() as conn:
                # Check if analysis exists and its status
                result = conn.execute(
                    text("""
                        SELECT 
                            CASE 
                                WHEN va.video_id IS NOT NULL THEN 'completed'
                                ELSE NULL
                            END as status
                        FROM videos v
                        LEFT JOIN video_analysis va ON v.id = va.video_id
                        WHERE v.id = :id
                    """),
                    {"id": media_id}
                ).fetchone()
                
                if result and result[0]:
                    return result[0]
            return None
        except Exception as e:
            logger.warning(f"Could not check analysis status: {e}")
            return None
    
    async def _mark_analysis_in_progress(self, media_id: str) -> bool:
        """Atomically mark analysis as in progress (idempotency)."""
        try:
            from sqlalchemy import create_engine, text
            import os
            
            DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:54322/postgres")
            engine = create_engine(DATABASE_URL)
            
            with engine.connect() as conn:
                # Try to create analysis record with in_progress status
                # This acts as a lock - only one worker can create it
                result = conn.execute(
                    text("""
                        INSERT INTO video_analysis (video_id, status, created_at)
                        VALUES (:video_id, 'in_progress', NOW())
                        ON CONFLICT (video_id) DO NOTHING
                        RETURNING video_id
                    """),
                    {"video_id": media_id}
                )
                conn.commit()
                
                # If we inserted a row, we got the lock
                return result.rowcount > 0
        except Exception as e:
            logger.warning(f"Could not mark analysis in progress: {e}")
            return False
    
    async def _verify_media_file(self, media_id: str) -> Dict[str, Any]:
        """Verify media file exists and is accessible (validation)."""
        try:
            video_path = await self._get_video_path(media_id)
            if not video_path:
                return {
                    "valid": False,
                    "error": f"No file path found for media {media_id}"
                }
            
            import os
            from pathlib import Path
            
            # Expand user path if needed
            file_path = os.path.expanduser(video_path)
            path = Path(file_path)
            
            if not path.exists():
                return {
                    "valid": False,
                    "error": f"File does not exist: {file_path}",
                    "file_path": str(path)
                }
            
            if not path.is_file():
                return {
                    "valid": False,
                    "error": f"Path is not a file: {file_path}",
                    "file_path": str(path)
                }
            
            if not os.access(file_path, os.R_OK):
                return {
                    "valid": False,
                    "error": f"File is not readable: {file_path}",
                    "file_path": str(path)
                }
            
            return {
                "valid": True,
                "file_path": str(path),
                "file_size": path.stat().st_size
            }
        except Exception as e:
            logger.error(f"File verification failed: {e}")
            return {
                "valid": False,
                "error": str(e)
            }


# Convenience function to create and start worker
async def start_analysis_worker(event_bus: Optional[EventBus] = None) -> AnalysisWorker:
    """Create and start an analysis worker."""
    worker = AnalysisWorker(event_bus)
    await worker.start()
    return worker
