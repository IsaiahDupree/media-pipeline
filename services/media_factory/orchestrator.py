"""
Media Factory Pipeline Orchestrator (MF-001)
=============================================
Coordinates the end-to-end media production pipeline from content brief to published video.

Pipeline Stages:
1. Script Generation - Brief → Script + Shot Plan
2. TTS Generation - Script → Voice Audio
3. Music Selection - Brief → Background Music
4. Visuals Assembly - Shot Plan → B-Roll + Matting
5. Remotion Render - Timeline → Final Video
6. Multi-Platform Publish - Video → Platform Posts

Each stage uses JSON contracts (MF-007) for provider swapping and testability.

Usage:
    >>> from services.media_factory.orchestrator import MediaFactoryOrchestrator
    >>> from services.media_factory.contracts import ContentBriefSchema
    >>>
    >>> orchestrator = MediaFactoryOrchestrator()
    >>>
    >>> # Create production job
    >>> job_id = await orchestrator.create_job(brief)
    >>>
    >>> # Execute pipeline
    >>> result = await orchestrator.execute_pipeline(job_id)
    >>>
    >>> # Check status
    >>> status = orchestrator.get_job_status(job_id)
"""

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from uuid import uuid4
from loguru import logger

from services.media_factory.contracts import (
    ContentBriefSchema,
    ScriptSchema,
    TimelineSchema,
    RenderJobSchema,
    PublishJobSchema,
    validate_content_brief,
    validate_script,
    validate_timeline,
)
from services.event_bus import EventBus, Topics
from config import get_settings


class PipelineStage(Enum):
    """Stages in the media factory pipeline."""
    SCRIPT_GENERATION = "script_generation"
    TTS_GENERATION = "tts_generation"
    MUSIC_SELECTION = "music_selection"
    VISUALS_ASSEMBLY = "visuals_assembly"
    REMOTION_RENDER = "remotion_render"
    PUBLISH = "publish"


class JobStatus(Enum):
    """Media factory job statuses."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineStageResult:
    """Result from a pipeline stage."""
    stage: PipelineStage
    status: JobStatus
    output_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage": self.stage.value,
            "status": self.status.value,
            "output_data": self.output_data,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class MediaFactoryJob:
    """Represents a media factory production job."""
    job_id: str
    brief: Dict[str, Any]  # ContentBriefSchema as dict
    status: JobStatus = JobStatus.PENDING
    current_stage: Optional[PipelineStage] = None
    stage_results: List[PipelineStageResult] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    # Intermediate outputs
    script: Optional[Dict[str, Any]] = None
    audio_url: Optional[str] = None
    music_url: Optional[str] = None
    visuals: Optional[List[Dict[str, Any]]] = None
    timeline: Optional[Dict[str, Any]] = None
    video_url: Optional[str] = None
    publish_results: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "brief": self.brief,
            "status": self.status.value,
            "current_stage": self.current_stage.value if self.current_stage else None,
            "stage_results": [sr.to_dict() for sr in self.stage_results],
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "script": self.script,
            "audio_url": self.audio_url,
            "music_url": self.music_url,
            "visuals": self.visuals,
            "timeline": self.timeline,
            "video_url": self.video_url,
            "publish_results": self.publish_results,
        }

    def get_stage_result(self, stage: PipelineStage) -> Optional[PipelineStageResult]:
        """Get result for specific stage."""
        for result in self.stage_results:
            if result.stage == stage:
                return result
        return None


class MediaFactoryOrchestrator:
    """
    Media Factory Pipeline Orchestrator (MF-001)

    Coordinates the end-to-end production pipeline with:
    - Stage-by-stage execution
    - Error handling and recovery
    - Progress tracking
    - Event emission for observability
    - Provider swapping via service registry
    """

    def __init__(self):
        """Initialize orchestrator."""
        self.settings = get_settings()
        self.event_bus = EventBus.get_instance()
        self.event_bus.set_source("media-factory-orchestrator")

        # Job storage (in-memory for now)
        self._jobs: Dict[str, MediaFactoryJob] = {}

        # Stage handlers (can be swapped via service registry)
        self._stage_handlers: Dict[PipelineStage, Callable] = {}

        logger.info("🏭 Media Factory Orchestrator initialized")

    def register_stage_handler(
        self,
        stage: PipelineStage,
        handler: Callable[[MediaFactoryJob], Awaitable[Dict[str, Any]]]
    ) -> None:
        """
        Register handler for a pipeline stage.

        Args:
            stage: Pipeline stage
            handler: Async function that processes the stage
        """
        self._stage_handlers[stage] = handler
        logger.info(f"✓ Registered handler for stage: {stage.value}")

    async def create_job(
        self,
        brief: Dict[str, Any],
        job_id: Optional[str] = None
    ) -> str:
        """
        Create a new media factory job.

        Args:
            brief: Content brief (ContentBriefSchema as dict)
            job_id: Optional custom job ID

        Returns:
            Job ID

        Raises:
            ValueError: If brief validation fails
        """
        # Validate brief
        try:
            validate_content_brief(brief)
        except Exception as e:
            raise ValueError(f"Invalid content brief: {e}")

        if not job_id:
            job_id = str(uuid4())

        job = MediaFactoryJob(
            job_id=job_id,
            brief=brief
        )

        self._jobs[job_id] = job

        # Emit job created event
        await self.event_bus.publish(
            Topics.MEDIA_FACTORY_JOB_CREATED,
            {
                "job_id": job_id,
                "brief_id": brief.get("brief_id"),
                "created_at": job.created_at.isoformat()
            }
        )

        logger.info(f"✓ Media factory job created: {job_id[:8]}")
        return job_id

    async def execute_pipeline(self, job_id: str) -> Dict[str, Any]:
        """
        Execute full pipeline for a job.

        Args:
            job_id: Job identifier

        Returns:
            Job result dictionary

        Raises:
            ValueError: If job not found
        """
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        if job.status != JobStatus.PENDING:
            logger.warning(f"Job {job_id[:8]} already {job.status.value}")
            return job.to_dict()

        job.status = JobStatus.PROCESSING
        job.started_at = datetime.now(timezone.utc)

        logger.info(f"🏭 Starting pipeline for job: {job_id[:8]}")

        # Define pipeline stages in order
        stages = [
            PipelineStage.SCRIPT_GENERATION,
            PipelineStage.TTS_GENERATION,
            PipelineStage.MUSIC_SELECTION,
            PipelineStage.VISUALS_ASSEMBLY,
            PipelineStage.REMOTION_RENDER,
            PipelineStage.PUBLISH,
        ]

        try:
            # Execute stages sequentially
            for stage in stages:
                await self._execute_stage(job, stage)

                # Stop if stage failed
                stage_result = job.get_stage_result(stage)
                if stage_result and stage_result.status == JobStatus.FAILED:
                    job.status = JobStatus.FAILED
                    job.error = stage_result.error
                    break

            # Mark as completed if all stages succeeded
            if job.status == JobStatus.PROCESSING:
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now(timezone.utc)

            # Emit completion event
            await self.event_bus.publish(
                Topics.MEDIA_FACTORY_JOB_COMPLETED,
                {
                    "job_id": job_id,
                    "status": job.status.value,
                    "video_url": job.video_url,
                    "duration_seconds": (
                        (job.completed_at - job.started_at).total_seconds()
                        if job.completed_at and job.started_at else 0
                    )
                }
            )

            logger.success(f"✓ Pipeline completed for job: {job_id[:8]} - {job.status.value}")
            return job.to_dict()

        except Exception as e:
            logger.error(f"Pipeline failed for job {job_id[:8]}: {e}")
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now(timezone.utc)

            await self.event_bus.publish(
                Topics.MEDIA_FACTORY_JOB_FAILED,
                {
                    "job_id": job_id,
                    "error": str(e),
                    "current_stage": job.current_stage.value if job.current_stage else None
                }
            )

            return job.to_dict()

    async def _execute_stage(
        self,
        job: MediaFactoryJob,
        stage: PipelineStage
    ) -> None:
        """
        Execute a single pipeline stage.

        Args:
            job: Media factory job
            stage: Pipeline stage to execute
        """
        job.current_stage = stage

        stage_result = PipelineStageResult(
            stage=stage,
            status=JobStatus.PROCESSING,
            started_at=datetime.now(timezone.utc)
        )

        logger.info(f"▶️  Executing stage: {stage.value} for job {job.job_id[:8]}")

        # Emit stage started event
        await self.event_bus.publish(
            Topics.MEDIA_FACTORY_STAGE_STARTED,
            {
                "job_id": job.job_id,
                "stage": stage.value,
                "started_at": stage_result.started_at.isoformat()
            }
        )

        try:
            # Get handler for stage
            handler = self._stage_handlers.get(stage)
            if not handler:
                # Default no-op handler for testing
                logger.warning(f"No handler registered for {stage.value}, using no-op")
                output_data = {"message": f"Stage {stage.value} - handler not implemented"}
            else:
                # Execute handler
                output_data = await handler(job)

            # Update stage result
            stage_result.status = JobStatus.COMPLETED
            stage_result.output_data = output_data
            stage_result.completed_at = datetime.now(timezone.utc)
            stage_result.duration_seconds = (
                stage_result.completed_at - stage_result.started_at
            ).total_seconds()

            # Store output in job
            self._store_stage_output(job, stage, output_data)

            logger.success(
                f"✓ Stage completed: {stage.value} for job {job.job_id[:8]} "
                f"({stage_result.duration_seconds:.2f}s)"
            )

        except Exception as e:
            logger.error(f"Stage {stage.value} failed for job {job.job_id[:8]}: {e}")

            stage_result.status = JobStatus.FAILED
            stage_result.error = str(e)
            stage_result.completed_at = datetime.now(timezone.utc)
            stage_result.duration_seconds = (
                stage_result.completed_at - stage_result.started_at
            ).total_seconds()

        finally:
            # Add stage result to job
            job.stage_results.append(stage_result)

            # Emit stage completed event
            await self.event_bus.publish(
                Topics.MEDIA_FACTORY_STAGE_COMPLETED,
                {
                    "job_id": job.job_id,
                    "stage": stage.value,
                    "status": stage_result.status.value,
                    "duration_seconds": stage_result.duration_seconds,
                    "error": stage_result.error
                }
            )

    def _store_stage_output(
        self,
        job: MediaFactoryJob,
        stage: PipelineStage,
        output_data: Dict[str, Any]
    ) -> None:
        """Store stage output in job."""
        if stage == PipelineStage.SCRIPT_GENERATION:
            job.script = output_data
        elif stage == PipelineStage.TTS_GENERATION:
            job.audio_url = output_data.get("audio_url")
        elif stage == PipelineStage.MUSIC_SELECTION:
            job.music_url = output_data.get("music_url")
        elif stage == PipelineStage.VISUALS_ASSEMBLY:
            job.visuals = output_data.get("visuals", [])
        elif stage == PipelineStage.REMOTION_RENDER:
            job.timeline = output_data.get("timeline")
            job.video_url = output_data.get("video_url")
        elif stage == PipelineStage.PUBLISH:
            job.publish_results = output_data.get("publish_results", [])

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of a job.

        Args:
            job_id: Job identifier

        Returns:
            Job status dictionary or None if not found
        """
        job = self._jobs.get(job_id)
        if not job:
            return None

        return job.to_dict()

    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """Get all jobs."""
        return [job.to_dict() for job in self._jobs.values()]

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job.

        Args:
            job_id: Job identifier

        Returns:
            True if cancelled, False if not found or already completed
        """
        job = self._jobs.get(job_id)
        if not job:
            return False

        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return False

        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now(timezone.utc)

        await self.event_bus.publish(
            Topics.MEDIA_FACTORY_JOB_CANCELLED,
            {
                "job_id": job_id,
                "cancelled_at": job.completed_at.isoformat()
            }
        )

        logger.info(f"✓ Job cancelled: {job_id[:8]}")
        return True

    def clear_completed_jobs(self, max_age_hours: int = 24) -> int:
        """
        Clear completed/failed jobs older than max_age_hours.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of jobs cleared
        """
        now = datetime.now(timezone.utc)
        cleared_count = 0

        job_ids_to_remove = []
        for job_id, job in self._jobs.items():
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                if job.completed_at:
                    age_hours = (now - job.completed_at).total_seconds() / 3600
                    if age_hours >= max_age_hours:
                        job_ids_to_remove.append(job_id)

        for job_id in job_ids_to_remove:
            del self._jobs[job_id]
            cleared_count += 1

        if cleared_count > 0:
            logger.info(f"✓ Cleared {cleared_count} old jobs")

        return cleared_count


# Global singleton
_orchestrator: Optional[MediaFactoryOrchestrator] = None


def get_orchestrator() -> MediaFactoryOrchestrator:
    """Get or create the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MediaFactoryOrchestrator()
    return _orchestrator


__all__ = [
    "MediaFactoryOrchestrator",
    "MediaFactoryJob",
    "PipelineStage",
    "JobStatus",
    "PipelineStageResult",
    "get_orchestrator",
]
