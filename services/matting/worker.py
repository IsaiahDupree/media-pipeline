"""
Matting Worker
==============
Event-driven worker for video matting operations.

Subscribes to:
    - matting.requested

Emits:
    - matting.started
    - matting.segmenting
    - matting.extracting
    - matting.compositing
    - matting.progress
    - matting.completed
    - matting.failed
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from services.event_bus import EventBus, Event, Topics
from services.workers.base import BaseWorker

from .models import MattingRequest, MattingResponse, MattingModel, MattingJobStatus, MattingConfig, MattingOperation
from .adapters.base import MattingAdapter

logger = logging.getLogger(__name__)


class MattingWorker(BaseWorker):
    """
    Worker for processing video matting requests.
    
    Supports multiple matting models via adapters:
        - RVM (Robust Video Matting) - Primary, production quality
        - MediaPipe - Fallback, fast and lightweight
        - BackgroundMattingV2 - Optional, with clean plate
        - rembg - Optional, simple batch processing
        - SAM 2 - Future, advanced segmentation
    
    Usage:
        worker = MattingWorker()
        await worker.start()
        
        # Worker will automatically process events from:
        # - matting.requested
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None, worker_id: Optional[str] = None):
        super().__init__(event_bus, worker_id)
        self._adapters: Dict[str, MattingAdapter] = {}
        self._jobs: Dict[str, MattingJobStatus] = {}
    
    def get_subscriptions(self) -> list:
        """Subscribe to matting-related events."""
        return [
            Topics.MATTING_REQUESTED,
        ]
    
    async def handle_event(self, event: Event) -> None:
        """Process matting events."""
        try:
            # Parse request from event payload
            request = self._parse_request(event.payload)
            
            if not request:
                logger.error(f"[{self.worker_id}] Invalid matting request in event")
                await self.emit(
                    Topics.MATTING_FAILED,
                    {
                        "error": "Invalid request payload",
                        "correlation_id": event.correlation_id
                    },
                    event.correlation_id
                )
                return
            
            # Create job status
            job_status = MattingJobStatus(
                job_id=request.job_id,
                status="pending",
                correlation_id=request.correlation_id
            )
            self._jobs[request.job_id] = job_status
            
            # Emit started event
            await self.emit(
                Topics.MATTING_STARTED,
                {
                    "job_id": request.job_id,
                    "model": request.model.value,
                    "operation": request.config.operation.value,
                    "correlation_id": request.correlation_id
                },
                request.correlation_id
            )
            
            # Process the request
            await self._process_request(request, job_status)
            
        except Exception as e:
            logger.error(f"[{self.worker_id}] Error processing matting event: {e}", exc_info=True)
            await self.emit(
                Topics.MATTING_FAILED,
                {
                    "error": str(e),
                    "correlation_id": event.correlation_id
                },
                event.correlation_id
            )
    
    def _parse_request(self, payload: Dict[str, Any]) -> Optional[MattingRequest]:
        """Parse matting request from event payload."""
        try:
            # Parse config if present
            config = None
            if "config" in payload:
                config_data = payload["config"]
                config = MattingConfig(
                    operation=MattingOperation(config_data.get("operation", "extract_person")),
                    target_description=config_data.get("target_description"),
                    clean_background_plate=config_data.get("clean_background_plate"),
                    preserve_alpha=config_data.get("preserve_alpha", True),
                    downsample_ratio=config_data.get("downsample_ratio", 0.25),
                    device=config_data.get("device", "auto"),
                    model_variant=config_data.get("model_variant")
                )
            
            return MattingRequest(
                source_video=payload["source_video"],
                target_video=payload.get("target_video"),
                model=MattingModel(payload.get("model", "rvm")),
                config=config,
                output_path=payload.get("output_path"),
                correlation_id=payload.get("correlation_id"),
                job_id=payload.get("job_id")
            )
        except Exception as e:
            logger.error(f"Failed to parse matting request: {e}")
            return None
    
    async def _process_request(self, request: MattingRequest, job_status: MattingJobStatus) -> None:
        """Process a matting request."""
        job_status.status = "processing"
        job_status.started_at = datetime.now(timezone.utc)
        
        try:
            # Get or create adapter for the model
            adapter = await self._get_adapter(request.model)
            
            if not adapter:
                error = f"Adapter not available for model: {request.model.value}"
                job_status.status = "failed"
                job_status.error = error
                await self.emit(
                    Topics.MATTING_FAILED,
                    {
                        "job_id": request.job_id,
                        "error": error,
                        "correlation_id": request.correlation_id
                    },
                    request.correlation_id
                )
                return
            
            # Emit progress based on operation
            if request.config.operation == MattingOperation.EXTRACT_PERSON or request.config.operation == MattingOperation.EXTRACT_OBJECT:
                await self.emit(
                    Topics.MATTING_SEGMENTING,
                    {
                        "job_id": request.job_id,
                        "progress": 0.1,
                        "message": "Segmenting objects...",
                        "correlation_id": request.correlation_id
                    },
                    request.correlation_id
                )
                job_status.status = "segmenting"
                job_status.progress = 0.1
            
            await self.emit(
                Topics.MATTING_EXTRACTING,
                {
                    "job_id": request.job_id,
                    "progress": 0.3,
                    "message": "Extracting foreground...",
                    "correlation_id": request.correlation_id
                },
                request.correlation_id
            )
            job_status.status = "extracting"
            job_status.progress = 0.3
            
            # Extract foreground
            response = await adapter.extract_foreground(request)
            
            if response.success:
                # If compositing, emit compositing event
                if request.config.operation == MattingOperation.COMPOSITE and request.target_video:
                    await self.emit(
                        Topics.MATTING_COMPOSITING,
                        {
                            "job_id": request.job_id,
                            "progress": 0.8,
                            "message": "Compositing into target...",
                            "correlation_id": request.correlation_id
                        },
                        request.correlation_id
                    )
                    job_status.status = "compositing"
                    job_status.progress = 0.8
                    
                    # TODO: Implement compositing logic
                    # For now, compositing would be done in Remotion service
                
                job_status.status = "completed"
                job_status.completed_at = datetime.now(timezone.utc)
                job_status.progress = 1.0
                job_status.response = response
                
                # Emit completion event
                await self.emit(
                    Topics.MATTING_COMPLETED,
                    {
                        "job_id": response.job_id,
                        "output_path": response.output_path,
                        "mask_path": response.mask_path,
                        "processing_time": response.processing_time,
                        "model_used": response.model_used,
                        "frames_processed": response.frames_processed,
                        "correlation_id": response.correlation_id
                    },
                    response.correlation_id
                )
            else:
                job_status.status = "failed"
                job_status.error = response.error
                job_status.completed_at = datetime.now(timezone.utc)
                
                await self.emit(
                    Topics.MATTING_FAILED,
                    {
                        "job_id": request.job_id,
                        "error": response.error,
                        "correlation_id": request.correlation_id
                    },
                    request.correlation_id
                )
                
        except Exception as e:
            logger.error(f"[{self.worker_id}] Matting processing error: {e}", exc_info=True)
            job_status.status = "failed"
            job_status.error = str(e)
            job_status.completed_at = datetime.now(timezone.utc)
            
            await self.emit(
                Topics.MATTING_FAILED,
                {
                    "job_id": request.job_id,
                    "error": str(e),
                    "correlation_id": request.correlation_id
                },
                request.correlation_id
            )
    
    async def _get_adapter(self, model: MattingModel) -> Optional[MattingAdapter]:
        """Get or create adapter for the specified model."""
        model_key = model.value
        
        if model_key not in self._adapters:
            # Create adapter based on model type
            if model == MattingModel.RVM:
                try:
                    from .adapters.rvm import RVMAdapter
                    adapter = RVMAdapter()
                    loaded = await adapter.load_model()
                    if loaded:
                        self._adapters[model_key] = adapter
                        logger.info(f"[{self.worker_id}] Loaded {model_key} adapter")
                    else:
                        logger.warning(f"[{self.worker_id}] Failed to load {model_key} adapter")
                        # Try fallback
                        return await self._get_adapter(MattingModel.MEDIAPIPE)
                except ImportError:
                    logger.warning(f"[{self.worker_id}] RVM adapter not available, trying fallback")
                    return await self._get_adapter(MattingModel.MEDIAPIPE)
            
            elif model == MattingModel.MEDIAPIPE:
                try:
                    from .adapters.mediapipe import MediaPipeAdapter
                    adapter = MediaPipeAdapter()
                    loaded = await adapter.load_model()
                    if loaded:
                        self._adapters[model_key] = adapter
                        logger.info(f"[{self.worker_id}] Loaded {model_key} adapter")
                    else:
                        logger.error(f"[{self.worker_id}] Failed to load {model_key} adapter")
                        return None
                except ImportError:
                    logger.error(f"[{self.worker_id}] MediaPipe adapter not available")
                    return None
            
            else:
                logger.warning(f"[{self.worker_id}] Model {model_key} not yet implemented")
                # Try fallback to MediaPipe
                if model != MattingModel.MEDIAPIPE:
                    return await self._get_adapter(MattingModel.MEDIAPIPE)
                return None
        
        return self._adapters.get(model_key)
    
    def get_job_status(self, job_id: str) -> Optional[MattingJobStatus]:
        """Get status of a matting job."""
        return self._jobs.get(job_id)
    
    async def shutdown(self) -> None:
        """Shutdown worker and unload models."""
        # Unload all adapters
        for model_key, adapter in self._adapters.items():
            try:
                await adapter.unload_model()
                logger.info(f"[{self.worker_id}] Unloaded {model_key} adapter")
            except Exception as e:
                logger.error(f"[{self.worker_id}] Error unloading {model_key}: {e}")
        
        await super().shutdown()

