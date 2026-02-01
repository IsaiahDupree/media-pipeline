"""
Experiment Tracker Worker
=========================
Event-driven worker that tracks experiment variant performance.

Subscribes to:
    - publish.completed (when variant content is published)
    - metrics.updated (when metrics are fetched)

Emits:
    - experiment.variant.metrics.updated
    - experiment.significance.reached
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from services.event_bus import EventBus, Event, Topics
from services.workers.base import BaseWorker

logger = logging.getLogger(__name__)


class ExperimentTrackerWorker(BaseWorker):
    """
    Worker that tracks experiment variant performance metrics.
    
    Automatically updates variant metrics when content is published
    or when new engagement metrics are fetched.
    
    Usage:
        worker = ExperimentTrackerWorker()
        await worker.start()
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None, worker_id: Optional[str] = None):
        super().__init__(event_bus, worker_id)
    
    def get_subscriptions(self) -> List[str]:
        """Subscribe to publish and metrics events."""
        return [
            Topics.PUBLISH_COMPLETED,
            Topics.METRICS_UPDATED,
        ]
    
    async def handle_event(self, event: Event) -> None:
        """Process tracking event."""
        if event.topic == Topics.PUBLISH_COMPLETED:
            await self._handle_publish(event)
        elif event.topic == Topics.METRICS_UPDATED:
            await self._handle_metrics_update(event)
    
    async def _handle_publish(self, event: Event) -> None:
        """Track when experiment variant content is published."""
        media_id = event.payload.get("media_id")
        platform = event.payload.get("platform")
        platform_url = event.payload.get("platform_url")
        
        if not media_id:
            return
        
        # Check if this media is part of an experiment
        experiment_info = await self._get_experiment_for_media(media_id)
        
        if experiment_info:
            experiment_id = experiment_info.get("experiment_id")
            variant_id = experiment_info.get("variant_id")
            
            logger.info(f"[{self.worker_id}] Variant published: experiment={experiment_id}, variant={variant_id}")
            
            # Increment impressions for this variant
            await self._increment_variant_impressions(variant_id)
            
            # Emit variant published event
            await self.emit(
                Topics.EXPERIMENT_VARIANT_PUBLISHED,
                {
                    "experiment_id": experiment_id,
                    "variant_id": variant_id,
                    "media_id": media_id,
                    "platform": platform,
                    "platform_url": platform_url,
                },
                event.correlation_id
            )
    
    async def _handle_metrics_update(self, event: Event) -> None:
        """Update variant metrics when engagement data is received."""
        media_id = event.payload.get("media_id")
        metrics = event.payload.get("metrics", {})
        
        if not media_id or not metrics:
            return
        
        # Check if this media is part of an experiment
        experiment_info = await self._get_experiment_for_media(media_id)
        
        if experiment_info:
            experiment_id = experiment_info.get("experiment_id")
            variant_id = experiment_info.get("variant_id")
            primary_metric = experiment_info.get("primary_metric", "views")
            
            # Calculate the primary metric value
            metric_value = self._calculate_metric_value(metrics, primary_metric)
            
            # Update variant metrics
            await self._update_variant_metrics(variant_id, metrics, metric_value)
            
            logger.info(f"[{self.worker_id}] Variant metrics updated: {variant_id}, {primary_metric}={metric_value}")
            
            # Check if experiment has reached significance
            await self._check_significance(experiment_id, event.correlation_id)
    
    def _calculate_metric_value(self, metrics: Dict[str, Any], primary_metric: str) -> float:
        """Calculate the primary metric value from raw metrics."""
        views = metrics.get("views", 0)
        likes = metrics.get("likes", 0)
        comments = metrics.get("comments", 0)
        shares = metrics.get("shares", 0)
        saves = metrics.get("saves", 0)
        
        if primary_metric == "views":
            return float(views)
        elif primary_metric == "likes":
            return float(likes)
        elif primary_metric == "engagement_rate":
            if views > 0:
                return ((likes + comments + shares) / views) * 100
            return 0.0
        elif primary_metric == "save_rate":
            if views > 0:
                return (saves / views) * 100
            return 0.0
        elif primary_metric == "share_rate":
            if views > 0:
                return (shares / views) * 100
            return 0.0
        elif primary_metric == "comment_rate":
            if views > 0:
                return (comments / views) * 100
            return 0.0
        else:
            return float(metrics.get(primary_metric, 0))
    
    async def _get_experiment_for_media(self, media_id: str) -> Optional[Dict[str, Any]]:
        """Check if media belongs to an experiment variant."""
        import httpx
        
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"http://localhost:5555/api/experiments/by-media/{media_id}"
                )
                
                if response.status_code == 200:
                    return response.json()
                return None
                
        except Exception as e:
            logger.debug(f"[{self.worker_id}] No experiment for media {media_id}: {e}")
            return None
    
    async def _increment_variant_impressions(self, variant_id: str) -> None:
        """Increment impression count for a variant."""
        import httpx
        
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(
                    f"http://localhost:5555/api/experiments/variant/{variant_id}/impression"
                )
        except Exception as e:
            logger.warning(f"[{self.worker_id}] Failed to increment impressions: {e}")
    
    async def _update_variant_metrics(
        self,
        variant_id: str,
        metrics: Dict[str, Any],
        primary_metric_value: float
    ) -> None:
        """Update variant with new metrics."""
        import httpx
        
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.put(
                    f"http://localhost:5555/api/experiments/variant/{variant_id}/metrics",
                    params={
                        "views": metrics.get("views", 0),
                        "primary_metric_value": primary_metric_value
                    }
                )
        except Exception as e:
            logger.warning(f"[{self.worker_id}] Failed to update variant metrics: {e}")
    
    async def _check_significance(self, experiment_id: str, correlation_id: str) -> None:
        """Check if experiment has reached statistical significance."""
        import httpx
        
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"http://localhost:5555/api/experiments/{experiment_id}/significance"
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get("is_significant"):
                        logger.info(f"[{self.worker_id}] Experiment {experiment_id} reached significance!")
                        
                        # Emit significance reached event
                        await self.emit(
                            Topics.EXPERIMENT_SIGNIFICANCE_REACHED,
                            {
                                "experiment_id": experiment_id,
                                "confidence": data.get("confidence"),
                                "winner_variant_id": data.get("winner_variant_id"),
                                "uplift": data.get("uplift"),
                            },
                            correlation_id
                        )
        except Exception as e:
            logger.debug(f"[{self.worker_id}] Significance check skipped: {e}")
