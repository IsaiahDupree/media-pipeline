"""
Notification Worker
====================
Event-driven worker that generates notifications for key system events.

Subscribes to:
    - publish.completed (post published successfully)
    - publish.failed (post failed to publish)
    - analysis.completed (analysis finished)
    - experiment.run.completed (experiment finished)
    - goal.completed (goal achieved)

Emits:
    - notification.created
    - notification.sent
    - mp.ui.evt.toast (for real-time UI updates)
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from services.event_bus import EventBus, Event, Topics
from services.workers.base import BaseWorker

logger = logging.getLogger(__name__)


class NotificationWorker(BaseWorker):
    """
    Worker that generates notifications for key system events.
    
    Subscribes to important lifecycle events and creates notifications
    that can be displayed in the UI or sent via external channels.
    
    Usage:
        worker = NotificationWorker()
        await worker.start()
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None, worker_id: Optional[str] = None):
        super().__init__(event_bus, worker_id)
        self._notification_count = 0
    
    def get_subscriptions(self) -> List[str]:
        """Subscribe to key events that should trigger notifications."""
        return [
            Topics.PUBLISH_COMPLETED,
            Topics.PUBLISH_FAILED,
            Topics.ANALYSIS_COMPLETED,
            Topics.EXPERIMENT_RUN_COMPLETED,
            Topics.GOAL_COMPLETED,
            Topics.AI_GENERATION_COMPLETED,
            Topics.AI_GENERATION_FAILED,
        ]
    
    async def handle_event(self, event: Event) -> None:
        """Process event and generate appropriate notification."""
        notification = await self._create_notification(event)
        
        if notification:
            # Emit notification created event
            await self.emit(
                Topics.NOTIFICATION_CREATED,
                notification,
                event.correlation_id
            )
            
            # Emit UI toast for real-time display
            await self.emit(
                Topics.UI_TOAST,
                {
                    "type": notification["type"],
                    "title": notification["title"],
                    "message": notification["message"],
                    "duration": notification.get("duration", 5000),
                },
                event.correlation_id
            )
            
            self._notification_count += 1
            logger.info(f"[{self.worker_id}] Created notification: {notification['title']}")
    
    async def _create_notification(self, event: Event) -> Optional[Dict[str, Any]]:
        """Create notification payload based on event type."""
        payload = event.payload or {}
        timestamp = datetime.now(timezone.utc).isoformat()
        
        if event.topic == Topics.PUBLISH_COMPLETED:
            return {
                "id": f"notif-{self._notification_count}",
                "type": "success",
                "category": "publishing",
                "title": "Content Published!",
                "message": f"Successfully posted to {payload.get('platform', 'platform')}",
                "data": {
                    "media_id": payload.get("media_id"),
                    "platform": payload.get("platform"),
                    "platform_url": payload.get("platform_url"),
                },
                "created_at": timestamp,
                "duration": 5000,
            }
        
        elif event.topic == Topics.PUBLISH_FAILED:
            return {
                "id": f"notif-{self._notification_count}",
                "type": "error",
                "category": "publishing",
                "title": "Publish Failed",
                "message": f"Failed to post to {payload.get('platform', 'platform')}: {payload.get('error', 'Unknown error')}",
                "data": {
                    "media_id": payload.get("media_id"),
                    "platform": payload.get("platform"),
                    "error": payload.get("error"),
                },
                "created_at": timestamp,
                "duration": 10000,
            }
        
        elif event.topic == Topics.ANALYSIS_COMPLETED:
            score = payload.get("pre_social_score")
            score_text = f" (Score: {score})" if score else ""
            return {
                "id": f"notif-{self._notification_count}",
                "type": "info",
                "category": "analysis",
                "title": "Analysis Complete",
                "message": f"Content analysis finished{score_text}",
                "data": {
                    "media_id": payload.get("media_id"),
                    "pre_social_score": score,
                },
                "created_at": timestamp,
                "duration": 4000,
            }
        
        elif event.topic == Topics.EXPERIMENT_RUN_COMPLETED:
            winner = payload.get("winner_variant_id")
            uplift = payload.get("uplift")
            return {
                "id": f"notif-{self._notification_count}",
                "type": "success",
                "category": "experiment",
                "title": "Experiment Complete!",
                "message": f"Winner found with {uplift:.1f}% uplift" if uplift else "Experiment completed",
                "data": {
                    "experiment_id": payload.get("experiment_id"),
                    "winner_variant_id": winner,
                    "uplift": uplift,
                },
                "created_at": timestamp,
                "duration": 8000,
            }
        
        elif event.topic == Topics.GOAL_COMPLETED:
            return {
                "id": f"notif-{self._notification_count}",
                "type": "success",
                "category": "goal",
                "title": "Goal Achieved! 🎉",
                "message": f"You've completed your goal",
                "data": {
                    "goal_id": payload.get("goal_id"),
                },
                "created_at": timestamp,
                "duration": 8000,
            }
        
        elif event.topic == Topics.AI_GENERATION_COMPLETED:
            return {
                "id": f"notif-{self._notification_count}",
                "type": "success",
                "category": "ai_generation",
                "title": "AI Video Ready",
                "message": f"Your AI-generated video is ready to view",
                "data": {
                    "job_id": payload.get("job_id"),
                    "output_url": payload.get("output_url"),
                },
                "created_at": timestamp,
                "duration": 6000,
            }
        
        elif event.topic == Topics.AI_GENERATION_FAILED:
            return {
                "id": f"notif-{self._notification_count}",
                "type": "error",
                "category": "ai_generation",
                "title": "AI Generation Failed",
                "message": f"Video generation failed: {payload.get('error', 'Unknown error')}",
                "data": {
                    "job_id": payload.get("job_id"),
                    "error": payload.get("error"),
                },
                "created_at": timestamp,
                "duration": 10000,
            }
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        stats = super().get_stats()
        stats["notifications_created"] = self._notification_count
        return stats
