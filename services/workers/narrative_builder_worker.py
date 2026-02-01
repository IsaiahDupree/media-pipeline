"""
Narrative Builder Worker
=========================
Event-driven worker that automatically updates narrative builder signals
when content is published, analyzed, or scheduled.

Subscribes to:
    - publish.completed (content published - update signals)
    - media.analysis.completed (new analysis - update topic momentum)
    - schedule.created (new scheduled post - update creative fatigue)

Emits:
    - narrative.signals.updated (signals refreshed)
    - narrative.goal.progress.updated (goal progress changed)
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from services.event_bus import EventBus, Event, Topics
from services.workers.base import BaseWorker
from database.connection import async_session_maker
from sqlalchemy import text

logger = logging.getLogger(__name__)


class NarrativeBuilderWorker(BaseWorker):
    """
    Worker that automatically updates narrative builder signals and goals.
    
    When content is published or analyzed, this worker:
    1. Updates signal metrics (creative fatigue, topic momentum, etc.)
    2. Updates goal progress if the post is associated with a goal
    3. Emits events so the frontend can refresh
    
    Usage:
        worker = NarrativeBuilderWorker()
        await worker.start()
        
        # Signals will auto-update when publish.completed events are emitted
    """
    
    def get_subscriptions(self) -> List[str]:
        """Subscribe to events that affect narrative signals."""
        return [
            Topics.PUBLISH_COMPLETED,
            Topics.ANALYSIS_COMPLETED,
            Topics.SCHEDULE_CREATED,
        ]
    
    async def handle_event(self, event: Event) -> None:
        """Update narrative signals based on event."""
        try:
            if event.topic == Topics.PUBLISH_COMPLETED:
                await self._handle_publish_completed(event)
            elif event.topic == Topics.ANALYSIS_COMPLETED:
                await self._handle_analysis_completed(event)
            elif event.topic == Topics.SCHEDULE_CREATED:
                await self._handle_schedule_created(event)
        except Exception as e:
            logger.error(f"Error updating narrative signals: {e}", exc_info=True)
    
    async def _handle_publish_completed(self, event: Event) -> None:
        """Update signals when content is published."""
        payload = event.payload
        media_id = payload.get("media_id")
        platform = payload.get("platform")
        goal_id = payload.get("goal_id")
        
        if not async_session_maker:
            logger.warning("Database not initialized, skipping narrative update")
            return
        
        try:
            async with async_session_maker() as session:
                # Update goal progress if goal_id is present
                if goal_id:
                    await self._update_goal_progress(session, goal_id, media_id, platform)
                
                # Emit signals updated event
                await self.emit(
                    "narrative.signals.updated",  # Custom topic
                    {
                        "trigger": "publish_completed",
                        "media_id": media_id,
                        "platform": platform,
                        "goal_id": goal_id
                    },
                    event.correlation_id
                )
                
                logger.info(f"Updated narrative signals for published content: {media_id}")
                
        except Exception as e:
            logger.error(f"Error handling publish.completed for narrative: {e}", exc_info=True)
    
    async def _handle_analysis_completed(self, event: Event) -> None:
        """Update topic momentum when analysis completes."""
        payload = event.payload
        media_id = payload.get("media_id")
        topics = payload.get("topics", [])
        
        if not topics:
            return
        
        try:
            # Emit signals updated event for topic momentum
            await self.emit(
                "narrative.signals.updated",
                {
                    "trigger": "analysis_completed",
                    "media_id": media_id,
                    "topics": topics,
                    "signal_type": "topic_momentum"
                },
                event.correlation_id
            )
            
            logger.debug(f"Updated topic momentum for analyzed content: {media_id}")
            
        except Exception as e:
            logger.error(f"Error handling analysis.completed for narrative: {e}", exc_info=True)
    
    async def _handle_schedule_created(self, event: Event) -> None:
        """Update creative fatigue when content is scheduled."""
        payload = event.payload
        content_id = payload.get("content_id")
        goal_id = payload.get("goal_id")
        
        try:
            # Emit signals updated event for creative fatigue
            await self.emit(
                "narrative.signals.updated",
                {
                    "trigger": "schedule_created",
                    "content_id": content_id,
                    "goal_id": goal_id,
                    "signal_type": "creative_fatigue"
                },
                event.correlation_id
            )
            
            logger.debug(f"Updated creative fatigue for scheduled content: {content_id}")
            
        except Exception as e:
            logger.error(f"Error handling schedule.created for narrative: {e}", exc_info=True)
    
    async def _update_goal_progress(
        self,
        session,
        goal_id: str,
        media_id: str,
        platform: str
    ) -> None:
        """Update goal progress when content is published."""
        try:
            # Get goal details
            goal_query = text("""
                SELECT 
                    id,
                    target_metric,
                    target_value,
                    current_value,
                    progress_percent
                FROM narrative_goals
                WHERE id = :goal_id
            """)
            
            result = await session.execute(goal_query, {"goal_id": goal_id})
            goal = result.fetchone()
            
            if not goal:
                logger.warning(f"Goal {goal_id} not found")
                return
            
            target_metric = goal[1]
            current_value = float(goal[3]) if goal[3] else 0.0
            target_value = float(goal[2]) if goal[2] else 1.0
            progress_percent = float(goal[4]) if goal[4] else 0.0
            
            # Increment current value based on metric type
            new_value = current_value + 1.0  # Simple increment for now
            new_progress = min(100.0, (new_value / target_value) * 100.0) if target_value > 0 else 0.0
            
            # Update goal progress
            update_query = text("""
                UPDATE narrative_goals
                SET 
                    current_value = :current_value,
                    progress_percent = :progress_percent,
                    updated_at = NOW()
                WHERE id = :goal_id
            """)
            
            await session.execute(update_query, {
                "goal_id": goal_id,
                "current_value": new_value,
                "progress_percent": new_progress
            })
            await session.commit()
            
            # Emit goal progress updated event
            await self.emit(
                Topics.NARRATIVE_GOAL_UPDATED,
                {
                    "goal_id": goal_id,
                    "action": "progress_updated",
                    "current_value": new_value,
                    "progress_percent": new_progress,
                    "trigger": "publish_completed",
                    "media_id": media_id
                },
                f"goal-{goal_id}"
            )
            
            logger.info(f"Updated goal {goal_id} progress: {new_progress:.1f}%")
            
        except Exception as e:
            logger.error(f"Error updating goal progress: {e}", exc_info=True)
            await session.rollback()

