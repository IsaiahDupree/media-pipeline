"""
Checkback Scheduler Worker (PTK-003)
====================================
Monitors for published posts and schedules engagement checkback periods.

This worker:
1. Listens for POST_PUBLISHED events
2. Schedules checkback periods (1h, 6h, 24h, 72h, 7d)
3. Triggers metrics collection at each checkback period
4. Integrates with Sleep Mode to wake system for checkbacks

Checkback Periods:
- 1 hour: Early engagement signal
- 6 hours: Short-term performance
- 24 hours: 1-day benchmark
- 72 hours: 3-day viral potential
- 7 days: Long-term performance
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from loguru import logger

from services.event_bus.bus import EventBus
from services.event_bus.topics import Topics
from services.sleep_mode_service import SleepModeService, WakeTriggerType
from database.connection import async_session_maker
from database.models import ScheduledPost, ContentMetricsSnapshot
from sqlalchemy import select


class CheckbackSchedulerWorker:
    """
    Checkback Scheduler Worker

    Listens for POST_PUBLISHED events and schedules engagement checkbacks.
    Integrates with Sleep Mode to wake system at checkback times.

    Usage:
        worker = CheckbackSchedulerWorker(event_bus)
        await worker.start()
    """

    def __init__(self, event_bus: EventBus):
        """Initialize checkback scheduler worker"""
        self.event_bus = event_bus
        self.event_bus.set_source("checkback-scheduler-worker")

        self._is_running = False
        self._checkback_task: Optional[asyncio.Task] = None
        self._sleep_service: Optional[SleepModeService] = None

        # Checkback periods
        self.checkback_periods = [
            {"hours": 1, "label": "1 hour"},
            {"hours": 6, "label": "6 hours"},
            {"hours": 24, "label": "1 day"},
            {"hours": 72, "label": "3 days"},
            {"hours": 168, "label": "7 days"},
        ]

        logger.info("📅 Checkback Scheduler Worker initialized")

    @property
    def sleep_service(self) -> Optional[SleepModeService]:
        """Lazy load sleep service"""
        if self._sleep_service is None:
            try:
                self._sleep_service = SleepModeService.get_instance()
            except Exception as e:
                logger.warning(f"Sleep service not available: {e}")
        return self._sleep_service

    async def start(self) -> None:
        """Start the worker"""
        if self._is_running:
            logger.warning("Checkback scheduler worker already running")
            return

        self._is_running = True

        # Subscribe to POST_PUBLISHED events
        self.event_bus.subscribe(Topics.POST_PUBLISHED, self._handle_post_published)

        # Start checkback execution loop
        self._checkback_task = asyncio.create_task(self._checkback_execution_loop())

        logger.success("✓ Checkback Scheduler Worker started")

    async def stop(self) -> None:
        """Stop the worker"""
        self._is_running = False

        if self._checkback_task:
            self._checkback_task.cancel()
            try:
                await self._checkback_task
            except asyncio.CancelledError:
                pass

        logger.info("🛑 Checkback Scheduler Worker stopped")

    async def _handle_post_published(self, event: Any) -> None:
        """
        Handle POST_PUBLISHED event by scheduling checkbacks

        Args:
            event: The post.published event
        """
        try:
            payload = event.payload if hasattr(event, 'payload') else event

            scheduled_post_id = payload.get('scheduled_post_id')
            platform = payload.get('platform')
            published_at_str = payload.get('published_at')

            if not scheduled_post_id or not published_at_str:
                logger.warning("Missing required fields in POST_PUBLISHED event")
                return

            published_at = datetime.fromisoformat(published_at_str.replace('Z', '+00:00'))

            logger.info(
                f"📊 Scheduling checkbacks | Post: {scheduled_post_id} | "
                f"Platform: {platform}"
            )

            # Schedule checkbacks and wake triggers
            for period in self.checkback_periods:
                checkback_time = published_at + timedelta(hours=period['hours'])

                # Schedule wake trigger (SLEEP-005: Checkback Period Wake Trigger)
                if self.sleep_service:
                    try:
                        wake_id = self.sleep_service.schedule_wake(
                            wake_time=checkback_time,
                            trigger_type=WakeTriggerType.CHECKBACK_PERIOD,
                            metadata={
                                "scheduled_post_id": scheduled_post_id,
                                "platform": platform,
                                "checkback_hours": period['hours'],
                                "checkback_label": period['label']
                            }
                        )
                        logger.debug(
                            f"⏰ Wake scheduled | Checkback: {period['label']} | "
                            f"Time: {checkback_time.isoformat()}"
                        )
                    except ValueError as e:
                        # Wake time is in the past (post was published long ago)
                        logger.debug(f"Skipping past checkback: {e}")

                # Emit CHECKBACK_SCHEDULED event
                await self.event_bus.publish(
                    Topics.CHECKBACK_SCHEDULED,
                    {
                        "scheduled_post_id": scheduled_post_id,
                        "platform": platform,
                        "checkback_time": checkback_time.isoformat(),
                        "checkback_hours": period['hours'],
                        "checkback_label": period['label']
                    }
                )

            logger.success(
                f"✓ Checkbacks scheduled | Post: {scheduled_post_id} | "
                f"Periods: {len(self.checkback_periods)}"
            )

        except Exception as e:
            logger.error(f"Error handling post published event: {e}")

    async def _checkback_execution_loop(self) -> None:
        """
        Background loop to check for due checkbacks and trigger metrics collection

        Runs every 5 minutes to check for posts that need metrics collected.
        """
        logger.info("🔄 Checkback execution loop started")

        while self._is_running:
            try:
                await self._check_and_execute_checkbacks()
                await asyncio.sleep(300)  # Check every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in checkback execution loop: {e}")
                await asyncio.sleep(300)

        logger.info("🛑 Checkback execution loop stopped")

    async def _check_and_execute_checkbacks(self) -> None:
        """
        Check for posts that need checkbacks and trigger metrics collection
        """
        try:
            now = datetime.now(timezone.utc)

            async with async_session_maker() as db:
                # Find published posts
                result = await db.execute(
                    select(ScheduledPost)
                    .where(ScheduledPost.status == 'published')
                    .where(ScheduledPost.published_at.isnot(None))
                )
                published_posts = result.scalars().all()

                for post in published_posts:
                    published_at = post.published_at

                    # Check each checkback period
                    for period in self.checkback_periods:
                        checkback_time = published_at + timedelta(hours=period['hours'])

                        # Check if checkback is due (within last 5 minutes)
                        time_diff = (now - checkback_time).total_seconds()

                        if 0 <= time_diff <= 300:  # Due in last 5 minutes
                            # Check if we already have a snapshot for this period
                            snapshot_result = await db.execute(
                                select(ContentMetricsSnapshot)
                                .where(ContentMetricsSnapshot.scheduled_post_id == post.id)
                                .where(ContentMetricsSnapshot.snapshot_at >= checkback_time)
                                .where(ContentMetricsSnapshot.snapshot_at < checkback_time + timedelta(minutes=10))
                            )
                            existing_snapshot = snapshot_result.scalar_one_or_none()

                            if not existing_snapshot:
                                # Trigger checkback
                                await self._trigger_checkback(post, period['hours'])

        except Exception as e:
            logger.error(f"Error checking for due checkbacks: {e}")

    async def _trigger_checkback(self, post: ScheduledPost, checkback_hours: int) -> None:
        """
        Trigger metrics collection for a checkback period

        Args:
            post: The scheduled post
            checkback_hours: Hours after publish for this checkback
        """
        try:
            logger.info(
                f"📊 Triggering checkback | Post: {post.id} | "
                f"Platform: {post.platform} | "
                f"Hours: {checkback_hours}"
            )

            # Emit CHECKBACK_TRIGGERED event (will be picked up by metrics workers)
            await self.event_bus.publish(
                Topics.CHECKBACK_TRIGGERED,
                {
                    "scheduled_post_id": str(post.id),
                    "platform": post.platform,
                    "platform_url": post.platform_url,
                    "platform_post_id": post.platform_post_id,
                    "checkback_hours": checkback_hours,
                    "triggered_at": datetime.now(timezone.utc).isoformat()
                }
            )

            logger.success(
                f"✓ Checkback triggered | Post: {post.id} | "
                f"Hours: {checkback_hours}"
            )

        except Exception as e:
            logger.error(f"Error triggering checkback: {e}")
