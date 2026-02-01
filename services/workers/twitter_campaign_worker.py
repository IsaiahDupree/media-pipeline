"""
Twitter Campaign Worker (ARCH-004)
===================================
Event-driven worker for Twitter campaign scheduling.

Subscribes to:
    - twitter.campaign.schedule_requested

Emits:
    - twitter.campaign.scheduled
    - twitter.campaign.failed
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from services.event_bus import EventBus, Event, Topics
from services.workers.base import BaseWorker
from services.twitter_campaign_service import TwitterCampaignService

logger = logging.getLogger(__name__)


class TwitterCampaignWorker(BaseWorker):
    """
    Worker for scheduling Twitter campaigns (ARCH-004).

    Handles:
        - Campaign scheduling requests from orchestrator
        - Tweet generation and scheduling
        - 2-hour interval management (12 tweets/day default)

    Usage:
        worker = TwitterCampaignWorker()
        await worker.start()

        # Request campaign scheduling via event
        await bus.publish("twitter.campaign.schedule_requested", {
            "pipeline_id": "...",
            "theme": "AI automation trends",
            "count": 12,
            "interval_minutes": 120,
            "offer_url": "https://..."
        })
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        worker_id: Optional[str] = None
    ):
        super().__init__(event_bus, worker_id)
        self.campaign_service = TwitterCampaignService(interval_minutes=120)

    def get_subscriptions(self) -> List[str]:
        """Subscribe to Twitter campaign events."""
        return [
            "twitter.campaign.schedule_requested",
        ]

    async def handle_event(self, event: Event) -> None:
        """Handle Twitter campaign events."""
        if event.topic == "twitter.campaign.schedule_requested":
            await self._handle_schedule_request(event)

    async def _handle_schedule_request(self, event: Event) -> None:
        """
        Handle campaign scheduling request (ARCH-004).

        Payload:
            - pipeline_id: Orchestrator pipeline ID
            - theme: Campaign theme
            - count: Number of tweets (default 12)
            - interval_minutes: Time between tweets (default 120 = 2 hours)
            - offer_url: Optional offer URL to include in tweets
        """
        payload = event.payload
        pipeline_id = payload.get("pipeline_id")
        theme = payload.get("theme")
        count = payload.get("count", 12)
        interval_minutes = payload.get("interval_minutes", 120)
        offer_url = payload.get("offer_url")

        if not theme:
            logger.error(f"[{self.worker_id}] Campaign request missing theme")
            await self.emit(
                "twitter.campaign.failed",
                {
                    "pipeline_id": pipeline_id,
                    "error": "Missing theme"
                },
                correlation_id=event.correlation_id
            )
            return

        logger.info(
            f"[{self.worker_id}] Scheduling campaign: "
            f"theme='{theme}', count={count}, interval={interval_minutes}min"
        )

        try:
            # Schedule campaign
            if offer_url:
                # Use offer-specific scheduling (REQ-TWITTER-002)
                tweet_ids = self.campaign_service.schedule_offer_tweets(
                    offer_url=offer_url,
                    offer_description=theme,
                    count=count,
                    interval_minutes=interval_minutes,
                    campaign_name=theme
                )
                campaign_id = f"offer_{theme}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"
            else:
                # Use theme-based scheduling
                campaign_id = self.campaign_service.schedule_campaign(
                    theme=theme,
                    count=count,
                    interval_minutes=interval_minutes
                )
                tweet_ids = []  # schedule_campaign doesn't return IDs yet

            # Emit success event
            await self.emit(
                "twitter.campaign.scheduled",
                {
                    "pipeline_id": pipeline_id,
                    "campaign_id": campaign_id,
                    "theme": theme,
                    "tweets_scheduled": count,
                    "interval_minutes": interval_minutes,
                    "offer_url": offer_url
                },
                correlation_id=event.correlation_id
            )

            logger.success(
                f"[{self.worker_id}] Campaign scheduled: {count} tweets, "
                f"interval={interval_minutes}min"
            )

        except Exception as e:
            logger.error(f"[{self.worker_id}] Campaign scheduling failed: {e}")
            await self.emit(
                "twitter.campaign.failed",
                {
                    "pipeline_id": pipeline_id,
                    "theme": theme,
                    "error": str(e)
                },
                correlation_id=event.correlation_id
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        stats = super().get_stats()
        stats.update({
            "campaigns_scheduled": self._events_processed,
            "campaigns_failed": self._events_failed,
        })
        return stats


# Singleton instance
_worker: Optional[TwitterCampaignWorker] = None

def get_twitter_campaign_worker(event_bus: Optional[EventBus] = None) -> TwitterCampaignWorker:
    """Get singleton TwitterCampaignWorker instance."""
    global _worker
    if _worker is None:
        _worker = TwitterCampaignWorker(event_bus)
    return _worker
