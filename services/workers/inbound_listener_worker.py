"""
Inbound Listener Worker
=======================
Listens for comments, DMs, and mentions across all platforms.

The Inbound Listener monitors platform webhooks and polls for inbound engagement,
creating unified touchpoint records for the responder to handle.

Implements OPS-015: Inbound Listener Worker
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from loguru import logger

from services.workers.base import BaseWorker
from services.event_bus import Event, Topics


class InboundListenerWorker(BaseWorker):
    """
    Inbound Listener Worker

    Listens for inbound engagement across platforms:
    - Comments on posts
    - Direct messages (DMs)
    - Mentions and tags
    - Email replies

    Captures inbound items and routes them to the responder worker.

    Consumes:
        - Platform webhook events (platform-specific)
        - inbound.poll.requested (for polling-based platforms)

    Emits:
        - inbound.received
        - inbound.respond.requested
        - comment.received
        - dm.received

    Usage:
        worker = InboundListenerWorker(event_bus)
        await worker.start()
    """

    def __init__(self, event_bus=None, worker_id: Optional[str] = None):
        """
        Initialize inbound listener worker.

        Args:
            event_bus: EventBus instance (uses singleton if not provided)
            worker_id: Unique worker identifier
        """
        # Track seen inbound items to prevent duplicates
        self._seen_items: Dict[str, datetime] = {}
        self._max_seen_items = 10000
        self._seen_item_ttl_hours = 24

        # Platform-specific handlers
        self._platform_handlers = {
            "x": self._handle_x_inbound,
            "instagram": self._handle_instagram_inbound,
            "tiktok": self._handle_tiktok_inbound,
            "youtube": self._handle_youtube_inbound,
            "linkedin": self._handle_linkedin_inbound,
            "threads": self._handle_threads_inbound,
            "email": self._handle_email_inbound
        }

        super().__init__(event_bus, worker_id)
        logger.info("👂 Inbound Listener Worker initialized")

    def get_subscriptions(self) -> List[str]:
        """Subscribe to inbound events from all platforms."""
        return [
            "inbound.poll.requested",
            "webhook.x.*",
            "webhook.instagram.*",
            "webhook.tiktok.*",
            "webhook.youtube.*",
            "webhook.linkedin.*",
            "webhook.threads.*",
            "webhook.email.*"
        ]

    async def handle_event(self, event: Event) -> None:
        """
        Handle inbound events.

        Args:
            event: Inbound event from platform or polling request
        """
        if event.topic == "inbound.poll.requested":
            await self._handle_poll_request(event)
        elif event.topic.startswith("webhook."):
            await self._handle_webhook_event(event)

    async def _handle_poll_request(self, event: Event) -> None:
        """
        Handle polling request for platforms without webhooks.

        Args:
            event: inbound.poll.requested event
        """
        payload = event.payload
        platform = payload.get("platform")

        if not platform:
            logger.error("Missing platform in poll request")
            return

        logger.debug(f"📡 Polling {platform} for inbound items...")

        try:
            # Poll platform for new items
            # (This would integrate with platform-specific APIs)
            # For now, emit a placeholder event
            await self.emit(
                "inbound.polling.started",
                {
                    "platform": platform,
                    "started_at": datetime.now(timezone.utc).isoformat()
                },
                correlation_id=event.correlation_id
            )

        except Exception as e:
            logger.error(f"❌ Polling failed for {platform}: {e}")
            raise

    async def _handle_webhook_event(self, event: Event) -> None:
        """
        Handle webhook event from a platform.

        Args:
            event: webhook.{platform}.* event
        """
        # Extract platform from topic
        topic_parts = event.topic.split(".")
        platform = topic_parts[1] if len(topic_parts) > 1 else "unknown"

        payload = event.payload
        external_event_id = payload.get("external_event_id")
        event_type = payload.get("event_type")

        logger.info(
            f"📨 Webhook received | platform={platform} | "
            f"type={event_type} | id={external_event_id}"
        )

        # Check for duplicates
        if await self._is_duplicate(platform, external_event_id):
            logger.debug(f"Skipping duplicate event: {external_event_id}")
            return

        # Route to platform-specific handler
        handler = self._platform_handlers.get(platform)
        if handler:
            await handler(payload, event.correlation_id)
        else:
            logger.warning(f"No handler for platform: {platform}")

    async def _handle_x_inbound(
        self,
        payload: Dict[str, Any],
        correlation_id: str
    ) -> None:
        """
        Handle X/Twitter inbound event.

        Args:
            payload: Webhook payload
            correlation_id: Workflow correlation ID
        """
        event_type = payload.get("event_type")

        if event_type == "comment":
            await self._process_comment("x", payload, correlation_id)
        elif event_type == "dm":
            await self._process_dm("x", payload, correlation_id)
        elif event_type == "mention":
            await self._process_mention("x", payload, correlation_id)

    async def _handle_instagram_inbound(
        self,
        payload: Dict[str, Any],
        correlation_id: str
    ) -> None:
        """
        Handle Instagram inbound event.

        Args:
            payload: Webhook payload
            correlation_id: Workflow correlation ID
        """
        event_type = payload.get("event_type")

        if event_type == "comment":
            await self._process_comment("instagram", payload, correlation_id)
        elif event_type == "dm":
            await self._process_dm("instagram", payload, correlation_id)

    async def _handle_tiktok_inbound(
        self,
        payload: Dict[str, Any],
        correlation_id: str
    ) -> None:
        """
        Handle TikTok inbound event.

        Args:
            payload: Webhook payload
            correlation_id: Workflow correlation ID
        """
        event_type = payload.get("event_type")

        if event_type == "comment":
            await self._process_comment("tiktok", payload, correlation_id)
        elif event_type == "dm":
            await self._process_dm("tiktok", payload, correlation_id)

    async def _handle_youtube_inbound(
        self,
        payload: Dict[str, Any],
        correlation_id: str
    ) -> None:
        """
        Handle YouTube inbound event.

        Args:
            payload: Webhook payload
            correlation_id: Workflow correlation ID
        """
        event_type = payload.get("event_type")

        if event_type == "comment":
            await self._process_comment("youtube", payload, correlation_id)

    async def _handle_linkedin_inbound(
        self,
        payload: Dict[str, Any],
        correlation_id: str
    ) -> None:
        """
        Handle LinkedIn inbound event.

        Args:
            payload: Webhook payload
            correlation_id: Workflow correlation ID
        """
        event_type = payload.get("event_type")

        if event_type == "comment":
            await self._process_comment("linkedin", payload, correlation_id)
        elif event_type == "dm":
            await self._process_dm("linkedin", payload, correlation_id)

    async def _handle_threads_inbound(
        self,
        payload: Dict[str, Any],
        correlation_id: str
    ) -> None:
        """
        Handle Threads inbound event.

        Args:
            payload: Webhook payload
            correlation_id: Workflow correlation ID
        """
        event_type = payload.get("event_type")

        if event_type == "comment":
            await self._process_comment("threads", payload, correlation_id)

    async def _handle_email_inbound(
        self,
        payload: Dict[str, Any],
        correlation_id: str
    ) -> None:
        """
        Handle email inbound event.

        Args:
            payload: Webhook payload
            correlation_id: Workflow correlation ID
        """
        await self._process_email(payload, correlation_id)

    async def _process_comment(
        self,
        platform: str,
        payload: Dict[str, Any],
        correlation_id: str
    ) -> None:
        """
        Process an inbound comment.

        Args:
            platform: Platform name
            payload: Comment payload
            correlation_id: Workflow correlation ID
        """
        comment_data = {
            "platform": platform,
            "channel": "comment",
            "platform_object_id": payload.get("comment_id"),
            "actor": "user",
            "text": payload.get("text"),
            "source_touchpoint_id": payload.get("post_id"),  # Parent post
            "author_username": payload.get("author_username"),
            "created_at": payload.get("created_at") or datetime.now(timezone.utc).isoformat()
        }

        # Emit comment received event
        await self.emit(
            Topics.COMMENT_RECEIVED,
            comment_data,
            correlation_id=correlation_id
        )

        # Emit general inbound event
        await self.emit(
            "inbound.received",
            comment_data,
            correlation_id=correlation_id
        )

        # Request response
        await self.emit(
            "inbound.respond.requested",
            {
                **comment_data,
                "strategy": "public_reply",
                "requested_at": datetime.now(timezone.utc).isoformat()
            },
            correlation_id=correlation_id
        )

        logger.info(
            f"💬 Comment received | platform={platform} | "
            f"author={comment_data.get('author_username')}"
        )

    async def _process_dm(
        self,
        platform: str,
        payload: Dict[str, Any],
        correlation_id: str
    ) -> None:
        """
        Process an inbound direct message.

        Args:
            platform: Platform name
            payload: DM payload
            correlation_id: Workflow correlation ID
        """
        dm_data = {
            "platform": platform,
            "channel": "dm",
            "platform_object_id": payload.get("message_id"),
            "actor": "user",
            "text": payload.get("text"),
            "author_username": payload.get("author_username"),
            "created_at": payload.get("created_at") or datetime.now(timezone.utc).isoformat()
        }

        # Emit DM received event
        await self.emit(
            "dm.received",
            dm_data,
            correlation_id=correlation_id
        )

        # Emit general inbound event
        await self.emit(
            "inbound.received",
            dm_data,
            correlation_id=correlation_id
        )

        # Request response (with DM permission gate)
        await self.emit(
            "inbound.respond.requested",
            {
                **dm_data,
                "strategy": "dm_flow",
                "requires_permission_gate": True,
                "requested_at": datetime.now(timezone.utc).isoformat()
            },
            correlation_id=correlation_id
        )

        logger.info(
            f"📩 DM received | platform={platform} | "
            f"author={dm_data.get('author_username')}"
        )

    async def _process_mention(
        self,
        platform: str,
        payload: Dict[str, Any],
        correlation_id: str
    ) -> None:
        """
        Process an inbound mention.

        Args:
            platform: Platform name
            payload: Mention payload
            correlation_id: Workflow correlation ID
        """
        mention_data = {
            "platform": platform,
            "channel": "mention",
            "platform_object_id": payload.get("mention_id"),
            "actor": "user",
            "text": payload.get("text"),
            "author_username": payload.get("author_username"),
            "created_at": payload.get("created_at") or datetime.now(timezone.utc).isoformat()
        }

        # Emit inbound event
        await self.emit(
            "inbound.received",
            mention_data,
            correlation_id=correlation_id
        )

        # Request response
        await self.emit(
            "inbound.respond.requested",
            {
                **mention_data,
                "strategy": "public_reply",
                "requested_at": datetime.now(timezone.utc).isoformat()
            },
            correlation_id=correlation_id
        )

        logger.info(
            f"🏷️ Mention received | platform={platform} | "
            f"author={mention_data.get('author_username')}"
        )

    async def _process_email(
        self,
        payload: Dict[str, Any],
        correlation_id: str
    ) -> None:
        """
        Process an inbound email.

        Args:
            payload: Email payload
            correlation_id: Workflow correlation ID
        """
        email_data = {
            "platform": "email",
            "channel": "email",
            "platform_object_id": payload.get("message_id"),
            "actor": "user",
            "text": payload.get("body"),
            "subject": payload.get("subject"),
            "from_address": payload.get("from"),
            "created_at": payload.get("created_at") or datetime.now(timezone.utc).isoformat()
        }

        # Emit inbound event
        await self.emit(
            "inbound.received",
            email_data,
            correlation_id=correlation_id
        )

        # Request response
        await self.emit(
            "inbound.respond.requested",
            {
                **email_data,
                "strategy": "email_reply",
                "requested_at": datetime.now(timezone.utc).isoformat()
            },
            correlation_id=correlation_id
        )

        logger.info(
            f"📧 Email received | from={email_data.get('from_address')}"
        )

    async def _is_duplicate(
        self,
        platform: str,
        external_event_id: str
    ) -> bool:
        """
        Check if an event has already been processed.

        Args:
            platform: Platform name
            external_event_id: External event ID

        Returns:
            True if duplicate, False otherwise
        """
        key = f"{platform}:{external_event_id}"

        # Clean up expired items
        now = datetime.now(timezone.utc)
        expired_keys = [
            k for k, v in self._seen_items.items()
            if (now - v).total_seconds() > (self._seen_item_ttl_hours * 3600)
        ]
        for k in expired_keys:
            del self._seen_items[k]

        # Check for duplicate
        if key in self._seen_items:
            return True

        # Mark as seen
        self._seen_items[key] = now

        # Trim to max size
        if len(self._seen_items) > self._max_seen_items:
            # Remove oldest items
            oldest_keys = sorted(
                self._seen_items.keys(),
                key=lambda k: self._seen_items[k]
            )[:100]
            for k in oldest_keys:
                del self._seen_items[k]

        return False

    def get_listener_stats(self) -> Dict[str, Any]:
        """
        Get listener statistics.

        Returns:
            Statistics about inbound listening
        """
        return {
            "seen_items_count": len(self._seen_items),
            "max_seen_items": self._max_seen_items,
            "ttl_hours": self._seen_item_ttl_hours,
            "supported_platforms": list(self._platform_handlers.keys())
        }
