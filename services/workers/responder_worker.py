"""
Responder Worker
================
Generates and sends responses to inbound items (comments, DMs, mentions).

The Responder Worker handles all inbound engagement responses,
enforcing DM permission gates and QA checks before sending.

Implements OPS-016: Responder Worker
"""

from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from loguru import logger

from services.workers.base import BaseWorker
from services.event_bus import Event, Topics
from services.dm_permission_service import DMPermissionService
from services.qa_gate_service import QAGateService
from services.ai_client import AIClient


class ResponderWorker(BaseWorker):
    """
    Responder Worker

    Generates and sends responses to inbound engagement:
    - Public replies to comments
    - DM responses (with permission gate)
    - Email replies
    - Mention responses

    Enforces DM permission gate to prevent unsolicited link sending.

    Consumes:
        - inbound.respond.requested

    Emits:
        - response.generate.started
        - response.generate.completed
        - response.qa.passed
        - response.qa.failed
        - response.sent
        - response.failed
        - dm.consent.requested

    Usage:
        worker = ResponderWorker(event_bus)
        await worker.start()
    """

    def __init__(self, event_bus=None, worker_id: Optional[str] = None):
        """
        Initialize responder worker.

        Args:
            event_bus: EventBus instance (uses singleton if not provided)
            worker_id: Unique worker identifier
        """
        self.dm_permission = DMPermissionService.get_instance()
        self.qa_service = QAGateService.get_instance()
        self.ai_client = AIClient()

        # Response configuration
        self.max_response_length = {
            "comment": 280,   # Short replies
            "dm": 1000,       # Longer DMs
            "email": 2000,    # Detailed emails
            "mention": 280    # Short mentions
        }

        # Track pending responses
        self._pending_responses: Dict[str, Dict[str, Any]] = {}

        super().__init__(event_bus, worker_id)
        logger.info("💬 Responder Worker initialized")

    def get_subscriptions(self) -> List[str]:
        """Subscribe to response requests."""
        return ["inbound.respond.requested"]

    async def handle_event(self, event: Event) -> None:
        """
        Handle response request.

        Flow:
        1. Validate inbound item
        2. Check DM permission if needed
        3. Generate response using AI
        4. Run QA gate
        5. Send response if QA passes

        Args:
            event: inbound.respond.requested event
        """
        payload = event.payload
        platform = payload.get("platform")
        channel = payload.get("channel")
        platform_object_id = payload.get("platform_object_id")
        strategy = payload.get("strategy")

        if not all([platform, channel, platform_object_id, strategy]):
            logger.error("Missing required fields in response request")
            raise ValueError("Missing required fields")

        logger.info(
            f"💬 Response requested | platform={platform} | "
            f"channel={channel} | strategy={strategy}"
        )

        try:
            # Generate response ID
            response_id = f"resp_{platform_object_id}_{datetime.now(timezone.utc).timestamp()}"

            # Track response
            self._pending_responses[response_id] = {
                "platform": platform,
                "channel": channel,
                "strategy": strategy,
                "inbound_data": payload,
                "started_at": datetime.now(timezone.utc),
                "status": "generating"
            }

            # Check DM permission if needed
            if strategy == "dm_flow" and payload.get("requires_permission_gate"):
                author_username = payload.get("author_username")
                if not await self._check_dm_permission(platform, author_username, response_id, event.correlation_id):
                    logger.info(f"⛔ DM permission denied for {author_username}")
                    return

            # Generate response
            response_text = await self._generate_response(payload, strategy, event.correlation_id)

            # Update tracking
            self._pending_responses[response_id]["status"] = "qa_checking"
            self._pending_responses[response_id]["response_text"] = response_text

            # Run QA gate
            qa_passed, qa_issues = await self._run_qa_gate(
                response_text,
                payload,
                strategy,
                event.correlation_id
            )

            if qa_passed:
                # Send response
                await self._send_response(
                    response_id,
                    platform,
                    channel,
                    platform_object_id,
                    response_text,
                    payload,
                    event.correlation_id
                )

                # Update tracking
                self._pending_responses[response_id]["status"] = "sent"
                self._pending_responses[response_id]["sent_at"] = datetime.now(timezone.utc)

                logger.success(
                    f"✅ Response sent | platform={platform} | "
                    f"channel={channel} | response_id={response_id}"
                )
            else:
                # QA failed - queue for human approval
                logger.warning(
                    f"⚠️ Response QA failed | issues={qa_issues}"
                )

                # Update tracking
                self._pending_responses[response_id]["status"] = "qa_failed"
                self._pending_responses[response_id]["qa_issues"] = qa_issues

                # Emit approval required event
                await self.emit(
                    "approval.required",
                    {
                        "response_id": response_id,
                        "response_text": response_text,
                        "qa_issues": qa_issues,
                        "inbound_data": payload,
                        "requested_at": datetime.now(timezone.utc).isoformat()
                    },
                    correlation_id=event.correlation_id
                )

        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")

            # Update tracking
            if response_id in self._pending_responses:
                self._pending_responses[response_id]["status"] = "failed"
                self._pending_responses[response_id]["error"] = str(e)

            # Emit failure event
            await self.emit(
                "response.failed",
                {
                    "response_id": response_id,
                    "error": str(e),
                    "failed_at": datetime.now(timezone.utc).isoformat()
                },
                correlation_id=event.correlation_id
            )

            raise

    async def _check_dm_permission(
        self,
        platform: str,
        username: str,
        response_id: str,
        correlation_id: str
    ) -> bool:
        """
        Check if DM permission is granted for a user.

        If no consent exists, request it first.

        Args:
            platform: Platform name
            username: User's username
            response_id: Response identifier
            correlation_id: Workflow correlation ID

        Returns:
            True if permission granted, False otherwise
        """
        # Check consent status
        consent = await self.dm_permission.check_consent(platform, username)

        if consent == "granted":
            logger.debug(f"✅ DM permission granted for {username}")
            return True
        elif consent == "denied":
            logger.info(f"⛔ DM permission denied for {username}")
            return False
        elif consent == "stopped":
            logger.info(f"🛑 User stopped all messages: {username}")
            return False
        else:
            # No consent exists - request it
            logger.info(f"📝 Requesting DM consent from {username}")

            await self.emit(
                Topics.DM_CONSENT_REQUESTED,
                {
                    "platform": platform,
                    "username": username,
                    "response_id": response_id,
                    "requested_at": datetime.now(timezone.utc).isoformat()
                },
                correlation_id=correlation_id
            )

            # For now, don't send until consent is granted
            return False

    async def _generate_response(
        self,
        inbound_data: Dict[str, Any],
        strategy: str,
        correlation_id: str
    ) -> str:
        """
        Generate AI response to inbound item.

        Args:
            inbound_data: Inbound item data
            strategy: Response strategy (public_reply, dm_flow, email_reply)
            correlation_id: Workflow correlation ID

        Returns:
            Generated response text
        """
        platform = inbound_data.get("platform")
        channel = inbound_data.get("channel")
        inbound_text = inbound_data.get("text")
        author_username = inbound_data.get("author_username")

        logger.info(f"🤖 Generating response | strategy={strategy}")

        # Emit generation started event
        await self.emit(
            "response.generate.started",
            {
                "platform": platform,
                "channel": channel,
                "strategy": strategy,
                "started_at": datetime.now(timezone.utc).isoformat()
            },
            correlation_id=correlation_id
        )

        # Build AI prompt based on strategy
        if strategy == "public_reply":
            prompt = self._build_public_reply_prompt(inbound_text, author_username)
        elif strategy == "dm_flow":
            prompt = self._build_dm_flow_prompt(inbound_text, author_username)
        elif strategy == "email_reply":
            prompt = self._build_email_reply_prompt(inbound_data)
        else:
            prompt = f"Generate a helpful response to: {inbound_text}"

        # Call AI
        response_text = await self.ai_client.generate_text(
            prompt=prompt,
            max_tokens=self.max_response_length.get(channel, 500),
            temperature=0.7
        )

        # Emit generation completed event
        await self.emit(
            "response.generate.completed",
            {
                "platform": platform,
                "channel": channel,
                "response_length": len(response_text),
                "completed_at": datetime.now(timezone.utc).isoformat()
            },
            correlation_id=correlation_id
        )

        return response_text

    def _build_public_reply_prompt(
        self,
        inbound_text: str,
        author_username: str
    ) -> str:
        """
        Build prompt for public reply.

        Args:
            inbound_text: Comment/mention text
            author_username: Author's username

        Returns:
            AI prompt
        """
        return f"""Generate a friendly, helpful public reply to this comment from @{author_username}:

"{inbound_text}"

Requirements:
- Keep it under 280 characters
- Be professional and helpful
- Don't include links (save for DMs)
- Match the tone of the comment
- Add value to the conversation"""

    def _build_dm_flow_prompt(
        self,
        inbound_text: str,
        author_username: str
    ) -> str:
        """
        Build prompt for DM response.

        Args:
            inbound_text: DM text
            author_username: Author's username

        Returns:
            AI prompt
        """
        return f"""Generate a helpful DM response to @{author_username}:

"{inbound_text}"

Requirements:
- Be personal and helpful
- Keep under 1000 characters
- Can include relevant links if appropriate
- Maintain professional tone
- Provide value and next steps"""

    def _build_email_reply_prompt(
        self,
        inbound_data: Dict[str, Any]
    ) -> str:
        """
        Build prompt for email reply.

        Args:
            inbound_data: Email data

        Returns:
            AI prompt
        """
        subject = inbound_data.get("subject", "")
        body = inbound_data.get("text", "")
        from_address = inbound_data.get("from_address", "")

        return f"""Generate a professional email reply to:

From: {from_address}
Subject: {subject}

Email:
{body}

Requirements:
- Professional email format
- Keep under 2000 characters
- Address all points raised
- Include clear call-to-action
- Maintain helpful, friendly tone"""

    async def _run_qa_gate(
        self,
        response_text: str,
        inbound_data: Dict[str, Any],
        strategy: str,
        correlation_id: str
    ) -> tuple[bool, List[str]]:
        """
        Run QA gate on generated response.

        Args:
            response_text: Generated response
            inbound_data: Original inbound data
            strategy: Response strategy
            correlation_id: Workflow correlation ID

        Returns:
            Tuple of (passed, issues)
        """
        logger.debug("🔍 Running QA gate on response")

        issues = []

        # Check length limits
        max_length = self.max_response_length.get(inbound_data.get("channel"), 500)
        if len(response_text) > max_length:
            issues.append(f"Response too long: {len(response_text)} > {max_length}")

        # Check for prohibited content (basic checks)
        prohibited = ["fuck", "shit", "damn", "hate", "stupid"]
        if any(word in response_text.lower() for word in prohibited):
            issues.append("Response contains prohibited language")

        # Check for DM permission gate enforcement
        if strategy == "dm_flow":
            # Verify no unsolicited links in first message
            if "http" in response_text.lower() and not await self._has_dm_permission(inbound_data):
                issues.append("Cannot send links without DM permission")

        # Run full QA gate check
        qa_result = await self.qa_service.check_content(response_text)
        if not qa_result.get("passed", False):
            issues.extend(qa_result.get("issues", []))

        passed = len(issues) == 0

        # Emit QA event
        if passed:
            await self.emit(
                "response.qa.passed",
                {
                    "response_length": len(response_text),
                    "checked_at": datetime.now(timezone.utc).isoformat()
                },
                correlation_id=correlation_id
            )
        else:
            await self.emit(
                "response.qa.failed",
                {
                    "response_length": len(response_text),
                    "issues": issues,
                    "checked_at": datetime.now(timezone.utc).isoformat()
                },
                correlation_id=correlation_id
            )

        return passed, issues

    async def _has_dm_permission(
        self,
        inbound_data: Dict[str, Any]
    ) -> bool:
        """
        Check if we have DM permission for this user.

        Args:
            inbound_data: Inbound item data

        Returns:
            True if permission granted
        """
        platform = inbound_data.get("platform")
        username = inbound_data.get("author_username")

        consent = await self.dm_permission.check_consent(platform, username)
        return consent == "granted"

    async def _send_response(
        self,
        response_id: str,
        platform: str,
        channel: str,
        platform_object_id: str,
        response_text: str,
        inbound_data: Dict[str, Any],
        correlation_id: str
    ) -> None:
        """
        Send response to platform.

        Args:
            response_id: Response identifier
            platform: Platform name
            channel: Channel type
            platform_object_id: Platform object ID to reply to
            response_text: Response text
            inbound_data: Original inbound data
            correlation_id: Workflow correlation ID
        """
        logger.info(f"📤 Sending response to {platform}/{channel}")

        # In a real implementation, this would call platform APIs
        # For now, emit an event for the platform adapter to handle

        await self.emit(
            "response.sent",
            {
                "response_id": response_id,
                "platform": platform,
                "channel": channel,
                "platform_object_id": platform_object_id,
                "response_text": response_text,
                "inbound_data": inbound_data,
                "sent_at": datetime.now(timezone.utc).isoformat()
            },
            correlation_id=correlation_id
        )

    def get_responder_stats(self) -> Dict[str, Any]:
        """
        Get responder statistics.

        Returns:
            Statistics about response generation
        """
        total_responses = len(self._pending_responses)
        status_counts = {}

        for response in self._pending_responses.values():
            status = response.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "total_responses": total_responses,
            "status_counts": status_counts,
            "max_response_lengths": self.max_response_length
        }
