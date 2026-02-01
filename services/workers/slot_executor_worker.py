"""
Slot Executor Worker
====================
Executes scheduled content slots from the weekly plan.

Flow: slot.execute.requested → generate → QA → publish

The Slot Executor listens for slot execution events and coordinates
the content generation pipeline for each slot in the plan.

Implements OPS-013: Slot Executor Worker
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from loguru import logger

from services.workers.base import BaseWorker
from services.event_bus import Event, Topics
from services.content_generation_pipeline import ContentGenerationPipeline
from services.qa_gate_service import QAGateService


class SlotExecutorWorker(BaseWorker):
    """
    Slot Executor Worker

    Executes scheduled content slots by coordinating the generation,
    QA, and publishing pipeline for each slot in the weekly plan.

    Consumes:
        - slot.execute.requested

    Emits:
        - draft.generate.requested
        - draft.qa.requested
        - draft.publish.requested
        - slot.execution.completed
        - slot.execution.failed

    Usage:
        worker = SlotExecutorWorker(event_bus)
        await worker.start()
    """

    def __init__(self, event_bus=None, worker_id: Optional[str] = None):
        """
        Initialize slot executor worker.

        Args:
            event_bus: EventBus instance (uses singleton if not provided)
            worker_id: Unique worker identifier
        """
        self.generation_pipeline = ContentGenerationPipeline.get_instance()
        self.qa_service = QAGateService.get_instance()

        # Track in-flight slot executions
        self._executing_slots: Dict[str, Dict[str, Any]] = {}

        super().__init__(event_bus, worker_id)
        logger.info("🎯 Slot Executor Worker initialized")

    def get_subscriptions(self) -> List[str]:
        """Subscribe to slot execution requests."""
        return ["slot.execute.requested"]

    async def handle_event(self, event: Event) -> None:
        """
        Handle slot execution request.

        Flow:
        1. Validate slot payload
        2. Request content generation
        3. Wait for generation completion
        4. Run QA gate
        5. Publish if QA passes, else queue for approval

        Args:
            event: slot.execute.requested event
        """
        payload = event.payload
        slot_id = payload.get("slot_id")

        if not slot_id:
            logger.error("Missing slot_id in payload")
            raise ValueError("slot_id is required")

        logger.info(f"🎯 Executing slot: {slot_id}")

        try:
            # Track execution start
            self._executing_slots[slot_id] = {
                "started_at": datetime.now(timezone.utc),
                "status": "generating",
                "correlation_id": event.correlation_id
            }

            # Step 1: Request content generation
            await self._request_generation(slot_id, payload, event.correlation_id)

            # Step 2: Wait for generation to complete (handled via events)
            # The generation pipeline will emit draft.generation.completed
            # which will trigger the next step

            logger.info(f"✅ Slot execution pipeline started: {slot_id}")

        except Exception as e:
            logger.error(f"❌ Slot execution failed for {slot_id}: {e}")

            # Mark as failed
            self._executing_slots[slot_id]["status"] = "failed"
            self._executing_slots[slot_id]["error"] = str(e)

            # Emit failure event
            await self.emit(
                "slot.execution.failed",
                {
                    "slot_id": slot_id,
                    "error": str(e),
                    "failed_at": datetime.now(timezone.utc).isoformat()
                },
                correlation_id=event.correlation_id
            )

            raise

    async def _request_generation(
        self,
        slot_id: str,
        slot_payload: Dict[str, Any],
        correlation_id: str
    ) -> None:
        """
        Request content generation for a slot.

        Args:
            slot_id: Slot identifier
            slot_payload: Complete slot payload from plan
            correlation_id: Workflow correlation ID
        """
        # Extract generation parameters from slot
        generation_request = {
            "slot_id": slot_id,
            "plan_id": slot_payload.get("plan_id"),
            "platform": slot_payload.get("platform"),
            "channel": slot_payload.get("channel", "post"),
            "awareness_level": slot_payload.get("awareness_level"),
            "fate_target": slot_payload.get("fate_target"),
            "cta_strength": slot_payload.get("cta_strength", "soft"),
            "target_offer_id": slot_payload.get("target_offer_id"),
            "target_icp_id": slot_payload.get("target_icp_id"),
            "template_hint_ids": slot_payload.get("template_hint_ids", []),
            "variants": slot_payload.get("variants", 3),
            "requested_at": datetime.now(timezone.utc).isoformat()
        }

        # Emit generation request event
        await self.emit(
            "draft.generate.requested",
            generation_request,
            correlation_id=correlation_id
        )

        logger.info(
            f"📝 Generation requested for slot {slot_id} | "
            f"platform={generation_request['platform']} | "
            f"awareness={generation_request['awareness_level']}"
        )

    async def _handle_generation_completed(
        self,
        slot_id: str,
        draft_ids: List[str],
        correlation_id: str
    ) -> None:
        """
        Handle completed generation - trigger QA gate.

        Args:
            slot_id: Slot identifier
            draft_ids: Generated draft IDs
            correlation_id: Workflow correlation ID
        """
        if slot_id not in self._executing_slots:
            logger.warning(f"Received generation completion for unknown slot: {slot_id}")
            return

        logger.info(f"✅ Generation completed for slot {slot_id} | drafts={len(draft_ids)}")

        # Update tracking
        self._executing_slots[slot_id]["status"] = "qa_checking"
        self._executing_slots[slot_id]["draft_ids"] = draft_ids

        # Request QA for each draft
        for draft_id in draft_ids:
            await self.emit(
                "draft.qa.requested",
                {
                    "slot_id": slot_id,
                    "draft_id": draft_id,
                    "requested_at": datetime.now(timezone.utc).isoformat()
                },
                correlation_id=correlation_id
            )

        logger.info(f"🔍 QA requested for {len(draft_ids)} drafts from slot {slot_id}")

    async def _handle_qa_completed(
        self,
        slot_id: str,
        draft_id: str,
        qa_passed: bool,
        qa_issues: List[str],
        correlation_id: str
    ) -> None:
        """
        Handle completed QA - publish or queue for approval.

        Args:
            slot_id: Slot identifier
            draft_id: Draft identifier
            qa_passed: Whether QA passed
            qa_issues: List of QA issues if failed
            correlation_id: Workflow correlation ID
        """
        if slot_id not in self._executing_slots:
            logger.warning(f"Received QA completion for unknown slot: {slot_id}")
            return

        logger.info(
            f"✅ QA completed for draft {draft_id} | "
            f"passed={qa_passed} | issues={len(qa_issues)}"
        )

        if qa_passed:
            # Publish immediately
            await self.emit(
                "draft.publish.requested",
                {
                    "slot_id": slot_id,
                    "draft_id": draft_id,
                    "auto_published": True,
                    "requested_at": datetime.now(timezone.utc).isoformat()
                },
                correlation_id=correlation_id
            )

            logger.info(f"📤 Publish requested for draft {draft_id}")
        else:
            # Queue for human approval
            await self.emit(
                "approval.required",
                {
                    "slot_id": slot_id,
                    "draft_id": draft_id,
                    "qa_issues": qa_issues,
                    "requested_at": datetime.now(timezone.utc).isoformat()
                },
                correlation_id=correlation_id
            )

            logger.warning(
                f"⚠️ Draft {draft_id} requires approval | issues={qa_issues}"
            )

    async def _handle_publish_completed(
        self,
        slot_id: str,
        draft_id: str,
        touchpoint_id: str,
        correlation_id: str
    ) -> None:
        """
        Handle completed publish - mark slot execution complete.

        Args:
            slot_id: Slot identifier
            draft_id: Draft identifier
            touchpoint_id: Published touchpoint ID
            correlation_id: Workflow correlation ID
        """
        if slot_id not in self._executing_slots:
            logger.warning(f"Received publish completion for unknown slot: {slot_id}")
            return

        logger.info(
            f"✅ Publish completed for draft {draft_id} | touchpoint={touchpoint_id}"
        )

        # Update tracking
        execution = self._executing_slots[slot_id]
        execution["status"] = "completed"
        execution["touchpoint_id"] = touchpoint_id
        execution["completed_at"] = datetime.now(timezone.utc)

        # Calculate execution duration
        duration_seconds = (
            execution["completed_at"] - execution["started_at"]
        ).total_seconds()

        # Emit completion event
        await self.emit(
            "slot.execution.completed",
            {
                "slot_id": slot_id,
                "draft_id": draft_id,
                "touchpoint_id": touchpoint_id,
                "duration_seconds": duration_seconds,
                "completed_at": execution["completed_at"].isoformat()
            },
            correlation_id=correlation_id
        )

        logger.success(
            f"🎉 Slot execution completed: {slot_id} | "
            f"duration={duration_seconds:.1f}s"
        )

        # Clean up tracking (keep for 5 minutes for debugging)
        await asyncio.sleep(300)
        if slot_id in self._executing_slots:
            del self._executing_slots[slot_id]

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics.

        Returns:
            Statistics about slot executions
        """
        return {
            "in_flight_executions": len(self._executing_slots),
            "executions": [
                {
                    "slot_id": slot_id,
                    "status": execution["status"],
                    "started_at": execution["started_at"].isoformat(),
                    "duration_seconds": (
                        datetime.now(timezone.utc) - execution["started_at"]
                    ).total_seconds()
                }
                for slot_id, execution in self._executing_slots.items()
            ]
        }
