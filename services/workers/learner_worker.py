"""
Learner Worker
==============
Updates template leaderboard, forks winning templates, demotes losers.

The Learner Worker analyzes template performance and updates the leaderboard,
automatically promoting high-performing templates and demoting low performers.

Implements OPS-014: Learner Worker
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from loguru import logger

from services.workers.base import BaseWorker
from services.event_bus import Event, Topics
from services.template_leaderboard import TemplateLeaderboard


class LearnerWorker(BaseWorker):
    """
    Learner Worker

    Analyzes template performance and updates the leaderboard:
    - Winners get 70% allocation
    - Losers get < 5% allocation
    - Automatically forks high-performing templates
    - Demotes low-performing templates

    Consumes:
        - learn.update.requested
        - metrics.snapshot.completed (for learning triggers)

    Emits:
        - template.leaderboard.updated
        - template.forked
        - template.demoted
        - learn.update.completed

    Usage:
        worker = LearnerWorker(event_bus)
        await worker.start()
    """

    def __init__(self, event_bus=None, worker_id: Optional[str] = None):
        """
        Initialize learner worker.

        Args:
            event_bus: EventBus instance (uses singleton if not provided)
            worker_id: Unique worker identifier
        """
        self.leaderboard = TemplateLeaderboard.get_instance()

        # Learning configuration
        self.winner_threshold = 0.70  # Top templates get 70% allocation
        self.loser_threshold = 0.05   # Bottom templates get < 5% allocation
        self.fork_threshold = 0.80    # Fork templates above 80% win rate
        self.min_samples = 10         # Minimum samples before learning

        # Track learning runs
        self._learning_runs: List[Dict[str, Any]] = []
        self._max_learning_history = 100

        super().__init__(event_bus, worker_id)
        logger.info("🧠 Learner Worker initialized")

    def get_subscriptions(self) -> List[str]:
        """Subscribe to learning requests and metrics updates."""
        return [
            "learn.update.requested",
            "metrics.snapshot.completed"
        ]

    async def handle_event(self, event: Event) -> None:
        """
        Handle learning events.

        Args:
            event: learn.update.requested or metrics.snapshot.completed
        """
        if event.topic == "learn.update.requested":
            await self._handle_learning_request(event)
        elif event.topic == "metrics.snapshot.completed":
            await self._handle_metrics_update(event)

    async def _handle_learning_request(self, event: Event) -> None:
        """
        Handle explicit learning request.

        Triggered by:
        - Daily cron job
        - Manual trigger via API
        - Significant metric changes

        Args:
            event: learn.update.requested event
        """
        payload = event.payload
        date = payload.get("date") or datetime.now(timezone.utc).date().isoformat()

        logger.info(f"🧠 Learning update requested for date: {date}")

        try:
            # Run learning cycle
            results = await self._run_learning_cycle(date, event.correlation_id)

            # Track learning run
            self._learning_runs.append({
                "date": date,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "templates_updated": results["templates_updated"],
                "templates_forked": results["templates_forked"],
                "templates_demoted": results["templates_demoted"],
                "correlation_id": event.correlation_id
            })

            # Trim history
            if len(self._learning_runs) > self._max_learning_history:
                self._learning_runs = self._learning_runs[-self._max_learning_history:]

            # Emit completion
            await self.emit(
                "learn.update.completed",
                {
                    "date": date,
                    "results": results,
                    "completed_at": datetime.now(timezone.utc).isoformat()
                },
                correlation_id=event.correlation_id
            )

            logger.success(
                f"✅ Learning cycle completed | "
                f"updated={results['templates_updated']} | "
                f"forked={results['templates_forked']} | "
                f"demoted={results['templates_demoted']}"
            )

        except Exception as e:
            logger.error(f"❌ Learning cycle failed: {e}")
            raise

    async def _handle_metrics_update(self, event: Event) -> None:
        """
        Handle metrics snapshot completion.

        Check if significant changes warrant a learning update.

        Args:
            event: metrics.snapshot.completed event
        """
        payload = event.payload
        touchpoint_id = payload.get("touchpoint_id")
        template_id = payload.get("template_id")

        if not template_id:
            # No template association, skip
            return

        logger.debug(f"📊 Metrics updated for template: {template_id}")

        # Check if we should trigger learning
        # (Could implement adaptive learning triggers here)
        # For now, rely on scheduled learning runs

    async def _run_learning_cycle(
        self,
        date: str,
        correlation_id: str
    ) -> Dict[str, Any]:
        """
        Run a complete learning cycle.

        1. Get template performance stats
        2. Update leaderboard rankings
        3. Fork high performers
        4. Demote low performers
        5. Adjust allocations

        Args:
            date: Date to run learning for (YYYY-MM-DD)
            correlation_id: Workflow correlation ID

        Returns:
            Learning cycle results
        """
        results = {
            "templates_updated": 0,
            "templates_forked": 0,
            "templates_demoted": 0,
            "allocation_changes": []
        }

        # Get all templates with performance data
        templates = await self.leaderboard.get_all_templates()

        for template in templates:
            template_id = template["template_id"]
            stats = template.get("stats", {})
            total_uses = stats.get("total_uses", 0)

            # Skip templates without enough samples
            if total_uses < self.min_samples:
                continue

            win_rate = stats.get("win_rate", 0.0)
            avg_engagement = stats.get("avg_engagement_rate", 0.0)

            # Check for high performers (fork candidates)
            if win_rate >= self.fork_threshold and total_uses >= 20:
                await self._fork_template(template_id, stats, correlation_id)
                results["templates_forked"] += 1

            # Update allocation based on performance
            old_allocation = template.get("allocation", 0.0)
            new_allocation = self._calculate_allocation(win_rate, avg_engagement)

            if abs(new_allocation - old_allocation) > 0.01:  # Significant change
                await self.leaderboard.update_allocation(template_id, new_allocation)
                results["allocation_changes"].append({
                    "template_id": template_id,
                    "old_allocation": old_allocation,
                    "new_allocation": new_allocation
                })

            # Check for low performers (demotion candidates)
            if new_allocation < self.loser_threshold:
                await self._demote_template(template_id, stats, correlation_id)
                results["templates_demoted"] += 1

            results["templates_updated"] += 1

        # Normalize allocations to sum to 1.0
        await self.leaderboard.normalize_allocations()

        return results

    def _calculate_allocation(
        self,
        win_rate: float,
        avg_engagement: float
    ) -> float:
        """
        Calculate new allocation for a template.

        Winners (high win rate) get up to 70% allocation.
        Losers (low win rate) get down to 5% allocation.

        Args:
            win_rate: Template win rate (0-1)
            avg_engagement: Average engagement rate

        Returns:
            New allocation (0-1)
        """
        # Combine win rate and engagement for score
        score = (win_rate * 0.7) + (avg_engagement * 0.3)

        # Map score to allocation range [0.05, 0.70]
        min_allocation = self.loser_threshold
        max_allocation = self.winner_threshold
        allocation = min_allocation + (score * (max_allocation - min_allocation))

        return round(allocation, 3)

    async def _fork_template(
        self,
        template_id: str,
        stats: Dict[str, Any],
        correlation_id: str
    ) -> None:
        """
        Fork a high-performing template for experimentation.

        Creates a new template variant based on the winner,
        allowing for further optimization while preserving the original.

        Args:
            template_id: Template to fork
            stats: Template performance stats
            correlation_id: Workflow correlation ID
        """
        logger.info(
            f"🍴 Forking high-performing template: {template_id} | "
            f"win_rate={stats.get('win_rate', 0):.2%}"
        )

        # Emit fork event (actual forking handled by template service)
        await self.emit(
            "template.forked",
            {
                "source_template_id": template_id,
                "reason": "high_performance",
                "stats": stats,
                "forked_at": datetime.now(timezone.utc).isoformat()
            },
            correlation_id=correlation_id
        )

    async def _demote_template(
        self,
        template_id: str,
        stats: Dict[str, Any],
        correlation_id: str
    ) -> None:
        """
        Demote a low-performing template.

        Reduces allocation to minimum and marks for review.

        Args:
            template_id: Template to demote
            stats: Template performance stats
            correlation_id: Workflow correlation ID
        """
        logger.warning(
            f"📉 Demoting low-performing template: {template_id} | "
            f"win_rate={stats.get('win_rate', 0):.2%}"
        )

        # Emit demotion event
        await self.emit(
            "template.demoted",
            {
                "template_id": template_id,
                "reason": "low_performance",
                "stats": stats,
                "demoted_at": datetime.now(timezone.utc).isoformat()
            },
            correlation_id=correlation_id
        )

    def get_learning_stats(self) -> Dict[str, Any]:
        """
        Get learning statistics.

        Returns:
            Statistics about learning runs
        """
        return {
            "total_learning_runs": len(self._learning_runs),
            "recent_runs": self._learning_runs[-10:],  # Last 10 runs
            "config": {
                "winner_threshold": self.winner_threshold,
                "loser_threshold": self.loser_threshold,
                "fork_threshold": self.fork_threshold,
                "min_samples": self.min_samples
            }
        }
