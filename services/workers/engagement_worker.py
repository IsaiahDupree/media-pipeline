"""
Engagement Worker

Processes engagement requests from the event bus.
Orchestrates the auto-engagement flow for each platform.

Subscribes to:
- engagement.requested

Emits:
- engagement.started
- engagement.post_found
- engagement.comment_generated
- engagement.comment_posted
- engagement.comment_skipped
- engagement.completed
- engagement.failed
- engagement.daily_limit_reached
"""

import os
import sys
import asyncio
import logging
import random
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from services.event_bus import Topics, Event
from services.workers.base import BaseWorker
from services.engagement.comment_tracker import CommentTracker, get_comment_tracker

logger = logging.getLogger(__name__)

# Delay between comments (seconds)
MIN_DELAY = 30
MAX_DELAY = 120


class EngagementWorker(BaseWorker):
    """
    Worker that processes engagement requests.
    
    For each request:
    1. Check daily limit
    2. Get platform engagement module
    3. Find posts, generate comments, post them
    4. Track comments to prevent duplicates
    5. Emit progress events
    """
    
    worker_type = "engagement"
    
    def __init__(self, event_bus=None, worker_id: Optional[str] = None):
        super().__init__(event_bus, worker_id)
        self._tracker: Optional[CommentTracker] = None
        self._platform_modules: Dict[str, Any] = {}
    
    @property
    def tracker(self) -> CommentTracker:
        """Get CommentTracker instance."""
        if self._tracker is None:
            self._tracker = get_comment_tracker()
        return self._tracker
    
    def get_subscriptions(self) -> List[str]:
        """Subscribe to engagement requests."""
        return [Topics.ENGAGEMENT_REQUESTED]
    
    async def handle_event(self, event: Event) -> None:
        """Process engagement request."""
        platform = event.payload.get('platform')
        count = event.payload.get('count', 1)
        correlation_id = event.correlation_id
        
        logger.info(f"[{self.worker_id}] Processing engagement request: {platform} x{count}")
        
        try:
            # Emit started
            await self.emit(
                Topics.ENGAGEMENT_STARTED,
                {
                    'platform': platform,
                    'count': count,
                    'worker_id': self.worker_id
                },
                correlation_id
            )
            
            # Check if enabled
            if not await self.tracker.is_enabled(platform):
                await self.emit(
                    Topics.ENGAGEMENT_PAUSED,
                    {'platform': platform, 'reason': 'Platform is paused'},
                    correlation_id
                )
                return
            
            # Check daily limit
            if await self.tracker.is_limit_reached(platform):
                remaining = await self.tracker.get_remaining(platform)
                await self.emit(
                    Topics.ENGAGEMENT_DAILY_LIMIT_REACHED,
                    {
                        'platform': platform,
                        'limit': await self.tracker.get_daily_limit(platform),
                        'remaining': remaining
                    },
                    correlation_id
                )
                return
            
            # Run engagement
            result = await self._run_engagement(platform, count, correlation_id)
            
            # Emit completed
            await self.emit(
                Topics.ENGAGEMENT_COMPLETED,
                {
                    'platform': platform,
                    'comments_posted': result['comments_posted'],
                    'comments_skipped': result['comments_skipped'],
                    'errors': result.get('errors', [])
                },
                correlation_id
            )
            
        except Exception as e:
            logger.error(f"[{self.worker_id}] Engagement failed: {e}")
            await self.emit(
                Topics.ENGAGEMENT_FAILED,
                {
                    'platform': platform,
                    'error': str(e),
                    'step': 'engagement_worker'
                },
                correlation_id
            )
            raise
    
    async def _run_engagement(
        self,
        platform: str,
        count: int,
        correlation_id: str
    ) -> Dict[str, Any]:
        """
        Run engagement for a platform.
        
        Args:
            platform: Platform name
            count: Number of comments to post
            correlation_id: Event correlation ID
            
        Returns:
            Result dict with counts
        """
        comments_posted = 0
        comments_skipped = 0
        errors = []
        
        # Get platform module
        engagement_module = await self._get_platform_module(platform)
        if not engagement_module:
            raise ValueError(f"No engagement module for {platform}")
        
        for i in range(count):
            try:
                # Check limit again (may have changed during processing)
                if await self.tracker.is_limit_reached(platform):
                    logger.info(f"Daily limit reached after {comments_posted} comments")
                    await self.emit(
                        Topics.ENGAGEMENT_DAILY_LIMIT_REACHED,
                        {'platform': platform, 'after_count': comments_posted},
                        correlation_id
                    )
                    break
                
                # Run single engagement
                result = await self._engage_single(
                    engagement_module, platform, correlation_id
                )
                
                if result['posted']:
                    comments_posted += 1
                else:
                    comments_skipped += 1
                
                # Delay between comments (except last one)
                if i < count - 1:
                    delay = random.randint(MIN_DELAY, MAX_DELAY)
                    logger.info(f"Waiting {delay}s before next engagement...")
                    await asyncio.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Error in engagement {i+1}: {e}")
                errors.append(str(e))
                comments_skipped += 1
        
        return {
            'comments_posted': comments_posted,
            'comments_skipped': comments_skipped,
            'errors': errors
        }
    
    async def _engage_single(
        self,
        module: Any,
        platform: str,
        correlation_id: str
    ) -> Dict[str, Any]:
        """
        Perform a single engagement (find post, comment).
        
        Args:
            module: Platform engagement module
            platform: Platform name
            correlation_id: Event correlation ID
            
        Returns:
            Dict with 'posted' bool and details
        """
        # Run the engagement
        result = module.engage_with_post()
        
        if not result.success:
            await self.emit(
                Topics.ENGAGEMENT_FAILED,
                {
                    'platform': platform,
                    'error': result.error or 'Unknown error',
                    'step': 'engage_with_post'
                },
                correlation_id
            )
            return {'posted': False, 'error': result.error}
        
        post_url = result.post_url or ''
        username = result.username or ''
        
        # Emit post found
        await self.emit(
            Topics.ENGAGEMENT_POST_FOUND,
            {
                'platform': platform,
                'post_url': post_url,
                'username': username
            },
            correlation_id
        )
        
        # Check for duplicate
        if post_url and await self.tracker.has_commented_on(platform, post_url):
            await self.emit(
                Topics.ENGAGEMENT_COMMENT_SKIPPED,
                {
                    'platform': platform,
                    'post_url': post_url,
                    'reason': 'duplicate'
                },
                correlation_id
            )
            return {'posted': False, 'reason': 'duplicate'}
        
        # Emit comment generated
        if result.generated_comment:
            await self.emit(
                Topics.ENGAGEMENT_COMMENT_GENERATED,
                {
                    'platform': platform,
                    'post_url': post_url,
                    'comment_text': result.generated_comment
                },
                correlation_id
            )
        
        # Check if actually posted
        if result.comment_posted:
            # Record in tracker
            try:
                await self.tracker.record_comment(
                    platform=platform,
                    post_url=post_url,
                    comment_text=result.generated_comment or '',
                    post_username=username,
                    proof_screenshot=result.proof_screenshot or ''
                )
            except ValueError as e:
                # Duplicate detected at insert time
                logger.warning(f"Duplicate detected: {e}")
                await self.emit(
                    Topics.ENGAGEMENT_COMMENT_SKIPPED,
                    {
                        'platform': platform,
                        'post_url': post_url,
                        'reason': 'duplicate_at_insert'
                    },
                    correlation_id
                )
                return {'posted': False, 'reason': 'duplicate_at_insert'}
            
            # Emit posted
            await self.emit(
                Topics.ENGAGEMENT_COMMENT_POSTED,
                {
                    'platform': platform,
                    'post_url': post_url,
                    'username': username,
                    'comment_text': result.generated_comment,
                    'proof_screenshot': result.proof_screenshot
                },
                correlation_id
            )
            
            return {'posted': True, 'post_url': post_url}
        else:
            await self.emit(
                Topics.ENGAGEMENT_COMMENT_SKIPPED,
                {
                    'platform': platform,
                    'post_url': post_url,
                    'reason': 'post_failed'
                },
                correlation_id
            )
            return {'posted': False, 'reason': 'post_failed'}
    
    async def _get_platform_module(self, platform: str) -> Any:
        """
        Get or create platform engagement module.
        
        Args:
            platform: Platform name
            
        Returns:
            Engagement module instance
        """
        if platform in self._platform_modules:
            return self._platform_modules[platform]
        
        # Import the appropriate module
        openai_key = os.environ.get('OPENAI_API_KEY')
        
        if platform == 'threads':
            from scripts.auto_engagement.threads_engagement import ThreadsEngagement
            module = ThreadsEngagement(openai_api_key=openai_key)
        elif platform == 'instagram':
            from scripts.auto_engagement.instagram_engagement import InstagramEngagement
            module = InstagramEngagement(openai_api_key=openai_key)
        elif platform == 'tiktok':
            from scripts.auto_engagement.tiktok_engagement import TikTokEngagement
            module = TikTokEngagement(openai_api_key=openai_key)
        elif platform == 'twitter':
            from scripts.auto_engagement.twitter_engagement import TwitterEngagement
            module = TwitterEngagement(openai_api_key=openai_key)
        else:
            return None
        
        self._platform_modules[platform] = module
        return module


# Factory function for worker registry
def create_engagement_worker(event_bus=None) -> EngagementWorker:
    """Create an EngagementWorker instance."""
    return EngagementWorker(event_bus)
