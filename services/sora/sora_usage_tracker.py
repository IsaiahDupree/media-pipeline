"""
Sora Usage Tracker
===================
Tracks Sora video generation usage with regular checks and database storage.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from dataclasses import dataclass, asdict
import os

logger = logging.getLogger(__name__)

# Timeouts for Sora automation
SORA_TIMEOUTS = {
    "page_load": 10,          # seconds to wait for page load
    "menu_open": 2,           # seconds to wait for menu to open
    "dialog_open": 3,         # seconds to wait for dialog to open
    "video_generation": 600,  # 10 minutes max for video generation
    "polling_interval": 30,   # seconds between status checks
    "batch_timeout": 900,     # 15 minutes for batch operations
}


@dataclass
class SoraUsage:
    """Sora usage data"""
    video_gens_left: int = 0
    free_count: int = 0
    paid_count: int = 0
    reset_date: str = ""
    checked_at: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SoraUsageTracker:
    """
    Tracks Sora usage with regular checks and database storage.
    
    Usage:
        tracker = SoraUsageTracker()
        usage = await tracker.check_and_store()
        print(f"Videos left: {usage.video_gens_left}")
    """
    
    CHECK_INTERVAL_MINUTES = 30  # How often to check usage
    
    def __init__(self):
        self._last_check: Optional[datetime] = None
        self._cached_usage: Optional[SoraUsage] = None
        self._check_task: Optional[asyncio.Task] = None
        self._running = False
    
    def get_timeouts(self) -> Dict:
        """Return all Sora automation timeouts"""
        return SORA_TIMEOUTS.copy()
    
    async def check_usage(self) -> SoraUsage:
        """Check Sora usage via Safari automation"""
        from automation.sora_full_automation import SoraFullAutomation
        
        try:
            sora = SoraFullAutomation()
            usage_data = sora.get_usage()
            
            usage = SoraUsage(
                video_gens_left=usage_data.get('video_gens_left', 0),
                free_count=usage_data.get('free_count', 0),
                paid_count=usage_data.get('paid_count', 0),
                reset_date=usage_data.get('reset_date', ''),
                checked_at=datetime.utcnow().isoformat()
            )
            
            self._cached_usage = usage
            self._last_check = datetime.utcnow()
            
            logger.info(f"Sora usage: {usage.video_gens_left} gens left, resets {usage.reset_date}")
            return usage
            
        except Exception as e:
            logger.error(f"Failed to check Sora usage: {e}")
            raise
    
    async def store_usage(self, usage: SoraUsage) -> bool:
        """Store usage in database"""
        try:
            import httpx
            
            supabase_url = os.environ.get('SUPABASE_URL', 'http://127.0.0.1:54321')
            supabase_key = os.environ.get('SUPABASE_ANON_KEY', '')
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{supabase_url}/rest/v1/sora_usage",
                    headers={
                        "apikey": supabase_key,
                        "Authorization": f"Bearer {supabase_key}",
                        "Content-Type": "application/json",
                        "Prefer": "return=minimal"
                    },
                    json={
                        "video_gens_left": usage.video_gens_left,
                        "free_count": usage.free_count,
                        "paid_count": usage.paid_count,
                        "reset_date": usage.reset_date,
                        "checked_at": usage.checked_at
                    }
                )
                
                if response.status_code in (200, 201, 204):
                    logger.info("Sora usage stored in database")
                    return True
                else:
                    logger.warning(f"Failed to store usage: {response.status_code} {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Database storage failed: {e}")
            return False
    
    async def check_and_store(self) -> SoraUsage:
        """Check usage and store in database"""
        usage = await self.check_usage()
        await self.store_usage(usage)
        return usage
    
    def get_cached_usage(self) -> Optional[SoraUsage]:
        """Get cached usage without checking"""
        return self._cached_usage
    
    def should_check(self) -> bool:
        """Check if enough time has passed since last check"""
        if self._last_check is None:
            return True
        elapsed = datetime.utcnow() - self._last_check
        return elapsed > timedelta(minutes=self.CHECK_INTERVAL_MINUTES)
    
    async def start_periodic_checks(self, interval_minutes: int = 30):
        """Start periodic usage checks in background"""
        self._running = True
        self.CHECK_INTERVAL_MINUTES = interval_minutes
        
        async def check_loop():
            while self._running:
                try:
                    if self.should_check():
                        await self.check_and_store()
                except Exception as e:
                    logger.error(f"Periodic check failed: {e}")
                
                await asyncio.sleep(interval_minutes * 60)
        
        self._check_task = asyncio.create_task(check_loop())
        logger.info(f"Started Sora usage periodic checks every {interval_minutes} minutes")
    
    def stop_periodic_checks(self):
        """Stop periodic usage checks"""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            self._check_task = None
        logger.info("Stopped Sora usage periodic checks")


# Singleton instance
_tracker: Optional[SoraUsageTracker] = None

def get_sora_usage_tracker() -> SoraUsageTracker:
    """Get singleton SoraUsageTracker instance"""
    global _tracker
    if _tracker is None:
        _tracker = SoraUsageTracker()
    return _tracker


class SoraVideoPoller:
    """
    Polls Sora /drafts page for completed videos and auto-downloads them.
    
    Typical video generation takes 8-12 minutes.
    """
    
    POLL_INTERVAL_SECONDS = 60  # Check every minute
    MAX_POLL_DURATION = 900     # 15 minutes max polling
    
    def __init__(self):
        self._polling = False
        self._poll_task: Optional[asyncio.Task] = None
        self._downloaded_ids: set = set()
        self._callbacks: list = []
    
    def add_completion_callback(self, callback):
        """Add callback to be called when video completes: callback(video_path, video_info)"""
        self._callbacks.append(callback)
    
    async def poll_and_download(self, timeout_minutes: int = 15) -> list:
        """
        Poll /drafts for new completed videos and download them.
        
        Args:
            timeout_minutes: Max time to poll (default 15 min for 8-12 min generation)
            
        Returns:
            List of downloaded video paths
        """
        from automation.sora_full_automation import SoraFullAutomation
        
        sora = SoraFullAutomation()
        downloaded = []
        start_time = datetime.utcnow()
        timeout = timedelta(minutes=timeout_minutes)
        
        # Get initial set of video IDs
        initial_videos = sora.get_completed_videos()
        known_ids = {v.get('id', '') for v in initial_videos}
        
        logger.info(f"Starting video poll (timeout: {timeout_minutes} min, known: {len(known_ids)} videos)")
        
        while datetime.utcnow() - start_time < timeout:
            try:
                # Navigate to drafts
                sora.navigate_to_drafts()
                await asyncio.sleep(2)
                
                # Get current videos
                current_videos = sora.get_completed_videos()
                
                # Find new completed videos
                for video in current_videos:
                    vid = video.get('id', '')
                    if vid and vid not in known_ids and vid not in self._downloaded_ids:
                        logger.info(f"New video completed: {vid}")
                        
                        # Download it
                        path = sora.download_video(video_id=vid)
                        if path:
                            downloaded.append(path)
                            self._downloaded_ids.add(vid)
                            known_ids.add(vid)
                            
                            # Call callbacks
                            for cb in self._callbacks:
                                try:
                                    cb(path, video)
                                except Exception as e:
                                    logger.error(f"Callback error: {e}")
                
                # Check queue - if empty and we have new downloads, we're done
                queue_count = sora.get_queue_count()
                elapsed = (datetime.utcnow() - start_time).seconds
                logger.info(f"Poll: {queue_count} generating, {len(downloaded)} downloaded, {elapsed}s elapsed")
                
                if queue_count == 0 and downloaded:
                    logger.info(f"✅ All videos complete! Downloaded {len(downloaded)}")
                    break
                    
            except Exception as e:
                logger.error(f"Poll error: {e}")
            
            await asyncio.sleep(self.POLL_INTERVAL_SECONDS)
        
        return downloaded
    
    async def start_background_polling(self, timeout_minutes: int = 15):
        """Start polling in background"""
        if self._polling:
            logger.warning("Already polling")
            return
        
        self._polling = True
        
        async def poll_loop():
            try:
                downloaded = await self.poll_and_download(timeout_minutes)
                logger.info(f"Background poll complete: {len(downloaded)} videos downloaded")
            finally:
                self._polling = False
        
        self._poll_task = asyncio.create_task(poll_loop())
        logger.info(f"Started background video polling (timeout: {timeout_minutes} min)")
    
    def stop_polling(self):
        """Stop background polling"""
        self._polling = False
        if self._poll_task:
            self._poll_task.cancel()
            self._poll_task = None
        logger.info("Stopped video polling")
    
    def is_polling(self) -> bool:
        return self._polling


# Singleton poller
_poller: Optional[SoraVideoPoller] = None

def get_sora_video_poller() -> SoraVideoPoller:
    """Get singleton SoraVideoPoller instance"""
    global _poller
    if _poller is None:
        _poller = SoraVideoPoller()
    return _poller
