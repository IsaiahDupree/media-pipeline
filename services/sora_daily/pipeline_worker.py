"""
Sora Daily Pipeline Worker
Full automation pipeline that runs daily to utilize all 30 Sora credits.
"""

import os
import asyncio
from datetime import datetime, timezone, date, time, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger

from .daily_scheduler import get_daily_scheduler, SoraJob, JobStatus
from .watermark_service import get_watermark_service
from .story_generator import get_story_generator
from .trend_collector import get_trend_collector
from .youtube_publisher import get_youtube_publisher


SORA_OUTPUT_DIR = Path("/Users/isaiahdupree/Documents/Software/MediaPoster/Backend/data/sora_downloads")
PROCESSED_OUTPUT_DIR = Path("/Users/isaiahdupree/Documents/Software/MediaPoster/Backend/data/sora_processed")


class SoraDailyPipelineWorker:
    """
    Full automation worker for daily Sora video pipeline.
    
    Pipeline Steps:
    1. Collect trends from engagement data
    2. Generate prompts for singles and 3-part movies
    3. Queue Sora generations
    4. Download completed videos
    5. Remove watermarks via BlankLogo
    6. Stitch 3-part movies
    7. Publish to YouTube with optimal scheduling
    """
    
    def __init__(self):
        self.scheduler = get_daily_scheduler()
        self.watermark = get_watermark_service()
        self.story_gen = get_story_generator()
        self.trend_collector = get_trend_collector()
        self.youtube = get_youtube_publisher()
        
        # Ensure directories exist
        SORA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        self.is_running = False
        logger.info("✅ SoraDailyPipelineWorker initialized")
    
    async def run_daily_pipeline(self) -> Dict:
        """
        Execute the full daily pipeline.
        
        This should be called once per day, typically early morning.
        """
        if self.is_running:
            return {"success": False, "error": "Pipeline already running"}
        
        self.is_running = True
        start_time = datetime.now(timezone.utc)
        
        try:
            logger.info("🚀 Starting daily Sora pipeline...")
            
            # Step 1: Initialize today's plan
            result = await self.scheduler.start_daily_run()
            if not result["success"]:
                return result
            
            plan = result["plan"]
            logger.info(f"📋 Plan created: {plan['id']}")
            
            # Step 2: Collect trends
            logger.info("📈 Collecting trends...")
            trends = await self.trend_collector.collect_all()
            trend_topics = [t.topic for t in trends[:20]]
            logger.info(f"📈 Found {len(trend_topics)} trends")
            
            # Step 3: Generate jobs
            logger.info("📝 Generating prompts...")
            jobs = await self.scheduler.generate_daily_jobs(trends=trend_topics)
            logger.info(f"📝 Generated {len(jobs)} jobs")
            
            # Step 4: Process jobs in batches
            await self._process_all_jobs(jobs)
            
            # Step 5: Mark plan complete
            self.scheduler.update_plan_status(plan["id"], "completed")
            
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"✅ Daily pipeline completed in {duration:.1f}s")
            
            return {
                "success": True,
                "plan_id": plan["id"],
                "jobs_processed": len(jobs),
                "duration_seconds": duration
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {"success": False, "error": str(e)}
            
        finally:
            self.is_running = False
    
    async def _process_all_jobs(self, jobs: List[SoraJob]):
        """
        Process all jobs through the pipeline.
        
        Handles:
        - Singles: Generate → Download → Remove Watermark → Publish
        - Movies: Generate 3 parts → Download → Remove Watermarks → Stitch → Publish
        """
        # Group movie jobs by movie_id
        movies = {}
        singles = []
        
        for job in jobs:
            if job.movie_id:
                if job.movie_id not in movies:
                    movies[job.movie_id] = []
                movies[job.movie_id].append(job)
            else:
                singles.append(job)
        
        # Process singles
        for job in singles:
            await self._process_single_job(job)
            await asyncio.sleep(2)  # Rate limiting
        
        # Process movies (3-part bundles)
        for movie_id, movie_jobs in movies.items():
            await self._process_movie_jobs(movie_jobs)
            await asyncio.sleep(5)  # Rate limiting
    
    async def _process_single_job(self, job: SoraJob):
        """Process a single video job through the full pipeline."""
        try:
            # Step 1: Generate with Sora (via Safari automation)
            self.scheduler.update_job_status(job.id, JobStatus.GENERATING.value)
            sora_result = await self._generate_with_sora(job.prompt)
            
            if not sora_result.get("success"):
                self.scheduler.update_job_status(job.id, JobStatus.FAILED.value)
                return
            
            # Step 2: Download video
            self.scheduler.update_job_status(job.id, JobStatus.DOWNLOADING.value)
            local_path = await self._download_sora_video(sora_result.get("video_url"), job.id)
            
            if not local_path:
                self.scheduler.update_job_status(job.id, JobStatus.FAILED.value)
                return
            
            self.scheduler.update_job_status(job.id, JobStatus.DOWNLOADING.value, local_path=local_path)
            
            # Step 3: Remove watermark
            self.scheduler.update_job_status(job.id, JobStatus.REMOVING_WATERMARK.value)
            watermark_result = await self.watermark.remove_watermark(local_path, job_id=job.id)
            
            clean_path = watermark_result.get("output_path", local_path)
            self.scheduler.update_job_status(
                job.id, 
                JobStatus.REMOVING_WATERMARK.value,
                watermark_removed_path=clean_path
            )
            
            # Step 4: Publish to YouTube
            self.scheduler.update_job_status(job.id, JobStatus.PUBLISHING.value)
            
            youtube_result = await self.youtube.publish_single(
                video_path=clean_path,
                prompt=job.prompt,
                theme=job.theme,
                trend=job.trend_source
            )
            
            if youtube_result.get("success"):
                self.scheduler.update_job_status(
                    job.id,
                    JobStatus.COMPLETED.value,
                    youtube_video_id=youtube_result.get("video_id")
                )
                logger.info(f"✅ Single published: {job.id}")
            else:
                self.scheduler.update_job_status(job.id, JobStatus.FAILED.value)
                
        except Exception as e:
            logger.error(f"Single job {job.id} failed: {e}")
            self.scheduler.update_job_status(job.id, JobStatus.FAILED.value)
    
    async def _process_movie_jobs(self, jobs: List[SoraJob]):
        """Process a 3-part movie through the pipeline."""
        # Sort by part number
        jobs.sort(key=lambda j: j.job_type)
        
        video_paths = []
        prompts = []
        theme = jobs[0].theme if jobs else "adventure"
        trend = jobs[0].trend_source if jobs else None
        
        try:
            for job in jobs:
                # Generate with Sora
                self.scheduler.update_job_status(job.id, JobStatus.GENERATING.value)
                sora_result = await self._generate_with_sora(job.prompt)
                
                if not sora_result.get("success"):
                    self.scheduler.update_job_status(job.id, JobStatus.FAILED.value)
                    continue
                
                # Download
                self.scheduler.update_job_status(job.id, JobStatus.DOWNLOADING.value)
                local_path = await self._download_sora_video(sora_result.get("video_url"), job.id)
                
                if not local_path:
                    self.scheduler.update_job_status(job.id, JobStatus.FAILED.value)
                    continue
                
                # Remove watermark
                self.scheduler.update_job_status(job.id, JobStatus.REMOVING_WATERMARK.value)
                watermark_result = await self.watermark.remove_watermark(local_path, job_id=job.id)
                
                clean_path = watermark_result.get("output_path", local_path)
                video_paths.append(clean_path)
                prompts.append(job.prompt)
                
                self.scheduler.update_job_status(
                    job.id,
                    JobStatus.COMPLETED.value,
                    watermark_removed_path=clean_path
                )
            
            # Publish all parts as a movie
            if len(video_paths) == 3:
                for job in jobs:
                    self.scheduler.update_job_status(job.id, JobStatus.PUBLISHING.value)
                
                movie_result = await self.youtube.publish_movie(
                    video_paths=video_paths,
                    prompts=prompts,
                    theme=theme,
                    trend=trend
                )
                
                if movie_result.get("success"):
                    for i, job in enumerate(jobs):
                        video_id = movie_result["parts"][i].get("video_id") if i < len(movie_result.get("parts", [])) else None
                        self.scheduler.update_job_status(
                            job.id,
                            JobStatus.COMPLETED.value,
                            youtube_video_id=video_id
                        )
                    logger.info(f"✅ Movie published: {jobs[0].movie_id}")
                    
        except Exception as e:
            logger.error(f"Movie processing failed: {e}")
            for job in jobs:
                self.scheduler.update_job_status(job.id, JobStatus.FAILED.value)
    
    async def _generate_with_sora(self, prompt: str) -> Dict:
        """
        Generate video with Sora via Safari automation.
        
        Uses existing sora_full_automation.py
        """
        try:
            # Import Sora automation
            import sys
            sys.path.insert(0, "/Users/isaiahdupree/Documents/Software/MediaPoster/Backend/automation")
            
            from sora.sora_full_automation import generate_video
            
            result = await asyncio.to_thread(generate_video, prompt, "@isaiahdupree")
            
            return {
                "success": result.get("success", False),
                "video_url": result.get("video_url"),
                "sora_job_id": result.get("job_id")
            }
            
        except ImportError:
            logger.warning("Sora automation not available, simulating...")
            # Simulation for testing
            return {
                "success": True,
                "video_url": "https://sora.example.com/video/test123.mp4",
                "sora_job_id": "sim_" + datetime.now().strftime("%Y%m%d%H%M%S")
            }
        except Exception as e:
            logger.error(f"Sora generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _download_sora_video(self, video_url: str, job_id: str) -> Optional[str]:
        """Download Sora video to local storage."""
        if not video_url:
            return None
        
        try:
            import aiohttp
            
            filename = f"sora_{job_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
            output_path = SORA_OUTPUT_DIR / filename
            
            async with aiohttp.ClientSession() as session:
                async with session.get(video_url) as response:
                    if response.status == 200:
                        content = await response.read()
                        with open(output_path, 'wb') as f:
                            f.write(content)
                        logger.info(f"📥 Downloaded: {output_path}")
                        return str(output_path)
                    else:
                        logger.error(f"Download failed: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Download error: {e}")
            return None
    
    async def emit_event(self, event_type: str, data: Dict):
        """Emit pub/sub event."""
        try:
            from services.event_bus import get_event_bus
            bus = get_event_bus()
            await bus.publish(event_type, data)
        except Exception as e:
            logger.debug(f"Event emission skipped: {e}")


# =============================================================================
# CRON SCHEDULE FUNCTIONS
# =============================================================================

async def run_scheduled_pipeline():
    """
    Entry point for scheduled daily run.
    
    Should be called by cron or scheduler at desired time (e.g., 5 AM).
    """
    worker = SoraDailyPipelineWorker()
    return await worker.run_daily_pipeline()


def get_next_run_time() -> datetime:
    """Get the next scheduled run time (5 AM local time)."""
    now = datetime.now()
    run_time = now.replace(hour=5, minute=0, second=0, microsecond=0)
    
    if now >= run_time:
        run_time += timedelta(days=1)
    
    return run_time


# =============================================================================
# SINGLETON
# =============================================================================

_worker_instance: Optional[SoraDailyPipelineWorker] = None

def get_pipeline_worker() -> SoraDailyPipelineWorker:
    """Get singleton instance of SoraDailyPipelineWorker."""
    global _worker_instance
    if _worker_instance is None:
        _worker_instance = SoraDailyPipelineWorker()
    return _worker_instance
