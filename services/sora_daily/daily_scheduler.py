"""
Daily Sora Scheduler Service
Orchestrates the daily Sora video generation pipeline.
"""

import os
import asyncio
from datetime import datetime, timezone, date, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
from loguru import logger
from sqlalchemy import create_engine, text


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@127.0.0.1:54322/postgres")

DAILY_SORA_CREDITS = 30
SINGLES_PER_DAY = 10
MOVIES_PER_DAY = 4  # 4 three-part movies = 12 videos
BUFFER_VIDEOS = 8   # For retries/experiments


class JobStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    GENERATING = "generating"
    DOWNLOADING = "downloading"
    REMOVING_WATERMARK = "removing_watermark"
    STITCHING = "stitching"
    PUBLISHING = "publishing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobType(str, Enum):
    SINGLE = "single"
    MOVIE_PART_1 = "movie_part_1"
    MOVIE_PART_2 = "movie_part_2"
    MOVIE_PART_3 = "movie_part_3"


@dataclass
class SoraJob:
    """Individual Sora generation job."""
    id: str = field(default_factory=lambda: str(uuid4()))
    plan_id: str = ""
    job_type: str = JobType.SINGLE.value
    movie_id: Optional[str] = None
    prompt: str = ""
    theme: str = ""
    trend_source: Optional[str] = None
    character: str = "@isaiahdupree"
    
    status: str = JobStatus.PENDING.value
    sora_job_id: Optional[str] = None
    video_url: Optional[str] = None
    local_path: Optional[str] = None
    watermark_removed_path: Optional[str] = None
    youtube_video_id: Optional[str] = None
    published_at: Optional[datetime] = None
    
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "plan_id": self.plan_id,
            "job_type": self.job_type,
            "movie_id": self.movie_id,
            "prompt": self.prompt,
            "theme": self.theme,
            "character": self.character,
            "status": self.status,
            "sora_job_id": self.sora_job_id,
            "video_url": self.video_url,
            "local_path": self.local_path,
            "watermark_removed_path": self.watermark_removed_path,
            "youtube_video_id": self.youtube_video_id,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


@dataclass
class DailyPlan:
    """Daily Sora generation plan."""
    id: str = field(default_factory=lambda: str(uuid4()))
    plan_date: date = field(default_factory=date.today)
    total_credits: int = DAILY_SORA_CREDITS
    used_credits: int = 0
    singles_planned: int = SINGLES_PER_DAY
    movies_planned: int = MOVIES_PER_DAY
    status: str = "pending"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "date": self.plan_date.isoformat(),
            "total_credits": self.total_credits,
            "used_credits": self.used_credits,
            "remaining_credits": self.total_credits - self.used_credits,
            "singles_planned": self.singles_planned,
            "movies_planned": self.movies_planned,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


class DailySoraScheduler:
    """
    Orchestrates daily Sora video generation.
    
    Manages the full pipeline:
    1. Plan daily content mix
    2. Generate prompts (singles + 3-part movies)
    3. Queue Sora generations
    4. Download videos
    5. Remove watermarks
    6. Stitch movies
    7. Publish to YouTube
    """
    
    def __init__(self):
        self.engine = create_engine(DATABASE_URL)
        self._ensure_tables()
        self.is_running = False
        self.current_plan: Optional[DailyPlan] = None
        logger.info("✅ DailySoraScheduler initialized")
    
    def _ensure_tables(self):
        """Create database tables if they don't exist."""
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sora_daily_plans (
                    id TEXT PRIMARY KEY,
                    plan_date DATE NOT NULL UNIQUE,
                    total_credits INTEGER DEFAULT 30,
                    used_credits INTEGER DEFAULT 0,
                    singles_planned INTEGER DEFAULT 10,
                    movies_planned INTEGER DEFAULT 4,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT NOW(),
                    completed_at TIMESTAMP
                )
            """))
            
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sora_generation_jobs (
                    id TEXT PRIMARY KEY,
                    plan_id TEXT REFERENCES sora_daily_plans(id),
                    job_type TEXT NOT NULL,
                    movie_id TEXT,
                    prompt TEXT NOT NULL,
                    theme TEXT,
                    trend_source TEXT,
                    character TEXT DEFAULT '@isaiahdupree',
                    status TEXT DEFAULT 'pending',
                    sora_job_id TEXT,
                    video_url TEXT,
                    local_path TEXT,
                    watermark_removed_path TEXT,
                    youtube_video_id TEXT,
                    published_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """))
            
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sora_trend_sources (
                    id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    relevance_score FLOAT DEFAULT 0.5,
                    used_in_story BOOLEAN DEFAULT FALSE,
                    discovered_at TIMESTAMP DEFAULT NOW()
                )
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_sora_jobs_plan 
                ON sora_generation_jobs(plan_id)
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_sora_jobs_status 
                ON sora_generation_jobs(status)
            """))
            
            conn.commit()
        
        logger.info("✅ Sora daily tables created")
    
    # -------------------------------------------------------------------------
    # PLAN MANAGEMENT
    # -------------------------------------------------------------------------
    
    def get_or_create_today_plan(self) -> DailyPlan:
        """Get or create today's generation plan."""
        today = date.today()
        
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT * FROM sora_daily_plans WHERE plan_date = :date
            """), {"date": today}).fetchone()
            
            if result:
                return DailyPlan(
                    id=result[0],
                    plan_date=result[1],
                    total_credits=result[2],
                    used_credits=result[3],
                    singles_planned=result[4],
                    movies_planned=result[5],
                    status=result[6],
                    created_at=result[7],
                    completed_at=result[8]
                )
        
        # Create new plan
        plan = DailyPlan(plan_date=today)
        
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO sora_daily_plans 
                (id, plan_date, total_credits, used_credits, singles_planned, movies_planned, status)
                VALUES (:id, :date, :total, :used, :singles, :movies, :status)
            """), {
                "id": plan.id,
                "date": plan.plan_date,
                "total": plan.total_credits,
                "used": plan.used_credits,
                "singles": plan.singles_planned,
                "movies": plan.movies_planned,
                "status": plan.status
            })
            conn.commit()
        
        logger.info(f"📋 Created daily plan: {plan.id}")
        return plan
    
    def update_plan_status(self, plan_id: str, status: str):
        """Update plan status."""
        with self.engine.connect() as conn:
            completed_at = datetime.now(timezone.utc) if status == "completed" else None
            conn.execute(text("""
                UPDATE sora_daily_plans 
                SET status = :status, completed_at = :completed_at
                WHERE id = :id
            """), {"id": plan_id, "status": status, "completed_at": completed_at})
            conn.commit()
    
    def increment_used_credits(self, plan_id: str, count: int = 1):
        """Increment used credits for a plan."""
        with self.engine.connect() as conn:
            conn.execute(text("""
                UPDATE sora_daily_plans 
                SET used_credits = used_credits + :count
                WHERE id = :id
            """), {"id": plan_id, "count": count})
            conn.commit()
    
    # -------------------------------------------------------------------------
    # JOB MANAGEMENT
    # -------------------------------------------------------------------------
    
    def create_job(self, job: SoraJob) -> SoraJob:
        """Create a new generation job."""
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO sora_generation_jobs 
                (id, plan_id, job_type, movie_id, prompt, theme, trend_source, character, status)
                VALUES (:id, :plan_id, :job_type, :movie_id, :prompt, :theme, :trend_source, :character, :status)
            """), {
                "id": job.id,
                "plan_id": job.plan_id,
                "job_type": job.job_type,
                "movie_id": job.movie_id,
                "prompt": job.prompt,
                "theme": job.theme,
                "trend_source": job.trend_source,
                "character": job.character,
                "status": job.status
            })
            conn.commit()
        
        return job
    
    def update_job_status(self, job_id: str, status: str, **kwargs):
        """Update job status and optional fields."""
        updates = ["status = :status", "updated_at = NOW()"]
        params = {"id": job_id, "status": status}
        
        for key, value in kwargs.items():
            if key in ["sora_job_id", "video_url", "local_path", "watermark_removed_path", "youtube_video_id"]:
                updates.append(f"{key} = :{key}")
                params[key] = value
        
        if status == "completed" and "published_at" not in kwargs:
            updates.append("published_at = NOW()")
        
        with self.engine.connect() as conn:
            conn.execute(text(f"""
                UPDATE sora_generation_jobs 
                SET {', '.join(updates)}
                WHERE id = :id
            """), params)
            conn.commit()
    
    def get_jobs_for_plan(self, plan_id: str) -> List[SoraJob]:
        """Get all jobs for a plan."""
        with self.engine.connect() as conn:
            results = conn.execute(text("""
                SELECT * FROM sora_generation_jobs 
                WHERE plan_id = :plan_id
                ORDER BY created_at
            """), {"plan_id": plan_id}).fetchall()
            
            return [self._row_to_job(r) for r in results]
    
    def get_pending_jobs(self, limit: int = 10) -> List[SoraJob]:
        """Get pending jobs ready for processing."""
        with self.engine.connect() as conn:
            results = conn.execute(text("""
                SELECT * FROM sora_generation_jobs 
                WHERE status = 'pending'
                ORDER BY created_at
                LIMIT :limit
            """), {"limit": limit}).fetchall()
            
            return [self._row_to_job(r) for r in results]
    
    def _row_to_job(self, row) -> SoraJob:
        """Convert database row to SoraJob."""
        return SoraJob(
            id=row[0],
            plan_id=row[1],
            job_type=row[2],
            movie_id=row[3],
            prompt=row[4],
            theme=row[5],
            trend_source=row[6],
            character=row[7],
            status=row[8],
            sora_job_id=row[9],
            video_url=row[10],
            local_path=row[11],
            watermark_removed_path=row[12],
            youtube_video_id=row[13],
            published_at=row[14],
            created_at=row[15],
            updated_at=row[16]
        )
    
    # -------------------------------------------------------------------------
    # DAILY EXECUTION
    # -------------------------------------------------------------------------
    
    async def start_daily_run(self) -> Dict:
        """Start the daily Sora generation run."""
        if self.is_running:
            return {"success": False, "error": "Daily run already in progress"}
        
        self.is_running = True
        self.current_plan = self.get_or_create_today_plan()
        
        if self.current_plan.status == "completed":
            self.is_running = False
            return {"success": False, "error": "Today's plan already completed"}
        
        self.update_plan_status(self.current_plan.id, "in_progress")
        
        # Emit event
        await self._emit_event("sora.daily.started", {
            "plan_id": self.current_plan.id,
            "date": self.current_plan.plan_date.isoformat()
        })
        
        logger.info(f"🚀 Started daily Sora run: {self.current_plan.id}")
        
        return {
            "success": True,
            "plan": self.current_plan.to_dict()
        }
    
    async def generate_daily_jobs(self, trends: List[str] = None) -> List[SoraJob]:
        """Generate jobs for the day based on content mix."""
        if not self.current_plan:
            self.current_plan = self.get_or_create_today_plan()
        
        from .story_generator import get_story_generator
        story_gen = get_story_generator()
        
        jobs = []
        
        # Generate 3-part movie jobs
        for movie_num in range(self.current_plan.movies_planned):
            movie_id = str(uuid4())
            theme = story_gen.get_random_theme()
            trend = trends[movie_num] if trends and movie_num < len(trends) else None
            
            for part in range(1, 4):
                job_type = f"movie_part_{part}"
                prompt = await story_gen.generate_movie_prompt(
                    part=part,
                    theme=theme,
                    trend=trend,
                    character="@isaiahdupree"
                )
                
                job = SoraJob(
                    plan_id=self.current_plan.id,
                    job_type=job_type,
                    movie_id=movie_id,
                    prompt=prompt,
                    theme=theme,
                    trend_source=trend,
                    character="@isaiahdupree"
                )
                
                self.create_job(job)
                jobs.append(job)
        
        # Generate single video jobs
        for i in range(self.current_plan.singles_planned):
            theme = story_gen.get_random_theme()
            trend = trends[self.current_plan.movies_planned + i] if trends and self.current_plan.movies_planned + i < len(trends) else None
            
            prompt = await story_gen.generate_single_prompt(
                theme=theme,
                trend=trend,
                character="@isaiahdupree"
            )
            
            job = SoraJob(
                plan_id=self.current_plan.id,
                job_type=JobType.SINGLE.value,
                prompt=prompt,
                theme=theme,
                trend_source=trend,
                character="@isaiahdupree"
            )
            
            self.create_job(job)
            jobs.append(job)
        
        logger.info(f"📝 Generated {len(jobs)} jobs for today")
        return jobs
    
    async def process_job(self, job: SoraJob) -> Dict:
        """Process a single job through the full pipeline."""
        try:
            # Step 1: Generate with Sora
            self.update_job_status(job.id, JobStatus.GENERATING.value)
            # TODO: Integrate with Sora automation
            # sora_result = await sora_automation.generate_video(job.prompt)
            
            # Step 2: Download video
            self.update_job_status(job.id, JobStatus.DOWNLOADING.value)
            # TODO: Download from Sora
            
            # Step 3: Remove watermark
            self.update_job_status(job.id, JobStatus.REMOVING_WATERMARK.value)
            from .watermark_service import get_watermark_service
            watermark_service = get_watermark_service()
            # watermark_result = await watermark_service.remove_watermark(job.local_path)
            
            # Step 4: Publish
            self.update_job_status(job.id, JobStatus.PUBLISHING.value)
            # TODO: Publish to YouTube
            
            self.update_job_status(job.id, JobStatus.COMPLETED.value)
            
            return {"success": True, "job_id": job.id}
            
        except Exception as e:
            logger.error(f"Job {job.id} failed: {e}")
            self.update_job_status(job.id, JobStatus.FAILED.value)
            return {"success": False, "job_id": job.id, "error": str(e)}
    
    async def _emit_event(self, event_type: str, data: Dict):
        """Emit pub/sub event."""
        try:
            from services.event_bus import get_event_bus
            bus = get_event_bus()
            await bus.publish(event_type, data)
        except Exception as e:
            logger.debug(f"Event emission skipped: {e}")
    
    def get_daily_status(self) -> Dict:
        """Get today's generation status."""
        plan = self.get_or_create_today_plan()
        jobs = self.get_jobs_for_plan(plan.id)
        
        status_counts = {}
        for job in jobs:
            status_counts[job.status] = status_counts.get(job.status, 0) + 1
        
        return {
            "plan": plan.to_dict(),
            "jobs_count": len(jobs),
            "status_breakdown": status_counts,
            "is_running": self.is_running
        }


# =============================================================================
# SINGLETON
# =============================================================================

_scheduler_instance: Optional[DailySoraScheduler] = None

def get_daily_scheduler() -> DailySoraScheduler:
    """Get singleton instance of DailySoraScheduler."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = DailySoraScheduler()
    return _scheduler_instance
