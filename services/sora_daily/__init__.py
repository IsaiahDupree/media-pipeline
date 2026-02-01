"""
Daily Sora Automation Package
Automated daily video generation with @isaiahdupree character.
"""

from .daily_scheduler import (
    DailySoraScheduler,
    DailyPlan,
    SoraJob,
    JobStatus,
    JobType,
    get_daily_scheduler
)

from .watermark_service import (
    WatermarkRemovalService,
    get_watermark_service
)

from .story_generator import (
    StoryGenerator,
    get_story_generator,
    STORY_THEMES
)

from .trend_collector import (
    TrendCollector,
    TrendSource,
    get_trend_collector
)

from .youtube_publisher import (
    YouTubePublisher,
    YouTubeUploadJob,
    get_youtube_publisher
)

from .pipeline_worker import (
    SoraDailyPipelineWorker,
    get_pipeline_worker,
    run_scheduled_pipeline,
    get_next_run_time
)

__all__ = [
    # Scheduler
    "DailySoraScheduler",
    "DailyPlan",
    "SoraJob",
    "JobStatus",
    "JobType",
    "get_daily_scheduler",
    
    # Watermark
    "WatermarkRemovalService",
    "get_watermark_service",
    
    # Story
    "StoryGenerator",
    "get_story_generator",
    "STORY_THEMES",
    
    # Trends
    "TrendCollector",
    "TrendSource",
    "get_trend_collector",
    
    # YouTube
    "YouTubePublisher",
    "YouTubeUploadJob",
    "get_youtube_publisher",
    
    # Pipeline
    "SoraDailyPipelineWorker",
    "get_pipeline_worker",
    "run_scheduled_pipeline",
    "get_next_run_time"
]
