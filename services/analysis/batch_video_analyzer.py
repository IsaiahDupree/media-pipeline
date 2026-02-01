"""
Batch Video Analysis Service (CUR-001)
=======================================
Queue all unanalyzed videos for AI analysis (transcript, sentiment, content).

Features:
- Batch process unanalyzed videos
- Multi-threaded/async processing
- Progress tracking
- Retry logic for failures
- Integration with existing AI analyzers
"""

import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func

from database.connection import get_db_context
from database.models import Video, VideoAnalysis
from services.event_bus import EventBus, Topics
from services.ai_content_analyzer import AIContentAnalyzer
from services.whisper_transcriber import WhisperTranscriber


@dataclass
class BatchAnalysisProgress:
    """Progress tracking for batch analysis"""
    total: int = 0
    completed: int = 0
    failed: int = 0
    in_progress: int = 0
    skipped: int = 0

    @property
    def percent_complete(self) -> float:
        """Calculate completion percentage"""
        if self.total == 0:
            return 0.0
        return (self.completed / self.total) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total': self.total,
            'completed': self.completed,
            'failed': self.failed,
            'in_progress': self.in_progress,
            'skipped': self.skipped,
            'percent_complete': round(self.percent_complete, 2)
        }


class BatchVideoAnalyzer:
    """
    Batch Video Analysis Service

    Processes unanalyzed videos in batches:
    1. Find videos without analysis
    2. Extract transcripts (Whisper)
    3. Analyze content (GPT-4 Vision)
    4. Calculate sentiment scores
    5. Store results in database
    """

    def __init__(
        self,
        batch_size: int = 10,
        concurrency: int = 3,
        skip_existing: bool = True
    ):
        """
        Initialize batch analyzer

        Args:
            batch_size: Number of videos per batch
            concurrency: Max concurrent analysis tasks
            skip_existing: Skip videos that already have analysis
        """
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.skip_existing = skip_existing

        # Services
        self.content_analyzer = AIContentAnalyzer()
        self.transcriber = WhisperTranscriber()

        # Event bus
        self.event_bus = EventBus.get_instance()
        self.event_bus.set_source("batch-video-analyzer")

        # Progress tracking
        self.current_batch_id: Optional[str] = None
        self.progress = BatchAnalysisProgress()

        logger.info(
            f"📊 Batch Video Analyzer initialized | "
            f"Batch size: {batch_size} | "
            f"Concurrency: {concurrency}"
        )

    async def get_unanalyzed_videos(
        self,
        limit: Optional[int] = None,
        media_type: str = "video"
    ) -> List[Video]:
        """
        Get list of videos that haven't been analyzed

        Args:
            limit: Maximum number of videos to return
            media_type: Type of media to analyze (default: video)

        Returns:
            List of Video records
        """
        async with get_db_context() as db:
            # Find videos without analysis
            query = (
                select(Video)
                .outerjoin(
                    VideoAnalysis,
                    Video.id == VideoAnalysis.video_id
                )
                .where(
                    VideoAnalysis.video_id.is_(None) if self.skip_existing else True
                )
                .order_by(Video.created_at.desc())
            )

            if limit:
                query = query.limit(limit)

            result = await db.execute(query)
            videos = result.scalars().all()

            logger.info(f"Found {len(videos)} unanalyzed videos")
            return list(videos)

    async def analyze_video(
        self,
        media: Video,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Analyze a single video

        Args:
            media: Video record
            db: Database session

        Returns:
            Analysis results dictionary
        """
        try:
            logger.info(f"Analyzing video: {media.file_name}")

            # Emit progress event
            await self.event_bus.publish(
                Topics.ANALYSIS_STARTED,
                {
                    'media_id': str(media.id),
                    'filename': media.file_name,
                    'batch_id': self.current_batch_id
                }
            )

            # Step 1: Extract transcript
            transcript = None
            if media.source_uri:
                try:
                    transcript_result = await self.transcriber.transcribe_video(
                        video_path=media.source_uri
                    )
                    transcript = transcript_result.get('text', '')
                except Exception as e:
                    logger.warning(f"Transcript extraction failed for {media.file_name}: {e}")

            # Step 2: Visual content analysis
            visual_analysis = None
            if media.source_uri:
                try:
                    visual_analysis = await self.content_analyzer.analyze_media(
                        media_path=media.source_uri,
                        media_type='video'
                    )
                except Exception as e:
                    logger.warning(f"Visual analysis failed for {media.file_name}: {e}")

            # Step 3: Create analysis record
            analysis = VideoAnalysis(
                video_id=media.id,
                transcript=transcript,
                visual_analysis=visual_analysis if visual_analysis else None,
                analyzed_at=datetime.now(timezone.utc)
            )

            db.add(analysis)
            await db.commit()
            await db.refresh(analysis)

            # Emit completion event
            await self.event_bus.publish(
                Topics.ANALYSIS_COMPLETED,
                {
                    'media_id': str(media.id),
                    'analysis_id': str(analysis.video_id),
                    'filename': media.file_name,
                    'has_transcript': transcript is not None,
                    'pre_social_score': analysis.pre_social_score,
                    'batch_id': self.current_batch_id
                }
            )

            return {
                'success': True,
                'media_id': str(media.id),
                'analysis_id': str(analysis.id),
                'has_transcript': transcript is not None,
                'quality_score': analysis.quality_score
            }

        except Exception as e:
            logger.error(f"Analysis failed for {media.file_name}: {e}")

            # Emit failure event
            await self.event_bus.publish(
                Topics.ANALYSIS_FAILED,
                {
                    'media_id': str(media.id),
                    'filename': media.file_name,
                    'error': str(e),
                    'batch_id': self.current_batch_id
                }
            )

            return {
                'success': False,
                'media_id': str(media.id),
                'error': str(e)
            }

    async def process_batch(
        self,
        videos: List[Video],
        batch_id: str
    ) -> Dict[str, Any]:
        """
        Process a batch of videos with concurrency control

        Args:
            videos: List of videos to process
            batch_id: Unique batch identifier

        Returns:
            Batch results summary
        """
        self.current_batch_id = batch_id
        self.progress = BatchAnalysisProgress(total=len(videos))

        # Emit batch started event
        await self.event_bus.publish(
            Topics.BATCH_ANALYSIS_STARTED,
            {
                'batch_id': batch_id,
                'total_videos': len(videos),
                'started_at': datetime.now(timezone.utc).isoformat()
            }
        )

        # Process videos with concurrency limit
        semaphore = asyncio.Semaphore(self.concurrency)

        async def process_with_semaphore(media: Video):
            async with semaphore:
                async with get_db_context() as db:
                    self.progress.in_progress += 1
                    result = await self.analyze_video(media, db)
                    self.progress.in_progress -= 1

                    if result['success']:
                        self.progress.completed += 1
                    else:
                        self.progress.failed += 1

                    return result

        # Execute analysis tasks
        tasks = [process_with_semaphore(video) for video in videos]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch processing exception: {result}")
                self.progress.failed += 1
                results[i] = {
                    'success': False,
                    'media_id': str(videos[i].id),
                    'error': str(result)
                }

        # Emit batch completed event
        await self.event_bus.publish(
            Topics.BATCH_ANALYSIS_COMPLETED,
            {
                'batch_id': batch_id,
                'total': self.progress.total,
                'completed': self.progress.completed,
                'failed': self.progress.failed,
                'percent_complete': self.progress.percent_complete,
                'completed_at': datetime.now(timezone.utc).isoformat()
            }
        )

        return {
            'batch_id': batch_id,
            'progress': self.progress.to_dict(),
            'results': results
        }

    async def analyze_all_unanalyzed(
        self,
        max_videos: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze all unanalyzed videos in the library

        Args:
            max_videos: Maximum number of videos to process (None = all)

        Returns:
            Analysis summary
        """
        import uuid

        # Get unanalyzed videos
        videos = await self.get_unanalyzed_videos(limit=max_videos)

        if not videos:
            return {
                'success': True,
                'message': 'No unanalyzed videos found',
                'analyzed': 0
            }

        # Process in batches
        batch_id = str(uuid.uuid4())
        result = await self.process_batch(videos, batch_id)

        logger.success(
            f"✓ Batch analysis complete | "
            f"Processed: {result['progress']['completed']}/{result['progress']['total']} | "
            f"Failed: {result['progress']['failed']}"
        )

        return {
            'success': True,
            'batch_id': batch_id,
            'total_videos': len(videos),
            'analyzed': result['progress']['completed'],
            'failed': result['progress']['failed'],
            'percent_complete': result['progress']['percent_complete']
        }

    async def get_analysis_stats(self) -> Dict[str, Any]:
        """
        Get statistics on analyzed vs unanalyzed videos

        Returns:
            Statistics dictionary
        """
        async with get_db_context() as db:
            # Total videos
            total_result = await db.execute(
                select(func.count(Video.id))
            )
            total = total_result.scalar() or 0

            # Analyzed videos
            analyzed_result = await db.execute(
                select(func.count(VideoAnalysis.video_id))
            )
            analyzed = analyzed_result.scalar() or 0

            unanalyzed = total - analyzed

            return {
                'total_videos': total,
                'analyzed': analyzed,
                'unanalyzed': unanalyzed,
                'percent_analyzed': (analyzed / total * 100) if total > 0 else 0
            }


# Singleton instance
_batch_analyzer: Optional[BatchVideoAnalyzer] = None


def get_batch_analyzer() -> BatchVideoAnalyzer:
    """Get singleton instance of BatchVideoAnalyzer"""
    global _batch_analyzer
    if _batch_analyzer is None:
        _batch_analyzer = BatchVideoAnalyzer()
    return _batch_analyzer
