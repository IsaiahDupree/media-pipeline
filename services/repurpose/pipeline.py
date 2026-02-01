"""
Content Repurposing Pipeline Orchestrator
==========================================
Main orchestrator for the content repurposing engine.

Coordinates:
- Video analysis
- Clip detection
- Rendering
- Database storage

PRD: docs/PRD_CONTENT_REPURPOSING_ENGINE.md
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import UUID, uuid4
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from .video_analyzer import VideoAnalyzer
from .clip_extractor import ClipExtractor, ClipConfig


class RepurposePipeline:
    """
    Content Repurposing Pipeline

    Full pipeline for repurposing long-form videos into short clips.

    Usage:
        pipeline = RepurposePipeline(
            database_url="postgresql://...",
            openai_api_key="sk-..."
        )

        result = await pipeline.process_video(
            video_path="long_video.mp4",
            user_id="user-uuid",
            title="My Podcast Episode #45"
        )
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ):
        """Initialize pipeline"""
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:54322/postgres"
        )
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        # Initialize components
        self.analyzer = VideoAnalyzer(api_key=self.openai_api_key)
        self.extractor = ClipExtractor()

        # Database
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)

    async def process_video(
        self,
        video_path: str,
        user_id: str,
        title: str,
        source_type: str = "upload",
        source_url: Optional[str] = None,
        auto_extract: bool = True
    ) -> Dict[str, Any]:
        """
        Process a video through the full repurposing pipeline

        Args:
            video_path: Path to source video file
            user_id: User UUID
            title: Video title
            source_type: "upload", "youtube", "podcast_rss", "twitch_vod"
            source_url: Original URL (if applicable)
            auto_extract: Automatically extract clips after analysis

        Returns:
            {
                "source_id": UUID,
                "status": "completed" | "failed",
                "highlights_count": int,
                "clips_extracted": int,
                "error": str (optional)
            }
        """
        source_id = None

        try:
            # Step 1: Create source record
            logger.info(f"Starting repurpose pipeline for: {title}")
            source_id = await self._create_source_record(
                user_id=user_id,
                title=title,
                source_type=source_type,
                source_url=source_url,
                file_path=video_path
            )

            await self._update_source_status(source_id, "processing")

            # Step 2: Get video info
            video_info = await self.extractor.get_video_info(video_path)
            duration = video_info.get("duration", 0)
            await self._update_source_duration(source_id, int(duration))

            # Step 3: Analyze video
            logger.info("Analyzing video for highlights...")
            analysis = await self.analyzer.analyze_video(video_path)

            # Step 4: Save transcript
            await self._save_transcript(
                source_id=source_id,
                full_text=analysis["transcript"],
                language=analysis["metadata"].get("language", "en"),
                words=analysis["metadata"].get("words_count", 0)
            )

            # Step 5: Save detected highlights
            highlights = analysis["highlights"]
            logger.info(f"Detected {len(highlights)} highlights")

            clip_ids = []
            for highlight in highlights:
                clip_id = await self._save_clip(
                    source_id=source_id,
                    highlight=highlight
                )
                clip_ids.append(clip_id)

            await self._update_source_clips_count(source_id, len(clip_ids))

            # Step 6: Auto-extract clips (optional)
            clips_extracted = 0
            if auto_extract and clip_ids:
                logger.info(f"Extracting {len(clip_ids)} clips...")
                clips_extracted = await self._extract_all_clips(
                    source_id=source_id,
                    video_path=video_path,
                    highlights=highlights
                )

            # Step 7: Mark as completed
            await self._update_source_status(source_id, "completed")

            logger.info(f"Pipeline completed: {len(highlights)} highlights, {clips_extracted} extracted")

            return {
                "source_id": str(source_id),
                "status": "completed",
                "highlights_count": len(highlights),
                "clips_extracted": clips_extracted
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")

            if source_id:
                await self._update_source_status(source_id, "failed", error=str(e))

            return {
                "source_id": str(source_id) if source_id else None,
                "status": "failed",
                "highlights_count": 0,
                "clips_extracted": 0,
                "error": str(e)
            }

    async def _create_source_record(
        self,
        user_id: str,
        title: str,
        source_type: str,
        source_url: Optional[str],
        file_path: str
    ) -> UUID:
        """Create repurpose_sources record"""
        with self.SessionLocal() as session:
            result = session.execute(text("""
                INSERT INTO repurpose_sources
                (user_id, title, source_type, source_url, file_path, status)
                VALUES (:user_id, :title, :source_type, :source_url, :file_path, 'pending')
                RETURNING id
            """), {
                "user_id": user_id,
                "title": title,
                "source_type": source_type,
                "source_url": source_url,
                "file_path": file_path
            })
            session.commit()
            return result.fetchone()[0]

    async def _update_source_status(
        self,
        source_id: UUID,
        status: str,
        error: Optional[str] = None
    ) -> None:
        """Update source status"""
        with self.SessionLocal() as session:
            session.execute(text("""
                UPDATE repurpose_sources
                SET status = :status, error_message = :error, updated_at = NOW()
                WHERE id = :id
            """), {
                "id": str(source_id),
                "status": status,
                "error": error
            })
            session.commit()

    async def _update_source_duration(self, source_id: UUID, duration: int) -> None:
        """Update source duration"""
        with self.SessionLocal() as session:
            session.execute(text("""
                UPDATE repurpose_sources
                SET duration_seconds = :duration
                WHERE id = :id
            """), {
                "id": str(source_id),
                "duration": duration
            })
            session.commit()

    async def _update_source_clips_count(self, source_id: UUID, count: int) -> None:
        """Update clips count"""
        with self.SessionLocal() as session:
            session.execute(text("""
                UPDATE repurpose_sources
                SET clips_generated = :count
                WHERE id = :id
            """), {
                "id": str(source_id),
                "count": count
            })
            session.commit()

    async def _save_transcript(
        self,
        source_id: UUID,
        full_text: str,
        language: str,
        words: int
    ) -> UUID:
        """Save transcript record"""
        with self.SessionLocal() as session:
            result = session.execute(text("""
                INSERT INTO repurpose_transcripts
                (source_id, full_text, language, words)
                VALUES (:source_id, :full_text, :language, :words)
                RETURNING id
            """), {
                "source_id": str(source_id),
                "full_text": full_text,
                "language": language,
                "words": "[]"  # TODO: Store actual word-level timing
            })
            session.commit()
            return result.fetchone()[0]

    async def _save_clip(
        self,
        source_id: UUID,
        highlight: Dict[str, Any]
    ) -> UUID:
        """Save clip record"""
        with self.SessionLocal() as session:
            result = session.execute(text("""
                INSERT INTO repurpose_clips
                (source_id, start_time, end_time, title, transcript_segment,
                 virality_score, hook_score, emotion_score, status, metadata)
                VALUES (:source_id, :start_time, :end_time, :title, :transcript,
                        :virality, :hook, :emotion, 'detected', :metadata)
                RETURNING id
            """), {
                "source_id": str(source_id),
                "start_time": highlight["start"],
                "end_time": highlight["end"],
                "title": highlight["title"],
                "transcript": highlight["transcript"],
                "virality": highlight["virality_score"],
                "hook": highlight["hook_score"],
                "emotion": highlight["emotion_score"],
                "metadata": "{}"
            })
            session.commit()
            return result.fetchone()[0]

    async def _extract_all_clips(
        self,
        source_id: UUID,
        video_path: str,
        highlights: List[Dict[str, Any]]
    ) -> int:
        """Extract all clips and save render records"""
        extracted_count = 0

        for highlight in highlights:
            # Find clip ID
            with self.SessionLocal() as session:
                result = session.execute(text("""
                    SELECT id FROM repurpose_clips
                    WHERE source_id = :source_id
                    AND start_time = :start
                    ORDER BY created_at DESC LIMIT 1
                """), {
                    "source_id": str(source_id),
                    "start": highlight["start"]
                })
                clip_row = result.fetchone()
                if not clip_row:
                    continue
                clip_id = clip_row[0]

            # Extract in 9:16 format for TikTok/Reels/Shorts
            config = ClipConfig(
                start_time=highlight["start"],
                end_time=highlight["end"],
                aspect_ratio="9:16",
                target_platform="tiktok"
            )

            result = await self.extractor.extract_clip(video_path, config)

            if result.success:
                # Save render record
                await self._save_render(
                    clip_id=clip_id,
                    aspect_ratio="9:16",
                    target_platform="tiktok",
                    file_path=result.output_path,
                    file_size=result.file_size_bytes,
                    status="completed"
                )
                extracted_count += 1
            else:
                logger.warning(f"Clip extraction failed: {result.error_message}")
                await self._save_render(
                    clip_id=clip_id,
                    aspect_ratio="9:16",
                    target_platform="tiktok",
                    file_path=None,
                    file_size=None,
                    status="failed",
                    error=result.error_message
                )

        return extracted_count

    async def _save_render(
        self,
        clip_id: UUID,
        aspect_ratio: str,
        target_platform: str,
        file_path: Optional[str],
        file_size: Optional[int],
        status: str,
        error: Optional[str] = None
    ) -> UUID:
        """Save render record"""
        with self.SessionLocal() as session:
            result = session.execute(text("""
                INSERT INTO repurpose_renders
                (clip_id, aspect_ratio, target_platform, file_path, file_size_bytes,
                 render_status, error_message)
                VALUES (:clip_id, :aspect_ratio, :target_platform, :file_path, :file_size,
                        :status, :error)
                RETURNING id
            """), {
                "clip_id": str(clip_id),
                "aspect_ratio": aspect_ratio,
                "target_platform": target_platform,
                "file_path": file_path,
                "file_size": file_size,
                "status": status,
                "error": error
            })
            session.commit()
            return result.fetchone()[0]

    async def get_source_status(self, source_id: str) -> Dict[str, Any]:
        """Get repurpose source status and clips"""
        with self.SessionLocal() as session:
            # Get source info
            source_result = session.execute(text("""
                SELECT id, title, status, clips_generated, duration_seconds, error_message
                FROM repurpose_sources
                WHERE id = :id
            """), {"id": source_id})

            source_row = source_result.fetchone()
            if not source_row:
                return {"error": "Source not found"}

            # Get clips
            clips_result = session.execute(text("""
                SELECT id, start_time, end_time, title, virality_score,
                       hook_score, emotion_score, status
                FROM repurpose_clips
                WHERE source_id = :source_id
                ORDER BY virality_score DESC
            """), {"source_id": source_id})

            clips = [
                {
                    "id": str(row[0]),
                    "start": row[1],
                    "end": row[2],
                    "title": row[3],
                    "virality_score": row[4],
                    "hook_score": row[5],
                    "emotion_score": row[6],
                    "status": row[7]
                }
                for row in clips_result.fetchall()
            ]

            return {
                "id": str(source_row[0]),
                "title": source_row[1],
                "status": source_row[2],
                "clips_generated": source_row[3],
                "duration_seconds": source_row[4],
                "error_message": source_row[5],
                "clips": clips
            }
