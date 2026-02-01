"""
Publish Worker
==============
Event-driven worker for social media publishing pipeline.

Subscribes to:
    - publish.requested (manual publish request)
    - schedule.due (scheduled post ready to publish)

Emits:
    - publish.started
    - publish.uploading
    - publish.upload.completed
    - publish.submitted
    - publish.polling
    - publish.completed
    - publish.failed
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from services.event_bus import EventBus, Event, Topics
from services.workers.base import BaseWorker
from services.content_guard.duplicate_detector import DuplicateDetector

logger = logging.getLogger(__name__)


class PublishWorker(BaseWorker):
    """
    Worker for processing publish requests.
    
    Pipeline steps:
        1. Verify media and account
        2. Upload to cloud storage
        3. Upload to Blotato
        4. Submit to platform
        5. Poll for platform URL
    
    Usage:
        worker = PublishWorker()
        await worker.start()
        
        # Trigger publishing via event:
        await event_bus.publish(Topics.PUBLISH_REQUESTED, {
            "media_id": "...",
            "platform": "instagram",
            "account_id": "807",
            "caption": "...",
            "hashtags": [...]
        })
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None, worker_id: Optional[str] = None):
        super().__init__(event_bus, worker_id)
        self._publish_service = None  # Lazy load
        self._background_publisher = None  # Lazy load
    
    def get_subscriptions(self) -> List[str]:
        """Subscribe to publish-related events."""
        return [
            Topics.PUBLISH_REQUESTED,
            Topics.SCHEDULE_DUE,
        ]
    
    async def handle_event(self, event: Event) -> None:
        """Process publish events."""
        if event.topic == Topics.SCHEDULE_DUE:
            # Extract from scheduled post
            post_id = event.payload.get("post_id")
            if post_id:
                # BUG FIX: Atomic status update - mark as publishing before processing
                if not await self._mark_post_publishing(post_id):
                    logger.warning(f"[{self.worker_id}] Post {post_id} already being processed, skipping")
                    return
                await self._publish_scheduled_post(post_id, event.correlation_id)
        else:
            # Direct publish request - no status to update
            await self._run_publish_pipeline(event.payload, event.correlation_id)
    
    async def _publish_scheduled_post(self, post_id: str, correlation_id: str) -> None:
        """Publish a scheduled post by ID."""
        try:
            # Get post details from database
            post_data = await self._get_scheduled_post(post_id)
            if not post_data:
                raise ValueError(f"Scheduled post {post_id} not found")
            
            await self._run_publish_pipeline(post_data, correlation_id)
            
        except Exception as e:
            logger.error(f"[{self.worker_id}] Failed to publish scheduled post {post_id}: {e}")
            await self.emit(
                Topics.PUBLISH_FAILED,
                {"post_id": post_id, "error": str(e)},
                correlation_id
            )
    
    async def _run_publish_pipeline(self, payload: Dict[str, Any], correlation_id: str) -> Dict[str, Any]:
        """
        Run the full publish pipeline with progress events.
        
        Steps:
            1. Verify (0-10%)
            2. Upload to cloud (10-30%)
            3. Upload to Blotato (30-50%)
            4. Submit to platform (50-70%)
            5. Poll for URL (70-100%)
        """
        media_id = payload.get("media_id") or payload.get("content_id")
        platform = payload.get("platform")
        account_id = payload.get("account_id")
        
        try:
            # Emit started event
            await self.emit(
                Topics.PUBLISH_STARTED,
                {
                    "media_id": media_id,
                    "platform": platform,
                    "account_id": account_id,
                    "step": "initializing"
                },
                correlation_id
            )
            
            # Step 0: Duplicate Check (prevent posting same content twice)
            await self.emit_progress("publish", 2, "checking_duplicates", correlation_id, media_id=media_id)
            duplicate_check = await self._check_for_duplicates(payload)
            if duplicate_check.get("is_duplicate"):
                raise ValueError(
                    f"Duplicate content detected: {duplicate_check.get('reason')} "
                    f"(similarity: {duplicate_check.get('similarity_score', 0):.0%})"
                )
            await self.emit_progress("publish", 5, "duplicate_check_passed", correlation_id, media_id=media_id)
            
            # Step 1: Verify
            await self.emit_progress("publish", 7, "verifying", correlation_id, media_id=media_id)
            verification = await self._verify_publish_request(payload)
            if not verification.get("valid"):
                raise ValueError(verification.get("error", "Verification failed"))
            await self.emit_progress("publish", 10, "verified", correlation_id, media_id=media_id)
            
            # Step 2: Upload to cloud
            await self.emit(
                Topics.PUBLISH_UPLOADING,
                {"media_id": media_id, "target": "cloud_storage"},
                correlation_id
            )
            await self.emit_progress("publish", 15, "uploading_cloud", correlation_id, media_id=media_id)
            cloud_url = await self._upload_to_cloud(verification.get("file_path"))
            await self.emit_progress("publish", 30, "cloud_uploaded", correlation_id, media_id=media_id)
            
            # Step 3: Upload to Blotato
            await self.emit(
                Topics.PUBLISH_UPLOADING,
                {"media_id": media_id, "target": "blotato"},
                correlation_id
            )
            await self.emit_progress("publish", 35, "uploading_blotato", correlation_id, media_id=media_id)
            blotato_media_id = await self._upload_to_blotato(cloud_url, account_id)
            await self.emit(
                Topics.PUBLISH_UPLOAD_COMPLETED,
                {"media_id": media_id, "blotato_media_id": blotato_media_id},
                correlation_id
            )
            await self.emit_progress("publish", 50, "blotato_uploaded", correlation_id, media_id=media_id)
            
            # Step 3.5: Auto-generate metadata if not provided (REQ-PUBLISH-002, ARCH-003)
            caption = payload.get("caption", "")
            title = payload.get("title", "")
            hashtags = payload.get("hashtags", [])

            # ARCH-003: Wire Content Analyzer → Publisher Integration
            # If analysis was provided by upstream (e.g., from Sora pipeline), use it directly
            if payload.get("analysis") and not caption:
                analysis = payload["analysis"]
                logger.info(f"[{self.worker_id}] Using pre-computed analysis for {media_id}")

                # Build caption from analysis
                caption = self._build_platform_caption(analysis, platform)
                if not title:
                    title = analysis.get("detected_hook", "")
                if not hashtags:
                    hashtags = analysis.get("hashtags", [])

                payload["generated_metadata"] = {
                    "caption": caption,
                    "title": title,
                    "hashtags": hashtags,
                    "viral_score": analysis.get("viral_score", 0),
                    "source": "pipeline_analysis"
                }
                logger.info(f"[{self.worker_id}] Using pipeline analysis: viral_score={analysis.get('viral_score', 0)}")

            # Fallback: Generate metadata if still not provided
            elif not caption and payload.get("auto_generate_metadata", True):
                await self.emit_progress("publish", 52, "generating_metadata", correlation_id, media_id=media_id)
                generated_metadata = await self._generate_ai_metadata(media_id, platform, payload)
                if generated_metadata:
                    caption = generated_metadata.get("caption", "")
                    title = generated_metadata.get("title", title)
                    hashtags = generated_metadata.get("hashtags", hashtags)
                    # Store generated metadata for reference
                    payload["generated_metadata"] = generated_metadata
                    logger.info(f"[{self.worker_id}] Auto-generated caption for {media_id}: {caption[:50]}...")
            
            # Step 4: Submit to platform
            await self.emit(
                Topics.PUBLISH_SUBMITTED,
                {"media_id": media_id, "platform": platform},
                correlation_id
            )
            await self.emit_progress("publish", 55, "submitting", correlation_id, media_id=media_id)
            submission_id = await self._submit_to_platform(
                blotato_media_id,
                account_id,
                caption,
                platform
            )
            await self.emit_progress("publish", 70, "submitted", correlation_id, media_id=media_id)
            
            # Step 5: Poll for URL
            await self.emit(
                Topics.PUBLISH_POLLING,
                {"media_id": media_id, "submission_id": submission_id},
                correlation_id
            )
            await self.emit_progress("publish", 75, "polling", correlation_id, media_id=media_id)
            platform_url = await self._poll_for_url(submission_id, correlation_id)
            await self.emit_progress("publish", 100, "complete", correlation_id, media_id=media_id)
            
            # Emit completion
            result = {
                "media_id": media_id,
                "platform": platform,
                "platform_url": platform_url,
                "submission_id": submission_id,
                "blotato_media_id": blotato_media_id,
                "completed_at": datetime.now(timezone.utc).isoformat()
            }
            
            await self.emit(Topics.PUBLISH_COMPLETED, result, correlation_id)
            
            # Register content fingerprint for future duplicate detection
            await self._register_content_fingerprint(media_id, account_id, platform)
            
            return {"success": True, **result}
            
        except Exception as e:
            logger.error(f"[{self.worker_id}] Publish failed for {media_id}: {e}")
            
            await self.emit(
                Topics.PUBLISH_FAILED,
                {
                    "media_id": media_id,
                    "platform": platform,
                    "error": str(e),
                    "failed_at": datetime.now(timezone.utc).isoformat()
                },
                correlation_id
            )
            raise
    
    async def _mark_post_publishing(self, post_id: str) -> bool:
        """Atomically mark post as publishing (idempotency check)."""
        try:
            from sqlalchemy import create_engine, text
            import os
            
            DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:54322/postgres")
            engine = create_engine(DATABASE_URL)
            
            with engine.connect() as conn:
                # BUG FIX: Atomic update - only if status is 'scheduled'
                result = conn.execute(text("""
                    UPDATE scheduled_posts
                    SET status = 'publishing', updated_at = NOW()
                    WHERE id = :id AND status = 'scheduled'
                    RETURNING id
                """), {"id": post_id})
                conn.commit()
                
                # If we updated a row, we got the lock
                return result.rowcount > 0
        except Exception as e:
            logger.error(f"Error marking post as publishing: {e}")
            return False
    
    async def _verify_publish_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Verify the publish request has all required data (validation)."""
        media_id = payload.get("media_id") or payload.get("content_id")
        account_id = payload.get("account_id")
        platform = payload.get("platform")
        
        # BUG FIX: Enhanced validation
        if not media_id:
            return {"valid": False, "error": "Missing media_id"}
        if not account_id:
            return {"valid": False, "error": "Missing account_id"}
        if not platform:
            return {"valid": False, "error": "Missing platform"}
        
        # Get file path
        file_path = await self._get_video_path(media_id)
        if not file_path:
            return {"valid": False, "error": f"Video file not found for {media_id}"}
        
        # BUG FIX: Enhanced file verification
        import os
        from pathlib import Path
        
        path = Path(file_path)
        if not path.exists():
            return {
                "valid": False,
                "error": f"File does not exist: {file_path}",
                "file_path": str(path)
            }
        
        if not path.is_file():
            return {
                "valid": False,
                "error": f"Path is not a file: {file_path}",
                "file_path": str(path)
            }
        
        if not os.access(file_path, os.R_OK):
            return {
                "valid": False,
                "error": f"File is not readable: {file_path}",
                "file_path": str(path)
            }
        
        return {
            "valid": True,
            "file_path": file_path,
            "file_size": path.stat().st_size
        }
    
    async def _upload_to_cloud(self, file_path: str) -> str:
        """Upload video to cloud storage (Google Drive or Supabase)."""
        service = self._get_publish_service()
        if service:
            result = await asyncio.to_thread(service.upload_to_cloud, file_path)
            return result.get("url") if result else None
        raise ValueError("Publish service not available")
    
    async def _upload_to_blotato(self, cloud_url: str, account_id: str) -> str:
        """Upload video to Blotato."""
        service = self._get_publish_service()
        if service:
            result = await asyncio.to_thread(service.upload_to_blotato, cloud_url, account_id)
            return result.get("media_id") if result else None
        raise ValueError("Publish service not available")
    
    async def _submit_to_platform(
        self,
        blotato_media_id: str,
        account_id: str,
        caption: str,
        platform: str
    ) -> str:
        """Submit content to platform via Blotato."""
        service = self._get_publish_service()
        if service:
            result = await asyncio.to_thread(
                service.publish_to_platform,
                blotato_media_id,
                account_id,
                caption,
                platform
            )
            return result.get("submission_id") if result else None
        raise ValueError("Publish service not available")
    
    async def _poll_for_url(self, submission_id: str, correlation_id: str) -> Optional[str]:
        """Poll Blotato for platform URL."""
        service = self._get_publish_service()
        if not service:
            return None
        
        max_attempts = 30
        poll_interval = 5
        
        for attempt in range(1, max_attempts + 1):
            result = await asyncio.to_thread(service.get_post_status, submission_id)
            
            if result and result.get("status") == "published":
                return result.get("platform_url")
            
            # Emit polling progress
            if attempt % 5 == 0:
                await self.emit(
                    Topics.PUBLISH_POLLING,
                    {
                        "submission_id": submission_id,
                        "attempt": attempt,
                        "max_attempts": max_attempts
                    },
                    correlation_id
                )
            
            await asyncio.sleep(poll_interval)
        
        logger.warning(f"URL polling timed out for {submission_id}")
        return None
    
    def _get_publish_service(self):
        """Lazy load the publish service."""
        if self._publish_service is None:
            try:
                from services.publish_service import PublishService
                self._publish_service = PublishService()
            except Exception as e:
                logger.warning(f"Could not load PublishService: {e}")
        return self._publish_service
    
    async def _get_video_path(self, media_id: str) -> Optional[str]:
        """Get video file path from database."""
        try:
            from sqlalchemy import create_engine, text
            import os
            
            DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:54322/postgres")
            engine = create_engine(DATABASE_URL)
            
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT source_uri FROM videos WHERE id = :id"),
                    {"id": media_id}
                ).fetchone()
                
                if result and result[0]:
                    return result[0]
            return None
        except Exception as e:
            logger.warning(f"Could not get video path: {e}")
            return None
    
    async def _check_for_duplicates(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Check if content is a duplicate before publishing."""
        try:
            account_id = payload.get("account_id")
            platform = payload.get("platform")
            media_id = payload.get("media_id") or payload.get("content_id")
            
            # Get transcript for the video
            transcript = await self._get_video_transcript(media_id)
            
            if not transcript or len(transcript) < 10:
                logger.debug(f"No transcript for {media_id}, skipping duplicate check")
                return {"is_duplicate": False, "reason": "No transcript available"}
            
            detector = DuplicateDetector()
            result = await detector.check_content(
                account_id=str(account_id),
                transcript=transcript,
                platform=platform
            )
            
            logger.info(f"Duplicate check for {media_id}: is_duplicate={result.is_duplicate}, score={result.similarity_score:.2f}")
            
            return {
                "is_duplicate": result.is_duplicate,
                "similarity_score": result.similarity_score,
                "reason": result.reason,
                "similar_post_id": result.similar_post_id
            }
        except Exception as e:
            logger.warning(f"Duplicate check failed (allowing publish): {e}")
            return {"is_duplicate": False, "reason": f"Check failed: {e}"}
    
    async def _register_content_fingerprint(
        self,
        media_id: str,
        account_id: str,
        platform: str
    ) -> bool:
        """Register content fingerprint after successful publish."""
        try:
            transcript = await self._get_video_transcript(media_id)
            if not transcript:
                return False
            
            detector = DuplicateDetector()
            return await detector.register_posted_content(
                content_id=str(media_id),
                account_id=str(account_id),
                platform=platform,
                transcript=transcript
            )
        except Exception as e:
            logger.warning(f"Failed to register fingerprint: {e}")
            return False
    
    async def _get_video_transcript(self, media_id: str) -> Optional[str]:
        """Get video transcript from database."""
        try:
            from sqlalchemy import create_engine, text
            import os
            
            DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:54322/postgres")
            engine = create_engine(DATABASE_URL)
            
            with engine.connect() as conn:
                # Try multiple tables for transcript
                result = conn.execute(
                    text("""
                        SELECT COALESCE(transcript, transcript_full, '')
                        FROM videos v
                        LEFT JOIN analyzed_videos av ON v.id = av.original_video_id
                        WHERE v.id = :id
                        LIMIT 1
                    """),
                    {"id": media_id}
                ).fetchone()
                
                if result and result[0]:
                    return result[0]
            return None
        except Exception as e:
            logger.warning(f"Could not get video transcript: {e}")
            return None
    
    async def _generate_ai_metadata(
        self,
        media_id: str,
        platform: str,
        payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate AI-powered titles, descriptions, and hashtags for content.
        
        REQ-PUBLISH-002: Auto-inject AI titles/descriptions into publish payload.
        
        Uses ContentAnalyzer if transcript is available, otherwise generates
        from any provided context (theme, title hints, etc.)
        """
        try:
            # First try to get existing analysis from database
            transcript = await self._get_video_transcript(media_id)
            
            if transcript:
                # Use ContentAnalyzer for full analysis
                try:
                    from services.content_analyzer import ContentAnalyzer
                    analyzer = ContentAnalyzer()
                    analysis = analyzer.analyze_transcript(transcript)
                    
                    # Build platform-specific caption
                    caption = self._build_platform_caption(analysis, platform)
                    
                    return {
                        "caption": caption,
                        "title": analysis.get("detected_hook", ""),
                        "hashtags": analysis.get("hashtags", []),
                        "hook": analysis.get("detected_hook", ""),
                        "viral_score": analysis.get("viral_score", 0),
                        "source": "content_analyzer"
                    }
                except Exception as e:
                    logger.warning(f"ContentAnalyzer failed: {e}")
            
            # Fallback: Generate from theme/context if provided
            theme = payload.get("theme") or payload.get("title") or ""
            if theme:
                return await self._generate_metadata_from_theme(theme, platform)
            
            # Last resort: Generic caption
            return {
                "caption": "Check out this video! 🔥 #viral #trending",
                "title": "",
                "hashtags": ["viral", "trending", "fyp"],
                "source": "fallback"
            }
            
        except Exception as e:
            logger.warning(f"AI metadata generation failed: {e}")
            return None
    
    def _build_platform_caption(self, analysis: Dict, platform: str) -> str:
        """Build platform-optimized caption from analysis."""
        hook = analysis.get("detected_hook", "")
        description = analysis.get("suggested_description", "")
        hashtags = analysis.get("hashtags", [])
        cta = analysis.get("cta", "Follow for more!")
        
        # Platform-specific formatting
        if platform == "tiktok":
            # TikTok: Short, punchy, hashtag-heavy
            caption = f"{hook}\n\n{cta}"
            if hashtags:
                caption += "\n\n" + " ".join([f"#{h}" for h in hashtags[:10]])
            return caption[:2200]  # TikTok limit
            
        elif platform == "instagram":
            # Instagram: Longer form okay, structured
            caption = f"{hook}\n\n{description}\n\n{cta}"
            if hashtags:
                caption += "\n\n" + " ".join([f"#{h}" for h in hashtags[:30]])
            return caption[:2200]
            
        elif platform == "youtube":
            # YouTube: SEO-focused title + description
            caption = f"{hook}\n\n{description}\n\n{cta}"
            if hashtags:
                caption += "\n\n" + " ".join([f"#{h}" for h in hashtags[:15]])
            return caption[:5000]
            
        elif platform == "twitter":
            # Twitter: Very short
            caption = f"{hook}"
            if hashtags:
                caption += " " + " ".join([f"#{h}" for h in hashtags[:3]])
            return caption[:280]
            
        else:
            # Default format
            caption = f"{hook}\n\n{cta}"
            if hashtags:
                caption += "\n\n" + " ".join([f"#{h}" for h in hashtags[:10]])
            return caption[:2000]
    
    async def _generate_metadata_from_theme(self, theme: str, platform: str) -> Dict[str, Any]:
        """Generate metadata from theme using AI."""
        try:
            from openai import OpenAI
            import os
            import json
            
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"""Generate social media metadata for {platform}.
Return JSON with: caption, title, hashtags (array), hook, cta"""
                    },
                    {
                        "role": "user",
                        "content": f"Create engaging {platform} caption for video about: {theme}"
                    }
                ],
                temperature=0.8,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            result["source"] = "ai_generated"
            return result
            
        except Exception as e:
            logger.warning(f"Theme-based metadata generation failed: {e}")
            return {
                "caption": f"Check out this video about {theme}! 🔥 #viral",
                "title": theme,
                "hashtags": ["viral", "trending"],
                "source": "fallback"
            }
    
    async def _get_scheduled_post(self, post_id: str) -> Optional[Dict[str, Any]]:
        """Get scheduled post details from database."""
        try:
            from sqlalchemy import create_engine, text
            import os
            
            DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:54322/postgres")
            engine = create_engine(DATABASE_URL)
            
            with engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT content_id, platform, account_id, caption, hashtags
                        FROM scheduled_posts WHERE id = :id
                    """),
                    {"id": post_id}
                ).fetchone()
                
                if result:
                    return {
                        "media_id": result[0],
                        "platform": result[1],
                        "account_id": result[2],
                        "caption": result[3],
                        "hashtags": result[4]
                    }
            return None
        except Exception as e:
            logger.warning(f"Could not get scheduled post: {e}")
            return None


# Convenience function to create and start worker
async def start_publish_worker(event_bus: Optional[EventBus] = None) -> PublishWorker:
    """Create and start a publish worker."""
    worker = PublishWorker(event_bus)
    await worker.start()
    return worker
