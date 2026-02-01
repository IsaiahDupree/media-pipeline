"""
YouTube Publisher for Sora Daily Automation
Publishes processed videos to YouTube with AI-generated metadata.
"""

import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger


YOUTUBE_ACCOUNT_ID = "228"  # Isaiah Dupree YouTube


@dataclass
class YouTubeUploadJob:
    """YouTube upload job details."""
    video_path: str
    title: str
    description: str
    tags: List[str]
    playlist_id: Optional[str] = None
    scheduled_time: Optional[datetime] = None
    is_short: bool = False
    category_id: str = "22"  # People & Blogs


class YouTubePublisher:
    """
    Publishes Sora videos to YouTube.
    
    Uses Blotato for actual upload via account ID 228.
    """
    
    def __init__(self):
        self.account_id = YOUTUBE_ACCOUNT_ID
        logger.info(f"✅ YouTubePublisher initialized (Account: {self.account_id})")
    
    async def generate_metadata(
        self,
        prompt: str,
        theme: str,
        is_movie_part: bool = False,
        part_number: int = 0,
        trend: Optional[str] = None
    ) -> Dict:
        """
        Generate YouTube metadata (title, description, tags) using AI.
        """
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            content_type = f"Part {part_number} of 3-part story" if is_movie_part else "standalone video"
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"""Generate YouTube metadata for an AI-generated video.

Video Type: {content_type}
Theme: {theme}
{f'Trending Topic: {trend}' if trend else ''}
Original Prompt: {prompt[:200]}

Return JSON with:
{{
    "title": "engaging title under 100 chars, include emoji",
    "description": "2-3 sentence description with CTA",
    "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"]
}}

Make it engaging for YouTube algorithm. Include relevant hashtags in description."""
                    },
                    {
                        "role": "user",
                        "content": "Generate the metadata."
                    }
                ],
                temperature=0.8,
                response_format={"type": "json_object"}
            )
            
            import json
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Metadata generation failed: {e}")
            # Fallback metadata
            return {
                "title": f"✨ {theme.replace('_', ' ').title()} | AI Generated",
                "description": f"AI-generated video exploring {theme}. Created with Sora.\n\n#AI #Sora #AIVideo",
                "tags": ["AI", "Sora", "AIVideo", theme, "creative"]
            }
    
    async def publish_single(
        self,
        video_path: str,
        prompt: str,
        theme: str,
        trend: Optional[str] = None,
        scheduled_time: Optional[datetime] = None
    ) -> Dict:
        """
        Publish a single standalone video to YouTube.
        """
        metadata = await self.generate_metadata(
            prompt=prompt,
            theme=theme,
            is_movie_part=False,
            trend=trend
        )
        
        return await self._upload_video(
            video_path=video_path,
            title=metadata["title"],
            description=metadata["description"],
            tags=metadata["tags"],
            scheduled_time=scheduled_time
        )
    
    async def publish_movie(
        self,
        video_paths: List[str],
        prompts: List[str],
        theme: str,
        trend: Optional[str] = None,
        scheduled_time: Optional[datetime] = None
    ) -> Dict:
        """
        Publish a 3-part movie to YouTube.
        Creates individual videos and optionally a playlist.
        """
        results = []
        
        for i, (path, prompt) in enumerate(zip(video_paths, prompts), 1):
            metadata = await self.generate_metadata(
                prompt=prompt,
                theme=theme,
                is_movie_part=True,
                part_number=i,
                trend=trend
            )
            
            # Add part number to title
            title = f"Part {i}/3: {metadata['title']}"
            
            # Calculate scheduled time with 5-min spacing
            part_schedule = None
            if scheduled_time:
                part_schedule = scheduled_time + timedelta(minutes=(i-1) * 5)
            
            result = await self._upload_video(
                video_path=path,
                title=title,
                description=metadata["description"],
                tags=metadata["tags"],
                scheduled_time=part_schedule
            )
            
            results.append(result)
        
        return {
            "success": all(r.get("success") for r in results),
            "parts": results,
            "theme": theme
        }
    
    async def _upload_video(
        self,
        video_path: str,
        title: str,
        description: str,
        tags: List[str],
        scheduled_time: Optional[datetime] = None,
        playlist_id: Optional[str] = None
    ) -> Dict:
        """
        Upload video to YouTube via Blotato.
        """
        try:
            # Use Blotato API for upload
            import aiohttp
            
            blotato_url = os.getenv("BLOTATO_URL", "http://localhost:16987")
            
            payload = {
                "account_id": int(self.account_id),
                "video_path": video_path,
                "title": title[:100],  # YouTube limit
                "description": description[:5000],
                "tags": tags[:500] if isinstance(tags, list) else tags,
                "scheduled_time": scheduled_time.isoformat() if scheduled_time else None
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{blotato_url}/api/youtube/upload",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"✅ Uploaded to YouTube: {title[:50]}...")
                        return {
                            "success": True,
                            "video_id": result.get("video_id"),
                            "url": result.get("url"),
                            "title": title
                        }
                    else:
                        error = await response.text()
                        logger.error(f"YouTube upload failed: {error}")
                        return {
                            "success": False,
                            "error": error,
                            "title": title
                        }
                        
        except Exception as e:
            logger.error(f"YouTube upload error: {e}")
            return {
                "success": False,
                "error": str(e),
                "title": title
            }
    
    def get_publishing_schedule(self, singles_count: int, movies_count: int) -> List[Dict]:
        """
        Generate optimal publishing schedule for the day.
        
        Singles: Spread throughout day
        Movies: 3-part bunches at peak times
        """
        from datetime import time
        
        schedule = []
        
        # Peak times for YouTube
        single_times = [
            time(6, 0), time(8, 0), time(10, 0), time(12, 0),
            time(14, 0), time(16, 0), time(18, 0), time(20, 0),
            time(21, 0), time(22, 0)
        ]
        
        movie_times = [
            time(9, 0), time(13, 0), time(17, 0), time(19, 0)
        ]
        
        # Schedule singles
        for i in range(min(singles_count, len(single_times))):
            schedule.append({
                "type": "single",
                "index": i,
                "time": single_times[i].isoformat()
            })
        
        # Schedule movies
        for i in range(min(movies_count, len(movie_times))):
            schedule.append({
                "type": "movie",
                "index": i,
                "time": movie_times[i].isoformat()
            })
        
        return schedule


# =============================================================================
# SINGLETON
# =============================================================================

_publisher_instance: Optional[YouTubePublisher] = None

def get_youtube_publisher() -> YouTubePublisher:
    """Get singleton instance of YouTubePublisher."""
    global _publisher_instance
    if _publisher_instance is None:
        _publisher_instance = YouTubePublisher()
    return _publisher_instance
