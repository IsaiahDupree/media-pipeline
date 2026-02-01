"""
Clip Extractor Service
======================
Extracts engaging short-form clips from long-form videos.

Based on SupoClip approach:
1. Analyze transcript for engaging segments
2. Score segments for viral potential
3. Smart crop for 9:16 format
4. Generate clips with subtitles
"""

import os
import logging
import json
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """A segment of transcript identified as clip-worthy."""
    id: str = field(default_factory=lambda: str(uuid4()))
    start_time: float = 0.0  # seconds
    end_time: float = 0.0
    text: str = ""
    relevance_score: float = 0.0
    reasoning: str = ""
    hooks: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "text": self.text,
            "relevance_score": self.relevance_score,
            "reasoning": self.reasoning,
            "hooks": self.hooks,
            "topics": self.topics
        }


@dataclass
class CropRegion:
    """Smart crop region for vertical video."""
    x_offset: int = 0
    y_offset: int = 0
    width: int = 1080
    height: int = 1920
    face_detected: bool = False
    confidence: float = 0.0


@dataclass
class ExtractedClip:
    """An extracted clip ready for posting."""
    id: str = field(default_factory=lambda: str(uuid4()))
    source_video_id: str = ""
    segment: TranscriptSegment = None
    crop_region: CropRegion = None
    output_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    status: str = "pending"  # pending, processing, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    
    # Narrative metadata
    pillar: Optional[str] = None
    suggested_caption: str = ""
    suggested_hashtags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_video_id": self.source_video_id,
            "segment": self.segment.to_dict() if self.segment else None,
            "crop_region": {
                "x_offset": self.crop_region.x_offset if self.crop_region else 0,
                "y_offset": self.crop_region.y_offset if self.crop_region else 0,
                "width": self.crop_region.width if self.crop_region else 1080,
                "height": self.crop_region.height if self.crop_region else 1920,
            } if self.crop_region else None,
            "output_path": self.output_path,
            "thumbnail_path": self.thumbnail_path,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "pillar": self.pillar,
            "suggested_caption": self.suggested_caption,
            "suggested_hashtags": self.suggested_hashtags
        }


@dataclass
class ExtractionConfig:
    """Configuration for clip extraction."""
    min_clip_duration: float = 10.0  # seconds
    max_clip_duration: float = 60.0
    optimal_duration: float = 30.0
    target_clips: int = 5
    min_relevance_score: float = 0.6
    output_format: str = "mp4"
    output_width: int = 1080
    output_height: int = 1920
    video_bitrate: str = "8000k"
    audio_bitrate: str = "256k"


class ClipExtractor:
    """
    Service for extracting clips from long-form videos.
    
    Pipeline:
    1. Load video and transcript
    2. Analyze transcript for engaging segments
    3. Score and rank segments
    4. Detect optimal crop regions
    5. Extract and encode clips
    6. Generate subtitles (optional)
    """
    
    def __init__(
        self,
        config: Optional[ExtractionConfig] = None,
        openai_api_key: Optional[str] = None
    ):
        self.config = config or ExtractionConfig()
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    
    async def analyze_transcript_for_clips(
        self,
        transcript: str,
        video_duration: float,
        topics: Optional[List[str]] = None
    ) -> List[TranscriptSegment]:
        """
        Analyze transcript to identify clip-worthy segments.
        
        Uses AI to find:
        - Strong hooks (attention-grabbing openings)
        - Valuable content (tips, insights)
        - Emotional moments (excitement, humor)
        - Complete thoughts (self-contained ideas)
        """
        if not self.openai_api_key:
            return self._basic_segment_analysis(transcript, video_duration)
        
        try:
            import openai
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            prompt = f"""Analyze this video transcript and identify 3-7 segments that would make great short-form clips (10-60 seconds).

TRANSCRIPT:
{transcript[:4000]}

VIDEO DURATION: {video_duration:.1f} seconds

TOPICS: {', '.join(topics or ['general'])}

For each segment, identify:
1. Start and end timestamps (estimate based on word count, ~150 words/minute)
2. The exact text of the segment
3. Relevance score (0.0-1.0) based on viral potential
4. Why this segment would work as a clip
5. Potential hooks in the segment

Selection Criteria:
- Strong hooks (attention-grabbing openings)
- Valuable content (tips, insights, interesting facts)
- Emotional moments (excitement, surprise, humor)
- Complete thoughts (self-contained ideas that don't need context)
- Optimal duration: 15-45 seconds

Return as JSON array:
[
  {{
    "start_time": 45.0,
    "end_time": 75.0,
    "text": "The exact transcript text...",
    "relevance_score": 0.85,
    "reasoning": "Strong hook with actionable tip",
    "hooks": ["Opening question", "Promise of value"]
  }}
]"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at identifying viral-worthy segments in video content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            result_text = response.choices[0].message.content
            
            # Parse JSON response
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            segments_data = json.loads(result_text.strip())
            
            segments = []
            for data in segments_data:
                segment = TranscriptSegment(
                    start_time=data.get("start_time", 0),
                    end_time=data.get("end_time", 30),
                    text=data.get("text", ""),
                    relevance_score=data.get("relevance_score", 0.5),
                    reasoning=data.get("reasoning", ""),
                    hooks=data.get("hooks", [])
                )
                
                # Validate duration
                if self.config.min_clip_duration <= segment.duration <= self.config.max_clip_duration:
                    if segment.relevance_score >= self.config.min_relevance_score:
                        segments.append(segment)
            
            # Sort by relevance score
            segments.sort(key=lambda s: s.relevance_score, reverse=True)
            
            return segments[:self.config.target_clips]
            
        except Exception as e:
            logger.error(f"AI segment analysis failed: {e}")
            return self._basic_segment_analysis(transcript, video_duration)
    
    def _basic_segment_analysis(
        self,
        transcript: str,
        video_duration: float
    ) -> List[TranscriptSegment]:
        """Fallback: basic segment analysis without AI."""
        segments = []
        
        # Split into sentences
        sentences = transcript.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return segments
        
        # Estimate words per minute (150 wpm standard)
        total_words = len(transcript.split())
        words_per_second = total_words / video_duration if video_duration > 0 else 2.5
        
        # Create segments of ~30 seconds
        current_start = 0
        current_words = 0
        current_text = []
        
        target_words = int(self.config.optimal_duration * words_per_second)
        
        for sentence in sentences:
            words = len(sentence.split())
            
            if current_words + words > target_words and current_text:
                # Create segment
                end_time = min(current_start + (current_words / words_per_second), video_duration)
                
                segments.append(TranscriptSegment(
                    start_time=current_start,
                    end_time=end_time,
                    text=" ".join(current_text),
                    relevance_score=0.6,  # Default score
                    reasoning="Automatic segmentation"
                ))
                
                current_start = end_time
                current_text = [sentence]
                current_words = words
            else:
                current_text.append(sentence)
                current_words += words
        
        # Add final segment
        if current_text:
            segments.append(TranscriptSegment(
                start_time=current_start,
                end_time=video_duration,
                text=" ".join(current_text),
                relevance_score=0.5,
                reasoning="Final segment"
            ))
        
        return segments[:self.config.target_clips]
    
    def detect_crop_region(
        self,
        video_path: str,
        segment: TranscriptSegment
    ) -> CropRegion:
        """
        Detect optimal crop region using face detection.
        
        Target: 9:16 vertical format
        """
        try:
            import cv2
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return CropRegion()
            
            # Get video dimensions
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate 9:16 crop dimensions
            target_ratio = 9 / 16
            current_ratio = width / height
            
            if current_ratio > target_ratio:
                # Video is wider than target - crop sides
                new_width = int(height * target_ratio)
                new_height = height
            else:
                # Video is taller than target - crop top/bottom
                new_width = width
                new_height = int(width / target_ratio)
            
            # Sample frames for face detection
            face_positions = []
            sample_interval = max(0.5, segment.duration / 10)  # Sample every 0.5s or 10 samples
            
            # Load face cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            current_time = segment.start_time
            while current_time < segment.end_time:
                cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
                ret, frame = cap.read()
                
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    for (x, y, w, h) in faces:
                        center_x = x + w // 2
                        center_y = y + h // 2
                        area = w * h
                        face_positions.append((center_x, center_y, area))
                
                current_time += sample_interval
            
            cap.release()
            
            # Calculate crop offset
            if face_positions:
                # Weighted average of face positions
                total_weight = sum(area for _, _, area in face_positions)
                weighted_x = sum(x * area for x, _, area in face_positions) / total_weight
                weighted_y = sum(y * area for _, y, area in face_positions) / total_weight
                
                # Center crop on faces with bias towards upper portion
                x_offset = int(max(0, min(weighted_x - new_width // 2, width - new_width)))
                y_offset = int(max(0, min(weighted_y - new_height // 2 - new_height * 0.1, height - new_height)))
                
                return CropRegion(
                    x_offset=x_offset,
                    y_offset=y_offset,
                    width=new_width,
                    height=new_height,
                    face_detected=True,
                    confidence=len(face_positions) / 10
                )
            else:
                # Center crop
                x_offset = (width - new_width) // 2
                y_offset = (height - new_height) // 2
                
                return CropRegion(
                    x_offset=x_offset,
                    y_offset=y_offset,
                    width=new_width,
                    height=new_height,
                    face_detected=False,
                    confidence=0.5
                )
                
        except ImportError:
            logger.warning("OpenCV not available for face detection")
            return CropRegion()
        except Exception as e:
            logger.error(f"Crop detection failed: {e}")
            return CropRegion()
    
    async def extract_clip(
        self,
        video_path: str,
        segment: TranscriptSegment,
        output_dir: str,
        crop_region: Optional[CropRegion] = None,
        add_subtitles: bool = False
    ) -> ExtractedClip:
        """
        Extract a single clip from video.
        
        Uses ffmpeg for efficient extraction and encoding.
        """
        clip = ExtractedClip(
            source_video_id=Path(video_path).stem,
            segment=segment,
            crop_region=crop_region or CropRegion(),
            status="processing"
        )
        
        try:
            output_path = os.path.join(
                output_dir,
                f"clip_{clip.id[:8]}_{int(segment.start_time)}-{int(segment.end_time)}.{self.config.output_format}"
            )
            
            # Build ffmpeg command
            cmd = self._build_ffmpeg_command(
                video_path=video_path,
                output_path=output_path,
                start_time=segment.start_time,
                duration=segment.duration,
                crop_region=clip.crop_region
            )
            
            # Execute ffmpeg
            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                clip.output_path = output_path
                clip.status = "completed"
                
                # Generate thumbnail
                thumb_path = output_path.replace(f".{self.config.output_format}", "_thumb.jpg")
                self._generate_thumbnail(output_path, thumb_path)
                clip.thumbnail_path = thumb_path
                
                logger.info(f"Clip extracted: {output_path}")
            else:
                clip.status = "failed"
                logger.error(f"FFmpeg failed: {result.stderr[:200]}")
                
        except Exception as e:
            clip.status = "failed"
            logger.error(f"Clip extraction failed: {e}")
        
        return clip
    
    def _build_ffmpeg_command(
        self,
        video_path: str,
        output_path: str,
        start_time: float,
        duration: float,
        crop_region: CropRegion
    ) -> List[str]:
        """Build ffmpeg command for clip extraction."""
        crop_filter = f"crop={crop_region.width}:{crop_region.height}:{crop_region.x_offset}:{crop_region.y_offset}"
        scale_filter = f"scale={self.config.output_width}:{self.config.output_height}"
        
        return [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", video_path,
            "-t", str(duration),
            "-vf", f"{crop_filter},{scale_filter}",
            "-c:v", "libx264",
            "-b:v", self.config.video_bitrate,
            "-c:a", "aac",
            "-b:a", self.config.audio_bitrate,
            "-preset", "medium",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            output_path
        ]
    
    def _generate_thumbnail(self, video_path: str, thumb_path: str):
        """Generate thumbnail from video."""
        try:
            import subprocess
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-ss", "1",
                "-vframes", "1",
                "-q:v", "2",
                thumb_path
            ]
            subprocess.run(cmd, capture_output=True)
        except Exception as e:
            logger.warning(f"Thumbnail generation failed: {e}")
    
    async def extract_clips_from_video(
        self,
        video_id: str,
        video_path: str,
        transcript: str,
        video_duration: float,
        output_dir: str,
        topics: Optional[List[str]] = None
    ) -> List[ExtractedClip]:
        """
        Full pipeline: analyze and extract all clips from a video.
        """
        logger.info(f"Starting clip extraction for video {video_id}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Analyze transcript for segments
        segments = await self.analyze_transcript_for_clips(
            transcript=transcript,
            video_duration=video_duration,
            topics=topics
        )
        
        logger.info(f"Found {len(segments)} clip-worthy segments")
        
        clips = []
        for segment in segments:
            # Step 2: Detect crop region
            crop_region = self.detect_crop_region(video_path, segment)
            
            # Step 3: Extract clip
            clip = await self.extract_clip(
                video_path=video_path,
                segment=segment,
                output_dir=output_dir,
                crop_region=crop_region
            )
            
            clip.source_video_id = video_id
            clips.append(clip)
        
        completed = sum(1 for c in clips if c.status == "completed")
        logger.info(f"Clip extraction complete: {completed}/{len(clips)} successful")
        
        return clips
