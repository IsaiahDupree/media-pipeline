"""
Clip Extraction Service
=======================
Extracts engaging short-form clips from long-form videos.

Based on SupoClip architecture:
1. Transcribe video with word-level timing (AssemblyAI)
2. AI analysis to find compelling segments (configurable provider)
3. Smart crop with face detection
4. Render clips with subtitles

AI Provider Configuration:
    Set AI_PROVIDER=openai|mock in environment
    Set OPENAI_API_KEY for OpenAI
    Use mock provider for testing

Usage:
    service = ClipExtractionService()
    result = await service.extract_clips(video_path, output_dir)
    
    # For testing with mock provider:
    service = ClipExtractionService(ai_provider="mock")
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """Represents a relevant segment of transcript with timing."""
    start_time: str  # MM:SS format
    end_time: str    # MM:SS format
    text: str
    relevance_score: float = 0.0
    reasoning: str = ""


@dataclass
class ClipResult:
    """Result of a single clip extraction."""
    clip_id: str
    filename: str
    path: str
    start_time: str
    end_time: str
    duration: float
    text: str
    relevance_score: float
    reasoning: str
    thumbnail_path: Optional[str] = None


@dataclass
class ExtractionResult:
    """Full extraction job result."""
    job_id: str
    source_video: str
    clips: List[ClipResult]
    transcript_text: str
    key_topics: List[str]
    summary: str
    total_duration: float
    processing_time: float
    success: bool
    error: Optional[str] = None


class ClipExtractionService:
    """
    Service for extracting short-form clips from long-form videos.
    
    Pipeline:
        1. Transcription (AssemblyAI with word-level timing)
        2. AI Segment Selection (configurable provider: openai, mock)
        3. Smart Cropping (face-centered 9:16)
        4. Clip Rendering (with subtitles)
    """
    
    def __init__(
        self,
        assemblyai_key: Optional[str] = None,
        ai_provider: Optional[str] = None,
        output_dir: Optional[Path] = None,
        font_family: str = "Arial",
        font_size: int = 24,
        font_color: str = "#FFFFFF"
    ):
        self.assemblyai_key = assemblyai_key or os.getenv("ASSEMBLYAI_API_KEY")
        self.ai_provider_name = ai_provider or os.getenv("AI_PROVIDER", "openai")
        self.output_dir = output_dir or Path("./clips")
        self.font_family = font_family
        self.font_size = font_size
        self.font_color = font_color
        self._ai_provider = None
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_ai_provider(self):
        """Get configured AI provider (lazy load)."""
        if self._ai_provider is None:
            from services.ai_providers import get_ai_provider
            self._ai_provider = get_ai_provider(self.ai_provider_name)
            logger.info(f"Using AI provider: {self._ai_provider.name}")
        return self._ai_provider
    
    async def extract_clips(
        self,
        video_path: str,
        output_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        min_clip_duration: int = 10,
        max_clip_duration: int = 60,
        max_clips: int = 7
    ) -> ExtractionResult:
        """
        Extract engaging clips from a long-form video.
        
        Args:
            video_path: Path to source video
            output_dir: Where to save clips (uses default if not provided)
            progress_callback: Optional callback(progress_pct, step_name)
            min_clip_duration: Minimum clip length in seconds
            max_clip_duration: Maximum clip length in seconds
            max_clips: Maximum number of clips to extract
        
        Returns:
            ExtractionResult with all clips and metadata
        """
        import time
        start_time = time.time()
        job_id = str(uuid4())
        
        output_dir = output_dir or self.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        def emit_progress(pct: int, step: str):
            if progress_callback:
                progress_callback(pct, step)
            logger.info(f"[{job_id[:8]}] {pct}% - {step}")
        
        try:
            video_path = Path(video_path)
            if not video_path.exists():
                raise FileNotFoundError(f"Video not found: {video_path}")
            
            emit_progress(5, "Starting transcription")
            
            # Step 1: Transcription
            transcript_data = await self._transcribe_video(video_path, emit_progress)
            emit_progress(30, "Transcription complete")
            
            # Step 2: AI Segment Selection
            emit_progress(35, "Analyzing transcript for engaging segments")
            segments = await self._identify_segments(
                transcript_data["formatted"],
                min_duration=min_clip_duration,
                max_duration=max_clip_duration,
                max_segments=max_clips
            )
            emit_progress(50, f"Found {len(segments)} engaging segments")
            
            # Step 3: Render clips
            emit_progress(55, "Rendering clips")
            clips = await self._render_clips(
                video_path,
                segments,
                transcript_data["words"],
                output_dir,
                emit_progress
            )
            emit_progress(95, f"Rendered {len(clips)} clips")
            
            processing_time = time.time() - start_time
            emit_progress(100, "Complete")
            
            return ExtractionResult(
                job_id=job_id,
                source_video=str(video_path),
                clips=clips,
                transcript_text=transcript_data["text"],
                key_topics=transcript_data.get("topics", []),
                summary=transcript_data.get("summary", ""),
                total_duration=sum(c.duration for c in clips),
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"[{job_id[:8]}] Extraction failed: {e}")
            return ExtractionResult(
                job_id=job_id,
                source_video=str(video_path),
                clips=[],
                transcript_text="",
                key_topics=[],
                summary="",
                total_duration=0,
                processing_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    async def _transcribe_video(
        self,
        video_path: Path,
        emit_progress: Callable
    ) -> Dict[str, Any]:
        """Transcribe video using AssemblyAI with word-level timing."""
        try:
            import assemblyai as aai
            
            aai.settings.api_key = self.assemblyai_key
            transcriber = aai.Transcriber()
            
            config = aai.TranscriptionConfig(
                speaker_labels=False,
                punctuate=True,
                format_text=True,
                speech_model=aai.SpeechModel.best
            )
            
            emit_progress(10, "Uploading to AssemblyAI")
            transcript = transcriber.transcribe(str(video_path), config=config)
            
            if transcript.status == aai.TranscriptStatus.error:
                raise Exception(f"Transcription failed: {transcript.error}")
            
            emit_progress(25, "Processing transcript")
            
            # Format transcript with timestamps
            formatted_lines = []
            words_data = []
            
            if transcript.words:
                current_segment = []
                current_start = None
                segment_word_count = 0
                max_words_per_segment = 8
                
                for word in transcript.words:
                    words_data.append({
                        'text': word.text,
                        'start': word.start,
                        'end': word.end,
                        'confidence': getattr(word, 'confidence', 1.0)
                    })
                    
                    if current_start is None:
                        current_start = word.start
                    
                    current_segment.append(word.text)
                    segment_word_count += 1
                    
                    if (segment_word_count >= max_words_per_segment or
                        word.text.endswith('.') or word.text.endswith('!') or word.text.endswith('?')):
                        
                        if current_segment:
                            start_ts = self._format_ms_to_timestamp(current_start)
                            end_ts = self._format_ms_to_timestamp(word.end)
                            text = ' '.join(current_segment)
                            formatted_lines.append(f"[{start_ts} - {end_ts}] {text}")
                        
                        current_segment = []
                        current_start = None
                        segment_word_count = 0
                
                # Handle remaining words
                if current_segment and current_start is not None:
                    start_ts = self._format_ms_to_timestamp(current_start)
                    end_ts = self._format_ms_to_timestamp(transcript.words[-1].end)
                    text = ' '.join(current_segment)
                    formatted_lines.append(f"[{start_ts} - {end_ts}] {text}")
            
            return {
                "text": transcript.text or "",
                "formatted": '\n'.join(formatted_lines),
                "words": words_data,
                "topics": [],
                "summary": ""
            }
            
        except ImportError:
            logger.warning("AssemblyAI not installed, using mock transcription")
            return await self._mock_transcribe(video_path)
    
    async def _mock_transcribe(self, video_path: Path) -> Dict[str, Any]:
        """Mock transcription for testing without AssemblyAI."""
        return {
            "text": "This is a mock transcript for testing purposes.",
            "formatted": "[00:00 - 00:30] This is a mock transcript\n[00:30 - 01:00] for testing purposes",
            "words": [
                {"text": "This", "start": 0, "end": 500, "confidence": 1.0},
                {"text": "is", "start": 500, "end": 700, "confidence": 1.0},
                {"text": "a", "start": 700, "end": 800, "confidence": 1.0},
                {"text": "mock", "start": 800, "end": 1200, "confidence": 1.0},
                {"text": "transcript", "start": 1200, "end": 2000, "confidence": 1.0},
            ],
            "topics": ["testing"],
            "summary": "Mock transcript for testing"
        }
    
    async def _identify_segments(
        self,
        formatted_transcript: str,
        min_duration: int = 10,
        max_duration: int = 60,
        max_segments: int = 7
    ) -> List[TranscriptSegment]:
        """Use AI provider to identify engaging segments from transcript."""
        try:
            # Use configured AI provider
            provider = self._get_ai_provider()
            
            analysis = await provider.analyze_transcript(
                transcript=formatted_transcript,
                min_duration=min_duration,
                max_duration=max_duration,
                max_segments=max_segments
            )
            
            # Convert provider segments to our format
            segments = []
            for seg in analysis.segments:
                segments.append(TranscriptSegment(
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    text=seg.text,
                    relevance_score=seg.relevance_score,
                    reasoning=seg.reasoning
                ))
            
            return segments
            
        except Exception as e:
            logger.warning(f"AI segment identification failed: {e}, using fallback")
            return await self._fallback_segment_detection(formatted_transcript, min_duration, max_duration)
    
    async def _identify_segments_legacy(
        self,
        formatted_transcript: str,
        min_duration: int = 10,
        max_duration: int = 60,
        max_segments: int = 7
    ) -> List[TranscriptSegment]:
        """Legacy: Use OpenAI directly to identify segments (deprecated)."""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            system_prompt = f"""You are an expert at analyzing video transcripts to find engaging segments for short-form content.

SELECTION CRITERIA:
1. Strong hooks - Attention-grabbing opening lines
2. Valuable content - Tips, insights, interesting facts
3. Emotional moments - Excitement, surprise, humor
4. Complete thoughts - Self-contained ideas that make sense alone
5. Entertaining - Content people would want to share

CONSTRAINTS:
- Segments must be {min_duration}-{max_duration} seconds
- Find up to {max_segments} best segments
- Use EXACT timestamps from the transcript (MM:SS format)
- start_time MUST be less than end_time

OUTPUT FORMAT (JSON):
{{
    "segments": [
        {{
            "start_time": "MM:SS",
            "end_time": "MM:SS", 
            "text": "transcript text",
            "relevance_score": 0.0-1.0,
            "reasoning": "why this is engaging"
        }}
    ],
    "summary": "brief video summary",
    "key_topics": ["topic1", "topic2"]
}}"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this transcript:\n\n{formatted_transcript}"}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            segments = []
            for seg in result.get("segments", []):
                # Validate segment
                start_secs = self._parse_timestamp(seg["start_time"])
                end_secs = self._parse_timestamp(seg["end_time"])
                duration = end_secs - start_secs
                
                if duration >= min_duration and duration <= max_duration and start_secs < end_secs:
                    segments.append(TranscriptSegment(
                        start_time=seg["start_time"],
                        end_time=seg["end_time"],
                        text=seg.get("text", ""),
                        relevance_score=float(seg.get("relevance_score", 0.5)),
                        reasoning=seg.get("reasoning", "")
                    ))
            
            # Sort by relevance and limit
            segments.sort(key=lambda x: x.relevance_score, reverse=True)
            return segments[:max_segments]
            
        except Exception as e:
            logger.warning(f"AI segment identification failed: {e}, using fallback")
            return await self._fallback_segment_detection(formatted_transcript, min_duration, max_duration)
    
    async def _fallback_segment_detection(
        self,
        formatted_transcript: str,
        min_duration: int,
        max_duration: int
    ) -> List[TranscriptSegment]:
        """Simple fallback segment detection without AI."""
        segments = []
        lines = formatted_transcript.strip().split('\n')
        
        # Group consecutive lines into segments
        current_segment_lines = []
        current_start = None
        current_end = None
        
        for line in lines:
            match = self._parse_transcript_line(line)
            if match:
                start, end, text = match
                
                if current_start is None:
                    current_start = start
                current_end = end
                current_segment_lines.append(text)
                
                duration = self._parse_timestamp(current_end) - self._parse_timestamp(current_start)
                
                if duration >= min_duration:
                    segments.append(TranscriptSegment(
                        start_time=current_start,
                        end_time=current_end,
                        text=' '.join(current_segment_lines),
                        relevance_score=0.5,
                        reasoning="Auto-detected segment"
                    ))
                    current_segment_lines = []
                    current_start = None
                    current_end = None
        
        return segments[:5]
    
    async def _render_clips(
        self,
        video_path: Path,
        segments: List[TranscriptSegment],
        words_data: List[Dict],
        output_dir: Path,
        emit_progress: Callable
    ) -> List[ClipResult]:
        """Render clips from segments with smart cropping and subtitles."""
        clips = []
        total_segments = len(segments)
        
        for i, segment in enumerate(segments):
            try:
                progress_pct = 55 + int((i / total_segments) * 35)
                emit_progress(progress_pct, f"Rendering clip {i+1}/{total_segments}")
                
                clip_result = await self._render_single_clip(
                    video_path,
                    segment,
                    words_data,
                    output_dir,
                    clip_index=i + 1
                )
                
                if clip_result:
                    clips.append(clip_result)
                    
            except Exception as e:
                logger.error(f"Failed to render clip {i+1}: {e}")
        
        return clips
    
    async def _render_single_clip(
        self,
        video_path: Path,
        segment: TranscriptSegment,
        words_data: List[Dict],
        output_dir: Path,
        clip_index: int
    ) -> Optional[ClipResult]:
        """Render a single clip with cropping and subtitles."""
        try:
            from moviepy import VideoFileClip, CompositeVideoClip, TextClip
            import cv2
            
            start_secs = self._parse_timestamp(segment.start_time)
            end_secs = self._parse_timestamp(segment.end_time)
            duration = end_secs - start_secs
            
            if duration <= 0:
                return None
            
            # Load video
            video = VideoFileClip(str(video_path))
            
            # Clamp to video duration
            end_secs = min(end_secs, video.duration)
            if start_secs >= video.duration:
                video.close()
                return None
            
            # Extract clip
            clip = video.subclipped(start_secs, end_secs)
            
            # Smart crop to 9:16
            cropped_clip = await self._smart_crop(clip, video, start_secs, end_secs)
            
            # Add subtitles
            final_clip = await self._add_subtitles(
                cropped_clip,
                words_data,
                start_secs,
                end_secs
            )
            
            # Generate filename
            clean_start = segment.start_time.replace(':', '')
            clean_end = segment.end_time.replace(':', '')
            filename = f"clip_{clip_index}_{clean_start}-{clean_end}.mp4"
            output_path = output_dir / filename
            
            # Render
            final_clip.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
                bitrate="8000k",
                audio_bitrate="256k",
                preset="medium",
                logger=None,
                ffmpeg_params=["-crf", "20", "-pix_fmt", "yuv420p"]
            )
            
            # Cleanup
            final_clip.close()
            clip.close()
            video.close()
            
            return ClipResult(
                clip_id=str(uuid4()),
                filename=filename,
                path=str(output_path),
                start_time=segment.start_time,
                end_time=segment.end_time,
                duration=duration,
                text=segment.text,
                relevance_score=segment.relevance_score,
                reasoning=segment.reasoning
            )
            
        except ImportError:
            logger.error("MoviePy not installed. Run: pip install moviepy")
            return None
        except Exception as e:
            logger.error(f"Error rendering clip: {e}")
            return None
    
    async def _smart_crop(
        self,
        clip,
        full_video,
        start_time: float,
        end_time: float,
        target_ratio: float = 9/16
    ):
        """Smart crop to 9:16 using face detection."""
        try:
            import cv2
            import numpy as np
            
            original_width, original_height = clip.size
            
            # Calculate target dimensions
            if original_width / original_height > target_ratio:
                new_width = int(original_height * target_ratio)
                new_height = original_height
            else:
                new_width = original_width
                new_height = int(original_width / target_ratio)
            
            # Make dimensions even for H.264
            new_width = new_width - (new_width % 2)
            new_height = new_height - (new_height % 2)
            
            # Try face detection
            face_center_x = None
            
            try:
                # Sample a frame
                frame = clip.get_frame(0.5)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    # Use largest face
                    largest = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = largest
                    face_center_x = x + w // 2
                    logger.debug(f"Face detected at x={face_center_x}")
            except Exception as e:
                logger.debug(f"Face detection failed: {e}")
            
            # Calculate crop position
            if face_center_x is not None:
                x_offset = max(0, min(face_center_x - new_width // 2, original_width - new_width))
            else:
                x_offset = (original_width - new_width) // 2
            
            y_offset = (original_height - new_height) // 2
            
            # Make offsets even
            x_offset = x_offset - (x_offset % 2)
            y_offset = y_offset - (y_offset % 2)
            
            return clip.cropped(
                x1=x_offset,
                y1=y_offset,
                x2=x_offset + new_width,
                y2=y_offset + new_height
            )
            
        except Exception as e:
            logger.warning(f"Smart crop failed, using center crop: {e}")
            # Fallback to center crop
            w, h = clip.size
            new_w = int(h * 9/16)
            x_offset = (w - new_w) // 2
            return clip.cropped(x1=x_offset, x2=x_offset + new_w)
    
    async def _add_subtitles(
        self,
        clip,
        words_data: List[Dict],
        clip_start_ms: float,
        clip_end_ms: float
    ):
        """Add word-level subtitles to clip."""
        try:
            from moviepy import CompositeVideoClip, TextClip
            
            clip_start_ms_int = int(clip_start_ms * 1000)
            clip_end_ms_int = int(clip_end_ms * 1000)
            clip_duration = clip_end_ms - clip_start_ms
            
            # Find words in clip range
            relevant_words = []
            for word in words_data:
                word_start = word['start']
                word_end = word['end']
                
                if word_start < clip_end_ms_int and word_end > clip_start_ms_int:
                    relative_start = max(0, (word_start - clip_start_ms_int) / 1000.0)
                    relative_end = min(clip_duration, (word_end - clip_start_ms_int) / 1000.0)
                    
                    if relative_end > relative_start:
                        relevant_words.append({
                            'text': word['text'],
                            'start': relative_start,
                            'end': relative_end
                        })
            
            if not relevant_words:
                return clip
            
            # Group into subtitle segments (3 words each)
            subtitle_clips = [clip]
            video_width, video_height = clip.size
            font_size = max(20, min(40, int(self.font_size * (video_width / 720))))
            
            words_per_subtitle = 3
            for i in range(0, len(relevant_words), words_per_subtitle):
                word_group = relevant_words[i:i + words_per_subtitle]
                if not word_group:
                    continue
                
                segment_start = word_group[0]['start']
                segment_end = word_group[-1]['end']
                segment_duration = segment_end - segment_start
                
                if segment_duration < 0.1:
                    continue
                
                text = ' '.join(w['text'] for w in word_group)
                
                try:
                    text_clip = TextClip(
                        text=text,
                        font_size=font_size,
                        color=self.font_color,
                        stroke_color='black',
                        stroke_width=1,
                        method='label',
                        text_align='center'
                    ).with_duration(segment_duration).with_start(segment_start)
                    
                    # Position at 75% down
                    vertical_pos = int(video_height * 0.75)
                    text_clip = text_clip.with_position(('center', vertical_pos))
                    
                    subtitle_clips.append(text_clip)
                except Exception as e:
                    logger.debug(f"Subtitle creation failed: {e}")
            
            if len(subtitle_clips) > 1:
                return CompositeVideoClip(subtitle_clips)
            return clip
            
        except Exception as e:
            logger.warning(f"Subtitle addition failed: {e}")
            return clip
    
    def _format_ms_to_timestamp(self, ms: int) -> str:
        """Format milliseconds to MM:SS."""
        seconds = ms // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:02d}"
    
    def _parse_timestamp(self, timestamp: str) -> float:
        """Parse MM:SS or HH:MM:SS to seconds."""
        try:
            parts = timestamp.strip().split(':')
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            return float(timestamp)
        except Exception:
            return 0.0
    
    def _parse_transcript_line(self, line: str) -> Optional[Tuple[str, str, str]]:
        """Parse a transcript line like '[00:05 - 00:12] text'."""
        import re
        match = re.match(r'\[(\d+:\d+)\s*-\s*(\d+:\d+)\]\s*(.+)', line)
        if match:
            return match.group(1), match.group(2), match.group(3)
        return None
