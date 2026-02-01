"""
Subtitle Generator for Extracted Clips
=======================================
Generates word-level subtitles for video clips.

Based on SupoClip approach:
- 3-4 words per subtitle for readability
- Positioned at 75% down the screen
- Styled text with stroke for visibility
"""

import os
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class WordTiming:
    """Word with timing information."""
    word: str
    start_time: float  # seconds
    end_time: float
    confidence: float = 1.0


@dataclass
class SubtitleSegment:
    """A subtitle segment (group of words)."""
    text: str
    start_time: float
    end_time: float
    words: List[WordTiming] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class SubtitleConfig:
    """Configuration for subtitle generation."""
    words_per_subtitle: int = 3
    font_family: str = "Arial-Bold"
    font_size: int = 48
    font_color: str = "#FFFFFF"
    stroke_color: str = "#000000"
    stroke_width: int = 2
    vertical_position: float = 0.75  # 75% down
    text_align: str = "center"
    background_color: Optional[str] = None  # None = transparent
    padding: int = 10


class SubtitleGenerator:
    """
    Generates subtitles for video clips.
    
    Features:
    - Word-level timing from transcripts
    - Grouped into readable segments (3-4 words)
    - Styled text with stroke for visibility
    - FFmpeg-based burning for compatibility
    """
    
    def __init__(self, config: Optional[SubtitleConfig] = None):
        self.config = config or SubtitleConfig()
    
    def parse_word_timings(
        self,
        transcript_data: Dict[str, Any]
    ) -> List[WordTiming]:
        """
        Parse word-level timings from transcript data.
        
        Supports formats from:
        - AssemblyAI
        - Whisper
        - Custom format
        """
        words = []
        
        # Try AssemblyAI format
        if "words" in transcript_data:
            for w in transcript_data["words"]:
                words.append(WordTiming(
                    word=w.get("text", w.get("word", "")),
                    start_time=w.get("start", 0) / 1000 if w.get("start", 0) > 100 else w.get("start", 0),
                    end_time=w.get("end", 0) / 1000 if w.get("end", 0) > 100 else w.get("end", 0),
                    confidence=w.get("confidence", 1.0)
                ))
        
        # Try segments format (Whisper)
        elif "segments" in transcript_data:
            for segment in transcript_data["segments"]:
                if "words" in segment:
                    for w in segment["words"]:
                        words.append(WordTiming(
                            word=w.get("word", ""),
                            start_time=w.get("start", 0),
                            end_time=w.get("end", 0),
                            confidence=w.get("probability", 1.0)
                        ))
        
        return words
    
    def estimate_word_timings(
        self,
        text: str,
        start_time: float,
        end_time: float
    ) -> List[WordTiming]:
        """
        Estimate word timings when not available.
        
        Uses even distribution based on word count.
        """
        words_list = text.split()
        if not words_list:
            return []
        
        duration = end_time - start_time
        word_duration = duration / len(words_list)
        
        words = []
        current_time = start_time
        
        for word in words_list:
            words.append(WordTiming(
                word=word,
                start_time=current_time,
                end_time=current_time + word_duration,
                confidence=0.8  # Lower confidence for estimated
            ))
            current_time += word_duration
        
        return words
    
    def group_into_segments(
        self,
        words: List[WordTiming]
    ) -> List[SubtitleSegment]:
        """
        Group words into subtitle segments.
        
        Groups by words_per_subtitle setting.
        """
        segments = []
        
        for i in range(0, len(words), self.config.words_per_subtitle):
            group = words[i:i + self.config.words_per_subtitle]
            
            if group:
                segment = SubtitleSegment(
                    text=" ".join(w.word for w in group),
                    start_time=group[0].start_time,
                    end_time=group[-1].end_time,
                    words=group
                )
                segments.append(segment)
        
        return segments
    
    def generate_srt(
        self,
        segments: List[SubtitleSegment],
        offset: float = 0.0
    ) -> str:
        """
        Generate SRT subtitle file content.
        
        Args:
            segments: List of subtitle segments
            offset: Time offset to apply (for clips from longer videos)
        """
        lines = []
        
        for i, segment in enumerate(segments, 1):
            start = self._format_srt_time(segment.start_time - offset)
            end = self._format_srt_time(segment.end_time - offset)
            
            lines.append(str(i))
            lines.append(f"{start} --> {end}")
            lines.append(segment.text)
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds to SRT timestamp (HH:MM:SS,mmm)."""
        if seconds < 0:
            seconds = 0
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def generate_ass(
        self,
        segments: List[SubtitleSegment],
        video_width: int = 1080,
        video_height: int = 1920,
        offset: float = 0.0
    ) -> str:
        """
        Generate ASS subtitle file content with styling.
        
        ASS format allows for more advanced styling than SRT.
        """
        # Calculate vertical position
        margin_v = int(video_height * (1 - self.config.vertical_position))
        
        # ASS header
        header = f"""[Script Info]
Title: Generated Subtitles
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{self.config.font_family},{self.config.font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,{self.config.stroke_width},0,2,10,10,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        lines = [header]
        
        for segment in segments:
            start = self._format_ass_time(segment.start_time - offset)
            end = self._format_ass_time(segment.end_time - offset)
            
            # Clean text for ASS
            text = segment.text.replace("\n", "\\N")
            
            lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")
        
        return "\n".join(lines)
    
    def _format_ass_time(self, seconds: float) -> str:
        """Format seconds to ASS timestamp (H:MM:SS.cc)."""
        if seconds < 0:
            seconds = 0
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centis = int((seconds % 1) * 100)
        
        return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"
    
    async def burn_subtitles(
        self,
        video_path: str,
        subtitle_path: str,
        output_path: str
    ) -> bool:
        """
        Burn subtitles into video using FFmpeg.
        
        Args:
            video_path: Path to input video
            subtitle_path: Path to SRT or ASS file
            output_path: Path for output video
        """
        try:
            import subprocess
            
            # Determine subtitle format
            is_ass = subtitle_path.endswith('.ass')
            
            if is_ass:
                # ASS subtitles - use ass filter
                filter_str = f"ass='{subtitle_path}'"
            else:
                # SRT subtitles - use subtitles filter with styling
                filter_str = (
                    f"subtitles='{subtitle_path}':"
                    f"force_style='Fontname={self.config.font_family},"
                    f"Fontsize={self.config.font_size},"
                    f"PrimaryColour=&H00FFFFFF,"
                    f"OutlineColour=&H00000000,"
                    f"Outline={self.config.stroke_width},"
                    f"Alignment=2'"
                )
            
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf", filter_str,
                "-c:v", "libx264",
                "-c:a", "copy",
                "-preset", "medium",
                "-crf", "20",
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Subtitles burned successfully: {output_path}")
                return True
            else:
                logger.error(f"FFmpeg subtitle burn failed: {result.stderr[:200]}")
                return False
                
        except Exception as e:
            logger.error(f"Subtitle burning failed: {e}")
            return False
    
    async def add_subtitles_to_clip(
        self,
        video_path: str,
        text: str,
        start_time: float,
        end_time: float,
        output_path: Optional[str] = None,
        transcript_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Full pipeline: generate and burn subtitles for a clip.
        
        Args:
            video_path: Path to video clip
            text: Transcript text for the clip
            start_time: Original start time in source video
            end_time: Original end time in source video
            output_path: Output path (default: adds _subtitled suffix)
            transcript_data: Word-level timing data if available
        
        Returns:
            Tuple of (success, output_path)
        """
        # Generate output path
        if not output_path:
            base, ext = os.path.splitext(video_path)
            output_path = f"{base}_subtitled{ext}"
        
        # Get word timings
        if transcript_data:
            words = self.parse_word_timings(transcript_data)
        else:
            words = self.estimate_word_timings(text, start_time, end_time)
        
        if not words:
            logger.warning("No words to subtitle")
            return False, video_path
        
        # Group into segments
        segments = self.group_into_segments(words)
        
        # Generate subtitle file
        subtitle_dir = os.path.dirname(output_path)
        subtitle_path = os.path.join(subtitle_dir, f"subtitles_{os.path.basename(video_path)}.ass")
        
        # Get video dimensions
        video_width, video_height = self._get_video_dimensions(video_path)
        
        ass_content = self.generate_ass(
            segments=segments,
            video_width=video_width,
            video_height=video_height,
            offset=start_time  # Offset for clip extraction
        )
        
        with open(subtitle_path, 'w', encoding='utf-8') as f:
            f.write(ass_content)
        
        # Burn subtitles
        success = await self.burn_subtitles(video_path, subtitle_path, output_path)
        
        # Cleanup subtitle file
        try:
            os.remove(subtitle_path)
        except Exception:
            pass
        
        return success, output_path if success else video_path
    
    def _get_video_dimensions(self, video_path: str) -> Tuple[int, int]:
        """Get video dimensions using ffprobe."""
        try:
            import subprocess
            
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                return int(parts[0]), int(parts[1])
        except Exception as e:
            logger.warning(f"Could not get video dimensions: {e}")
        
        # Default to vertical format
        return 1080, 1920
