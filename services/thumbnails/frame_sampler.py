"""
Frame Sampling Service for Content Intelligence
Extracts frames from videos at specific intervals using FFmpeg
"""
import subprocess
import os
from pathlib import Path
from typing import List, Optional
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


class FrameSamplerService:
    """Service for extracting frames from videos using FFmpeg"""
    
    def __init__(self, output_dir: str = "/tmp/frames"):
        """
        Initialize frame sampler
        
        Args:
            output_dir: Directory to store extracted frames
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def check_ffmpeg_installed(self) -> bool:
        """Check if FFmpeg is installed"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def get_video_duration(self, video_path: str) -> Optional[float]:
        """
        Get video duration in seconds using ffprobe
        
        Args:
            video_path: Path to video file
            
        Returns:
            Duration in seconds or None if error
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    video_path
                ],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
            return None
            
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError) as e:
            logger.error(f"Error getting video duration: {e}")
            return None
    
    def extract_frame_at_time(
        self,
        video_path: str,
        time_s: float,
        output_filename: str,
        quality: int = 2
    ) -> Optional[str]:
        """
        Extract a single frame at a specific timestamp
        
        Args:
            video_path: Path to video file
            time_s: Time in seconds
            output_filename: Name for output file (without extension)
            quality: JPEG quality (1-31, lower is better, 2 is very high quality)
            
        Returns:
            Path to extracted frame or None if error
        """
        output_path = self.output_dir / f"{output_filename}.jpg"
        
        try:
            # Use FFmpeg to extract frame
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-ss", str(time_s),  # Seek to timestamp
                    "-i", video_path,
                    "-frames:v", "1",  # Extract 1 frame
                    "-q:v", str(quality),  # Quality
                    "-y",  # Overwrite if exists
                    str(output_path)
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and output_path.exists():
                logger.info(f"Extracted frame at {time_s}s to {output_path}")
                return str(output_path)
            else:
                logger.error(f"FFmpeg error: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout extracting frame at {time_s}s")
            return None
        except Exception as e:
            logger.error(f"Error extracting frame: {e}")
            return None
    
    def sample_frames_uniform(
        self,
        video_path: str,
        interval_s: float = 1.0,
        video_id: Optional[str] = None
    ) -> List[dict]:
        """
        Sample frames at uniform intervals
        
        Args:
            video_path: Path to video file
            interval_s: Interval between frames in seconds
            video_id: Optional video ID for naming
            
        Returns:
            List of frame data dicts with {time_s, frame_path}
        """
        duration = self.get_video_duration(video_path)
        if not duration:
            logger.error("Could not determine video duration")
            return []
        
        frames = []
        current_time = 0.0
        frame_index = 0
        
        while current_time <= duration:
            # Generate filename
            video_prefix = video_id if video_id else Path(video_path).stem
            filename = f"{video_prefix}_frame_{frame_index:04d}_{int(current_time*1000):06d}ms"
            
            # Extract frame
            frame_path = self.extract_frame_at_time(
                video_path,
                current_time,
                filename
            )
            
            if frame_path:
                frames.append({
                    "time_s": float(current_time),
                    "frame_path": frame_path,
                    "frame_index": frame_index
                })
            
            current_time += interval_s
            frame_index += 1
        
        logger.info(f"Sampled {len(frames)} frames from {video_path}")
        return frames
    
    def sample_frames_at_times(
        self,
        video_path: str,
        timestamps: List[float],
        video_id: Optional[str] = None
    ) -> List[dict]:
        """
        Sample frames at specific timestamps
        
        Args:
            video_path: Path to video file
            timestamps: List of timestamps in seconds
            video_id: Optional video ID for naming
            
        Returns:
            List of frame data dicts with {time_s, frame_path}
        """
        frames = []
        
        for idx, time_s in enumerate(timestamps):
            # Generate filename
            video_prefix = video_id if video_id else Path(video_path).stem
            filename = f"{video_prefix}_frame_t{int(time_s*1000):06d}ms"
            
            # Extract frame
            frame_path = self.extract_frame_at_time(
                video_path,
                time_s,
                filename
            )
            
            if frame_path:
                frames.append({
                    "time_s": float(time_s),
                    "frame_path": frame_path,
                    "frame_index": idx
                })
        
        logger.info(f"Sampled {len(frames)} frames at specific times from {video_path}")
        return frames
    
    def sample_frames_adaptive(
        self,
        video_path: str,
        min_interval_s: float = 0.5,
        max_interval_s: float = 2.0,
        video_id: Optional[str] = None
    ) -> List[dict]:
        """
        Sample frames adaptively based on scene changes
        (Future enhancement: detect scene changes and sample more densely)
        
        For now, uses uniform sampling at min_interval_s
        
        Args:
            video_path: Path to video file
            min_interval_s: Minimum interval between frames
            max_interval_s: Maximum interval between frames
            video_id: Optional video ID for naming
            
        Returns:
            List of frame data dicts
        """
        # TODO: Implement scene change detection using FFmpeg's scene filter
        # For now, use uniform sampling
        return self.sample_frames_uniform(video_path, min_interval_s, video_id)
    
    def cleanup_frames(self, video_id: Optional[str] = None):
        """
        Clean up extracted frames
        
        Args:
            video_id: If provided, only delete frames for this video
        """
        if video_id:
            # Delete frames for specific video
            pattern = f"{video_id}_frame_*"
            for frame_file in self.output_dir.glob(pattern):
                try:
                    frame_file.unlink()
                    logger.debug(f"Deleted frame: {frame_file}")
                except Exception as e:
                    logger.error(f"Error deleting frame {frame_file}: {e}")
        else:
            # Delete all frames
            for frame_file in self.output_dir.glob("*.jpg"):
                try:
                    frame_file.unlink()
                except Exception as e:
                    logger.error(f"Error deleting frame {frame_file}: {e}")
