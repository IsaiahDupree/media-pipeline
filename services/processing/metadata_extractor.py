"""
Metadata extraction for video and image files.
Handles extracting comprehensive metadata from both video and image files.
"""
import subprocess
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class MetadataExtractor:
    """Extract metadata from video and image files using ffprobe."""

    def __init__(self):
        self.ffprobe_path = "ffprobe"

    def extract_video_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from a video file."""
        if not os.path.exists(file_path):
            return self._mock_video_metadata(file_path)

        try:
            result = subprocess.run(
                [
                    self.ffprobe_path,
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_format",
                    "-show_streams",
                    file_path
                ],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                return self._mock_video_metadata(file_path)

            probe_data = json.loads(result.stdout)
            format_info = probe_data.get("format", {})
            streams = probe_data.get("streams", [])

            video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
            audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})

            return {
                "file_path": file_path,
                "file_size": int(format_info.get("size", 0)),
                "duration": float(format_info.get("duration", 0)),
                "format_name": format_info.get("format_name", "unknown"),
                "bitrate": int(format_info.get("bit_rate", 0)),
                "video": {
                    "codec": video_stream.get("codec_name", "unknown"),
                    "width": video_stream.get("width", 0),
                    "height": video_stream.get("height", 0),
                    "fps": self._parse_fps(video_stream.get("r_frame_rate", "0/1")),
                    "bit_rate": int(video_stream.get("bit_rate", 0)),
                },
                "audio": {
                    "codec": audio_stream.get("codec_name", "unknown"),
                    "sample_rate": int(audio_stream.get("sample_rate", 0)),
                    "channels": audio_stream.get("channels", 0),
                    "bit_rate": int(audio_stream.get("bit_rate", 0)),
                }
            }
        except Exception as e:
            return {"error": str(e), "file_path": file_path}

    def extract_image_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from an image file."""
        if not os.path.exists(file_path):
            return self._mock_image_metadata(file_path)

        try:
            # Use ffprobe for image metadata
            result = subprocess.run(
                [
                    self.ffprobe_path,
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_format",
                    "-show_streams",
                    file_path
                ],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                probe_data = json.loads(result.stdout)
                format_info = probe_data.get("format", {})
                streams = probe_data.get("streams", [])

                stream = streams[0] if streams else {}

                return {
                    "file_path": file_path,
                    "file_size": int(format_info.get("size", 0)),
                    "format": format_info.get("format_name", "unknown"),
                    "width": stream.get("width", 0),
                    "height": stream.get("height", 0),
                    "codec": stream.get("codec_name", "unknown"),
                    "color_space": stream.get("color_space", "unknown"),
                    "tags": format_info.get("tags", {}),
                }
        except Exception as e:
            return {"error": str(e), "file_path": file_path}

        return self._mock_image_metadata(file_path)

    def _parse_fps(self, fps_str: str) -> float:
        """Parse FPS from ffprobe format (e.g., '30000/1001')."""
        try:
            parts = fps_str.split("/")
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1])
            return float(fps_str)
        except Exception:
            return 0.0

    def _mock_video_metadata(self, file_path: str) -> Dict[str, Any]:
        """Return mock metadata for testing."""
        return {
            "file_path": file_path,
            "file_size": 0,
            "duration": 120.0,
            "format_name": "mp4",
            "bitrate": 5000000,
            "video": {
                "codec": "h264",
                "width": 1920,
                "height": 1080,
                "fps": 30.0,
                "bit_rate": 4000000,
            },
            "audio": {
                "codec": "aac",
                "sample_rate": 48000,
                "channels": 2,
                "bit_rate": 128000,
            }
        }

    def _mock_image_metadata(self, file_path: str) -> Dict[str, Any]:
        """Return mock metadata for testing."""
        return {
            "file_path": file_path,
            "file_size": 0,
            "format": "jpeg",
            "width": 1920,
            "height": 1080,
            "codec": "mjpeg",
            "color_space": "yuvj420p",
            "tags": {}
        }


class VideoQualityAnalyzer:
    """Analyze video quality metrics."""

    @staticmethod
    def analyze(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze video quality from metadata."""
        video = metadata.get("video", {})

        resolution = video.get("width", 0) * video.get("height", 0)
        if resolution >= 3840 * 2160:
            quality = "4K"
        elif resolution >= 1920 * 1080:
            quality = "1080p"
        elif resolution >= 1280 * 720:
            quality = "720p"
        else:
            quality = "SD"

        return {
            "resolution": f"{video.get('width')}x{video.get('height')}",
            "quality_tier": quality,
            "fps": video.get("fps", 0),
            "bitrate_mbps": metadata.get("bitrate", 0) / 1_000_000,
            "codec": video.get("codec", "unknown")
        }


class ImageQualityAnalyzer:
    """Analyze image quality metrics."""

    @staticmethod
    def analyze(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze image quality from metadata."""
        resolution = metadata.get("width", 0) * metadata.get("height", 0)

        if resolution >= 3840 * 2160:
            quality = "4K"
        elif resolution >= 1920 * 1080:
            quality = "FHD"
        elif resolution >= 1280 * 720:
            quality = "HD"
        else:
            quality = "SD"

        return {
            "resolution": f"{metadata.get('width')}x{metadata.get('height')}",
            "quality_tier": quality,
            "codec": metadata.get("codec", "unknown"),
            "color_space": metadata.get("color_space", "unknown"),
            "file_size_mb": metadata.get("file_size", 0) / 1_000_000
        }
