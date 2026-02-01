"""
Media Pipeline service configuration.
"""
import os
from pathlib import Path

# Service settings
SERVICE_NAME = "media-pipeline"
SERVICE_VERSION = "1.0.0"
SERVICE_PORT = int(os.getenv("PORT", 6004))

# Paths
BASE_DIR = Path(__file__).parent.parent
SERVICES_DIR = BASE_DIR / "services"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/tmp/media-pipeline/output"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", "/tmp/media-pipeline/cache"))

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# External services
MEDIAPOSTER_URL = os.getenv("MEDIAPOSTER_URL", "http://localhost:5555")
CONTENT_INTEL_URL = os.getenv("CONTENT_INTEL_URL", "http://localhost:6006")

# Processing settings
MAX_THUMBNAIL_COUNT = 10
DEFAULT_THUMBNAIL_COUNT = 5
SUPPORTED_VIDEO_FORMATS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".webp", ".gif"]

# FFmpeg settings
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")
FFPROBE_PATH = os.getenv("FFPROBE_PATH", "ffprobe")
