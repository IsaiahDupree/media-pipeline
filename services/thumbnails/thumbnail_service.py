"""
Thumbnail Generation Service
Extracts thumbnails from videos and serves images.
"""
import os
import subprocess
import hashlib
from pathlib import Path
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Thumbnail storage directory
THUMBNAIL_DIR = Path(os.getenv("TEMP_DIR", "/tmp/mediaposter")) / "thumbnails"
THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)

# Thread pool for CPU-bound operations - lazily initialized
_executor = None

def get_executor():
    """Get or create thread pool executor, handling shutdown gracefully."""
    global _executor
    if _executor is None or _executor._shutdown:
        _executor = ThreadPoolExecutor(max_workers=4)
    return _executor


def get_thumbnail_path(file_path: str, size: str = "medium") -> Path:
    """Get the path where a thumbnail should be stored."""
    # Create hash of file path for unique filename
    path_hash = hashlib.md5(file_path.encode()).hexdigest()[:16]
    filename = Path(file_path).stem
    return THUMBNAIL_DIR / f"{filename}_{path_hash}_{size}.jpg"


def generate_video_thumbnail(
    video_path: str,
    output_path: Optional[str] = None,
    timestamp: float = 1.0,
    width: int = 400
) -> Optional[str]:
    """
    Generate a thumbnail from a video file using ffmpeg.
    
    Args:
        video_path: Path to the video file
        output_path: Where to save the thumbnail (auto-generated if None)
        timestamp: Timestamp in seconds to extract frame from
        width: Width of thumbnail (height auto-calculated)
    
    Returns:
        Path to generated thumbnail or None if failed
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return None
    
    if output_path is None:
        output_path = str(get_thumbnail_path(str(video_path)))
    
    output = Path(output_path)
    
    # Return existing thumbnail if already generated
    if output.exists():
        return str(output)
    
    try:
        # Use ffmpeg to extract a frame with proper color handling
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-ss', str(timestamp),  # Seek to timestamp
            '-i', str(video_path),
            '-vframes', '1',  # Extract 1 frame
            '-vf', f'scale={width}:-1',  # Scale width, auto height
            '-pix_fmt', 'yuvj420p',  # Preserve color information
            '-q:v', '2',  # High quality JPEG
            str(output)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=30
        )
        
        if result.returncode == 0 and output.exists():
            return str(output)
        
        # Try at 0 seconds if timestamp failed
        if timestamp > 0:
            return generate_video_thumbnail(video_path, output_path, 0, width)
        
        return None
        
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        print(f"Error generating thumbnail: {e}")
        return None


def generate_image_thumbnail(
    image_path: str,
    output_path: Optional[str] = None,
    width: int = 400
) -> Optional[str]:
    """
    Generate a resized thumbnail from an image file.
    Supports HEIC, JPEG, PNG, and other formats.
    
    Args:
        image_path: Path to the image file
        output_path: Where to save the thumbnail
        width: Width of thumbnail
    
    Returns:
        Path to generated thumbnail or None if failed
    """
    image_path = Path(image_path)
    if not image_path.exists():
        return None
    
    if output_path is None:
        output_path = str(get_thumbnail_path(str(image_path)))
    
    output = Path(output_path)
    
    # Return existing thumbnail if already generated
    if output.exists():
        return str(output)
    
    ext = image_path.suffix.lower()
    
    try:
        # For HEIC files, use macOS sips command (best color support)
        if ext in {'.heic', '.heif'}:
            return _generate_heic_thumbnail(str(image_path), str(output), width)
        
        # For other images, try PIL first (better quality), then ffmpeg
        try:
            from PIL import Image
            with Image.open(str(image_path)) as img:
                # Convert to RGB if needed (handles RGBA, P, etc.)
                if img.mode in ('RGBA', 'P', 'LA'):
                    img = img.convert('RGB')
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Calculate new height maintaining aspect ratio
                ratio = width / img.width
                new_height = int(img.height * ratio)
                
                # Resize with high quality
                img_resized = img.resize((width, new_height), Image.Resampling.LANCZOS)
                img_resized.save(str(output), 'JPEG', quality=90)
                
                if output.exists():
                    return str(output)
        except ImportError:
            pass  # PIL not available, fall back to ffmpeg
        except Exception as pil_error:
            print(f"PIL error, falling back to ffmpeg: {pil_error}")
        
        # Fallback: Use ffmpeg for image conversion
        cmd = [
            'ffmpeg',
            '-y',
            '-i', str(image_path),
            '-vf', f'scale={width}:-1',
            '-pix_fmt', 'yuvj420p',
            '-q:v', '2',
            str(output)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=30
        )
        
        if result.returncode == 0 and output.exists():
            return str(output)
        
        return None
        
    except Exception as e:
        print(f"Error generating image thumbnail: {e}")
        return None


def _generate_heic_thumbnail(image_path: str, output_path: str, width: int = 400) -> Optional[str]:
    """
    Generate thumbnail from HEIC file using macOS sips command.
    Preserves color information properly.
    """
    try:
        # First convert HEIC to JPEG using sips (macOS built-in)
        temp_jpeg = output_path.replace('.jpg', '_temp.jpg')
        
        # sips can convert HEIC to JPEG with proper color handling
        convert_cmd = [
            'sips',
            '-s', 'format', 'jpeg',
            '-s', 'formatOptions', '90',  # Quality
            '--resampleWidth', str(width),
            str(image_path),
            '--out', temp_jpeg
        ]
        
        result = subprocess.run(
            convert_cmd,
            capture_output=True,
            timeout=30
        )
        
        if result.returncode == 0 and Path(temp_jpeg).exists():
            # Rename to final output
            Path(temp_jpeg).rename(output_path)
            return output_path
        
        # Fallback: try pillow-heif if available
        try:
            from pillow_heif import register_heif_opener
            from PIL import Image
            
            register_heif_opener()
            
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                ratio = width / img.width
                new_height = int(img.height * ratio)
                img_resized = img.resize((width, new_height), Image.Resampling.LANCZOS)
                img_resized.save(output_path, 'JPEG', quality=90)
                
                if Path(output_path).exists():
                    return output_path
        except ImportError:
            print("pillow-heif not available, sips failed")
        
        return None
        
    except Exception as e:
        print(f"Error converting HEIC: {e}")
        return None


def get_media_type(file_path: str) -> str:
    """Determine if file is video or image."""
    ext = Path(file_path).suffix.lower()
    video_exts = {'.mov', '.mp4', '.m4v', '.avi', '.mkv', '.webm'}
    return 'video' if ext in video_exts else 'image'


def generate_thumbnail(file_path: str, size: str = "medium") -> Optional[str]:
    """
    Generate thumbnail for any media file.
    
    Args:
        file_path: Path to media file
        size: 'small' (200px), 'medium' (400px), or 'large' (800px)
    
    Returns:
        Path to thumbnail or None
    """
    sizes = {
        'small': 200,
        'medium': 400,
        'large': 800
    }
    width = sizes.get(size, 400)
    
    media_type = get_media_type(file_path)
    
    if media_type == 'video':
        return generate_video_thumbnail(file_path, width=width)
    else:
        return generate_image_thumbnail(file_path, width=width)


async def generate_thumbnail_async(file_path: str, size: str = "medium") -> Optional[str]:
    """Async wrapper for thumbnail generation."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(get_executor(), generate_thumbnail, file_path, size)


def get_thumbnail_url(media_id: str, file_path: str) -> str:
    """Get the URL for a media's thumbnail."""
    return f"/api/media/thumbnail/{media_id}"


# Pre-generate common thumbnail sizes
async def pregenerate_thumbnails(file_path: str):
    """Pre-generate all thumbnail sizes for a file."""
    for size in ['small', 'medium', 'large']:
        await generate_thumbnail_async(file_path, size)


# =============================================================================
# SMART RESUME - State Tracking
# =============================================================================

import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

# State file location
THUMBNAIL_STATE_FILE = THUMBNAIL_DIR / "thumbnail_state.json"


@dataclass
class ThumbnailState:
    """Track thumbnail generation state for smart resume."""
    state_file: Path = field(default=THUMBNAIL_STATE_FILE)
    generated: Dict[str, Dict[str, str]] = field(default_factory=dict)  # file_path -> {size: thumb_path}
    failed: Dict[str, str] = field(default_factory=dict)  # file_path -> error_message
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        if isinstance(self.state_file, str):
            self.state_file = Path(self.state_file)
        self.load()
    
    def load(self):
        """Load state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.generated = data.get('generated', {})
                    self.failed = data.get('failed', {})
                    self.last_updated = data.get('last_updated', datetime.now().isoformat())
            except Exception as e:
                print(f"Error loading thumbnail state: {e}")
    
    def save(self):
        """Save state to file."""
        self.last_updated = datetime.now().isoformat()
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump({
                    'generated': self.generated,
                    'failed': self.failed,
                    'last_updated': self.last_updated
                }, f, indent=2)
        except Exception as e:
            print(f"Error saving thumbnail state: {e}")
    
    def mark_generated(self, file_path: str, size: str, thumb_path: str):
        """Mark a thumbnail as successfully generated."""
        if file_path not in self.generated:
            self.generated[file_path] = {}
        self.generated[file_path][size] = thumb_path
        
        # Clear from failed if present
        if file_path in self.failed:
            del self.failed[file_path]
    
    def mark_failed(self, file_path: str, error: str):
        """Mark a thumbnail as failed."""
        self.failed[file_path] = error
    
    def is_generated(self, file_path: str, size: str) -> bool:
        """Check if a thumbnail has been generated."""
        return file_path in self.generated and size in self.generated[file_path]
    
    def is_failed(self, file_path: str) -> bool:
        """Check if a thumbnail has failed."""
        return file_path in self.failed
    
    def get_thumb_path(self, file_path: str, size: str) -> Optional[str]:
        """Get the generated thumbnail path if available."""
        if self.is_generated(file_path, size):
            thumb_path = self.generated[file_path][size]
            # Verify it still exists
            if Path(thumb_path).exists():
                return thumb_path
            # Remove stale entry
            del self.generated[file_path][size]
        return None
    
    def clear_failed(self, file_path: str):
        """Clear failed status for retry."""
        if file_path in self.failed:
            del self.failed[file_path]
    
    def get_stats(self) -> dict:
        """Get statistics about thumbnail generation."""
        total_generated = sum(len(sizes) for sizes in self.generated.values())
        return {
            'files_with_thumbnails': len(self.generated),
            'total_thumbnails': total_generated,
            'failed_count': len(self.failed),
            'last_updated': self.last_updated
        }


@dataclass
class ThumbnailBatchJob:
    """Manage batch thumbnail generation with resume support."""
    files: List[str]
    sizes: List[str] = field(default_factory=lambda: ['medium'])
    state: ThumbnailState = field(default_factory=ThumbnailState)
    completed: Dict[str, set] = field(default_factory=dict)  # file -> set of completed sizes
    
    def __post_init__(self):
        # Initialize completed from state
        for file_path in self.files:
            self.completed[file_path] = set()
            for size in self.sizes:
                if self.state.is_generated(file_path, size):
                    self.completed[file_path].add(size)
    
    @property
    def total_files(self) -> int:
        return len(self.files)
    
    @property
    def total_thumbnails(self) -> int:
        return len(self.files) * len(self.sizes)
    
    @property
    def processed_count(self) -> int:
        return sum(len(sizes) for sizes in self.completed.values())
    
    @property
    def progress(self) -> float:
        if self.total_thumbnails == 0:
            return 1.0
        return self.processed_count / self.total_thumbnails
    
    def mark_complete(self, file_path: str, size: str, thumb_path: str = None):
        """Mark a thumbnail as complete."""
        if file_path not in self.completed:
            self.completed[file_path] = set()
        self.completed[file_path].add(size)
        
        if thumb_path:
            self.state.mark_generated(file_path, size, thumb_path)
    
    def mark_failed(self, file_path: str, error: str):
        """Mark a file as failed."""
        self.state.mark_failed(file_path, error)
    
    def get_remaining(self) -> List[Tuple[str, str]]:
        """Get list of (file_path, size) tuples that still need processing."""
        remaining = []
        for file_path in self.files:
            completed_sizes = self.completed.get(file_path, set())
            for size in self.sizes:
                if size not in completed_sizes and not self.state.is_failed(file_path):
                    remaining.append((file_path, size))
        return remaining
    
    def is_complete(self) -> bool:
        """Check if all thumbnails are generated."""
        return len(self.get_remaining()) == 0
    
    def save_state(self):
        """Save state for resume."""
        self.state.save()


# Smart thumbnail generation with state tracking
_global_state: Optional[ThumbnailState] = None


def get_thumbnail_state() -> ThumbnailState:
    """Get or create global thumbnail state."""
    global _global_state
    if _global_state is None:
        _global_state = ThumbnailState()
    return _global_state


def generate_thumbnail_smart(file_path: str, size: str = "medium") -> Optional[str]:
    """
    Generate thumbnail with smart resume support.
    
    Checks state first to avoid regenerating existing thumbnails.
    Updates state after generation.
    """
    state = get_thumbnail_state()
    
    # Check if already generated
    existing = state.get_thumb_path(file_path, size)
    if existing:
        return existing
    
    # Check if previously failed (skip unless retry requested)
    if state.is_failed(file_path):
        return None
    
    # Generate thumbnail
    try:
        result = generate_thumbnail(file_path, size)
        
        if result:
            state.mark_generated(file_path, size, result)
            state.save()
            return result
        else:
            state.mark_failed(file_path, "Generation returned None")
            state.save()
            return None
            
    except Exception as e:
        state.mark_failed(file_path, str(e))
        state.save()
        return None


async def generate_thumbnails_batch(
    files: List[str],
    sizes: List[str] = ['medium'],
    on_progress: Optional[callable] = None
) -> ThumbnailBatchJob:
    """
    Generate thumbnails for multiple files with resume support.
    
    Args:
        files: List of file paths
        sizes: List of sizes to generate
        on_progress: Optional callback(job) for progress updates
    
    Returns:
        ThumbnailBatchJob with results
    """
    job = ThumbnailBatchJob(files, sizes)
    
    remaining = job.get_remaining()
    total_remaining = len(remaining)
    
    for i, (file_path, size) in enumerate(remaining):
        try:
            result = await generate_thumbnail_async(file_path, size)
            
            if result:
                job.mark_complete(file_path, size, result)
            else:
                job.mark_failed(file_path, "Generation failed")
                
        except Exception as e:
            job.mark_failed(file_path, str(e))
        
        # Save state periodically
        if (i + 1) % 10 == 0:
            job.save_state()
            if on_progress:
                on_progress(job)
    
    # Final save
    job.save_state()
    
    return job
