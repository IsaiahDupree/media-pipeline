"""
Content Sourcing Engine (PIPE-001)
===================================
Automatically discover and ingest content from configured sources.

Features:
- Scan local media folders for new content
- Auto-import images, videos, and clips
- Detect duplicates and near-duplicates
- Extract metadata (duration, resolution, format)
- Support external sources (YouTube downloads, etc.)

Usage:
    sourcer = ContentSourcer.get_instance()
    await sourcer.start()

    # Manual scan
    results = await sourcer.scan_folders()

    # Add a specific file
    content_item = await sourcer.import_file("/path/to/video.mp4")
"""

import os
import hashlib
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timezone
from loguru import logger
import asyncio

try:
    from PIL import Image
    import cv2
    HAS_MEDIA_LIBS = True
except ImportError:
    HAS_MEDIA_LIBS = False
    logger.warning("PIL or cv2 not available - media analysis will be limited")

from database.connection import async_session_maker
from database.models_content_pipeline import ContentItem, ContentStatus
from sqlalchemy import select, and_
from sqlalchemy.dialects.postgresql import insert


class ContentSourcer:
    """
    Content Sourcing Engine

    Scans configured media folders and imports new content into the pipeline.
    Implements PIPE-001 requirements (CS-001 to CS-005).
    """

    _instance: Optional["ContentSourcer"] = None

    def __init__(self):
        """Initialize content sourcer"""
        if ContentSourcer._instance is not None:
            raise RuntimeError("Use ContentSourcer.get_instance()")

        # Default media folders to scan
        self.scan_folders: List[Path] = []
        self._is_running = False
        self._scan_task: Optional[asyncio.Task] = None

        # Track seen file hashes to detect duplicates
        self._seen_hashes: Set[str] = set()

        # Supported media types
        self.supported_extensions = {
            'image': {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.heic'},
            'video': {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'},
            'clip': {'.mp4', '.mov', '.webm'}  # Short-form video clips
        }

        # Scanning configuration
        self.scan_interval_seconds = 300  # 5 minutes
        self.min_file_size_bytes = 1024  # 1KB minimum
        self.max_file_size_bytes = 5 * 1024 * 1024 * 1024  # 5GB maximum

        logger.info("📂 Content Sourcer initialized")

    @classmethod
    def get_instance(cls) -> "ContentSourcer":
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def add_scan_folder(self, folder_path: str) -> None:
        """
        Add a folder to scan for content (CS-001)

        Args:
            folder_path: Path to media folder
        """
        path = Path(folder_path)
        if not path.exists():
            logger.warning(f"Folder does not exist: {folder_path}")
            return

        if not path.is_dir():
            logger.warning(f"Path is not a directory: {folder_path}")
            return

        if path not in self.scan_folders:
            self.scan_folders.append(path)
            logger.info(f"✓ Added scan folder: {folder_path}")

    def remove_scan_folder(self, folder_path: str) -> None:
        """Remove a folder from scanning"""
        path = Path(folder_path)
        if path in self.scan_folders:
            self.scan_folders.remove(path)
            logger.info(f"✓ Removed scan folder: {folder_path}")

    async def start(self, scan_interval_seconds: Optional[int] = None) -> None:
        """
        Start the content sourcing service

        Args:
            scan_interval_seconds: How often to scan folders (default: 300)
        """
        if self._is_running:
            logger.warning("Content sourcer already running")
            return

        if scan_interval_seconds:
            self.scan_interval_seconds = scan_interval_seconds

        self._is_running = True
        self._scan_task = asyncio.create_task(self._scan_loop())

        # Load existing content hashes to avoid re-importing
        await self._load_existing_hashes()

        logger.success(f"✓ Content Sourcer started (scan interval: {self.scan_interval_seconds}s)")

    async def stop(self) -> None:
        """Stop the content sourcing service"""
        self._is_running = False

        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass

        logger.info("🛑 Content Sourcer stopped")

    async def _scan_loop(self) -> None:
        """Background loop to scan folders periodically"""
        logger.info(f"🔍 Content scan loop started (interval: {self.scan_interval_seconds}s)")

        while self._is_running:
            try:
                await self.scan_folders_once()
                await asyncio.sleep(self.scan_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scan loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

        logger.info("🛑 Content scan loop stopped")

    async def scan_folders_once(self) -> Dict[str, Any]:
        """
        Scan all configured folders once and import new content (CS-001, CS-002)

        Returns:
            Statistics about the scan
        """
        if not self.scan_folders:
            logger.debug("No folders configured for scanning")
            return {
                "scanned_folders": 0,
                "files_found": 0,
                "files_imported": 0,
                "duplicates_skipped": 0,
                "errors": 0
            }

        stats = {
            "scanned_folders": len(self.scan_folders),
            "files_found": 0,
            "files_imported": 0,
            "duplicates_skipped": 0,
            "errors": 0
        }

        logger.info(f"📂 Scanning {len(self.scan_folders)} folders for new content...")

        for folder in self.scan_folders:
            try:
                folder_stats = await self._scan_folder(folder)
                stats["files_found"] += folder_stats["files_found"]
                stats["files_imported"] += folder_stats["files_imported"]
                stats["duplicates_skipped"] += folder_stats["duplicates_skipped"]
                stats["errors"] += folder_stats["errors"]
            except Exception as e:
                logger.error(f"Error scanning folder {folder}: {e}")
                stats["errors"] += 1

        logger.success(
            f"✓ Scan complete | "
            f"Found: {stats['files_found']} | "
            f"Imported: {stats['files_imported']} | "
            f"Duplicates: {stats['duplicates_skipped']} | "
            f"Errors: {stats['errors']}"
        )

        return stats

    async def _scan_folder(self, folder: Path) -> Dict[str, int]:
        """
        Scan a single folder for media files

        Args:
            folder: Path to scan

        Returns:
            Statistics for this folder
        """
        stats = {
            "files_found": 0,
            "files_imported": 0,
            "duplicates_skipped": 0,
            "errors": 0
        }

        try:
            # Find all media files recursively
            for file_path in folder.rglob("*"):
                if not file_path.is_file():
                    continue

                # Check if supported media type
                if not self._is_supported_file(file_path):
                    continue

                stats["files_found"] += 1

                # Check file size
                file_size = file_path.stat().st_size
                if file_size < self.min_file_size_bytes or file_size > self.max_file_size_bytes:
                    logger.debug(f"Skipping file due to size: {file_path} ({file_size} bytes)")
                    continue

                # Check if already imported (duplicate detection - CS-003)
                file_hash = self._compute_file_hash(file_path)
                if file_hash in self._seen_hashes:
                    stats["duplicates_skipped"] += 1
                    continue

                # Import the file
                try:
                    await self.import_file(str(file_path), file_hash=file_hash)
                    stats["files_imported"] += 1
                    self._seen_hashes.add(file_hash)
                except Exception as e:
                    logger.error(f"Error importing {file_path}: {e}")
                    stats["errors"] += 1

        except Exception as e:
            logger.error(f"Error scanning folder {folder}: {e}")
            stats["errors"] += 1

        return stats

    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if file extension is supported"""
        ext = file_path.suffix.lower()
        for media_type, extensions in self.supported_extensions.items():
            if ext in extensions:
                return True
        return False

    def _get_content_type(self, file_path: Path) -> str:
        """Determine content type from file extension"""
        ext = file_path.suffix.lower()

        if ext in self.supported_extensions['image']:
            return 'image'
        elif ext in self.supported_extensions['video']:
            # Determine if it's a clip (short video) based on duration later
            return 'video'

        return 'unknown'

    def _compute_file_hash(self, file_path: Path) -> str:
        """
        Compute SHA-256 hash of file for duplicate detection (CS-003)

        Args:
            file_path: Path to file

        Returns:
            SHA-256 hash string
        """
        sha256 = hashlib.sha256()

        try:
            with open(file_path, 'rb') as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)

            return sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error computing hash for {file_path}: {e}")
            # Fallback to file path + size + mtime
            stat = file_path.stat()
            fallback = f"{file_path}{stat.st_size}{stat.st_mtime}"
            return hashlib.sha256(fallback.encode()).hexdigest()

    async def import_file(
        self,
        file_path: str,
        source_type: str = "media_library",
        file_hash: Optional[str] = None
    ) -> Optional[str]:
        """
        Import a single file into the content pipeline (CS-002, CS-004)

        Args:
            file_path: Path to media file
            source_type: Source of the file (media_library, upload, import)
            file_hash: Pre-computed file hash (optional)

        Returns:
            Content item ID if successful, None otherwise
        """
        path = Path(file_path)

        if not path.exists():
            logger.error(f"File does not exist: {file_path}")
            return None

        # Extract metadata (CS-004)
        metadata = await self._extract_metadata(path)

        if not metadata:
            logger.error(f"Failed to extract metadata for {file_path}")
            return None

        # Compute hash if not provided
        if not file_hash:
            file_hash = self._compute_file_hash(path)

        # Create content item in database
        async with async_session_maker() as session:
            try:
                # Check if already exists
                stmt = select(ContentItem).where(ContentItem.source_path == str(path))
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()

                if existing:
                    logger.debug(f"Content already imported: {file_path}")
                    return str(existing.id)

                # Create new content item
                content_item = ContentItem(
                    source_type=source_type,
                    source_path=str(path),
                    status=ContentStatus.PENDING_ANALYSIS.value,
                    content_type=metadata['content_type'],
                    duration_sec=metadata.get('duration_sec'),
                    resolution=metadata.get('resolution'),
                    aspect_ratio=metadata.get('aspect_ratio'),
                    file_size=metadata.get('file_size'),
                    ai_analysis={"file_hash": file_hash}
                )

                session.add(content_item)
                await session.commit()
                await session.refresh(content_item)

                logger.info(f"✓ Imported: {path.name} (ID: {str(content_item.id)[:8]})")

                return str(content_item.id)

            except Exception as e:
                logger.error(f"Error creating content item for {file_path}: {e}")
                await session.rollback()
                return None

    async def _extract_metadata(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Extract media metadata (CS-004)

        Args:
            file_path: Path to media file

        Returns:
            Metadata dictionary or None if extraction fails
        """
        try:
            metadata = {
                "content_type": self._get_content_type(file_path),
                "file_size": file_path.stat().st_size,
            }

            # Extract image metadata
            if metadata["content_type"] == "image":
                if HAS_MEDIA_LIBS:
                    try:
                        with Image.open(file_path) as img:
                            width, height = img.size
                            metadata["resolution"] = f"{width}x{height}"
                            metadata["aspect_ratio"] = f"{width}:{height}"
                    except Exception as e:
                        logger.warning(f"Could not extract image metadata: {e}")

            # Extract video metadata
            elif metadata["content_type"] in ["video", "clip"]:
                if HAS_MEDIA_LIBS:
                    try:
                        cap = cv2.VideoCapture(str(file_path))
                        if cap.isOpened():
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                            if fps > 0:
                                duration = frame_count / fps
                                metadata["duration_sec"] = round(duration, 2)

                                # Classify as clip if < 60 seconds
                                if duration < 60:
                                    metadata["content_type"] = "clip"

                            metadata["resolution"] = f"{width}x{height}"
                            metadata["aspect_ratio"] = f"{width}:{height}"

                            cap.release()
                    except Exception as e:
                        logger.warning(f"Could not extract video metadata: {e}")

            return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata from {file_path}: {e}")
            return None

    async def _load_existing_hashes(self) -> None:
        """Load hashes of existing content to avoid re-importing"""
        async with async_session_maker() as session:
            try:
                stmt = select(ContentItem)
                result = await session.execute(stmt)
                items = result.scalars().all()

                for item in items:
                    if item.ai_analysis and "file_hash" in item.ai_analysis:
                        self._seen_hashes.add(item.ai_analysis["file_hash"])

                logger.info(f"📋 Loaded {len(self._seen_hashes)} existing content hashes")

            except Exception as e:
                logger.error(f"Error loading existing hashes: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get content sourcer statistics"""
        return {
            "is_running": self._is_running,
            "scan_folders": [str(f) for f in self.scan_folders],
            "scan_interval_seconds": self.scan_interval_seconds,
            "known_content_items": len(self._seen_hashes),
            "supported_extensions": {
                k: list(v) for k, v in self.supported_extensions.items()
            }
        }


# Export public API
__all__ = [
    "ContentSourcer",
]
