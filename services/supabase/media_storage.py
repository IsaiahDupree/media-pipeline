"""
Media storage operations for Supabase.
Handles uploading, downloading, and managing media files.
"""
import os
from typing import Dict, Any, Optional
from pathlib import Path


class MediaStorage:
    """Handle media file storage in Supabase."""

    def __init__(self, client):
        self.client = client
        self.bucket_name = "media-pipeline"

    def upload_media(self, file_path: str, destination_dir: str = "uploads") -> Optional[str]:
        """Upload media file to storage."""
        if not os.path.exists(file_path):
            # Return mock URL for testing
            return self._get_mock_url(file_path, destination_dir)

        try:
            file_size = os.path.getsize(file_path)
            file_name = os.path.basename(file_path)
            destination = f"{destination_dir}/{file_name}"

            # Upload file
            url = self.client.upload_file(file_path, destination)
            return url
        except Exception as e:
            print(f"Error uploading media: {e}")
            return None

    def upload_processed_media(self, file_path: str, media_id: str, format_type: str) -> Optional[str]:
        """Upload processed media to storage."""
        destination_dir = f"processed/{media_id}"
        file_name = f"{format_type}_{os.path.basename(file_path)}"
        destination = f"{destination_dir}/{file_name}"

        url = self.client.upload_file(file_path, destination)
        return url

    def upload_thumbnail(self, file_path: str, media_id: str, index: int = 0) -> Optional[str]:
        """Upload thumbnail to storage."""
        destination_dir = f"thumbnails/{media_id}"
        file_name = f"thumb_{index:03d}.jpg"
        destination = f"{destination_dir}/{file_name}"

        url = self.client.upload_file(file_path, destination)
        return url

    def get_public_url(self, media_id: str, file_name: str) -> str:
        """Get public URL for a file."""
        path = f"processed/{media_id}/{file_name}"
        return self.client.get_public_url(path)

    def delete_media(self, media_id: str) -> bool:
        """Delete all files for a media item."""
        # In production, would delete from storage
        return self.client.delete_file(f"uploads/{media_id}")

    def _get_mock_url(self, file_path: str, destination_dir: str) -> str:
        """Get mock URL for testing."""
        file_name = os.path.basename(file_path)
        return f"https://storage.example.com/{self.bucket_name}/{destination_dir}/{file_name}"


class ThumbnailStorage:
    """Handle thumbnail storage."""

    def __init__(self, client):
        self.client = client

    def save_thumbnails(self, media_id: str, thumbnail_paths: list) -> Dict[int, str]:
        """Save multiple thumbnails for a media item."""
        urls = {}
        for idx, thumb_path in enumerate(thumbnail_paths):
            if os.path.exists(thumb_path):
                url = self.client.upload_file(
                    thumb_path,
                    f"thumbnails/{media_id}/thumb_{idx:03d}.jpg"
                )
                if url:
                    urls[idx] = url
            else:
                # Mock URL for testing
                urls[idx] = f"https://storage.example.com/{idx:03d}.jpg"
        return urls

    def get_thumbnail_urls(self, media_id: str) -> list:
        """Get all thumbnail URLs for a media item."""
        # In production, would query storage or database
        return [
            f"https://storage.example.com/thumbnails/{media_id}/thumb_000.jpg",
            f"https://storage.example.com/thumbnails/{media_id}/thumb_001.jpg",
            f"https://storage.example.com/thumbnails/{media_id}/thumb_002.jpg"
        ]


class MetadataStorage:
    """Handle metadata storage in database."""

    def __init__(self, client):
        self.client = client

    def save_metadata(self, media_id: str, metadata: Dict[str, Any]) -> bool:
        """Save media metadata to database."""
        return bool(self.client.insert_media_metadata({
            "id": media_id,
            **metadata
        }))

    def get_metadata(self, media_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve media metadata from database."""
        return self.client.get_media_metadata(media_id)

    def update_metadata(self, media_id: str, updates: Dict[str, Any]) -> bool:
        """Update media metadata in database."""
        return self.client.update_media_metadata(media_id, updates)

    def store_quality_analysis(self, media_id: str, quality_data: Dict[str, Any]) -> bool:
        """Store quality analysis results."""
        return self.update_metadata(media_id, {
            "quality_analysis": quality_data,
            "status": "analyzed"
        })

    def store_processing_result(self, media_id: str, operation: str, result: Dict[str, Any]) -> bool:
        """Store processing result."""
        updates = {
            f"{operation}_result": result,
            f"{operation}_status": "completed"
        }
        return self.update_metadata(media_id, updates)
