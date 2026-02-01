"""Media processing validation"""
from services.validation_framework import ValidationResult, ValidationSeverity
from api.media_processing_db import map_host_to_container_path
import os
from pathlib import Path


async def validate_media_processing() -> ValidationResult:
    """Validate media processing setup"""
    result = ValidationResult(component="media_processing")
    
    # Check thumbnail directory
    thumb_dir = Path("/tmp/mediaposter/thumbnails")
    if not thumb_dir.exists():
        try:
            thumb_dir.mkdir(parents=True, exist_ok=True)
            result.metadata["thumb_dir_created"] = True
        except Exception as e:
            result.add_issue(
                ValidationSeverity.WARNING,
                f"Cannot create thumbnail directory: {thumb_dir}",
                details=str(e)
            )
    
    # Check path mapping function
    try:
        test_path = "~/Documents/IphoneImport/test.mp4"
        container_path = map_host_to_container_path(test_path)
        result.metadata["path_mapping_works"] = True
        result.metadata["test_container_path"] = container_path
    except Exception as e:
        result.add_issue(
            ValidationSeverity.WARNING,
            f"Path mapping function error: {str(e)}",
            details="Container path mapping may not work correctly"
        )
    
    return result

