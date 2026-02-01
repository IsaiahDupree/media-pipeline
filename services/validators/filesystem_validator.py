"""File system validation"""
from services.validation_framework import ValidationResult, ValidationSeverity
import os
from pathlib import Path
import shutil


async def validate_file_system() -> ValidationResult:
    """Validate file system paths and permissions"""
    result = ValidationResult(component="file_system")
    
    # Check iPhone import directory (My Passport or local fallback)
    from config.paths import get_iphone_import_dir
    iphone_import = get_iphone_import_dir()
    if not iphone_import.exists():
        result.add_issue(
            ValidationSeverity.WARNING,
            f"iPhone import directory does not exist: {iphone_import}",
            fix_suggestion=f"Create directory: mkdir -p {iphone_import}"
        )
    elif not os.access(iphone_import, os.R_OK):
        result.add_issue(
            ValidationSeverity.CRITICAL,
            f"iPhone import directory is not readable: {iphone_import}",
            fix_suggestion=f"Fix permissions: chmod +r {iphone_import}"
        )
    else:
        result.metadata["iphone_import_exists"] = True
        result.metadata["iphone_import_readable"] = True
    
    # Check temp directory
    temp_dir = Path("/tmp/mediaposter")
    if not temp_dir.exists():
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)
            result.metadata["temp_dir_created"] = True
        except Exception as e:
            result.add_issue(
                ValidationSeverity.CRITICAL,
                f"Cannot create temp directory: {temp_dir}",
                details=str(e)
            )
    
    # Check disk space
    try:
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024**3)
        result.metadata["disk_free_gb"] = free_gb
        
        if free_gb < 1:
            result.add_issue(
                ValidationSeverity.CRITICAL,
                f"Low disk space: {free_gb:.2f} GB free",
                fix_suggestion="Free up disk space"
            )
        elif free_gb < 5:
            result.add_issue(
                ValidationSeverity.WARNING,
                f"Disk space is low: {free_gb:.2f} GB free",
                fix_suggestion="Consider freeing up disk space"
            )
    except Exception:
        pass  # Can't check on all systems
    
    return result

