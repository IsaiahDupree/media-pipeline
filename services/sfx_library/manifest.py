"""
SFX Manifest Management

Load, save, and query the SFX library manifest.
"""

import json
import os
from pathlib import Path
from typing import Optional

from .types import SfxManifest, SfxItem


def load_manifest(manifest_path: str | Path) -> SfxManifest:
    """
    Load and validate an SFX manifest from JSON file.
    
    Args:
        manifest_path: Path to manifest.json
        
    Returns:
        Validated SfxManifest
        
    Raises:
        FileNotFoundError: If manifest doesn't exist
        ValidationError: If manifest is invalid
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return SfxManifest.model_validate(data)


def save_manifest(manifest: SfxManifest, manifest_path: str | Path) -> None:
    """
    Save an SFX manifest to JSON file.
    
    Args:
        manifest: The manifest to save
        manifest_path: Path to write to
    """
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(by_alias=True), f, indent=2, default=str)


def get_sfx_by_id(manifest: SfxManifest, sfx_id: str) -> Optional[SfxItem]:
    """
    Get an SFX item by its ID.
    
    Args:
        manifest: The manifest to search
        sfx_id: The ID to find
        
    Returns:
        SfxItem if found, None otherwise
    """
    return manifest.get_by_id(sfx_id)


def search_sfx_by_tags(
    manifest: SfxManifest,
    tags: list[str],
    category: Optional[str] = None,
    max_results: int = 20
) -> list[SfxItem]:
    """
    Search SFX items by tags and optional category.
    
    Args:
        manifest: The manifest to search
        tags: Tags to match (OR logic)
        category: Optional category filter
        max_results: Maximum results to return
        
    Returns:
        List of matching SfxItems sorted by relevance
    """
    tag_set = {t.lower() for t in tags}
    scored = []
    
    for item in manifest.items:
        # Category filter
        if category and item.category and item.category.lower() != category.lower():
            continue
        
        # Score by tag overlap
        item_tags = {t.lower() for t in item.tags}
        overlap = len(tag_set & item_tags)
        
        if overlap > 0:
            scored.append((item, overlap))
    
    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)
    
    return [item for item, _ in scored[:max_results]]


def get_sfx_file_path(
    manifest: SfxManifest,
    sfx_id: str,
    sfx_root_dir: str | Path
) -> Optional[Path]:
    """
    Get the full file path for an SFX item.
    
    Args:
        manifest: The manifest
        sfx_id: The SFX ID
        sfx_root_dir: Root directory containing SFX files
        
    Returns:
        Full path to the audio file, or None if not found
    """
    item = manifest.get_by_id(sfx_id)
    if not item:
        return None
    
    return Path(sfx_root_dir) / item.file


def validate_sfx_files_exist(
    manifest: SfxManifest,
    sfx_root_dir: str | Path
) -> list[str]:
    """
    Check which SFX files are missing from disk.
    
    Args:
        manifest: The manifest
        sfx_root_dir: Root directory containing SFX files
        
    Returns:
        List of missing SFX IDs
    """
    root = Path(sfx_root_dir)
    missing = []
    
    for item in manifest.items:
        file_path = root / item.file
        if not file_path.exists():
            missing.append(item.id)
    
    return missing


def create_empty_manifest(version: str = "1.0") -> SfxManifest:
    """
    Create a new empty manifest.
    
    Args:
        version: Version string
        
    Returns:
        Empty SfxManifest
    """
    return SfxManifest(version=version, items=[])


def add_sfx_item(manifest: SfxManifest, item: SfxItem) -> SfxManifest:
    """
    Add an SFX item to the manifest (immutable).
    
    Args:
        manifest: The manifest
        item: Item to add
        
    Returns:
        New manifest with item added
    """
    # Check for duplicate ID
    if manifest.get_by_id(item.id):
        raise ValueError(f"SFX ID already exists: {item.id}")
    
    return SfxManifest(
        version=manifest.version,
        items=[*manifest.items, item]
    )


def remove_sfx_item(manifest: SfxManifest, sfx_id: str) -> SfxManifest:
    """
    Remove an SFX item from the manifest (immutable).
    
    Args:
        manifest: The manifest
        sfx_id: ID to remove
        
    Returns:
        New manifest with item removed
    """
    return SfxManifest(
        version=manifest.version,
        items=[item for item in manifest.items if item.id != sfx_id]
    )


def get_categories(manifest: SfxManifest) -> list[str]:
    """
    Get all unique categories in the manifest.
    
    Args:
        manifest: The manifest
        
    Returns:
        Sorted list of categories
    """
    categories = {item.category for item in manifest.items if item.category}
    return sorted(categories)


def get_all_tags(manifest: SfxManifest) -> list[str]:
    """
    Get all unique tags in the manifest.
    
    Args:
        manifest: The manifest
        
    Returns:
        Sorted list of tags
    """
    tags = set()
    for item in manifest.items:
        tags.update(item.tags)
    return sorted(tags)
