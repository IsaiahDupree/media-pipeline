"""
SFX Auto-Fix

Deterministic closest-match finder for hallucinated SFX IDs.
Uses tag/description similarity scoring without embeddings.
"""

import re
from typing import Optional

from .types import SfxManifest


def tokenize_text(text: str) -> set[str]:
    """
    Tokenize text for matching.
    
    Args:
        text: Text to tokenize
        
    Returns:
        Set of lowercase tokens
    """
    # Remove non-alphanumeric, split on whitespace
    cleaned = re.sub(r"[^a-z0-9\s_-]", " ", text.lower())
    tokens = cleaned.split()
    return {t for t in tokens if t}


def best_sfx_match(
    manifest: SfxManifest,
    requested_id_or_hint: str,
    hint_tags: Optional[list[str]] = None,
    hint_category: Optional[str] = None,
    min_score: int = 1
) -> Optional[dict]:
    """
    Find the best matching SFX for a hallucinated/unknown ID.
    
    Scoring:
    - +3 per overlapping tag
    - +2 if category matches
    - +1 per description token overlap
    
    Args:
        manifest: The SFX manifest
        requested_id_or_hint: The unknown ID or hint text
        hint_tags: Optional additional tags to match
        hint_category: Optional category preference
        min_score: Minimum score to return a match
        
    Returns:
        Dict with {id, score} or None if no good match
    """
    hint_tokens = tokenize_text(requested_id_or_hint)
    hint_tag_set = {t.lower() for t in (hint_tags or [])}
    hint_cat = hint_category.lower() if hint_category else None
    
    best: Optional[dict] = None
    
    for item in manifest.items:
        score = 0
        
        # Tag overlap with hint tokens
        item_tags = {t.lower() for t in item.tags}
        for tag in item_tags:
            if tag in hint_tokens:
                score += 3
            if tag in hint_tag_set:
                score += 3
        
        # Category match
        if hint_cat and item.category and item.category.lower() == hint_cat:
            score += 2
        
        # Description token overlap
        desc_tokens = tokenize_text(item.description or "")
        for token in hint_tokens:
            if token in desc_tokens:
                score += 1
        
        # ID token overlap (e.g., "whoosh" in "whoosh_fast_02")
        id_tokens = tokenize_text(item.id.replace("_", " "))
        for token in hint_tokens:
            if token in id_tokens:
                score += 2
        
        if best is None or score > best["score"]:
            best = {"id": item.id, "score": score}
    
    if best and best["score"] >= min_score:
        return best
    
    return None


def find_similar_sfx(
    manifest: SfxManifest,
    sfx_id: str,
    max_results: int = 5
) -> list[dict]:
    """
    Find SFX items similar to a given ID.
    
    Args:
        manifest: The SFX manifest
        sfx_id: The reference SFX ID
        max_results: Maximum results to return
        
    Returns:
        List of {id, score} dicts sorted by similarity
    """
    item = manifest.get_by_id(sfx_id)
    if not item:
        return []
    
    # Use the item's tags and description as hints
    hint_tokens = tokenize_text(item.description or "")
    hint_tokens.update(t.lower() for t in item.tags)
    
    scored = []
    
    for other in manifest.items:
        if other.id == sfx_id:
            continue
        
        score = 0
        
        # Tag overlap
        other_tags = {t.lower() for t in other.tags}
        for tag in other_tags:
            if tag in {t.lower() for t in item.tags}:
                score += 3
        
        # Category match
        if item.category and other.category:
            if item.category.lower() == other.category.lower():
                score += 2
        
        # Description overlap
        other_desc_tokens = tokenize_text(other.description or "")
        for token in hint_tokens:
            if token in other_desc_tokens:
                score += 1
        
        if score > 0:
            scored.append({"id": other.id, "score": score})
    
    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)
    
    return scored[:max_results]


def suggest_sfx_for_action(
    manifest: SfxManifest,
    action: str,
    max_results: int = 5
) -> list[dict]:
    """
    Suggest SFX for a given action type.
    
    Args:
        manifest: The SFX manifest
        action: Action type (hook, reveal, transition, etc.)
        max_results: Maximum results
        
    Returns:
        List of {id, score} dicts
    """
    # Map actions to typical tags
    action_tag_map = {
        "hook": ["whoosh", "impact", "attention", "intro", "punch"],
        "reveal": ["pop", "sparkle", "magic", "ding", "appear"],
        "transition": ["whoosh", "swoosh", "slide", "sweep", "zoom"],
        "punchline": ["pop", "ding", "tada", "success", "win"],
        "cta": ["click", "button", "tap", "ui", "confirm"],
        "explain": ["soft", "gentle", "subtle", "light", "ambient"],
    }
    
    hint_tags = action_tag_map.get(action.lower(), [])
    
    scored = []
    hint_tag_set = {t.lower() for t in hint_tags}
    
    for item in manifest.items:
        score = 0
        item_tags = {t.lower() for t in item.tags}
        
        for tag in item_tags:
            if tag in hint_tag_set:
                score += 3
        
        # Also check description
        desc_tokens = tokenize_text(item.description or "")
        for tag in hint_tag_set:
            if tag in desc_tokens:
                score += 1
        
        if score > 0:
            scored.append({"id": item.id, "score": score})
    
    scored.sort(key=lambda x: x["score"], reverse=True)
    
    return scored[:max_results]
