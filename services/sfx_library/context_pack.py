"""
SFX Context Pack Generator

Creates token-efficient context packs for LLM prompts and
generates SFX selection prompts.
"""

import json
from typing import Optional

from .types import SfxManifest, SfxContextPack, SfxContextItem, Beat
from .autofix import tokenize_text


# Default rules for AI
DEFAULT_SFX_RULES = [
    "You MUST ONLY use sfxId values that exist in sfxIndex.id.",
    "Return ONLY JSON. No extra commentary.",
    "Prefer exact tag matches; if unsure, choose a safe generic UI pop or soft whoosh.",
    "Keep volumes between 0.3 and 1.0 unless asked otherwise.",
    "Maximum 1 SFX per beat unless action is 'transition'.",
    "Avoid consecutive identical sfxIds.",
]


def build_sfx_context_pack(
    manifest: SfxManifest,
    max_items: int = 500,
    rules: Optional[list[str]] = None,
) -> SfxContextPack:
    """
    Build a compact context pack from the full manifest.
    
    Args:
        manifest: The full SFX manifest
        max_items: Maximum items to include
        rules: Custom rules (uses defaults if None)
        
    Returns:
        Token-efficient SfxContextPack
    """
    sfx_index = []
    
    for item in manifest.items[:max_items]:
        sfx_index.append(SfxContextItem(
            id=item.id,
            tags=item.tags[:10],  # Limit tags
            desc=item.description[:120] if item.description else "",
            intensity=item.intensity,
            category=item.category,
        ))
    
    return SfxContextPack(
        version=manifest.version,
        rules=rules or DEFAULT_SFX_RULES,
        sfx_index=sfx_index,
    )


def build_filtered_context_pack(
    manifest: SfxManifest,
    beat_text: str,
    max_items: int = 80,
    rules: Optional[list[str]] = None,
) -> SfxContextPack:
    """
    Build a filtered context pack based on beat/scene text.
    
    This reduces tokens by only including relevant SFX for the context.
    
    Args:
        manifest: The full SFX manifest
        beat_text: Text describing the beat/scene
        max_items: Maximum items to include
        rules: Custom rules
        
    Returns:
        Filtered SfxContextPack
    """
    text_tokens = tokenize_text(beat_text)
    
    # Score each item by relevance
    scored = []
    for item in manifest.items:
        score = 0
        
        # Tag matches
        for tag in item.tags:
            if tag.lower() in text_tokens:
                score += 3
        
        # Description token matches
        desc_tokens = tokenize_text(item.description or "")
        for token in text_tokens:
            if token in desc_tokens:
                score += 1
        
        scored.append((item, score))
    
    # Sort by score, filter to items with score > 0
    scored.sort(key=lambda x: x[1], reverse=True)
    relevant = [item for item, score in scored if score > 0][:max_items]
    
    # If not enough relevant items, add some generic ones
    if len(relevant) < 10:
        generic_categories = ["ui", "transition"]
        for item in manifest.items:
            if len(relevant) >= max_items:
                break
            if item.category in generic_categories and item not in relevant:
                relevant.append(item)
    
    # Build context pack from filtered items
    sfx_index = [
        SfxContextItem(
            id=item.id,
            tags=item.tags[:10],
            desc=item.description[:120] if item.description else "",
            intensity=item.intensity,
            category=item.category,
        )
        for item in relevant
    ]
    
    return SfxContextPack(
        version=manifest.version,
        rules=rules or DEFAULT_SFX_RULES,
        sfx_index=sfx_index,
    )


def make_sfx_selection_prompt(
    context_pack: SfxContextPack,
    beats: list[Beat],
    fps: int,
) -> str:
    """
    Generate a prompt for LLM to select SFX for beats.
    
    Args:
        context_pack: The SFX context pack
        beats: List of narrative beats
        fps: Frames per second
        
    Returns:
        Formatted prompt string
    """
    context_json = context_pack.model_dump(by_alias=True)
    beats_json = [b.model_dump(by_alias=True) for b in beats]
    
    prompt = f"""You are selecting sound effects for a video timeline.

SFX_LIBRARY_CONTEXT:
{json.dumps(context_json, indent=2)}

VIDEO_CONTEXT:
- fps: {fps}
- beats: {json.dumps(beats_json, indent=2)}

TASK:
Create audio events for SFX only.
Output JSON with this exact shape:
{{
  "fps": {fps},
  "events": [
    {{ "type": "sfx", "sfxId": "...", "frame": 0, "volume": 0.8 }}
  ]
}}

CONSTRAINTS:
- sfxId MUST match one of SFX_LIBRARY_CONTEXT.sfxIndex[].id exactly.
- Use frames from beats or nearby offsets (±10 frames).
- Avoid spamming: max 1 sfx per beat unless action is "transition".
- Return ONLY the JSON, no explanation."""

    return prompt.strip()


def make_sfx_merge_prompt(
    base_events_json: str,
    sfx_events_json: str,
    fps: int,
) -> str:
    """
    Generate a prompt to merge base audio events with new SFX.
    
    Args:
        base_events_json: Existing events (voiceover, music)
        sfx_events_json: New SFX events to merge
        fps: Frames per second
        
    Returns:
        Merged events JSON prompt
    """
    return f"""Merge these audio events into a single timeline.

BASE_EVENTS (voiceover + music):
{base_events_json}

SFX_EVENTS:
{sfx_events_json}

OUTPUT:
Return a single JSON object combining all events:
{{
  "fps": {fps},
  "events": [
    // all voiceover events
    // all music events  
    // all sfx events
  ]
}}

Sort events by frame number. Keep all events. Return ONLY JSON."""


def estimate_prompt_tokens(context_pack: SfxContextPack) -> int:
    """
    Estimate the number of tokens in a context pack.
    
    Rough estimate: ~4 characters per token.
    
    Args:
        context_pack: The context pack
        
    Returns:
        Estimated token count
    """
    json_str = json.dumps(context_pack.model_dump(by_alias=True))
    return len(json_str) // 4


def get_context_pack_stats(context_pack: SfxContextPack) -> dict:
    """
    Get statistics about a context pack.
    
    Args:
        context_pack: The context pack
        
    Returns:
        Dict with stats
    """
    categories = {}
    for item in context_pack.sfx_index:
        cat = item.category or "uncategorized"
        categories[cat] = categories.get(cat, 0) + 1
    
    return {
        "total_items": len(context_pack.sfx_index),
        "estimated_tokens": estimate_prompt_tokens(context_pack),
        "categories": categories,
        "rules_count": len(context_pack.rules),
    }
