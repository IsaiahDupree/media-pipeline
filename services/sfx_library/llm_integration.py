"""
LLM Integration for SFX Selection

Integrates with OpenAI to generate SFX selections based on
script beats and context packs.
"""

import json
import re
import os
from typing import Optional
import openai
from pydantic import BaseModel

from .types import AudioEvents, SfxAudioEvent, SfxManifest
from .context_pack import build_filtered_context_pack, make_sfx_selection_prompt
from .validator import validate_and_fix_events
from .beat_extractor import ExtractedBeat


class LLMSfxResult(BaseModel):
    """Result from LLM SFX generation."""
    events: AudioEvents
    raw_response: str
    prompt_tokens: int
    completion_tokens: int
    fixed_count: int
    rejected_count: int


def extract_json_from_response(text: str) -> Optional[str]:
    """
    Extract JSON from LLM response that may contain markdown or commentary.
    
    Args:
        text: Raw LLM response
        
    Returns:
        Extracted JSON string or None
    """
    # Try fenced JSON blocks first
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    
    # Try to find object from first { to last }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        return text[first_brace:last_brace + 1].strip()
    
    # Try array
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")
    if first_bracket >= 0 and last_bracket > first_bracket:
        return text[first_bracket:last_bracket + 1].strip()
    
    return None


def parse_llm_audio_events(raw_response: str, fps: int) -> AudioEvents:
    """
    Parse and validate audio events from LLM response.
    
    Args:
        raw_response: Raw LLM text response
        fps: Frames per second
        
    Returns:
        Parsed AudioEvents
        
    Raises:
        ValueError: If response cannot be parsed
    """
    json_str = extract_json_from_response(raw_response)
    if not json_str:
        raise ValueError("LLM response did not contain valid JSON")
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from LLM: {str(e)[:100]}")
    
    # Handle both full AudioEvents format and just events array
    if isinstance(data, list):
        data = {"fps": fps, "events": data}
    
    if "fps" not in data:
        data["fps"] = fps
    
    # Convert sfxId to sfx_id if needed
    events = []
    for ev in data.get("events", []):
        if ev.get("type") == "sfx":
            events.append(SfxAudioEvent(
                type="sfx",
                sfx_id=ev.get("sfxId") or ev.get("sfx_id"),
                frame=ev.get("frame", 0),
                volume=ev.get("volume", 1.0),
            ))
    
    return AudioEvents(fps=data["fps"], events=events)


async def generate_sfx_events_with_llm(
    manifest: SfxManifest,
    beats: list[ExtractedBeat],
    fps: int,
    script_text: Optional[str] = None,
    max_context_items: int = 80,
    allow_auto_fix: bool = True,
    model: str = "gpt-4o-mini",
) -> LLMSfxResult:
    """
    Generate SFX events using OpenAI LLM.
    
    Args:
        manifest: SFX manifest
        beats: Extracted beats from script
        fps: Frames per second
        script_text: Optional full script for context filtering
        max_context_items: Max SFX items in context pack
        allow_auto_fix: Whether to auto-fix invalid IDs
        model: OpenAI model to use
        
    Returns:
        LLMSfxResult with events and metadata
    """
    # Build filtered context pack
    context_text = script_text or " ".join(b.text for b in beats)
    context_pack = build_filtered_context_pack(
        manifest=manifest,
        beat_text=context_text,
        max_items=max_context_items,
    )
    
    # Convert beats to input format
    beat_dicts = [
        {
            "beatId": b.beat_id,
            "frame": b.frame,
            "text": b.text,
            "action": b.action,
        }
        for b in beats
    ]
    
    # Generate prompt
    prompt = make_sfx_selection_prompt(
        context_pack=context_pack,
        beats=[],  # We'll add beats manually
        fps=fps,
    )
    
    # Add beats to prompt
    prompt = prompt.replace(
        '"beats": []',
        f'"beats": {json.dumps(beat_dicts)}'
    )
    
    # Call OpenAI
    client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert sound designer selecting sound effects for video timelines. Return only valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=2000,
    )
    
    raw_response = response.choices[0].message.content or ""
    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
    completion_tokens = response.usage.completion_tokens if response.usage else 0
    
    # Parse response
    proposed_events = parse_llm_audio_events(raw_response, fps)
    
    # Validate and fix
    cleaned_events, report = validate_and_fix_events(
        events=proposed_events,
        manifest=manifest,
        allow_auto_fix=allow_auto_fix,
    )
    
    return LLMSfxResult(
        events=cleaned_events,
        raw_response=raw_response,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        fixed_count=len(report.fixed),
        rejected_count=len(report.rejected),
    )


def generate_sfx_events_sync(
    manifest: SfxManifest,
    beats: list[ExtractedBeat],
    fps: int,
    script_text: Optional[str] = None,
    max_context_items: int = 80,
    allow_auto_fix: bool = True,
    model: str = "gpt-4o-mini",
) -> LLMSfxResult:
    """
    Synchronous version of generate_sfx_events_with_llm.
    """
    import asyncio
    return asyncio.run(generate_sfx_events_with_llm(
        manifest=manifest,
        beats=beats,
        fps=fps,
        script_text=script_text,
        max_context_items=max_context_items,
        allow_auto_fix=allow_auto_fix,
        model=model,
    ))
