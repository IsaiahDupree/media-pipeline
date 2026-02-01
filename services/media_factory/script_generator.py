"""
Script Generator Service (MF-002)
==================================
Transforms content briefs into structured scripts with timing and shot plans.

Input: ContentBriefSchema (brief.json)
Output: ScriptSchema (script.json)

Uses OpenAI API to generate:
1. Hook (pattern interrupt)
2. Script segments with timing
3. On-screen text suggestions
4. Emphasis words
5. Visual style recommendations

Usage:
    >>> from services.media_factory.script_generator import ScriptGeneratorService
    >>> from services.media_factory.contracts import ContentBriefSchema
    >>>
    >>> generator = ScriptGeneratorService()
    >>> brief = ContentBriefSchema(...)
    >>> script = await generator.generate_script(brief)
    >>> # Returns ScriptSchema with segments and timing
"""

import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from loguru import logger
from openai import AsyncOpenAI

from services.media_factory.contracts.content_brief import ContentBriefSchema
from services.media_factory.contracts.script import ScriptSchema, ScriptBeatSchema
from config import get_settings


class ScriptGeneratorService:
    """
    Script Generator Service

    Generates production-ready scripts from content briefs using OpenAI.
    Outputs structured JSON with timing, intent, and visual directions.
    """

    _instance: Optional["ScriptGeneratorService"] = None

    def __init__(self):
        """Initialize script generator."""
        if ScriptGeneratorService._instance is not None:
            raise RuntimeError("Use ScriptGeneratorService.get_instance()")

        self.settings = get_settings()
        self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)

        # Script generation parameters
        self.model = "gpt-4o"  # Use latest GPT-4 for best script quality
        self.temperature = 0.7  # Balance creativity with consistency
        self.max_tokens = 2000

        logger.info("✓ Script Generator Service initialized")

    @classmethod
    def get_instance(cls) -> "ScriptGeneratorService":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def generate_script(
        self,
        brief: ContentBriefSchema,
        words_per_second: float = 2.5
    ) -> ScriptSchema:
        """
        Generate script from content brief.

        Args:
            brief: Content brief to generate script from
            words_per_second: Speaking rate for TTS (default: 2.5 wps)

        Returns:
            ScriptSchema with segments, timing, and visual directions

        Raises:
            ValueError: If brief is invalid or generation fails
        """
        logger.info(f"🎬 Generating script for brief: {brief.brief_id}")

        # Build prompt with brief context
        prompt = self._build_script_prompt(brief)

        try:
            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )

            # Parse response
            content = response.choices[0].message.content
            script_data = json.loads(content)

            # Calculate timing
            script_data = self._calculate_timing(script_data, words_per_second)

            # Add brief ID
            script_data["brief_id"] = brief.brief_id

            # Validate and construct ScriptSchema
            script = ScriptSchema(**script_data)

            logger.success(
                f"✓ Script generated | Brief: {brief.brief_id} | "
                f"Segments: {len(script.segments)} | "
                f"Duration: {script.metadata.get('total_duration_sec', 0):.1f}s"
            )

            return script

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI response: {e}")
            raise ValueError(f"Invalid JSON response from OpenAI: {e}")

        except Exception as e:
            logger.error(f"Script generation failed: {e}")
            raise ValueError(f"Script generation failed: {e}")

    def _get_system_prompt(self) -> str:
        """Get system prompt for script generation."""
        return """You are an expert video scriptwriter specializing in short-form content (Shorts, Reels, TikTok).

Your scripts follow these principles:
1. HOOK FIRST - Open with a pattern interrupt (1-2 seconds)
2. PACE - Keep segments tight (2-12 second beats)
3. VISUAL - Suggest on-screen text and emphasis words
4. INTENT - Mark each segment's intent (hook, problem, solution, proof, cta)
5. STRUCTURE - Follow FATE framework (Focus, Authority, Tribe, Emotion)

Output Format (JSON):
{
  "title": "Video title",
  "hook": "Opening hook text",
  "segments": [
    {
      "id": "seg_001",
      "t": "0-2",  // Time range
      "text": "Script text for this segment",
      "intent": "hook",  // hook|problem|solution|proof|cta|example
      "on_screen": ["KEY", "WORDS"],  // 1-3 words to show on screen
      "visual_style": "big_text_punch_in",  // Style suggestion
      "emphasis_words": ["emphasize", "these"]  // Words to stress in TTS
    }
  ],
  "metadata": {
    "word_count": 150,
    "format": "shorts"
  }
}

Visual Style Options:
- big_text_punch_in: Large text with zoom animation
- diagram: Diagram or chart overlay
- meme: Meme template
- b_roll: B-roll footage
- split_screen: Before/after or comparison
- text_overlay: Simple text overlay

Keep scripts:
- Conversational and direct
- Under 150 words for 45s shorts
- Front-loaded (hook in first 2 seconds)
- Action-oriented (clear CTA at end)"""

    def _build_script_prompt(self, brief: ContentBriefSchema) -> str:
        """Build user prompt from content brief."""

        # Extract key information
        title = brief.title or "Untitled"
        hook = brief.hook or ""
        promise = brief.promise or ""
        unique_lens = brief.unique_lens or ""
        format_type = brief.format
        length_sec = brief.length_sec
        hook_sec = brief.hook_sec

        # Build angle context if available
        angle_context = ""
        if brief.angle:
            angle_context = f"""
Angle Details:
- Target Audience: {brief.angle.audience_role}
- User Intent: {brief.angle.intent}
- Stakes: {brief.angle.stakes}
- Format: {brief.angle.format}
- Promise: {brief.angle.promise}
- Unique Lens: {brief.angle.unique_lens}
"""

        # Build cluster/trend context if available
        trend_context = ""
        if brief.cluster:
            trend_context = f"""
Trend Context:
- Name: {brief.cluster.name}
- Why Now: {brief.cluster.why_now or 'N/A'}
"""

        prompt = f"""Generate a {format_type} video script with these requirements:

Title: {title}
Hook: {hook}
Promise: {promise}
Unique Lens: {unique_lens}
Target Length: {length_sec} seconds
Hook Duration: {hook_sec} seconds

{angle_context}
{trend_context}

Instructions:
1. Start with a strong hook in the first {hook_sec} seconds
2. Break the script into 4-6 segments
3. Keep total duration around {length_sec} seconds
4. Include on-screen text suggestions for key moments
5. Mark emphasis words for TTS
6. Suggest visual styles for each segment
7. End with a clear call-to-action

Make it conversational, direct, and optimized for short attention spans."""

        return prompt

    def _calculate_timing(
        self,
        script_data: Dict[str, Any],
        words_per_second: float
    ) -> Dict[str, Any]:
        """
        Calculate timing for script segments based on word count.

        Args:
            script_data: Raw script data from OpenAI
            words_per_second: Speaking rate (default: 2.5 wps)

        Returns:
            Script data with calculated timing
        """
        segments = script_data.get("segments", [])

        total_words = 0
        current_time = 0.0

        for segment in segments:
            text = segment.get("text", "")
            word_count = len(text.split())
            total_words += word_count

            # Calculate segment duration
            duration = word_count / words_per_second

            # Update timing
            start_time = current_time
            end_time = current_time + duration
            segment["t"] = f"{start_time:.1f}-{end_time:.1f}"

            current_time = end_time

        # Add metadata
        if "metadata" not in script_data:
            script_data["metadata"] = {}

        script_data["metadata"]["total_duration_sec"] = current_time
        script_data["metadata"]["word_count"] = total_words
        script_data["metadata"]["estimated_tts_duration"] = current_time

        return script_data

    async def regenerate_segment(
        self,
        script: ScriptSchema,
        segment_id: str,
        feedback: str
    ) -> ScriptBeatSchema:
        """
        Regenerate a specific script segment based on feedback.

        Args:
            script: Current script
            segment_id: ID of segment to regenerate
            feedback: User feedback (e.g., "Make it more engaging", "Too long")

        Returns:
            New ScriptBeatSchema for the segment

        Raises:
            ValueError: If segment not found or regeneration fails
        """
        # Find segment
        segment = next((s for s in script.segments if s.id == segment_id), None)
        if not segment:
            raise ValueError(f"Segment not found: {segment_id}")

        logger.info(f"🔄 Regenerating segment: {segment_id} | Feedback: {feedback}")

        # Build regeneration prompt
        prompt = f"""Regenerate this script segment based on feedback:

Original Segment:
- Text: {segment.text}
- Intent: {segment.intent}
- Time Range: {segment.t}
- On-Screen: {', '.join(segment.on_screen)}

Feedback: {feedback}

Generate an improved version in JSON format:
{{
  "id": "{segment_id}",
  "t": "{segment.t}",
  "text": "New script text...",
  "intent": "{segment.intent}",
  "on_screen": ["KEY", "WORDS"],
  "visual_style": "style_name",
  "emphasis_words": ["word1", "word2"]
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,  # Slightly higher for variation
                max_tokens=500,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            segment_data = json.loads(content)

            new_segment = ScriptBeatSchema(**segment_data)

            logger.success(f"✓ Segment regenerated: {segment_id}")

            return new_segment

        except Exception as e:
            logger.error(f"Segment regeneration failed: {e}")
            raise ValueError(f"Segment regeneration failed: {e}")


# Export
__all__ = ["ScriptGeneratorService"]
