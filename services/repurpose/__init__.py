"""
Content Repurposing Engine
===========================
Transform long-form videos into multiple platform-optimized short clips.

Features:
- AI-powered highlight detection
- Smart reframing (9:16, 1:1, 16:9, 4:5)
- Animated captions
- Virality scoring
- Multi-platform export

PRD: docs/PRD_CONTENT_REPURPOSING_ENGINE.md
"""

from .video_analyzer import VideoAnalyzer
from .clip_extractor import ClipExtractor
from .pipeline import RepurposePipeline

__all__ = ['VideoAnalyzer', 'ClipExtractor', 'RepurposePipeline']
