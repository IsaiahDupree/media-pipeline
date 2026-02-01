"""
AI-Powered Clip Selection Service
Analyzes video segments to suggest optimal clips for different platforms
"""
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
import openai
import os

from database.models import VideoClip, VideoSegment, VideoFrame, AnalyzedVideo
import uuid

logger = logging.getLogger(__name__)


@dataclass
class ClipSuggestion:
    """AI-generated clip suggestion"""
    start_time: float
    end_time: float
    duration: float
    ai_score: float
    clip_type: str
    reasoning: str
    hook_quality: float
    visual_engagement: float
    emotion_arc: float
    platform_fit: float
    cta_presence: float
    suggested_title: str
    segment_ids: List[str]
    hook_segment_id: Optional[str]
    platform_recommendations: Dict[str, Any]


class ClipSelector:
    """
    AI-powered clip selection from analyzed videos
    
    Scores potential clips based on:
    - Hook quality (FATE/AIDA psychology)
    - Visual engagement (pattern interrupts, faces)
    - Emotion arc through the clip
    - Platform fit (optimal length, format)
    - CTA presence and clarity
    """
    
    def __init__(self, db: Session, openai_api_key: Optional[str] = None):
        """
        Initialize clip selector
        
        Args:
            db: Database session
            openai_api_key: OpenAI API key for GPT-powered analysis
        """
        self.db = db
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
    
    async def suggest_clips(
        self,
        video_id: str,
        platform: Optional[str] = None,
        max_clips: int = 5,
        min_duration: float = 15.0,
        max_duration: float = 180.0
    ) -> List[ClipSuggestion]:
        """
        Generate AI-powered clip suggestions from analyzed video
        
        Args:
            video_id: UUID of video to analyze
            platform: Target platform (tiktok, youtube, etc.) or None for all
            max_clips: Maximum number of suggestions to return
            min_duration: Minimum clip duration in seconds
            max_duration: Maximum clip duration in seconds
            
        Returns:
            List of ClipSuggestion objects sorted by AI score
        """
        # Get video and its analyzed segments
        video = self.db.query(AnalyzedVideo).filter(AnalyzedVideo.id == video_id).first()
        if not video:
            raise ValueError(f"Video {video_id} not found")
        
        segments = self.db.query(VideoSegment).filter(
            VideoSegment.video_id == video_id
        ).order_by(VideoSegment.start_s).all()
        
        if not segments:
            logger.warning(f"No segments found for video {video_id}, video may not be analyzed yet")
            return []
        
        # Get frame data for visual engagement analysis
        frames = self.db.query(VideoFrame).filter(
            VideoFrame.video_id == video_id
        ).all()
        
        # Generate candidate clips
        candidates = self._generate_clip_candidates(
            segments=segments,
            frames=frames,
            min_duration=min_duration,
            max_duration=max_duration,
            platform=platform
        )
        
        # Score each candidate
        scored_clips = []
        for candidate in candidates:
            score_data = await self._score_clip_candidate(
                candidate=candidate,
                segments=segments,
                frames=frames,
                platform=platform
            )
            
            suggestion = ClipSuggestion(
                start_time=candidate["start_time"],
                end_time=candidate["end_time"],
                duration=candidate["end_time"] - candidate["start_time"],
                ai_score=score_data["overall_score"],
                clip_type="ai_generated",
                reasoning=score_data["reasoning"],
                hook_quality=score_data["hook_quality"],
                visual_engagement=score_data["visual_engagement"],
                emotion_arc=score_data["emotion_arc"],
                platform_fit=score_data["platform_fit"],
                cta_presence=score_data["cta_presence"],
                suggested_title=score_data["suggested_title"],
                segment_ids=candidate["segment_ids"],
                hook_segment_id=candidate.get("hook_segment_id"),
                platform_recommendations=score_data["platform_recommendations"]
            )
            
            scored_clips.append(suggestion)
        
        # Sort by score and return top clips
        scored_clips.sort(key=lambda x: x.ai_score, reverse=True)
        
        return scored_clips[:max_clips]
    
    def _generate_clip_candidates(
        self,
        segments: List[VideoSegment],
        frames: List[VideoFrame],
        min_duration: float,
        max_duration: float,
        platform: Optional[str]
    ) -> List[Dict]:
        """
        Generate candidate clip ranges from segments
        
        Strategy:
        1. Look for natural segment boundaries
        2. Ensure each clip has a hook
        3. Prefer clips with clear narrative arcs
        4. Consider platform-specific length preferences
        """
        candidates = []
        
        # Platform-specific duration preferences
        platform_durations = {
            "tiktok": (15, 60),
            "instagram_reel": (15, 90),
            "youtube_short": (15, 60),
            "youtube": (60, 180),
            "linkedin": (30, 90),
            "facebook": (30, 120)
        }
        
        if platform and platform in platform_durations:
            min_dur, max_dur = platform_durations[platform]
        else:
            min_dur, max_dur = min_duration, max_duration
        
        # Strategy 1: Single segment clips (if segment is good length)
        for seg in segments:
            dur = seg.end_s - seg.start_s
            if min_dur <= dur <= max_dur:
                candidates.append({
                    "start_time": seg.start_s,
                    "end_time": seg.end_s,
                    "segment_ids": [str(seg.id)],
                    "hook_segment_id": str(seg.id) if seg.segment_type == "hook" else None,
                    "strategy": "single_segment"
                })
        
        # Strategy 2: Hook + Body combinations
        for i, seg in enumerate(segments):
            if seg.segment_type == "hook":
                # Try combining with next 1-3 segments
                for j in range(i + 1, min(i + 4, len(segments))):
                    start = seg.start_s
                    end = segments[j].end_s
                    dur = end - start
                    
                    if min_dur <= dur <= max_dur:
                        segment_ids = [str(segments[k].id) for k in range(i, j + 1)]
                        candidates.append({
                            "start_time": start,
                            "end_time": end,
                            "segment_ids": segment_ids,
                            "hook_segment_id": str(seg.id),
                            "strategy": "hook_plus_body"
                        })
        
        # Strategy 3: Full narrative arcs (Hook + Body + CTA)
        i = 0
        while i < len(segments):
            if segments[i].segment_type == "hook":
                # Look for a complete arc
                j = i + 1
                has_body = False
                has_cta = False
                
                while j < len(segments):
                    if segments[j].segment_type == "body":
                        has_body = True
                    elif segments[j].segment_type == "cta":
                        has_cta = True
                        # Found complete arc
                        dur = segments[j].end_s - segments[i].start_s
                        if min_dur <= dur <= max_dur:
                            segment_ids = [str(segments[k].id) for k in range(i, j + 1)]
                            candidates.append({
                                "start_time": segments[i].start_s,
                                "end_time": segments[j].end_s,
                                "segment_ids": segment_ids,
                                "hook_segment_id": str(segments[i].id),
                                "strategy": "full_arc",
                                "has_cta": True
                            })
                        break
                    j += 1
            i += 1
        
        return candidates
    
    async def _score_clip_candidate(
        self,
        candidate: Dict,
        segments: List[VideoSegment],
        frames: List[VideoFrame],
        platform: Optional[str]
    ) -> Dict[str, Any]:
        """
        Score a clip candidate on multiple dimensions
        
        Returns dict with:
        - overall_score: 0-1 weighted score
        - hook_quality: 0-1 score
        - visual_engagement: 0-1 score
        - emotion_arc: 0-1 score
        - platform_fit: 0-1 score
        - cta_presence: 0-1 score
        - reasoning: text explanation
        - suggested_title: GPT-generated title
        """
        # Get segments in this clip
        clip_segments = [
            seg for seg in segments
            if str(seg.id) in candidate["segment_ids"]
        ]
        
        # Get frames in this clip
        clip_frames = [
            f for f in frames
            if candidate["start_time"] <= f.timestamp <= candidate["end_time"]
        ]
        
        # 1. Hook Quality (30% weight)
        hook_score = self._score_hook_quality(clip_segments)
        
        # 2. Visual Engagement (25% weight)
        visual_score = self._score_visual_engagement(clip_frames)
        
        # 3. Emotion Arc (20% weight)
        emotion_score = self._score_emotion_arc(clip_segments)
        
        # 4. Platform Fit (15% weight)
        platform_score = self._score_platform_fit(candidate, platform)
        
        # 5. CTA Presence (10% weight)
        cta_score = self._score_cta_presence(clip_segments)
        
        # Calculate overall weighted score
        overall_score = (
            hook_score * 0.30 +
            visual_score * 0.25 +
            emotion_score * 0.20 +
            platform_score * 0.15 +
            cta_score * 0.10
        )
        
        # Generate reasoning and title using GPT
        reasoning, suggested_title = await self._generate_clip_insights(
            clip_segments=clip_segments,
            scores={
                "hook": hook_score,
                "visual": visual_score,
                "emotion": emotion_score,
                "platform": platform_score,
                "cta": cta_score
            },
            platform=platform
        )
        
        # Platform-specific recommendations
        platform_recs = self._generate_platform_recommendations(
            duration=candidate["end_time"] - candidate["start_time"],
            scores={
                "hook": hook_score,
                "visual": visual_score,
                "cta": cta_score
            }
        )
        
        return {
            "overall_score": round(overall_score, 3),
            "hook_quality": round(hook_score, 3),
            "visual_engagement": round(visual_score, 3),
            "emotion_arc": round(emotion_score, 3),
            "platform_fit": round(platform_score, 3),
            "cta_presence": round(cta_score, 3),
            "reasoning": reasoning,
            "suggested_title": suggested_title,
            "platform_recommendations": platform_recs
        }
    
    def _score_hook_quality(self, segments: List[VideoSegment]) -> float:
        """Score the hook quality using FATE/AIDA psychology"""
        hook_segments = [s for s in segments if s.segment_type == "hook"]
        
        if not hook_segments:
            return 0.2  # Low score if no hook
        
        hook = hook_segments[0]  # Use first hook
        score = 0.5  # Base score for having a hook
        
        # Check for FATE patterns (stored in segment metadata)
        if hook.psychology_tags:
            fate_patterns = hook.psychology_tags.get("fate_patterns", [])
            if "fear" in fate_patterns or "pain" in fate_patterns:
                score += 0.2  # Fear/pain hooks are strong
            if "aspiration" in fate_patterns or "identity" in fate_patterns:
                score += 0.15  # Aspiration hooks are good
        
        # Check for AIDA  (Attention)
        if hook.psychology_tags and "aida_stage" in hook.psychology_tags:
            if hook.psychology_tags["aida_stage"] == "attention":
                score += 0.15
        
        return min(score, 1.0)
    
    def _score_visual_engagement(self, frames: List[VideoFrame]) -> float:
        """Score visual engagement from pattern interrupts and faces"""
        if not frames:
            return 0.5  # Neutral if no frame data
        
        score = 0.5  # Base score
        
        # Count pattern interrupts
        pattern_interrupts = sum(1 for f in frames if f.has_pattern_interrupt)
        if len(frames) > 0:
            interrupt_rate = pattern_interrupts / len(frames)
            score += interrupt_rate * 0.3  # Up to +0.3 for high interrupt rate
        
        # Count faces
        faces_present = sum(1 for f in frames if f.metadata and f.metadata.get("faces_detected", 0) > 0)
        if len(frames) > 0:
            face_rate = faces_present / len(frames)
            score += face_rate * 0.2  # Up to +0.2 for consistent face presence
        
        return min(score, 1.0)
    
    def _score_emotion_arc(self, segments: List[VideoSegment]) -> float:
        """Score emotional progression through the clip"""
        if not segments:
            return 0.5
        
        # Check for emotional diversity
        emotions = []
        for seg in segments:
            if seg.psychology_tags and "emotions" in seg.psychology_tags:
                emotions.extend(seg.psychology_tags["emotions"])
        
        unique_emotions = len(set(emotions))
        
        # Good clips have 2-4 different emotions (variety but not chaos)
        if unique_emotions == 0:
            return 0.4
        elif unique_emotions == 1:
            return 0.6
        elif 2 <= unique_emotions <= 4:
            return 0.9
        else:
            return 0.7  # Too many emotions can be confusing
    
    def _score_platform_fit(self, candidate: Dict, platform: Optional[str]) -> float:
        """Score how well clip duration fits platform preferences"""
        duration = candidate["end_time"] - candidate["start_time"]
        
        # Optimal ranges by platform
        optimal_ranges = {
            "tiktok": (20, 45),
            "instagram_reel": (20, 60),
            "youtube_short": (30, 60),
            "youtube": (90, 180),
            "linkedin": (45, 90),
            "facebook": (60, 120)
        }
        
        if platform and platform in optimal_ranges:
            opt_min, opt_max = optimal_ranges[platform]
            if opt_min <= duration <= opt_max:
                return 1.0
            elif duration < opt_min:
                return 0.6 + (duration / opt_min) * 0.4
            else:
                return 0.6 + (opt_max / duration) * 0.4
        else:
            # General scoring: prefer 30-90 seconds
            if 30 <= duration <= 90:
                return 0.9
            elif duration < 30:
                return 0.6
            else:
                return 0.7
    
    def _score_cta_presence(self, segments: List[VideoSegment]) -> float:
        """Score presence and quality of call-to-action"""
        cta_segments = [s for s in segments if s.segment_type == "cta"]
        
        if not cta_segments:
            return 0.3  # Low score if no CTA
        
        # Has CTA
        score = 0.7
        
        # Check for strong CTA keywords
        cta = cta_segments[0]
        if cta.cta_keywords:
            strong_ctas = ["click", "download", "subscribe", "follow", "buy", "get"]
            if any(keyword in cta.cta_keywords for keyword in strong_ctas):
                score += 0.3
        
        return min(score, 1.0)
    
    async def _generate_clip_insights(
        self,
        clip_segments: List[VideoSegment],
        scores: Dict[str, float],
        platform: Optional[str]
    ) -> tuple[str, str]:
        """Use GPT to generate reasoning and suggested title"""
        if not self.openai_api_key:
            return "AI scoring based on segment analysis", "Untitled Clip"
        
        # Build context from segments
        segment_summary = []
        for seg in clip_segments:
            summary = f"- {seg.segment_type.upper()}: "
            if seg.psychology_tags:
                if "fate_patterns" in seg.psychology_tags:
                    summary += f"Patterns: {', '.join(seg.psychology_tags['fate_patterns'])}. "
                if "emotions" in seg.psychology_tags:
                    summary += f"Emotions: {', '.join(seg.psychology_tags['emotions'][:2])}. "
            segment_summary.append(summary)
        
        prompt = f"""Analyze this video clip for {platform or 'social media'}:

Segments:
{chr(10).join(segment_summary)}

Scores:
- Hook Quality: {scores['hook']:.2f}
- Visual Engagement: {scores['visual']:.2f}
- Emotion Arc: {scores['emotion']:.2f}
- Platform Fit: {scores['platform']:.2f}
- CTA Presence: {scores['cta']:.2f}

Generate:
1. A brief reason why this clip would perform well (2-3 sentences)
2. A catchy title for this clip (max 60 characters)

Format:
REASONING: [your reasoning]
TITLE: [your title]"""
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing video content for social media virality."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            result = response.choices[0].message.content
            
            # Parse response
            lines = result.split("\n")
            reasoning = ""
            title = "Untitled Clip"
            
            for line in lines:
                if line.startswith("REASONING:"):
                    reasoning = line.replace("REASONING:", "").strip()
                elif line.startswith("TITLE:"):
                    title = line.replace("TITLE:", "").strip()
            
            return reasoning, title
            
        except Exception as e:
            logger.error(f"Error generating clip insights: {e}")
            return "High-scoring clip based on AI analysis", "Untitled Clip"
    
    def _generate_platform_recommendations(
        self,
        duration: float,
        scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate platform-specific recommendations"""
        recommendations = {}
        
        # TikTok/Reels/Shorts (short-form vertical)
        if 15 <= duration <= 60:
            recommendations["short_form"] = {
                "platforms": ["tiktok", "instagram_reel", "youtube_short"],
                "fit_score": scores["hook"] * 0.4 + scores["visual"] * 0.6,
                "recommendation": "Excellent for short-form vertical video"
            }
        
        # YouTube (long-form horizontal)
        if 60 <= duration <= 180:
            recommendations["youtube"] = {
                "platforms": ["youtube"],
                "fit_score": scores["emotion"] * 0.4 + scores["cta"] * 0.3 + scores["hook"] * 0.3,
                "recommendation": "Good for YouTube educational/entertainment content"
            }
        
        # LinkedIn (professional)
        if 30 <= duration <= 90:
            recommendations["linkedin"] = {
                "platforms": ["linkedin"],
                "fit_score": (scores["hook"] + scores["cta"]) / 2,
                "recommendation": "Suitable for LinkedIn if content is professional"
            }
        
        return recommendations
