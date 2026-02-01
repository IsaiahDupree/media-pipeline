"""
Video Analyzer Service (REPURPOSE-001)
=======================================
Analyzes videos to detect highlights, emotional peaks, and viral moments.

Features:
- Whisper transcription with word-level timing
- Audio energy analysis
- Sentiment detection
- Topic segmentation
- Highlight scoring
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger
import httpx

# Import OpenAI for transcription
try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.warning("OpenAI not installed. Install with: pip install openai")


@dataclass
class TranscriptWord:
    """Word-level transcript data with timing"""
    word: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class TranscriptSegment:
    """Sentence or phrase segment"""
    text: str
    start: float
    end: float
    words: List[TranscriptWord] = field(default_factory=list)

    # Analysis scores
    sentiment_score: float = 0.0  # -1 (negative) to 1 (positive)
    emotion_intensity: float = 0.0  # 0-1
    energy_level: float = 0.0  # 0-1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "words": [{"word": w.word, "start": w.start, "end": w.end, "confidence": w.confidence} for w in self.words],
            "sentiment_score": self.sentiment_score,
            "emotion_intensity": self.emotion_intensity,
            "energy_level": self.energy_level
        }


@dataclass
class Highlight:
    """Detected highlight clip"""
    start: float
    end: float
    title: str
    transcript: str

    # Scoring
    virality_score: int  # 0-100
    hook_score: int  # 0-100
    emotion_score: int  # 0-100

    # Metadata
    reason: str  # Why this is a highlight
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "title": self.title,
            "transcript": self.transcript,
            "virality_score": self.virality_score,
            "hook_score": self.hook_score,
            "emotion_score": self.emotion_score,
            "reason": self.reason,
            "tags": self.tags
        }


class VideoAnalyzer:
    """
    Video Analyzer Service

    Analyzes video content to detect highlights and viral moments.

    Usage:
        analyzer = VideoAnalyzer(api_key="sk-...")
        result = await analyzer.analyze_video("path/to/video.mp4")
        highlights = result["highlights"]
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize analyzer with OpenAI API key"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key")

        if not HAS_OPENAI:
            raise ImportError("OpenAI package required. Install with: pip install openai")

        self.client = AsyncOpenAI(api_key=self.api_key)

    async def analyze_video(
        self,
        video_path: str,
        min_clip_duration: int = 15,
        max_clip_duration: int = 60,
        target_clip_count: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze video and detect highlights

        Args:
            video_path: Path to video file
            min_clip_duration: Minimum clip length in seconds
            max_clip_duration: Maximum clip length in seconds
            target_clip_count: Target number of clips to extract

        Returns:
            {
                "transcript": full transcript text,
                "segments": list of transcript segments,
                "highlights": list of detected highlights,
                "metadata": {
                    "duration": video duration,
                    "language": detected language,
                    "words_count": word count
                }
            }
        """
        logger.info(f"Analyzing video: {video_path}")

        # Step 1: Transcribe with Whisper
        logger.info("Step 1: Transcribing video...")
        transcript_data = await self._transcribe_video(video_path)

        # Step 2: Segment into phrases
        logger.info("Step 2: Segmenting transcript...")
        segments = self._segment_transcript(transcript_data)

        # Step 3: Analyze segments for emotional peaks
        logger.info("Step 3: Analyzing emotional content...")
        await self._analyze_segments(segments)

        # Step 4: Detect highlights
        logger.info("Step 4: Detecting highlights...")
        highlights = self._detect_highlights(
            segments,
            min_duration=min_clip_duration,
            max_duration=max_clip_duration,
            target_count=target_clip_count
        )

        # Step 5: Score highlights
        logger.info("Step 5: Scoring highlights...")
        await self._score_highlights(highlights)

        logger.info(f"Analysis complete. Found {len(highlights)} highlights")

        return {
            "transcript": transcript_data["text"],
            "segments": [seg.to_dict() for seg in segments],
            "highlights": [h.to_dict() for h in highlights],
            "metadata": {
                "duration": transcript_data.get("duration", 0),
                "language": transcript_data.get("language", "en"),
                "words_count": len(transcript_data.get("words", [])),
                "highlights_count": len(highlights)
            }
        }

    async def _transcribe_video(self, video_path: str) -> Dict[str, Any]:
        """
        Transcribe video using Whisper API

        Returns:
            {
                "text": full transcript,
                "language": detected language,
                "duration": video duration,
                "words": [{"word": str, "start": float, "end": float}]
            }
        """
        try:
            # Open video file
            with open(video_path, "rb") as video_file:
                # Call Whisper API with word-level timestamps
                response = await self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=video_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word"]
                )

            # Extract word-level timing
            words = []
            if hasattr(response, 'words') and response.words:
                words = [
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end
                    }
                    for word in response.words
                ]

            return {
                "text": response.text,
                "language": getattr(response, 'language', 'en'),
                "duration": getattr(response, 'duration', 0),
                "words": words
            }

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Failed to transcribe video: {e}")

    def _segment_transcript(self, transcript_data: Dict[str, Any]) -> List[TranscriptSegment]:
        """
        Segment transcript into phrases based on punctuation and timing

        Returns list of TranscriptSegment objects
        """
        words = transcript_data.get("words", [])
        if not words:
            # Fallback: use full text
            return [TranscriptSegment(
                text=transcript_data["text"],
                start=0,
                end=transcript_data.get("duration", 60),
                words=[]
            )]

        segments = []
        current_words = []
        current_text = []

        for word_data in words:
            word = word_data["word"]
            current_words.append(TranscriptWord(
                word=word,
                start=word_data["start"],
                end=word_data["end"]
            ))
            current_text.append(word)

            # End segment on punctuation or after ~10 words
            if word.endswith(('.', '!', '?', ',')) or len(current_words) >= 10:
                if current_words:
                    segments.append(TranscriptSegment(
                        text=' '.join(current_text),
                        start=current_words[0].start,
                        end=current_words[-1].end,
                        words=current_words.copy()
                    ))
                    current_words = []
                    current_text = []

        # Add remaining words
        if current_words:
            segments.append(TranscriptSegment(
                text=' '.join(current_text),
                start=current_words[0].start,
                end=current_words[-1].end,
                words=current_words
            ))

        return segments

    async def _analyze_segments(self, segments: List[TranscriptSegment]) -> None:
        """
        Analyze each segment for sentiment and emotion

        Updates segments in-place with scores
        """
        # Batch segments for efficient API calls
        batch_size = 20
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i+batch_size]
            texts = [seg.text for seg in batch]

            # Call GPT to analyze sentiment/emotion
            try:
                analysis = await self._batch_sentiment_analysis(texts)
                for seg, scores in zip(batch, analysis):
                    seg.sentiment_score = scores.get("sentiment", 0.0)
                    seg.emotion_intensity = scores.get("emotion", 0.0)
                    seg.energy_level = scores.get("energy", 0.0)
            except Exception as e:
                logger.warning(f"Segment analysis failed: {e}")
                # Use defaults
                for seg in batch:
                    seg.sentiment_score = 0.0
                    seg.emotion_intensity = 0.5
                    seg.energy_level = 0.5

    async def _batch_sentiment_analysis(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment/emotion for batch of texts using GPT-4

        Returns list of {"sentiment": float, "emotion": float, "energy": float}
        """
        prompt = f"""Analyze the sentiment and emotion of these speech segments. For each segment, provide:
- sentiment: -1 (negative) to 1 (positive)
- emotion: 0 (flat) to 1 (highly emotional)
- energy: 0 (calm) to 1 (energetic/excited)

Segments:
{chr(10).join([f"{i+1}. {text}" for i, text in enumerate(texts)])}

Return JSON array: [{{"sentiment": 0.5, "emotion": 0.7, "energy": 0.8}}, ...]"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )

            # Parse response
            import json
            result = json.loads(response.choices[0].message.content)

            # Handle different response formats
            if "analyses" in result:
                return result["analyses"]
            elif "segments" in result:
                return result["segments"]
            elif isinstance(result, list):
                return result
            else:
                # Fallback
                return [{"sentiment": 0.0, "emotion": 0.5, "energy": 0.5} for _ in texts]

        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return [{"sentiment": 0.0, "emotion": 0.5, "energy": 0.5} for _ in texts]

    def _detect_highlights(
        self,
        segments: List[TranscriptSegment],
        min_duration: int,
        max_duration: int,
        target_count: int
    ) -> List[Highlight]:
        """
        Detect highlight clips based on emotional peaks and content quality

        Returns sorted list of highlights (highest scoring first)
        """
        highlights = []

        # Find peaks in emotion/energy
        for i, seg in enumerate(segments):
            # Calculate peak score
            peak_score = (seg.emotion_intensity + seg.energy_level) / 2

            # Look for sustained peaks (current + next few segments)
            if i + 2 < len(segments):
                next_scores = [s.emotion_intensity + s.energy_level for s in segments[i:i+3]]
                avg_score = sum(next_scores) / (2 * len(next_scores))

                if avg_score > 0.6:  # Threshold for highlight
                    # Build clip from consecutive high-scoring segments
                    start_idx = i
                    end_idx = i + 1

                    # Extend while scores remain high and within duration limits
                    while end_idx < len(segments):
                        duration = segments[end_idx].end - segments[start_idx].start
                        if duration > max_duration:
                            break

                        next_score = (segments[end_idx].emotion_intensity + segments[end_idx].energy_level) / 2
                        if next_score < 0.4:  # Drop threshold
                            break

                        end_idx += 1

                    # Check minimum duration
                    duration = segments[end_idx - 1].end - segments[start_idx].start
                    if duration >= min_duration:
                        clip_segments = segments[start_idx:end_idx]
                        transcript = ' '.join([s.text for s in clip_segments])

                        # Generate title from first segment
                        title = self._generate_clip_title(clip_segments[0].text)

                        highlights.append(Highlight(
                            start=clip_segments[0].start,
                            end=clip_segments[-1].end,
                            title=title,
                            transcript=transcript,
                            virality_score=0,  # Scored later
                            hook_score=0,
                            emotion_score=int(avg_score * 100),
                            reason="Emotional peak detected",
                            tags=["highlight", "emotion"]
                        ))

        # Remove overlapping highlights (keep highest scoring)
        highlights = self._remove_overlaps(highlights)

        # Limit to target count
        if len(highlights) > target_count:
            # Sort by emotion score and take top N
            highlights.sort(key=lambda h: h.emotion_score, reverse=True)
            highlights = highlights[:target_count]

        return highlights

    def _generate_clip_title(self, text: str, max_length: int = 50) -> str:
        """Generate a clip title from text"""
        # Take first sentence or phrase
        words = text.split()
        if len(words) <= 6:
            return text

        # Take first 5-6 words
        title = ' '.join(words[:6])
        if len(title) > max_length:
            title = title[:max_length-3] + "..."

        return title

    def _remove_overlaps(self, highlights: List[Highlight]) -> List[Highlight]:
        """Remove overlapping highlights, keeping highest scoring"""
        if not highlights:
            return []

        # Sort by start time
        sorted_highlights = sorted(highlights, key=lambda h: h.start)

        result = [sorted_highlights[0]]
        for current in sorted_highlights[1:]:
            last = result[-1]

            # Check for overlap
            if current.start < last.end:
                # Keep the one with higher emotion score
                if current.emotion_score > last.emotion_score:
                    result[-1] = current
            else:
                result.append(current)

        return result

    async def _score_highlights(self, highlights: List[Highlight]) -> None:
        """
        Score highlights for virality and hook strength using GPT-4

        Updates highlights in-place with scores
        """
        for highlight in highlights:
            try:
                scores = await self._calculate_virality_score(
                    highlight.transcript,
                    highlight.start,
                    highlight.end
                )
                highlight.virality_score = scores["virality"]
                highlight.hook_score = scores["hook"]
            except Exception as e:
                logger.warning(f"Scoring failed for highlight: {e}")
                # Use defaults
                highlight.virality_score = 50
                highlight.hook_score = 50

    async def _calculate_virality_score(
        self,
        transcript: str,
        start: float,
        end: float
    ) -> Dict[str, int]:
        """
        Calculate virality and hook scores for a clip

        Returns {"virality": 0-100, "hook": 0-100}
        """
        duration = end - start
        is_short = duration < 20

        prompt = f"""Score this video clip for social media virality (0-100):

Transcript: "{transcript}"
Duration: {duration:.1f}s
Position: {start:.1f}s into video

Scoring criteria:
- Hook strength (first 3 seconds compelling?)
- Emotional impact
- Shareability
- Quotability
- Clarity of message

Return JSON: {{"virality": 75, "hook": 85, "reason": "Strong emotional hook with clear message"}}"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )

            import json
            result = json.loads(response.choices[0].message.content)

            return {
                "virality": min(100, max(0, result.get("virality", 50))),
                "hook": min(100, max(0, result.get("hook", 50)))
            }

        except Exception as e:
            logger.warning(f"Virality scoring failed: {e}")
            # Simple heuristic fallback
            virality = 60 if is_short else 50
            hook = 70 if start < 60 else 50  # Early clips often have better hooks
            return {"virality": virality, "hook": hook}
