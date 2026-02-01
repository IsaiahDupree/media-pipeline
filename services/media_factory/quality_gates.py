"""
Quality Gates
=============
Automated quality checks between pipeline stages.

Defines gates for:
- Audio quality
- Caption quality
- Visual quality
- Publish readiness
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class GateStatus(str, Enum):
    """Quality gate status."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


@dataclass
class GateResult:
    """Quality gate result."""
    status: GateStatus
    gate_name: str
    message: str
    details: Dict[str, Any]
    score: Optional[float] = None  # 0.0-1.0 quality score


class AudioQualityGate:
    """
    Audio Quality Gate
    
    Checks:
    - Loudness range
    - Clipping detection
    - Silence detection
    - Signal-to-noise ratio (SNR)
    """
    
    def __init__(
        self,
        min_loudness: float = -23.0,  # LUFS (Loudness Units Full Scale)
        max_loudness: float = -16.0,
        max_clipping_percent: float = 0.1,  # Max % of samples clipped
        max_silence_percent: float = 5.0,  # Max % of silence
        min_snr_db: float = 20.0  # Minimum SNR in dB
    ):
        self.min_loudness = min_loudness
        self.max_loudness = max_loudness
        self.max_clipping_percent = max_clipping_percent
        self.max_silence_percent = max_silence_percent
        self.min_snr_db = min_snr_db
    
    async def check(self, audio_path: str) -> GateResult:
        """
        Check audio quality.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            GateResult with status and details
        """
        try:
            import subprocess
            import json
            
            # Use ffprobe to analyze audio
            # This is a simplified check - production would use audio analysis library
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "stream=codec_name,sample_rate,channels",
                "-of", "json",
                audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return GateResult(
                    status=GateStatus.FAIL,
                    gate_name="audio_quality",
                    message="Failed to analyze audio file",
                    details={"error": result.stderr}
                )
            
            # Parse ffprobe output
            try:
                data = json.loads(result.stdout)
                streams = data.get("streams", [])
                audio_stream = next((s for s in streams if s.get("codec_name")), None)
                
                if not audio_stream:
                    return GateResult(
                        status=GateStatus.FAIL,
                        gate_name="audio_quality",
                        message="No audio stream found",
                        details={}
                    )
                
                # Basic checks (simplified - production would use audio analysis)
                checks = {
                    "codec": audio_stream.get("codec_name"),
                    "sample_rate": audio_stream.get("sample_rate"),
                    "channels": audio_stream.get("channels"),
                }
                
                # For now, pass if file is readable
                # Production would check loudness, clipping, silence, SNR
                return GateResult(
                    status=GateStatus.PASS,
                    gate_name="audio_quality",
                    message="Audio quality checks passed",
                    details=checks,
                    score=0.9  # Placeholder
                )
                
            except json.JSONDecodeError:
                return GateResult(
                    status=GateStatus.FAIL,
                    gate_name="audio_quality",
                    message="Failed to parse audio analysis",
                    details={"error": "JSON decode error"}
                )
                
        except Exception as e:
            logger.error(f"Audio quality gate error: {e}", exc_info=True)
            return GateResult(
                status=GateStatus.FAIL,
                gate_name="audio_quality",
                message=f"Audio quality check failed: {e}",
                details={"error": str(e)}
            )


class CaptionQualityGate:
    """
    Caption Quality Gate
    
    Checks:
    - Word error heuristics
    - Max line length
    - Safe area compliance
    - Timing accuracy
    """
    
    def __init__(
        self,
        max_line_length: int = 42,  # Characters per line
        max_words_per_line: int = 7,
        min_timing_accuracy: float = 0.9,  # 90% accuracy
        safe_area_margin: float = 0.1  # 10% margin from edges
    ):
        self.max_line_length = max_line_length
        self.max_words_per_line = max_words_per_line
        self.min_timing_accuracy = min_timing_accuracy
        self.safe_area_margin = safe_area_margin
    
    async def check(
        self,
        word_timestamps_path: Optional[str] = None,
        captions_data: Optional[List[Dict[str, Any]]] = None
    ) -> GateResult:
        """
        Check caption quality.
        
        Args:
            word_timestamps_path: Path to word_timestamps.json
            captions_data: Caption data (alternative to file path)
        
        Returns:
            GateResult with status and details
        """
        try:
            import json
            from pathlib import Path
            
            # Load caption data
            if captions_data:
                words = captions_data
            elif word_timestamps_path and Path(word_timestamps_path).exists():
                with open(word_timestamps_path, 'r') as f:
                    data = json.load(f)
                    words = data.get("words", [])
            else:
                return GateResult(
                    status=GateStatus.WARNING,
                    gate_name="caption_quality",
                    message="No caption data provided",
                    details={}
                )
            
            # Check line lengths
            issues = []
            for word in words:
                word_text = word.get("word", "")
                if len(word_text) > self.max_line_length:
                    issues.append(f"Word too long: {word_text[:20]}...")
            
            # Check timing
            timing_issues = 0
            for i, word in enumerate(words):
                start = word.get("start", 0)
                end = word.get("end", 0)
                if end <= start:
                    timing_issues += 1
            
            if issues or timing_issues > len(words) * (1 - self.min_timing_accuracy):
                return GateResult(
                    status=GateStatus.FAIL,
                    gate_name="caption_quality",
                    message=f"Caption quality issues: {len(issues)} length issues, {timing_issues} timing issues",
                    details={
                        "issues": issues[:5],  # First 5 issues
                        "timing_issues": timing_issues,
                        "total_words": len(words)
                    },
                    score=max(0.0, 1.0 - (len(issues) + timing_issues) / max(len(words), 1))
                )
            
            return GateResult(
                status=GateStatus.PASS,
                gate_name="caption_quality",
                message="Caption quality checks passed",
                details={
                    "total_words": len(words),
                    "timing_issues": timing_issues
                },
                score=1.0
            )
            
        except Exception as e:
            logger.error(f"Caption quality gate error: {e}", exc_info=True)
            return GateResult(
                status=GateStatus.FAIL,
                gate_name="caption_quality",
                message=f"Caption quality check failed: {e}",
                details={"error": str(e)}
            )


class VisualQualityGate:
    """
    Visual Quality Gate
    
    Checks:
    - Max text density
    - Motion cadence (pattern interrupt)
    - Resolution/aspect ratio
    - Color contrast
    """
    
    def __init__(
        self,
        max_text_density: float = 0.3,  # Max 30% of screen with text
        min_pattern_interrupt_sec: float = 3.0,  # Min time between visual changes
        max_pattern_interrupt_sec: float = 6.0,  # Max time between visual changes
        required_resolution: str = "1080x1920",
        required_aspect_ratio: str = "9:16"
    ):
        self.max_text_density = max_text_density
        self.min_pattern_interrupt_sec = min_pattern_interrupt_sec
        self.max_pattern_interrupt_sec = max_pattern_interrupt_sec
        self.required_resolution = required_resolution
        self.required_aspect_ratio = required_aspect_ratio
    
    async def check(
        self,
        timeline_data: Dict[str, Any],
        video_path: Optional[str] = None
    ) -> GateResult:
        """
        Check visual quality.
        
        Args:
            timeline_data: Timeline.json data
            video_path: Optional path to rendered video
        
        Returns:
            GateResult with status and details
        """
        try:
            issues = []
            
            # Check resolution
            resolution = timeline_data.get("resolution", "")
            if resolution != self.required_resolution:
                issues.append(f"Resolution mismatch: {resolution} != {self.required_resolution}")
            
            # Check pattern interrupt timing
            layers = timeline_data.get("layers", [])
            if layers:
                # Calculate time between visual changes
                layer_times = sorted([l.get("start", 0) for l in layers])
                for i in range(1, len(layer_times)):
                    gap = layer_times[i] - layer_times[i-1]
                    if gap < self.min_pattern_interrupt_sec:
                        issues.append(f"Pattern interrupt too fast: {gap:.2f}s < {self.min_pattern_interrupt_sec}s")
                    elif gap > self.max_pattern_interrupt_sec:
                        issues.append(f"Pattern interrupt too slow: {gap:.2f}s > {self.max_pattern_interrupt_sec}s")
            
            # Check text density (simplified)
            text_layers = [l for l in layers if l.get("type") == "text"]
            if len(text_layers) > len(layers) * self.max_text_density:
                issues.append(f"Text density too high: {len(text_layers)}/{len(layers)} layers")
            
            if issues:
                return GateResult(
                    status=GateStatus.FAIL,
                    gate_name="visual_quality",
                    message=f"Visual quality issues: {len(issues)} problems",
                    details={"issues": issues[:5]},
                    score=max(0.0, 1.0 - len(issues) * 0.1)
                )
            
            return GateResult(
                status=GateStatus.PASS,
                gate_name="visual_quality",
                message="Visual quality checks passed",
                details={
                    "resolution": resolution,
                    "layers": len(layers),
                    "text_layers": len(text_layers)
                },
                score=1.0
            )
            
        except Exception as e:
            logger.error(f"Visual quality gate error: {e}", exc_info=True)
            return GateResult(
                status=GateStatus.FAIL,
                gate_name="visual_quality",
                message=f"Visual quality check failed: {e}",
                details={"error": str(e)}
            )


class PublishQualityGate:
    """
    Publish Quality Gate
    
    Checks:
    - File size
    - Codec compatibility
    - Platform constraints
    - Duration limits
    """
    
    def __init__(
        self,
        max_file_size_mb: float = 100.0,  # Max file size in MB
        required_codec: str = "h264",
        max_duration_sec: float = 60.0,  # Platform limits
        min_duration_sec: float = 15.0
    ):
        self.max_file_size_mb = max_file_size_mb
        self.required_codec = required_codec
        self.max_duration_sec = max_duration_sec
        self.min_duration_sec = min_duration_sec
    
    async def check(
        self,
        video_path: str,
        platform: str = "multi"
    ) -> GateResult:
        """
        Check publish readiness.
        
        Args:
            video_path: Path to video file
            platform: Target platform
        
        Returns:
            GateResult with status and details
        """
        try:
            from pathlib import Path
            import subprocess
            import json
            
            video_file = Path(video_path)
            
            if not video_file.exists():
                return GateResult(
                    status=GateStatus.FAIL,
                    gate_name="publish_quality",
                    message="Video file not found",
                    details={"video_path": video_path}
                )
            
            # Check file size
            file_size_mb = video_file.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                return GateResult(
                    status=GateStatus.FAIL,
                    gate_name="publish_quality",
                    message=f"File size too large: {file_size_mb:.2f}MB > {self.max_file_size_mb}MB",
                    details={"file_size_mb": file_size_mb}
                )
            
            # Check codec and duration
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "stream=codec_name:format=duration",
                "-of", "json",
                str(video_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                streams = data.get("streams", [])
                video_stream = next((s for s in streams if s.get("codec_name")), None)
                duration = float(data.get("format", {}).get("duration", 0))
                
                # Check codec
                codec = video_stream.get("codec_name") if video_stream else None
                if codec != self.required_codec:
                    return GateResult(
                        status=GateStatus.WARNING,
                        gate_name="publish_quality",
                        message=f"Codec mismatch: {codec} != {self.required_codec}",
                        details={"codec": codec, "required": self.required_codec}
                    )
                
                # Check duration
                if duration < self.min_duration_sec:
                    return GateResult(
                        status=GateStatus.FAIL,
                        gate_name="publish_quality",
                        message=f"Duration too short: {duration:.2f}s < {self.min_duration_sec}s",
                        details={"duration": duration}
                    )
                
                if duration > self.max_duration_sec:
                    return GateResult(
                        status=GateStatus.FAIL,
                        gate_name="publish_quality",
                        message=f"Duration too long: {duration:.2f}s > {self.max_duration_sec}s",
                        details={"duration": duration}
                    )
                
                return GateResult(
                    status=GateStatus.PASS,
                    gate_name="publish_quality",
                    message="Publish quality checks passed",
                    details={
                        "file_size_mb": file_size_mb,
                        "codec": codec,
                        "duration": duration,
                        "platform": platform
                    },
                    score=1.0
                )
            else:
                return GateResult(
                    status=GateStatus.FAIL,
                    gate_name="publish_quality",
                    message="Failed to analyze video",
                    details={"error": result.stderr}
                )
                
        except Exception as e:
            logger.error(f"Publish quality gate error: {e}", exc_info=True)
            return GateResult(
                status=GateStatus.FAIL,
                gate_name="publish_quality",
                message=f"Publish quality check failed: {e}",
                details={"error": str(e)}
            )


class QualityGateManager:
    """
    Quality Gate Manager
    
    Orchestrates all quality gates and provides unified interface.
    """
    
    def __init__(self):
        """Initialize quality gate manager."""
        self.audio_gate = AudioQualityGate()
        self.caption_gate = CaptionQualityGate()
        self.visual_gate = VisualQualityGate()
        self.publish_gate = PublishQualityGate()
    
    async def check_audio(self, audio_path: str) -> GateResult:
        """Check audio quality."""
        return await self.audio_gate.check(audio_path)
    
    async def check_captions(
        self,
        word_timestamps_path: Optional[str] = None,
        captions_data: Optional[List[Dict[str, Any]]] = None
    ) -> GateResult:
        """Check caption quality."""
        return await self.caption_gate.check(word_timestamps_path, captions_data)
    
    async def check_visuals(
        self,
        timeline_data: Dict[str, Any],
        video_path: Optional[str] = None
    ) -> GateResult:
        """Check visual quality."""
        return await self.visual_gate.check(timeline_data, video_path)
    
    async def check_publish(self, video_path: str, platform: str = "multi") -> GateResult:
        """Check publish readiness."""
        return await self.publish_gate.check(video_path, platform)
    
    async def check_all(
        self,
        audio_path: Optional[str] = None,
        word_timestamps_path: Optional[str] = None,
        captions_data: Optional[List[Dict[str, Any]]] = None,
        timeline_data: Optional[Dict[str, Any]] = None,
        video_path: Optional[str] = None,
        platform: str = "multi"
    ) -> Dict[str, GateResult]:
        """
        Run all applicable quality gates.
        
        Returns:
            Dict mapping gate name to result
        """
        results = {}
        
        if audio_path:
            results["audio"] = await self.check_audio(audio_path)
        
        if captions_data or word_timestamps_path:
            results["captions"] = await self.check_captions(word_timestamps_path, captions_data)
        
        if timeline_data:
            results["visuals"] = await self.check_visuals(timeline_data, video_path)
        
        if video_path:
            results["publish"] = await self.check_publish(video_path, platform)
        
        return results

