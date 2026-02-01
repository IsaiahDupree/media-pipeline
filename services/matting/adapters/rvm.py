"""
RVM (Robust Video Matting) Adapter
===================================
Adapter for RVM - highest quality video matting with temporal memory.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from .base import MattingAdapter
from ..models import MattingRequest, MattingResponse, MattingConfig

logger = logging.getLogger(__name__)


class RVMAdapter(MattingAdapter):
    """
    Adapter for Robust Video Matting (RVM).
    
    Highest quality matting with temporal memory for stable results.
    Requires GPU (CUDA) for best performance, but can run on CPU.
    """
    
    def __init__(self):
        super().__init__("rvm")
        self._model_available = False
        self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if RVM is available."""
        try:
            # Try importing inference module
            # RVM uses: from inference import convert_video
            import importlib.util
            
            # Check if rvm package is installed
            spec = importlib.util.find_spec("inference")
            if spec is None:
                logger.warning("RVM not installed. Install with: pip install git+https://github.com/PeterL1n/RobustVideoMatting.git")
                return False
            
            self._model_available = True
            return True
        except Exception as e:
            logger.warning(f"RVM availability check failed: {e}")
            return False
    
    async def extract_foreground(
        self,
        request: MattingRequest,
        output_path: Optional[str] = None
    ) -> MattingResponse:
        """Extract foreground using RVM."""
        if not self._model_available:
            return MattingResponse(
                job_id=request.job_id,
                success=False,
                error="RVM not available. Install with: pip install git+https://github.com/PeterL1n/RobustVideoMatting.git",
                correlation_id=request.correlation_id
            )
        
        output_file = self._ensure_output_path(request, output_path)
        source_path = Path(request.source_video)
        
        if not source_path.exists():
            return MattingResponse(
                job_id=request.job_id,
                success=False,
                error=f"Source video not found: {request.source_video}",
                correlation_id=request.correlation_id
            )
        
        import time
        start_time = time.time()
        
        try:
            # Import RVM inference
            from inference import convert_video
            
            # Determine model variant
            model_variant = request.config.model_variant or "mobilenetv3"  # or "resnet50"
            model_path = f"rvm_{model_variant}.pth"
            
            # Determine device
            device = self._get_device(request.config)
            
            # Convert video
            logger.info(f"RVM: Processing {source_path.name} with {model_variant} on {device}")
            
            convert_video(
                source=str(source_path),
                output=str(output_file),
                model=model_path,
                downsample_ratio=request.config.downsample_ratio,
                device=device
            )
            
            processing_time = time.time() - start_time
            
            if output_file.exists():
                # Get frame count (simplified)
                frames = self._estimate_frame_count(output_file)
                
                return MattingResponse(
                    job_id=request.job_id,
                    success=True,
                    output_path=str(output_file),
                    processing_time=processing_time,
                    model_used=self.model_name,
                    frames_processed=frames,
                    correlation_id=request.correlation_id
                )
            else:
                return MattingResponse(
                    job_id=request.job_id,
                    success=False,
                    error="RVM processing completed but output file not found",
                    correlation_id=request.correlation_id
                )
                
        except ImportError as e:
            logger.error(f"RVM import error: {e}")
            return MattingResponse(
                job_id=request.job_id,
                success=False,
                error=f"RVM not properly installed: {e}",
                correlation_id=request.correlation_id
            )
        except Exception as e:
            logger.error(f"RVM processing error: {e}", exc_info=True)
            return MattingResponse(
                job_id=request.job_id,
                success=False,
                error=str(e),
                correlation_id=request.correlation_id
            )
    
    def _estimate_frame_count(self, video_path: Path) -> int:
        """Estimate frame count using ffprobe."""
        try:
            import subprocess
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-count_packets",
                    "-show_entries", "stream=nb_read_packets",
                    "-of", "csv=p=0",
                    str(video_path)
                ],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception:
            pass
        
        # Fallback: estimate from duration
        try:
            import subprocess
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(video_path)
                ],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                duration = float(result.stdout.strip())
                # Assume 30 FPS
                return int(duration * 30)
        except Exception:
            pass
        
        return 0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get RVM model information."""
        return {
            "name": "Robust Video Matting",
            "version": "1.0",
            "provider": "ByteDance",
            "capabilities": {
                "video_matting": True,
                "temporal_memory": True,
                "alpha_channel": True,
                "real_time": True,
                "gpu_required": False,  # Works on CPU but slower
                "gpu_recommended": True,
            },
            "model_variants": ["mobilenetv3", "resnet50"],
            "performance": {
                "4k_fps": 76,  # On GTX 1080 Ti
                "hd_fps": 104,  # On GTX 1080 Ti
            },
            "github": "https://github.com/PeterL1n/RobustVideoMatting",
        }
    
    async def load_model(self) -> bool:
        """Load RVM model."""
        if not self._model_available:
            return False
        
        # RVM loads model on first use
        # Could pre-load here if needed
        self._loaded = True
        return True
    
    async def unload_model(self) -> bool:
        """Unload RVM model."""
        self._loaded = False
        return True

