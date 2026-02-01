"""
MediaPipe Selfie Segmentation Adapter
======================================
Adapter for MediaPipe - fast, lightweight matting for UGC content.
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from .base import MattingAdapter
from ..models import MattingRequest, MattingResponse, MattingConfig

logger = logging.getLogger(__name__)


class MediaPipeAdapter(MattingAdapter):
    """
    Adapter for MediaPipe Selfie Segmentation.
    
    Fast, lightweight matting that works on CPU.
    Best for UGC content, talking heads, selfies.
    """
    
    def __init__(self):
        super().__init__("mediapipe")
        self._model_available = False
        self._segmentation = None
        self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if MediaPipe is available."""
        try:
            import mediapipe as mp
            self._model_available = True
            return True
        except ImportError:
            logger.warning("MediaPipe not installed. Install with: pip install mediapipe")
            return False
    
    async def extract_foreground(
        self,
        request: MattingRequest,
        output_path: Optional[str] = None
    ) -> MattingResponse:
        """Extract foreground using MediaPipe."""
        if not self._model_available:
            return MattingResponse(
                job_id=request.job_id,
                success=False,
                error="MediaPipe not available. Install with: pip install mediapipe",
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
            import mediapipe as mp
            
            # Initialize MediaPipe
            mp_selfie_segmentation = mp.solutions.selfie_segmentation
            selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
                model_selection=1  # 0 = general, 1 = landscape (better quality)
            )
            
            # Open video
            cap = cv2.VideoCapture(str(source_path))
            if not cap.isOpened():
                return MattingResponse(
                    job_id=request.job_id,
                    success=False,
                    error=f"Could not open video: {request.source_video}",
                    correlation_id=request.correlation_id
                )
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Setup video writer (MOV with alpha channel)
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
            # Note: OpenCV doesn't support alpha channel in video directly
            # We'll output mask separately and composite with FFmpeg
            
            mask_output = output_file.parent / f"{output_file.stem}_mask.mp4"
            temp_output = output_file.parent / f"{output_file.stem}_temp.mp4"
            
            out = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
            mask_out = cv2.VideoWriter(str(mask_output), fourcc, fps, (width, height), isColor=False)
            
            frames_processed = 0
            
            logger.info(f"MediaPipe: Processing {total_frames} frames...")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = selfie_segmentation.process(rgb_frame)
                
                # Get segmentation mask
                mask = results.segmentation_mask
                
                # Convert mask to 8-bit
                mask_8bit = (mask * 255).astype(np.uint8)
                
                # Apply mask to frame (for preview)
                masked_frame = cv2.bitwise_and(frame, frame, mask=mask_8bit)
                
                # Write frames
                out.write(masked_frame)
                mask_out.write(mask_8bit)
                
                frames_processed += 1
                
                if frames_processed % 30 == 0:
                    logger.info(f"MediaPipe: Processed {frames_processed}/{total_frames} frames")
            
            # Release resources
            cap.release()
            out.release()
            mask_out.release()
            selfie_segmentation.close()
            
            # Composite with FFmpeg to add alpha channel
            logger.info("MediaPipe: Compositing with alpha channel...")
            self._composite_with_alpha(str(temp_output), str(mask_output), str(output_file))
            
            # Cleanup temp files
            if temp_output.exists():
                temp_output.unlink()
            if mask_output.exists():
                mask_output.unlink()
            
            processing_time = time.time() - start_time
            
            if output_file.exists():
                return MattingResponse(
                    job_id=request.job_id,
                    success=True,
                    output_path=str(output_file),
                    mask_path=None,  # Mask is embedded in output
                    processing_time=processing_time,
                    model_used=self.model_name,
                    frames_processed=frames_processed,
                    correlation_id=request.correlation_id
                )
            else:
                return MattingResponse(
                    job_id=request.job_id,
                    success=False,
                    error="MediaPipe processing completed but output file not found",
                    correlation_id=request.correlation_id
                )
                
        except Exception as e:
            logger.error(f"MediaPipe processing error: {e}", exc_info=True)
            return MattingResponse(
                job_id=request.job_id,
                success=False,
                error=str(e),
                correlation_id=request.correlation_id
            )
    
    def _composite_with_alpha(self, video_path: str, mask_path: str, output_path: str):
        """Composite video with alpha channel using FFmpeg."""
        import subprocess
        
        # Use FFmpeg to composite video with alpha channel
        # This creates a MOV file with alpha channel
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", mask_path,
            "-filter_complex", "[1:v]format=gray,geq=lum='lum(X,Y)':a='if(gt(lum(X,Y),10),255,0)'[alpha];[0:v][alpha]alphamerge[out]",
            "-map", "[out]",
            "-c:v", "libx264",
            "-pix_fmt", "yuva420p",  # Alpha channel support
            "-preset", "medium",
            "-crf", "23",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode != 0:
            raise Exception(f"FFmpeg compositing failed: {result.stderr.decode()}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get MediaPipe model information."""
        return {
            "name": "MediaPipe Selfie Segmentation",
            "version": "1.0",
            "provider": "Google",
            "capabilities": {
                "video_matting": True,
                "temporal_memory": False,
                "alpha_channel": True,
                "real_time": True,
                "gpu_required": False,
                "gpu_recommended": False,  # Works well on CPU
            },
            "model_variants": ["general", "landscape"],
            "performance": {
                "cpu_fps": 30,  # Approximate on modern CPU
                "gpu_fps": 60,  # Approximate with GPU
            },
            "github": "https://github.com/google/mediapipe",
        }
    
    async def load_model(self) -> bool:
        """Load MediaPipe model."""
        if not self._model_available:
            return False
        
        try:
            import mediapipe as mp
            mp_selfie_segmentation = mp.solutions.selfie_segmentation
            self._segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
            self._loaded = True
            return True
        except Exception as e:
            logger.error(f"Failed to load MediaPipe: {e}")
            return False
    
    async def unload_model(self) -> bool:
        """Unload MediaPipe model."""
        if self._segmentation:
            try:
                self._segmentation.close()
            except Exception:
                pass
            self._segmentation = None
        self._loaded = False
        return True

