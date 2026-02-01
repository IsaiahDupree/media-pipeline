"""
BlankLogo Watermark Removal Service
Integrates with local BlankLogo for Sora video watermark removal.
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime, timezone
from loguru import logger


BLANKLOGO_PATH = "/Users/isaiahdupree/Documents/Software/ai-video-platform"
OUTPUT_DIR = "/Users/isaiahdupree/Documents/Software/MediaPoster/Backend/data/sora_processed"


class WatermarkRemovalService:
    """
    Service to remove watermarks from Sora videos using BlankLogo.
    """
    
    def __init__(self, blanklogo_path: str = BLANKLOGO_PATH):
        self.blanklogo_path = Path(blanklogo_path)
        self.output_dir = Path(OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify BlankLogo exists
        self.is_available = self._check_blanklogo()
        if self.is_available:
            logger.info(f"✅ WatermarkRemovalService initialized (BlankLogo at {blanklogo_path})")
        else:
            logger.warning(f"⚠️  BlankLogo not found at {blanklogo_path}")
    
    def _check_blanklogo(self) -> bool:
        """Check if BlankLogo is available."""
        app_path = self.blanklogo_path / "apps" / "watermark-remover"
        return app_path.exists() or self.blanklogo_path.exists()
    
    async def remove_watermark(
        self,
        input_path: str,
        output_filename: Optional[str] = None,
        job_id: Optional[str] = None
    ) -> Dict:
        """
        Remove watermark from a video file.
        
        Args:
            input_path: Path to input video with watermark
            output_filename: Optional custom output filename
            job_id: Optional job ID for tracking
        
        Returns:
            Dict with success status and output path
        """
        input_file = Path(input_path)
        
        if not input_file.exists():
            return {
                "success": False,
                "error": f"Input file not found: {input_path}",
                "job_id": job_id
            }
        
        # Generate output filename
        if output_filename:
            output_path = self.output_dir / output_filename
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = input_file.stem
            output_path = self.output_dir / f"{stem}_clean_{timestamp}.mp4"
        
        try:
            # Method 1: Try BlankLogo Node.js app
            if self._try_blanklogo_node(input_path, str(output_path)):
                logger.info(f"✅ Watermark removed via BlankLogo: {output_path}")
                return {
                    "success": True,
                    "output_path": str(output_path),
                    "method": "blanklogo_node",
                    "job_id": job_id
                }
            
            # Method 2: Try FFmpeg-based removal (crop/blur approach)
            if self._try_ffmpeg_removal(input_path, str(output_path)):
                logger.info(f"✅ Watermark removed via FFmpeg: {output_path}")
                return {
                    "success": True,
                    "output_path": str(output_path),
                    "method": "ffmpeg",
                    "job_id": job_id
                }
            
            # Method 3: Copy original as fallback
            shutil.copy2(input_path, output_path)
            logger.warning(f"⚠️  Using original video (watermark removal failed): {output_path}")
            return {
                "success": True,
                "output_path": str(output_path),
                "method": "fallback_copy",
                "job_id": job_id,
                "warning": "Watermark removal failed, using original"
            }
            
        except Exception as e:
            logger.error(f"Watermark removal failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "job_id": job_id
            }
    
    def _try_blanklogo_node(self, input_path: str, output_path: str) -> bool:
        """Try BlankLogo Node.js watermark remover."""
        try:
            # Check if the Node.js app exists
            app_dir = self.blanklogo_path / "apps" / "watermark-remover"
            if not app_dir.exists():
                # Try alternate structure
                app_dir = self.blanklogo_path
            
            # Look for the main script
            possible_scripts = [
                app_dir / "index.js",
                app_dir / "src" / "index.js",
                app_dir / "dist" / "index.js",
                self.blanklogo_path / "scripts" / "remove-watermark.js"
            ]
            
            script_path = None
            for p in possible_scripts:
                if p.exists():
                    script_path = p
                    break
            
            if not script_path:
                logger.debug("BlankLogo script not found, trying alternate methods")
                return False
            
            # Run the Node.js script
            result = subprocess.run(
                ["node", str(script_path), "--input", input_path, "--output", output_path],
                cwd=str(self.blanklogo_path),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0 and Path(output_path).exists():
                return True
            
            logger.debug(f"BlankLogo failed: {result.stderr}")
            return False
            
        except subprocess.TimeoutExpired:
            logger.warning("BlankLogo timed out")
            return False
        except Exception as e:
            logger.debug(f"BlankLogo error: {e}")
            return False
    
    def _try_ffmpeg_removal(self, input_path: str, output_path: str) -> bool:
        """
        Try FFmpeg-based watermark removal.
        Uses a combination of delogo filter and cropping.
        """
        try:
            # Sora watermark is typically in bottom-right corner
            # We'll use delogo filter with coordinates for common Sora watermark position
            
            # First, get video dimensions
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=s=x:p=0",
                input_path
            ]
            
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
            if probe_result.returncode != 0:
                return False
            
            dimensions = probe_result.stdout.strip().split('x')
            if len(dimensions) != 2:
                return False
            
            width, height = int(dimensions[0]), int(dimensions[1])
            
            # Sora watermark typical position (bottom-right, ~200x50 pixels)
            logo_x = width - 220
            logo_y = height - 60
            logo_w = 200
            logo_h = 50
            
            # Apply delogo filter
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-vf", f"delogo=x={logo_x}:y={logo_y}:w={logo_w}:h={logo_h}:show=0",
                "-c:a", "copy",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                output_path
            ]
            
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0 and Path(output_path).exists():
                return True
            
            logger.debug(f"FFmpeg delogo failed: {result.stderr[:200]}")
            return False
            
        except Exception as e:
            logger.debug(f"FFmpeg removal error: {e}")
            return False
    
    async def batch_remove_watermarks(
        self,
        input_paths: list,
        job_ids: Optional[list] = None
    ) -> list:
        """
        Remove watermarks from multiple videos.
        
        Args:
            input_paths: List of input video paths
            job_ids: Optional list of job IDs for tracking
        
        Returns:
            List of results for each video
        """
        results = []
        
        for i, input_path in enumerate(input_paths):
            job_id = job_ids[i] if job_ids and i < len(job_ids) else None
            result = await self.remove_watermark(input_path, job_id=job_id)
            results.append(result)
            
            # Small delay between processing
            import asyncio
            await asyncio.sleep(0.5)
        
        return results


# =============================================================================
# SINGLETON
# =============================================================================

_watermark_instance: Optional[WatermarkRemovalService] = None

def get_watermark_service() -> WatermarkRemovalService:
    """Get singleton instance of WatermarkRemovalService."""
    global _watermark_instance
    if _watermark_instance is None:
        _watermark_instance = WatermarkRemovalService()
    return _watermark_instance
