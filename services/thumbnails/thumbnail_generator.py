"""
Intelligent Thumbnail Generator
Selects best frames and generates platform-specific thumbnails with AI enhancements
Uses ModelRegistry for configurable model selection (Groq by default, 100% cost savings)
"""
import logging
import os
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import subprocess
import json
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from dataclasses import dataclass
import base64
from io import BytesIO

from config.model_registry import TaskType, ModelRegistry
from services.ai_client import AIClient

logger = logging.getLogger(__name__)


@dataclass
class PlatformDimensions:
    """Social media platform thumbnail dimensions"""
    width: int
    height: int
    aspect_ratio: str
    platform: str
    orientation: str  # 'landscape', 'portrait', 'square'


# Platform-specific dimensions
PLATFORM_DIMENSIONS = {
    "youtube": PlatformDimensions(1280, 720, "16:9", "YouTube", "landscape"),
    "youtube_short": PlatformDimensions(1080, 1920, "9:16", "YouTube Shorts", "portrait"),
    "tiktok": PlatformDimensions(1080, 1920, "9:16", "TikTok", "portrait"),
    "instagram_feed": PlatformDimensions(1080, 1080, "1:1", "Instagram Feed", "square"),
    "instagram_story": PlatformDimensions(1080, 1920, "9:16", "Instagram Story", "portrait"),
    "instagram_reel": PlatformDimensions(1080, 1920, "9:16", "Instagram Reel", "portrait"),
    "facebook": PlatformDimensions(1200, 630, "1.91:1", "Facebook", "landscape"),
    "twitter": PlatformDimensions(1200, 675, "16:9", "Twitter/X", "landscape"),
    "linkedin": PlatformDimensions(1200, 627, "1.91:1", "LinkedIn", "landscape"),
    "pinterest": PlatformDimensions(1000, 1500, "2:3", "Pinterest", "portrait"),
    "snapchat": PlatformDimensions(1080, 1920, "9:16", "Snapchat", "portrait"),
    "threads": PlatformDimensions(1080, 1350, "4:5", "Threads", "portrait"),
}


class ThumbnailGenerator:
    """
    Intelligent thumbnail generation with best frame selection
    and AI-powered enhancements
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize thumbnail generator using ModelRegistry
        
        Args:
            openai_api_key: Optional API key (deprecated, use ModelRegistry instead)
        """
        # Get model configuration from registry
        self.config = ModelRegistry.get_model_config(TaskType.THUMBNAIL_GENERATION)
        try:
            self.client = AIClient(self.config)
            self.ai_enabled = True
            logger.info(f"ThumbnailGenerator using {self.config.provider}/{self.config.model}")
        except Exception as e:
            logger.warning(f"ThumbnailGenerator AI disabled: {e}")
            self.client = None
            self.ai_enabled = False
    
    def extract_frames(
        self,
        video_path: str,
        num_frames: int = 10,
        output_dir: Optional[str] = None
    ) -> List[str]:
        """
        Extract candidate frames from video for analysis
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            output_dir: Directory to save frames
            
        Returns:
            List of frame file paths
        """
        if output_dir is None:
            output_dir = "/tmp/thumbnail_candidates"
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Check if input is an image
        ext = Path(video_path).suffix.lower()
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
        heic_extensions = {'.heic', '.heif'}
        
        if ext in image_extensions:
            logger.info(f"Input is an image {ext}, skipping frame extraction")
            return [video_path]
            
        if ext in heic_extensions:
            logger.info(f"Input is HEIC image {ext}, converting to color JPG")
            output_path = f"{output_dir}/{Path(video_path).stem}.jpg"
            
            # Try pillow-heif first (better color handling)
            try:
                import pillow_heif
                pillow_heif.register_heif_opener()
                heic_img = Image.open(video_path)
                
                # Ensure RGB color mode
                if heic_img.mode != 'RGB':
                    heic_img = heic_img.convert('RGB')
                
                # Apply slight color enhancement to ensure vibrant output
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Color(heic_img)
                heic_img = enhancer.enhance(1.05)  # Slight boost
                
                heic_img.save(output_path, "JPEG", quality=95)
                logger.info(f"Converted HEIC to color JPG using pillow-heif: {output_path}")
                return [output_path]
                
            except ImportError:
                logger.warning("pillow-heif not installed, falling back to ffmpeg")
            except Exception as e:
                logger.warning(f"pillow-heif failed: {e}, trying ffmpeg")
            
            # Fallback to ffmpeg with improved color handling
            # Probe for streams to find the main color image (usually largest resolution)
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v",
                "-show_entries", "stream=index,width,height",
                "-of", "json",
                video_path
            ]
            
            try:
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
                streams = json.loads(probe_result.stdout).get("streams", [])
                
                # Find the stream with the largest resolution (main color image)
                best_stream = 0
                max_pixels = 0
                for stream in streams:
                    pixels = stream.get("width", 0) * stream.get("height", 0)
                    if pixels > max_pixels:
                        max_pixels = pixels
                        best_stream = stream.get("index", 0)
                
                logger.info(f"HEIC: Selected stream {best_stream} with {max_pixels} pixels")
                
            except Exception as e:
                logger.warning(f"Could not probe HEIC streams: {e}")
                best_stream = 0
            
            # Convert using ffmpeg with proper color settings
            convert_cmd = [
                "ffmpeg",
                "-i", video_path,
                "-map", f"0:v:{best_stream}",  # Select best resolution stream
                "-vframes", "1",
                "-pix_fmt", "yuvj444p",  # Full color range
                "-colorspace", "bt709",  # Standard color space
                "-color_primaries", "bt709",
                "-color_trc", "bt709",
                "-q:v", "2",  # High quality
                "-y",
                output_path
            ]
            
            try:
                result = subprocess.run(convert_cmd, capture_output=True, check=True)
                
                # Verify the output has color by checking if it's not grayscale
                verify_img = Image.open(output_path)
                if verify_img.mode == 'L':  # Grayscale
                    logger.warning("HEIC converted to grayscale, forcing RGB")
                    verify_img = verify_img.convert('RGB')
                    verify_img.save(output_path, "JPEG", quality=95)
                
                logger.info(f"Converted HEIC to color JPG using ffmpeg: {output_path}")
                return [output_path]
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Error converting HEIC: {e.stderr.decode() if e.stderr else e}")
                return []
        
        # It's a video, proceed with frame extraction
        
        # Get video duration
        probe_cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            video_path
        ]
        
        try:
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            if "format" not in data or "duration" not in data["format"]:
                logger.warning(f"Could not determine duration for {video_path}, treating as image/short video")
                # Try to extract just one frame at 0s
                output_path = f"{output_dir}/frame_000.jpg"
                extract_cmd = [
                    "ffmpeg",
                    "-i", video_path,
                    "-vframes", "1",
                    "-y",
                    output_path
                ]
                subprocess.run(extract_cmd, capture_output=True, check=True)
                return [output_path]
                
            duration = float(data["format"]["duration"])
            
            # Extract frames at evenly spaced intervals
            frame_paths = []
            interval = duration / (num_frames + 1)
            
            for i in range(1, num_frames + 1):
                timestamp = interval * i
                output_path = f"{output_dir}/frame_{i:03d}.jpg"
                
                extract_cmd = [
                    "ffmpeg",
                    "-ss", str(timestamp),
                    "-i", video_path,
                    "-vframes", "1",
                    "-q:v", "2",  # High quality
                    "-y",
                    output_path
                ]
                
                subprocess.run(extract_cmd, capture_output=True, check=True)
                frame_paths.append(output_path)
                
                logger.info(f"Extracted frame at {timestamp:.2f}s -> {output_path}")
            
            return frame_paths
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return []
    
    def analyze_frame_quality(self, frame_path: str) -> Dict[str, Any]:
        """
        Analyze a frame's suitability for thumbnail use
        
        Considers:
        - Sharpness/clarity
        - Brightness/contrast
        - Face detection
        - Color vibrancy
        - Composition (rule of thirds)
        
        Args:
            frame_path: Path to frame image
            
        Returns:
            Quality metrics dict
        """
        try:
            img = Image.open(frame_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate sharpness (Laplacian variance)
            import cv2
            import numpy as np
            
            cv_img = cv2.imread(frame_path)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 500, 1.0)  # Normalize
            
            # Calculate brightness
            brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Prefer mid-range
            
            # Calculate contrast
            contrast = gray.std() / 128.0
            contrast_score = min(contrast, 1.0)
            
            # Detect faces (simple Haar cascade)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            face_score = min(len(faces) * 0.3, 1.0)  # Bonus for faces
            
            # Calculate color vibrancy
            hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
            saturation = np.mean(hsv[:, :, 1]) / 255.0
            vibrancy_score = saturation
            
            # Overall score (weighted average)
            overall_score = (
                sharpness_score * 0.3 +
                brightness_score * 0.2 +
                contrast_score * 0.2 +
                face_score * 0.2 +
                vibrancy_score * 0.1
            )
            
            return {
                "sharpness": round(sharpness_score, 3),
                "brightness": round(brightness_score, 3),
                "contrast": round(contrast_score, 3),
                "faces_detected": len(faces),
                "face_score": round(face_score, 3),
                "vibrancy": round(vibrancy_score, 3),
                "overall_score": round(overall_score, 3),
                "frame_path": frame_path
            }
            
        except Exception as e:
            logger.error(f"Error analyzing frame {frame_path}: {e}")
            return {"overall_score": 0.0, "frame_path": frame_path}
    
    def select_best_frame(
        self,
        video_path: str,
        num_candidates: int = 10
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Select the best frame from video for thumbnail
        
        Args:
            video_path: Path to video file
            num_candidates: Number of candidate frames to analyze
            
        Returns:
            Tuple of (best_frame_path, analysis_data)
        """
        # Extract candidate frames
        frames = self.extract_frames(video_path, num_frames=num_candidates)
        
        if not frames:
            raise ValueError("No frames could be extracted from video")
        
        # Analyze each frame
        analyses = []
        for frame_path in frames:
            analysis = self.analyze_frame_quality(frame_path)
            analyses.append(analysis)
        
        # Sort by overall score
        analyses.sort(key=lambda x: x["overall_score"], reverse=True)
        
        best_frame = analyses[0]
        
        logger.info(f"Best frame selected: {best_frame['frame_path']} (score: {best_frame['overall_score']})")
        
        return best_frame["frame_path"], best_frame

    def select_best_from_frames(
        self,
        frame_paths: List[str]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Select best frame from a list of existing frame paths
        """
        if not frame_paths:
            raise ValueError("No frames provided")
            
        # Analyze each frame
        analyses = []
        for frame_path in frame_paths:
            analysis = self.analyze_frame_quality(frame_path)
            analyses.append(analysis)
        
        # Sort by overall score
        analyses.sort(key=lambda x: x["overall_score"], reverse=True)
        
        best_frame = analyses[0]
        logger.info(f"Best frame selected: {best_frame['frame_path']} (score: {best_frame['overall_score']})")
        
        return best_frame["frame_path"], best_frame
    
    def generate_thumbnail(
        self,
        source_image: str,
        platform: str,
        output_path: str,
        crop_mode: str = "smart"
    ) -> str:
        """
        Generate platform-specific thumbnail from source image
        
        Args:
            source_image: Path to source image
            platform: Platform key (e.g., 'youtube', 'tiktok')
            output_path: Where to save thumbnail
            crop_mode: 'smart', 'center', 'top', 'bottom'
            
        Returns:
            Path to generated thumbnail
        """
        if platform not in PLATFORM_DIMENSIONS:
            raise ValueError(f"Unknown platform: {platform}")
        
        dims = PLATFORM_DIMENSIONS[platform]
        img = Image.open(source_image)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Calculate aspect ratios
        source_ratio = img.width / img.height
        target_ratio = dims.width / dims.height
        
        # Smart crop - maintain focal point
        if crop_mode == "smart":
            if source_ratio > target_ratio:
                # Source is wider - crop sides
                new_width = int(img.height * target_ratio)
                left = (img.width - new_width) // 2
                img = img.crop((left, 0, left + new_width, img.height))
            else:
                # Source is taller - crop top/bottom
                new_height = int(img.width / target_ratio)
                top = (img.height - new_height) // 3  # Crop more from bottom
                img = img.crop((0, top, img.width, top + new_height))
        
        # Resize to target dimensions
        img = img.resize((dims.width, dims.height), Image.Resampling.LANCZOS)
        
        # Enhance for thumbnail viewing
        img = self._enhance_thumbnail(img)
        
        # Save
        img.save(output_path, "JPEG", quality=95, optimize=True)
        
        logger.info(f"Generated {platform} thumbnail: {output_path}")
        
        return output_path
    
    def generate_all_platforms(
        self,
        source_image: str,
        output_dir: str,
        base_name: str = "thumbnail"
    ) -> Dict[str, str]:
        """
        Generate thumbnails for all social media platforms
        
        Args:
            source_image: Path to source image
            output_dir: Directory to save thumbnails
            base_name: Base filename for thumbnails
            
        Returns:
            Dict mapping platform to thumbnail path
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        thumbnails = {}
        
        for platform in PLATFORM_DIMENSIONS.keys():
            output_path = f"{output_dir}/{base_name}_{platform}.jpg"
            
            try:
                self.generate_thumbnail(source_image, platform, output_path)
                thumbnails[platform] = output_path
            except Exception as e:
                logger.error(f"Error generating {platform} thumbnail: {e}")
        
        return thumbnails
    
    def _enhance_thumbnail(self, img: Image.Image) -> Image.Image:
        """Apply enhancements to make thumbnail more appealing"""
        # Increase sharpness slightly
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.2)
        
        # Increase saturation slightly
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.1)
        
        # Increase contrast slightly
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)
        
        return img
    
    async def add_ai_text_overlay(
        self,
        image_path: str,
        title: str,
        output_path: str,
        style: str = "bold"
    ) -> str:
        """
        Add AI-generated catchy text overlay to thumbnail
        
        Args:
            image_path: Path to thumbnail image
            title: Video title to base text on
            output_path: Where to save enhanced thumbnail
            style: Text style ('bold', 'minimal', 'colorful')
            
        Returns:
            Path to enhanced thumbnail
        """
        if not self.ai_enabled or not self.client:
            logger.warning("AI not enabled - skipping AI enhancement")
            # Just copy the file
            import shutil
            shutil.copy(image_path, output_path)
            return output_path
        
        try:
            # Generate catchy thumbnail text using AI
            prompt = f"""Create a short, catchy text overlay for a YouTube thumbnail.

Video title: "{title}"

Generate ONLY the text that should appear on the thumbnail. Make it:
- Very short (1-5 words maximum)
- Attention-grabbing
- Easy to read
- Creates curiosity

Return only the text, nothing else."""
            
            # Use AIClient unified interface
            overlay_text = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are an expert at creating viral YouTube thumbnails."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=20
            ).strip().strip('"')
            
            # Add text overlay to image
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)
            
            # Try to load a bold font, fallback to default
            try:
                font_size = int(img.width * 0.1)  # 10% of width
                font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()
            
            # Get text size
            bbox = draw.textbbox((0, 0), overlay_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Position text (bottom third, centered)
            x = (img.width - text_width) // 2
            y = int(img.height * 0.7)
            
            # Draw text with outline for visibility
            outline_width = max(2, int(font_size * 0.05))
            
            # Draw outline
            for adj_x in range(-outline_width, outline_width + 1):
                for adj_y in range(-outline_width, outline_width + 1):
                    draw.text((x + adj_x, y + adj_y), overlay_text, font=font, fill=(0, 0, 0, 255))
            
            # Draw main text
            draw.text((x, y), overlay_text, font=font, fill=(255, 255, 255, 255))
            
            # Save
            img.save(output_path, "JPEG", quality=95)
            
            logger.info(f"Added AI text overlay: '{overlay_text}' to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error adding AI text overlay: {e}")
            # Fallback - just copy the original
            import shutil
            shutil.copy(image_path, output_path)
            return output_path
