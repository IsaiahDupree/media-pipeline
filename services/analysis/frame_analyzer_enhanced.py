"""
Enhanced Frame Analyzer for Viral Video Analysis

Analyzes video frames at a detailed level for:
- Shot types (close-up, medium, wide, etc.)
- Camera movement (static, pan, zoom, etc.)
- Face presence and eye contact
- On-screen text detection
- Visual clutter/composition
- Visual hooks (memes, emojis, objects)
- Scene changes
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import os


class ShotType(str, Enum):
    """Types of camera shots"""
    EXTREME_CLOSE_UP = "extreme_close_up"  # Eyes, mouth
    CLOSE_UP = "close_up"  # Face, head
    MEDIUM_CLOSE_UP = "medium_close_up"  # Chest up
    MEDIUM = "medium"  # Waist up
    MEDIUM_WIDE = "medium_wide"  # Full body
    WIDE = "wide"  # Environment visible
    EXTREME_WIDE = "extreme_wide"  # Landscape
    SCREEN_RECORD = "screen_record"  # Computer screen


class CameraMotion(str, Enum):
    """Types of camera movement"""
    STATIC = "static"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    HANDHELD = "handheld"
    SMOOTH_GIMBAL = "smooth_gimbal"


@dataclass
class FrameAnalysis:
    """Analysis results for a single frame"""
    frame_number: int
    timestamp_s: float
    
    # Shot composition
    shot_type: Optional[str] = None
    camera_motion: Optional[str] = None
    
    # Presence detection
    has_face: bool = False
    face_count: int = 0
    eye_contact_detected: bool = False
    face_size_ratio: Optional[float] = None  # Face area / frame area
    
    # Visual elements
    has_text: bool = False
    text_area_ratio: Optional[float] = None
    has_emoji: bool = False
    emoji_count: int = 0
    
    # Composition
    visual_clutter_score: Optional[float] = None  # 0-1, higher = more cluttered
    contrast_score: Optional[float] = None  # 0-1, higher = better contrast
    color_palette: Optional[List[str]] = None  # Dominant colors
    
    # Motion
    motion_score: Optional[float] = None  # 0-1, amount of motion in frame
    scene_change: bool = False
    
    # Objects detected
    objects_detected: Optional[List[str]] = None
    
    # Meme/viral elements
    has_meme_format: bool = False
    meme_type: Optional[str] = None


class FrameAnalyzerEnhanced:
    """Enhanced frame analyzer with computer vision"""
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize frame analyzer
        
        Args:
            use_gpu: Whether to use GPU acceleration for OpenCV
        """
        self.use_gpu = use_gpu
        
        # Load face cascade for detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Load eye cascade for eye contact detection
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    
    def extract_frames(
        self, 
        video_path: str, 
        interval_s: float = 0.5,
        max_frames: Optional[int] = None
    ) -> List[Tuple[int, float, np.ndarray]]:
        """
        Extract frames from video at regular intervals
        
        Args:
            video_path: Path to video file
            interval_s: Seconds between frames
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of (frame_number, timestamp, frame_array) tuples
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps == 0:
            raise ValueError("Could not read video FPS")
        
        frame_interval = int(fps * interval_s)
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                timestamp_s = frame_count / fps
                frames.append((frame_count, timestamp_s, frame))
                
                if max_frames and len(frames) >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def analyze_frame(
        self, 
        frame: np.ndarray, 
        frame_number: int,
        timestamp_s: float,
        prev_frame: Optional[np.ndarray] = None
    ) -> FrameAnalysis:
        """
        Analyze a single frame
        
        Args:
            frame: Frame array (BGR format from OpenCV)
            frame_number: Frame index
            timestamp_s: Timestamp in video
            prev_frame: Previous frame for motion detection
            
        Returns:
            FrameAnalysis object
        """
        analysis = FrameAnalysis(
            frame_number=frame_number,
            timestamp_s=timestamp_s
        )
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        analysis.has_face = len(faces) > 0
        analysis.face_count = len(faces)
        
        # Analyze largest face
        if len(faces) > 0:
            # Sort by area to get largest face
            faces_sorted = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            x, y, w, h = faces_sorted[0]
            
            # Calculate face size ratio
            frame_area = frame.shape[0] * frame.shape[1]
            face_area = w * h
            analysis.face_size_ratio = face_area / frame_area
            
            # Estimate shot type from face size
            analysis.shot_type = self._estimate_shot_type(analysis.face_size_ratio)
            
            # Check for eye contact (eyes in top half of face)
            face_roi = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(face_roi)
            analysis.eye_contact_detected = len(eyes) >= 2
        else:
            # No face - likely screen record or wide shot
            analysis.shot_type = ShotType.SCREEN_RECORD
        
        # Analyze visual clutter
        analysis.visual_clutter_score = self._calculate_clutter(frame)
        
        # Analyze contrast
        analysis.contrast_score = self._calculate_contrast(gray)
        
        # Detect text regions (simple method using edge detection)
        analysis.has_text, analysis.text_area_ratio = self._detect_text_regions(frame)
        
        # Extract dominant colors
        analysis.color_palette = self._extract_color_palette(frame, n_colors=3)
        
        # Detect motion if previous frame available
        if prev_frame is not None:
            analysis.motion_score = self._calculate_motion(prev_frame, frame)
            analysis.scene_change = self._detect_scene_change(prev_frame, frame)
        
        return analysis
    
    def analyze_video(
        self, 
        video_path: str,
        interval_s: float = 0.5,
        max_frames: Optional[int] = None
    ) -> List[FrameAnalysis]:
        """
        Analyze all frames in a video
        
        Args:
            video_path: Path to video file
            interval_s: Seconds between sampled frames
            max_frames: Maximum frames to analyze
            
        Returns:
            List of FrameAnalysis objects
        """
        frames = self.extract_frames(video_path, interval_s, max_frames)
        analyses = []
        prev_frame = None
        
        for frame_num, timestamp, frame in frames:
            analysis = self.analyze_frame(
                frame, 
                frame_num, 
                timestamp,
                prev_frame
            )
            analyses.append(analysis)
            prev_frame = frame
        
        return analyses
    
    def _estimate_shot_type(self, face_size_ratio: float) -> str:
        """Estimate shot type from face size"""
        if face_size_ratio > 0.4:
            return ShotType.EXTREME_CLOSE_UP
        elif face_size_ratio > 0.25:
            return ShotType.CLOSE_UP
        elif face_size_ratio > 0.15:
            return ShotType.MEDIUM_CLOSE_UP
        elif face_size_ratio > 0.08:
            return ShotType.MEDIUM
        elif face_size_ratio > 0.03:
            return ShotType.MEDIUM_WIDE
        else:
            return ShotType.WIDE
    
    def _calculate_clutter(self, frame: np.ndarray) -> float:
        """
        Calculate visual clutter score (0-1)
        Based on edge density - more edges = more clutter
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Normalize to 0-1 (typical values are 0-0.3)
        return min(edge_density * 3.0, 1.0)
    
    def _calculate_contrast(self, gray: np.ndarray) -> float:
        """
        Calculate contrast score (0-1)
        Higher score = better contrast
        """
        # Use standard deviation as proxy for contrast
        std = np.std(gray)
        
        # Normalize (std typically ranges 0-70)
        return min(std / 70.0, 1.0)
    
    def _detect_text_regions(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Detect text regions using MSER
        
        Returns:
            (has_text, text_area_ratio)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use MSER (Maximally Stable Extremal Regions) for text detection
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        # Count regions that might be text (aspect ratio, size filters)
        text_like_regions = 0
        total_text_area = 0
        
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            aspect_ratio = w / h if h > 0 else 0
            
            # Text typically has certain aspect ratios and sizes
            if 0.2 < aspect_ratio < 10 and 100 < w * h < 50000:
                text_like_regions += 1
                total_text_area += w * h
        
        frame_area = frame.shape[0] * frame.shape[1]
        text_area_ratio = total_text_area / frame_area if frame_area > 0 else 0
        
        has_text = text_like_regions > 5 or text_area_ratio > 0.02
        
        return has_text, text_area_ratio
    
    def _extract_color_palette(self, frame: np.ndarray, n_colors: int = 3) -> List[str]:
        """
        Extract dominant colors from frame
        
        Returns:
            List of hex color codes
        """
        # Resize for faster processing
        small = cv2.resize(frame, (150, 150))
        
        # Reshape to be a list of pixels
        pixels = small.reshape(-1, 3).astype(np.float32)
        
        # Use k-means to find dominant colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, 
            n_colors, 
            None, 
            criteria, 
            10, 
            cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Convert to hex colors
        colors = []
        for center in centers:
            b, g, r = center.astype(int)
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            colors.append(hex_color)
        
        return colors
    
    def _calculate_motion(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        """
        Calculate motion score between frames (0-1)
        """
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(prev_gray, curr_gray)
        
        # Threshold to get changed pixels
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Calculate percentage of changed pixels
        changed_pixels = np.sum(thresh > 0)
        total_pixels = thresh.size
        motion_score = changed_pixels / total_pixels
        
        return min(motion_score * 5.0, 1.0)  # Amplify small motions
    
    def _detect_scene_change(
        self, 
        prev_frame: np.ndarray, 
        curr_frame: np.ndarray,
        threshold: float = 0.3
    ) -> bool:
        """Detect if this is a scene change"""
        motion = self._calculate_motion(prev_frame, curr_frame)
        return motion > threshold
    
    def get_composition_metrics(self, analyses: List[FrameAnalysis]) -> Dict[str, any]:
        """
        Calculate aggregate composition metrics
        
        Returns:
            Dict with face_presence_pct, avg_clutter, etc.
        """
        if not analyses:
            return {}
        
        total = len(analyses)
        
        return {
            'face_presence_pct': sum(1 for a in analyses if a.has_face) / total * 100,
            'eye_contact_pct': sum(1 for a in analyses if a.eye_contact_detected) / total * 100,
            'text_presence_pct': sum(1 for a in analyses if a.has_text) / total * 100,
            'avg_visual_clutter': np.mean([a.visual_clutter_score for a in analyses if a.visual_clutter_score]),
            'avg_contrast': np.mean([a.contrast_score for a in analyses if a.contrast_score]),
            'avg_motion': np.mean([a.motion_score for a in analyses if a.motion_score]),
            'scene_change_count': sum(1 for a in analyses if a.scene_change),
            'shot_type_distribution': self._get_shot_distribution(analyses)
        }
    
    def _get_shot_distribution(self, analyses: List[FrameAnalysis]) -> Dict[str, int]:
        """Get distribution of shot types"""
        distribution = {}
        for analysis in analyses:
            if analysis.shot_type:
                distribution[analysis.shot_type] = distribution.get(analysis.shot_type, 0) + 1
        return distribution


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python frame_analyzer_enhanced.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    print(f"Analyzing video: {video_path}")
    analyzer = FrameAnalyzerEnhanced()
    
    # Analyze video (sample every 1 second, max 30 frames)
    analyses = analyzer.analyze_video(video_path, interval_s=1.0, max_frames=30)
    
    print(f"\nAnalyzed {len(analyses)} frames")
    print("=" * 60)
    
    # Show first 5 frames
    for i, analysis in enumerate(analyses[:5]):
        print(f"\nFrame {i+1} @ {analysis.timestamp_s:.1f}s:")
        print(f"  Shot type: {analysis.shot_type}")
        print(f"  Has face: {analysis.has_face} (count: {analysis.face_count})")
        print(f"  Eye contact: {analysis.eye_contact_detected}")
        print(f"  Has text: {analysis.has_text}")
        print(f"  Visual clutter: {analysis.visual_clutter_score:.2f}")
        print(f"  Contrast: {analysis.contrast_score:.2f}")
        print(f"  Motion: {analysis.motion_score:.2f if analysis.motion_score else 'N/A'}")
        print(f"  Scene change: {analysis.scene_change}")
    
    # Show aggregate metrics
    print("\n" + "=" * 60)
    print("Aggregate Metrics:")
    print("=" * 60)
    metrics = analyzer.get_composition_metrics(analyses)
    for key, value in metrics.items():
        if key != 'shot_type_distribution':
            print(f"{key}: {value}")
    
    print("\nShot Type Distribution:")
    for shot_type, count in metrics['shot_type_distribution'].items():
        print(f"  {shot_type}: {count}")
