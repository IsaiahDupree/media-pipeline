"""
Frame Analyzer Service
Uses AI Vision to analyze video frames for visual context, objects, and text.
Uses ModelRegistry for configurable model selection (GPT-4o Mini by default, 80% cheaper)
"""
import base64
import os
import logging
from typing import List, Dict, Any

from config.model_registry import TaskType, ModelRegistry
from services.ai_client import AIClient

logger = logging.getLogger(__name__)

class FrameAnalyzer:
    def __init__(self, api_key: str = None):
        """
        Initialize frame analyzer using ModelRegistry
        
        Args:
            api_key: Optional API key (deprecated, use ModelRegistry instead)
        """
        # Get model configuration from registry
        self.config = ModelRegistry.get_model_config(TaskType.FRAME_ANALYSIS)
        self.client = AIClient(self.config)
        
        logger.info(f"FrameAnalyzer using {self.config.provider}/{self.config.model}")

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_frames(self, frame_paths: List[str]) -> Dict[str, Any]:
        """
        Analyze a list of video frames using AI Vision.
        Returns a summary of visual content, objects, and text.
        """
        if not frame_paths:
            return {"error": "No frames provided"}

        logger.info(f"Analyzing {len(frame_paths)} frames with {self.config.provider}/{self.config.model}")

        # For multiple frames, analyze each and combine
        # For single frame, use vision_analysis directly
        prompt = "Analyze these video frames. Describe the visual setting, key objects, any text on screen, and the overall visual style/mood. Summarize the visual progression."

        try:
            # If only one frame, use vision_analysis directly
            if len(frame_paths) == 1 and os.path.exists(frame_paths[0]):
                analysis_text = self.client.vision_analysis(
                    image_path=frame_paths[0],
                    prompt=prompt,
                    max_tokens=500
                )
            else:
                # For multiple frames, analyze first few and combine
                summaries = []
                for i, path in enumerate(frame_paths[:5]):  # Limit to 5 frames
                    if os.path.exists(path):
                        frame_analysis = self.client.vision_analysis(
                            image_path=path,
                            prompt=f"Analyze frame {i+1}: Describe what you see - objects, setting, mood, any text.",
                            max_tokens=200
                        )
                        summaries.append(f"Frame {i+1}: {frame_analysis}")
                
                # Combine summaries
                analysis_text = "\n".join(summaries) if summaries else "No valid frames to analyze"
            
            return {
                "visual_summary": analysis_text,
                "frame_count": len(frame_paths)
            }

        except Exception as e:
            logger.error(f"Frame analysis failed ({self.config.provider}): {e}")
            return {"error": str(e)}
