"""
Base Matting Adapter
====================
Abstract base class for matting model adapters.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional

from ..models import MattingRequest, MattingResponse, MattingConfig


class MattingAdapter(ABC):
    """
    Abstract base class for matting model adapters.
    
    Each adapter implements the interface to a specific matting model,
    allowing the matting service to support multiple models seamlessly.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize adapter.
        
        Args:
            model_name: Name of the model (e.g., "rvm", "mediapipe")
        """
        self.model_name = model_name
        self._loaded = False
    
    @abstractmethod
    async def extract_foreground(
        self,
        request: MattingRequest,
        output_path: Optional[str] = None
    ) -> MattingResponse:
        """
        Extract foreground (person/object) from video with alpha channel.
        
        Args:
            request: Matting request
            output_path: Optional output path (auto-generated if None)
        
        Returns:
            MattingResponse with output video path and metadata
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dict with model metadata (name, version, capabilities, etc.)
        """
        pass
    
    @abstractmethod
    async def load_model(self) -> bool:
        """
        Load the model into memory.
        
        Returns:
            True if loaded successfully
        """
        pass
    
    @abstractmethod
    async def unload_model(self) -> bool:
        """
        Unload the model from memory.
        
        Returns:
            True if unloaded successfully
        """
        pass
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
    
    def _ensure_output_path(self, request: MattingRequest, output_path: Optional[str] = None) -> Path:
        """
        Generate output path if not provided.
        
        Args:
            request: Matting request
            output_path: Optional explicit path
        
        Returns:
            Path object for output file
        """
        if output_path:
            return Path(output_path)
        
        # Generate default path
        output_dir = Path("data/matting_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use MOV format to preserve alpha channel
        filename = f"{request.job_id}.mov"
        return output_dir / filename
    
    def _get_device(self, config: MattingConfig) -> str:
        """
        Determine device to use (cuda/cpu).
        
        Args:
            config: Matting configuration
        
        Returns:
            Device string ("cuda" or "cpu")
        """
        if config.device != "auto":
            return config.device
        
        # Auto-detect: try CUDA if available
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        
        return "cpu"

