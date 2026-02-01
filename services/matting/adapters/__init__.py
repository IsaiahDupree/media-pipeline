"""
Matting Adapters
================
Adapter implementations for different matting models.
"""

from .base import MattingAdapter

# Import available adapters
try:
    from .rvm import RVMAdapter
except ImportError:
    RVMAdapter = None

try:
    from .mediapipe import MediaPipeAdapter
except ImportError:
    MediaPipeAdapter = None

# Future adapters
# from .background_matting_v2 import BackgroundMattingV2Adapter
# from .rembg import RembgAdapter
# from .sam2 import SAM2Adapter

__all__ = [
    "MattingAdapter",
    "RVMAdapter",
    "MediaPipeAdapter",
]

