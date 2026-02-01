"""
Video Matting Service
=====================
Service for extracting objects/people from videos with alpha channel support.
Supports multiple matting solutions via adapters.
"""

from .worker import MattingWorker
from .adapters.base import MattingAdapter
from .models import MattingRequest, MattingResponse, MattingJobStatus, MattingModel

__all__ = [
    "MattingWorker",
    "MattingAdapter",
    "MattingRequest",
    "MattingResponse",
    "MattingJobStatus",
    "MattingModel",
]

