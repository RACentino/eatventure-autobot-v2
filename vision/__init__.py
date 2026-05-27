from .capture import ForbiddenAreaOverlay, WindowCapture, WindowCaptureError, WindowNotAvailableError
from .matcher import ImageMatcher

__all__ = [
    "ForbiddenAreaOverlay",
    "ImageMatcher",
    "VisionScannerMixin",
    "WindowCapture",
    "WindowCaptureError",
    "WindowNotAvailableError",
]


def __getattr__(name: str):
    if name == "VisionScannerMixin":
        from .scanner import VisionScannerMixin

        return VisionScannerMixin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
