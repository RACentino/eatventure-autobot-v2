import logging
import threading
from typing import Any

import numpy as np

from eatventure_autobot.capture._pywinctl_base import PyWinCtlWindowCapture
from eatventure_autobot.domain.errors import CaptureError

logger = logging.getLogger(__name__)


class WindowsWindowCapture(PyWinCtlWindowCapture):
    def __init__(
        self,
        title: str,
        target_width: int,
        target_height: int,
        stop_event: threading.Event | None = None,
    ) -> None:
        import mss

        try:
            self._backend: Any = mss.mss()
        except Exception as exc:
            raise CaptureError(f"Cannot initialize Windows capture backend: {exc}") from exc
        super().__init__(title, target_width, target_height, stop_event)

    def capture(self, max_y: int | None = None) -> np.ndarray:
        with self._lock:
            bounds = self.get_window_rect()
            height = bounds.height if max_y is None else min(bounds.height, max_y)
            if height <= 0:
                raise CaptureError(f"Invalid capture size: {bounds.width}x{height}")
            try:
                region = {
                    "left": bounds.left,
                    "top": bounds.top,
                    "width": bounds.width,
                    "height": height,
                }
                image = np.asarray(self._backend.grab(region))
            except Exception as exc:
                raise CaptureError(f"Window capture failed: {exc}") from exc
            if image.ndim != 3 or image.shape[:2] != (height, bounds.width) or image.shape[2] < 3:
                raise CaptureError(f"Unexpected capture shape: {image.shape!r}")
            bgr = image[:, :, :3]
            return bgr if bgr.flags.c_contiguous else np.ascontiguousarray(bgr)

    def close(self) -> None:
        with self._lock:
            self._window = None
            try:
                self._backend.close()
            except Exception:
                logger.exception("Failed to close Windows capture backend")
