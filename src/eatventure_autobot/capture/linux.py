import logging
import os
import threading
from typing import Any

import numpy as np

from eatventure_autobot.capture._pywinctl_base import PyWinCtlWindowCapture
from eatventure_autobot.domain.errors import CaptureError

logger = logging.getLogger(__name__)

_SESSION_TYPE = os.getenv("XDG_SESSION_TYPE", "").lower()
if _SESSION_TYPE == "wayland" and os.getenv("DISPLAY"):
    # Force PyWinCtl onto its X11 backend so it can control an XWayland-only target.
    # Native Wayland is out of scope (GREENFIELD_PLAN.md decision 1).
    os.environ["XDG_SESSION_TYPE"] = "x11"


class LinuxWindowCapture(PyWinCtlWindowCapture):
    """X11 and Wayland-via-XWayland only. Native Wayland capture/input is out of scope."""

    def __init__(
        self,
        title: str,
        target_width: int,
        target_height: int,
        stop_event: threading.Event | None = None,
    ) -> None:
        if _SESSION_TYPE == "wayland" and not os.getenv("DISPLAY"):
            raise CaptureError("Wayland requires the target running through XWayland (no DISPLAY)")
        from Xlib import display

        try:
            self._xdisplay = display.Display()
        except Exception as exc:
            raise CaptureError(f"Cannot open X11 display: {exc}") from exc
        super().__init__(title, target_width, target_height, stop_event)

    def _resize_hint(self) -> str:
        if _SESSION_TYPE == "wayland":
            return " Ensure the target is a floating XWayland window."
        return ""

    def _handle(self) -> int:
        try:
            getter = self._window.getHandle
            return int(getter() if callable(getter) else self._window.handle)
        except Exception as exc:
            raise CaptureError(f"Window has no native X11 handle: {exc}") from exc

    def capture(self, max_y: int | None = None) -> np.ndarray:
        with self._lock:
            bounds = self.get_window_rect()
            height = bounds.height if max_y is None else min(bounds.height, max_y)
            if height <= 0:
                raise CaptureError(f"Invalid capture size: {bounds.width}x{height}")
            try:
                from Xlib import X

                xwindow: Any = self._xdisplay.create_resource_object("window", self._handle())
                raw = xwindow.get_image(0, 0, bounds.width, height, X.ZPixmap, 0xFFFFFFFF)
                if raw is None or len(raw.data) != height * bounds.width * 4:
                    raise ValueError("X11 returned an incomplete capture buffer")
                image = np.frombuffer(raw.data, dtype=np.uint8).reshape(height, bounds.width, 4)
            except Exception as exc:
                raise CaptureError(f"Window capture failed: {exc}") from exc
            bgr = image[:, :, :3]
            return bgr if bgr.flags.c_contiguous else np.ascontiguousarray(bgr)

    def close(self) -> None:
        with self._lock:
            self._window = None
            try:
                self._xdisplay.close()
            except Exception:
                logger.exception("Failed to close X11 display")
