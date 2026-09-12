import logging
import os
import threading

import numpy as np

from eatventure_autobot.capture._pywinctl_base import PyWinCtlWindowCapture
from eatventure_autobot.domain.errors import CaptureError

logger = logging.getLogger(__name__)

_SESSION_TYPE = os.getenv("XDG_SESSION_TYPE", "").lower()
if _SESSION_TYPE == "wayland" and os.getenv("DISPLAY"):
    # Force PyWinCtl onto its X11 backend so it can control an XWayland-only target.
    # Native Wayland is intentionally out of scope — see the class docstring below.
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
        try:
            super().__init__(title, target_width, target_height, stop_event)
        except Exception:
            self._xdisplay.close()
            raise

    def _resize_hint(self) -> str:
        if _SESSION_TYPE == "wayland":
            return " Ensure the target is a floating XWayland window."
        return ""

    def capture(self, max_y: int | None = None) -> np.ndarray:
        with self._lock:
            bounds = self.get_window_rect()
            height = bounds.height if max_y is None else min(bounds.height, max_y)
            if height <= 0:
                raise CaptureError(f"Invalid capture size: {bounds.width}x{height}")
            try:
                from Xlib import X

                # Grab from the root window at the client rect's screen-absolute position,
                # not from the target window's own local (0, 0) — that origin belongs to
                # whatever X11 gives us for getHandle() (often the outer/decorated window),
                # which sits (border, title-bar) pixels above-and-left of the actual client
                # area that get_window_rect()/getClientFrame() and every click computation
                # use. Matches windows.py's mss.grab(), which already grabs by screen-
                # absolute bounds.left/top rather than a window-local origin.
                root = self._xdisplay.screen().root
                raw = root.get_image(
                    bounds.left, bounds.top, bounds.width, height, X.ZPixmap, 0xFFFFFFFF
                )
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
