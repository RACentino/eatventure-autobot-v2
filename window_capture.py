import logging
import os
import sys
import threading
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_ACTUAL_SESSION = os.getenv("XDG_SESSION_TYPE", "").lower()
if sys.platform.startswith("linux") and _ACTUAL_SESSION == "wayland" and os.getenv("DISPLAY"):
    # PyWinCtl can control this intentionally XWayland-only target through its X11 backend.
    os.environ["XDG_SESSION_TYPE"] = "x11"

try:
    import pywinctl
except Exception as exc:
    pywinctl = None
    _PYWINCTL_ERROR = exc
else:
    _PYWINCTL_ERROR = None


class WindowCaptureError(RuntimeError):
    pass


class WindowNotAvailableError(WindowCaptureError):
    pass


def _bounds(geometry: Any) -> tuple[int, int, int, int]:
    if hasattr(geometry, "left"):
        left, top = int(geometry.left), int(geometry.top)
        if hasattr(geometry, "width"):
            return left, top, int(geometry.width), int(geometry.height)
        return left, top, int(geometry.right) - left, int(geometry.bottom) - top
    if isinstance(geometry, dict):
        left_value = geometry.get("left", geometry.get("x"))
        top_value = geometry.get("top", geometry.get("y"))
        if left_value is None or top_value is None:
            raise ValueError("geometry is missing left/top coordinates")
        left, top = int(left_value), int(top_value)
        width = geometry.get("width", geometry.get("w"))
        height = geometry.get("height", geometry.get("h"))
        if width is None or height is None:
            width, height = int(geometry["right"]) - left, int(geometry["bottom"]) - top
        return left, top, int(width), int(height)
    values = tuple(map(int, geometry))
    if len(values) != 4:
        raise ValueError(f"geometry must contain four values, got {values!r}")
    return values


class WindowCapture:
    def __init__(
        self,
        title: str,
        target_width: int,
        target_height: int,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.window_title = str(title)
        self.target_width = int(target_width)
        self.target_height = int(target_height)
        if self.target_width <= 0 or self.target_height <= 0:
            raise WindowCaptureError(
                f"Invalid target client size: {self.target_width}x{self.target_height}"
            )
        self.hwnd = None
        self._window = None
        self._lock = threading.RLock()
        self._backend = None
        self._xdisplay = None
        self._stop_event = stop_event
        if pywinctl is None:
            raise WindowCaptureError(f"Cannot initialize window backend: {_PYWINCTL_ERROR}")
        self._open_capture_backend()
        try:
            self.ensure_window(resize=True)
        except WindowCaptureError as exc:
            logger.warning("%s", exc)

    def _open_capture_backend(self) -> None:
        if sys.platform.startswith("linux"):
            if _ACTUAL_SESSION == "wayland" and not os.getenv("DISPLAY"):
                raise WindowCaptureError("Wayland requires scrcpy running through XWayland (DISPLAY is missing)")
            try:
                from Xlib import display

                self._xdisplay = display.Display()
            except Exception as exc:
                raise WindowCaptureError(f"Cannot initialize X11 capture: {exc}") from exc
        elif sys.platform == "win32":
            try:
                import mss

                self._backend = mss.mss()
            except Exception as exc:
                raise WindowCaptureError(f"Cannot initialize Windows capture: {exc}") from exc
        else:
            raise WindowCaptureError(f"Unsupported platform: {sys.platform}")

    def _wait(self, duration: float) -> bool:
        return not self._stop_event.wait(duration) if self._stop_event else not time.sleep(duration)

    @staticmethod
    def _alive(window: Any) -> bool:
        try:
            value = getattr(window, "isAlive", True)
            return bool(value() if callable(value) else value)
        except Exception:
            return False

    @staticmethod
    def _active(window: Any) -> bool:
        try:
            value = getattr(window, "isActive", False)
            return bool(value() if callable(value) else value)
        except Exception:
            return False

    @staticmethod
    def _title(window: Any) -> str:
        try:
            return str(window.title)
        except Exception:
            return ""

    @staticmethod
    def _handle(window: Any) -> Any:
        getter = getattr(window, "getHandle", None)
        return getter() if callable(getter) else getattr(window, "handle", None)

    def _find(self) -> Any | None:
        backend = pywinctl
        if backend is None:
            raise WindowCaptureError(f"Window backend is unavailable: {_PYWINCTL_ERROR}")
        try:
            matches = [
                window
                for window in (backend.getWindowsWithTitle(self.window_title) or [])
                if self._alive(window) and self._title(window) == self.window_title
            ]
        except Exception as exc:
            raise WindowCaptureError(f"Cannot search for '{self.window_title}': {exc}") from exc
        if len(matches) > 1:
            raise WindowCaptureError(f"Multiple live windows have title '{self.window_title}'")
        return matches[0] if matches else None

    def ensure_window(self, resize: bool = False) -> Any:
        with self._lock:
            window = self._find()
            if window is None:
                self._window = None
                self.hwnd = None
                raise WindowNotAvailableError(f"Window '{self.window_title}' not found")
            self._window = window
            self.hwnd = self._handle(window)
            if self.hwnd is None:
                raise WindowCaptureError(f"Window '{self.window_title}' has no native handle")
            if resize:
                self._resize()
            return self._window

    def invalidate_window(self) -> None:
        with self._lock:
            self._window = None
            self.hwnd = None

    def _window_bounds(self, window: Any) -> tuple[int, int, int, int]:
        try:
            bounds = _bounds(window.getClientFrame())
            if bounds[2] <= 0 or bounds[3] <= 0:
                raise ValueError(f"invalid target size: {bounds[2]}x{bounds[3]}")
            return bounds
        except Exception as exc:
            raise WindowCaptureError(f"Cannot read window client bounds: {exc}") from exc

    def get_window_rect(self) -> tuple[int, int, int, int]:
        with self._lock:
            bounds = self._window_bounds(self.ensure_window())
            if bounds[2] <= 0 or bounds[3] <= 0:
                raise WindowCaptureError(f"Invalid target size: {bounds[2]}x{bounds[3]}")
            return bounds

    def _resize(self) -> None:
        window = self._window
        if window is None:
            raise WindowNotAvailableError(f"Window '{self.window_title}' not found")
        current = self._window_bounds(window)
        if current[2:] == (self.target_width, self.target_height):
            return
        used_x11 = self._xdisplay is not None
        try:
            if getattr(window, "isMinimized", False) or getattr(
                window, "isMaximized", False
            ):
                window.restore(wait=False)
                current = self._window_bounds(window)
                if current[2:] == (self.target_width, self.target_height):
                    return
            if used_x11:
                if self._xdisplay is None or self.hwnd is None:
                    raise ValueError("X11 window handle is unavailable")
                xwindow = self._xdisplay.create_resource_object("window", int(self.hwnd))
                xwindow.configure(width=self.target_width, height=self.target_height)
                self._xdisplay.sync()
            else:
                outer = _bounds(window.box)
                frame_width = max(0, outer[2] - current[2])
                frame_height = max(0, outer[3] - current[3])
                window.resizeTo(
                    self.target_width + frame_width,
                    self.target_height + frame_height,
                    wait=False,
                )
        except Exception as exc:
            raise WindowCaptureError(f"Window resize failed: {exc}") from exc
        actual = self._window_bounds(window)[2:]
        for _ in range(10):
            if actual == (self.target_width, self.target_height):
                break
            if not self._wait(0.05):
                raise WindowCaptureError("Window resize interrupted")
            actual = self._window_bounds(window)[2:]
        if actual != (self.target_width, self.target_height):
            hint = (
                " Ensure scrcpy is a floating XWayland window."
                if _ACTUAL_SESSION == "wayland"
                else ""
            )
            raise WindowCaptureError(
                f"Window client must be exactly {self.target_width}x{self.target_height}; "
                f"got {actual[0]}x{actual[1]}.{hint}"
            )
        logger.info(
            "Window client resized from %sx%s to %sx%s",
            current[2],
            current[3],
            self.target_width,
            self.target_height,
        )

    def get_input_window_rect(self) -> tuple[int, int, int, int]:
        with self._lock:
            window = self.ensure_window()
            if not self._active(window):
                raise WindowNotAvailableError(f"Window '{self.window_title}' is not active")
            return self.get_window_rect()

    def is_window_active(self) -> bool:
        with self._lock:
            try:
                window = self.ensure_window()
            except WindowCaptureError:
                return False
            return self._title(window) == self.window_title and self._active(window)

    def capture(self, max_y: Any = None) -> np.ndarray:
        with self._lock:
            left, top, width, height = self.get_window_rect()
            if max_y is not None:
                try:
                    height = min(height, int(max_y))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise WindowCaptureError(f"Invalid capture height: {max_y!r}") from exc
            if height <= 0:
                raise WindowCaptureError(f"Invalid capture size: {width}x{height}")
            try:
                if self._xdisplay is not None:
                    from Xlib import X

                    if self.hwnd is None:
                        raise ValueError("X11 window handle is unavailable")
                    window = self._xdisplay.create_resource_object("window", int(self.hwnd))
                    raw = window.get_image(0, 0, width, height, X.ZPixmap, 0xFFFFFFFF)
                    if raw is None or len(raw.data) != height * width * 4:
                        raise ValueError("X11 returned an incomplete capture buffer")
                    image = np.frombuffer(raw.data, dtype=np.uint8).reshape(height, width, 4)
                else:
                    if self._backend is None:
                        raise ValueError("capture backend is closed")
                    image = np.asarray(
                        self._backend.grab(
                            {"left": left, "top": top, "width": width, "height": height}
                        )
                    )
            except Exception as exc:
                raise WindowCaptureError(f"Window capture failed: {exc}") from exc
            if image.ndim != 3 or image.shape[:2] != (height, width) or image.shape[2] < 3:
                raise WindowCaptureError(f"Unexpected capture shape: {image.shape!r}")
            bgr = image[:, :, :3]
            return bgr if bgr.flags.c_contiguous else np.ascontiguousarray(bgr)

    def reset_backend(self) -> None:
        with self._lock:
            self._close_capture_backend()
            self._open_capture_backend()
            self.invalidate_window()

    def _close_capture_backend(self) -> None:
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception:
                logger.exception("Failed to close Windows capture backend")
            self._backend = None
        if self._xdisplay is not None:
            try:
                self._xdisplay.close()
            except Exception:
                logger.exception("Failed to close X11 capture backend")
            self._xdisplay = None

    def close(self) -> None:
        with self._lock:
            self._close_capture_backend()
            self.invalidate_window()
