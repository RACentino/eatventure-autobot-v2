import logging
import threading
from typing import Any

import numpy as np

_MSS_IMPORT_ERROR: Exception | None
_PYWINCTL_IMPORT_ERROR: Exception | None
mss: Any
pywinctl: Any

try:
    import mss
except Exception as exc:
    mss = None
    _MSS_IMPORT_ERROR = exc
else:
    _MSS_IMPORT_ERROR = None

try:
    import pywinctl
except Exception as exc:
    pywinctl = None
    _PYWINCTL_IMPORT_ERROR = exc
else:
    _PYWINCTL_IMPORT_ERROR = None

logger = logging.getLogger(__name__)

WindowRect = tuple[int, int, int, int]
_MISSING = object()


class WindowCaptureError(RuntimeError):
    pass


class WindowNotAvailableError(WindowCaptureError):
    pass


def _geometry_attribute_value(geometry: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if hasattr(geometry, name):
            return getattr(geometry, name)
    return _MISSING


def _geometry_mapping_value(geometry: Any, names: tuple[str, ...]) -> Any:
    if not isinstance(geometry, dict):
        return _MISSING
    for name in names:
        if name in geometry:
            return geometry[name]
    return _MISSING


def _geometry_value(geometry: Any, names: tuple[str, ...], index: int) -> Any:
    value = _geometry_attribute_value(geometry, names)
    if value is not _MISSING:
        return value
    value = _geometry_mapping_value(geometry, names)
    if value is not _MISSING:
        return value
    return geometry[index]


def _bounds_from_geometry(geometry: Any) -> WindowRect:
    left = int(_geometry_value(geometry, ("left", "x"), 0))
    top = int(_geometry_value(geometry, ("top", "y"), 1))
    if hasattr(geometry, "right") or (
        isinstance(geometry, dict) and "right" in geometry
    ):
        right = int(_geometry_value(geometry, ("right",), 2))
        bottom = int(_geometry_value(geometry, ("bottom",), 3))
        return left, top, right - left, bottom - top
    width = int(_geometry_value(geometry, ("width", "w"), 2))
    height = int(_geometry_value(geometry, ("height", "h"), 3))
    return left, top, width, height


class WindowCapture:
    def __init__(
        self, window_title: str, target_width: int = 800, target_height: int = 600
    ) -> None:
        self.window_title = str(window_title)
        self.target_width = int(target_width)
        self.target_height = int(target_height)
        if self.target_width <= 0 or self.target_height <= 0:
            raise WindowCaptureError(
                f"Invalid target window size: {self.target_width}x{self.target_height}"
            )
        self.hwnd = None
        self._window = None
        self._lock = threading.RLock()
        if mss is None:
            raise WindowCaptureError(
                f"Cannot initialize screenshot backend: {_MSS_IMPORT_ERROR}"
            )
        try:
            self._screenshotter = mss.mss()
        except Exception as exc:
            raise WindowCaptureError(
                f"Cannot initialize screenshot backend: {exc}"
            ) from exc
        try:
            self.ensure_window(resize=True)
        except WindowCaptureError as exc:
            logger.warning("%s", exc)

    @staticmethod
    def _alive(window: Any) -> bool:
        alive = getattr(window, "isAlive", None)
        try:
            return bool(
                alive() if callable(alive) else True if alive is None else alive
            )
        except Exception:
            return False

    @staticmethod
    def _handle(window: Any) -> Any:
        getter = getattr(window, "getHandle", None)
        if callable(getter):
            try:
                return getter()
            except Exception:
                return None
        return getattr(window, "handle", None)

    def _find_window(self) -> Any | None:
        if pywinctl is None:
            raise WindowCaptureError(
                f"Could not initialize window backend: {_PYWINCTL_IMPORT_ERROR}"
            )
        try:
            windows = pywinctl.getWindowsWithTitle(self.window_title) or []
        except Exception as exc:
            raise WindowCaptureError(
                f"Could not search for window '{self.window_title}': {exc}"
            ) from exc
        live_windows = [window for window in windows if self._alive(window)]
        for window in live_windows:
            if getattr(window, "title", None) == self.window_title:
                return window
        return live_windows[0] if live_windows else None

    def ensure_window(self, resize: bool = False) -> Any:
        with self._lock:
            window = self._ensure_window_reference()
            if resize:
                self._resize()
            return window

    def _ensure_window_reference(self) -> Any:
        if self._window is not None and self._alive(self._window):
            return self._window
        self._window = self._find_window()
        if self._window is None:
            self.hwnd = None
            raise WindowNotAvailableError(f"Window '{self.window_title}' not found")
        self.hwnd = self._handle(self._window)
        logger.info("Window found: %s (handle: %s)", self.window_title, self.hwnd)
        return self._window

    def _resize(self) -> None:
        if self._window is None or not self._alive(self._window):
            return
        try:
            resized = self._window.resizeTo(
                self.target_width, self.target_height, wait=True
            )
        except Exception as exc:
            raise WindowCaptureError(
                f"Resizing window '{self.window_title}' failed: {exc}"
            ) from exc
        if resized is False:
            raise WindowCaptureError(f"Resizing window '{self.window_title}' failed")
        logger.info("Window resized to %sx%s", self.target_width, self.target_height)

    def find_window(self) -> Any:
        return self.ensure_window()

    def get_hwnd(self) -> Any:
        self.ensure_window()
        return self.hwnd

    def resize_window(self) -> None:
        with self._lock:
            self.ensure_window()
            self._resize()

    def get_window_rect(self) -> WindowRect:
        with self._lock:
            window = self.ensure_window()
            x, y, width, height = self._window_bounds(window)
            if width <= 0 or height <= 0:
                raise WindowCaptureError(
                    f"Window '{self.window_title}' has invalid size: {width}x{height}"
                )
            return x, y, width, height

    def _window_bounds(self, window: Any) -> WindowRect:
        try:
            return _bounds_from_geometry(window.getClientFrame())
        except Exception:
            return self._window_box_bounds(window)

    def _window_box_bounds(self, window: Any) -> WindowRect:
        try:
            return _bounds_from_geometry(window.box)
        except Exception as exc:
            raise WindowCaptureError(
                f"Cannot read window bounds for '{self.window_title}': {exc}"
            ) from exc

    def capture(self, max_y: Any = None) -> np.ndarray:
        with self._lock:
            x, y, width, height = self.get_window_rect()
            if max_y is not None:
                height = min(height, int(max_y))
            if width <= 0 or height <= 0:
                raise WindowCaptureError(
                    f"Window '{self.window_title}' cannot be captured with size {width}x{height}"
                )
            image = self._grab_window_image(x, y, width, height)
            return self._decoded_bgr_image(image)

    def _grab_window_image(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        try:
            screenshot = self._screenshotter.grab(
                {"left": x, "top": y, "width": width, "height": height}
            )
            return np.asarray(screenshot)
        except Exception as exc:
            if not self.is_window_active():
                raise WindowNotAvailableError(
                    f"Window '{self.window_title}' is no longer available"
                ) from exc
            raise WindowCaptureError(
                f"Capturing window '{self.window_title}' failed: {exc}"
            ) from exc

    @staticmethod
    def _decoded_bgr_image(image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] < 3:
            raise WindowCaptureError(f"Captured image has invalid shape: {image.shape}")
        decoded = image[:, :, :3].astype(np.uint8, copy=False)
        return decoded if decoded.flags.c_contiguous else np.ascontiguousarray(decoded)

    def is_window_active(self) -> bool:
        with self._lock:
            if self._window is not None and self._alive(self._window):
                return True
            self._window = self._find_window()
            self.hwnd = self._handle(self._window) if self._window is not None else None
            return self._window is not None

    def close(self) -> None:
        close_screenshotter = getattr(self._screenshotter, "close", None)
        if callable(close_screenshotter):
            try:
                close_screenshotter()
            except Exception as exc:
                logger.debug("Screenshot backend close failed: %s", exc)
