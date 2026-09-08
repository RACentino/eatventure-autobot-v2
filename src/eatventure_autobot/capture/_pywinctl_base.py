"""Shared window discovery/resize logic for capture backends, built on PyWinCtl (cross-platform
window management). Subclasses provide only the pixel-grab mechanism, which differs per platform."""

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from eatventure_autobot.domain.errors import CaptureError, WindowNotAvailableError
from eatventure_autobot.domain.types import WindowBounds

logger = logging.getLogger(__name__)

pywinctl: Any = None
_PYWINCTL_IMPORT_ERROR: Exception | None = None
try:
    import pywinctl
except Exception as exc:  # pragma: no cover - exercised only when the dependency is missing
    _PYWINCTL_IMPORT_ERROR = exc


class PyWinCtlWindowCapture(ABC):
    def __init__(
        self,
        title: str,
        target_width: int,
        target_height: int,
        stop_event: threading.Event | None = None,
    ) -> None:
        if target_width <= 0 or target_height <= 0:
            raise CaptureError(f"Invalid target size: {target_width}x{target_height}")
        if pywinctl is None:
            raise CaptureError(f"Window backend unavailable: {_PYWINCTL_IMPORT_ERROR}")
        self._title = title
        self._target_width = target_width
        self._target_height = target_height
        self._stop_event = stop_event
        self._lock = threading.RLock()
        self._window: Any = None
        try:
            self._locate_and_resize()
        except CaptureError as exc:
            logger.warning("%s", exc)

    def _wait(self, duration: float) -> bool:
        if self._stop_event is None:
            time.sleep(duration)
            return True
        return not self._stop_event.wait(duration)

    def _find(self) -> Any:
        assert pywinctl is not None
        try:
            matches = [
                window
                for window in pywinctl.getWindowsWithTitle(self._title) or []
                if self._is_alive(window) and self._window_title(window) == self._title
            ]
        except Exception as exc:
            raise CaptureError(f"Cannot search for window '{self._title}': {exc}") from exc
        if len(matches) > 1:
            raise CaptureError(f"Multiple live windows have title '{self._title}'")
        if not matches:
            raise WindowNotAvailableError(f"Window '{self._title}' not found")
        return matches[0]

    @staticmethod
    def _is_alive(window: Any) -> bool:
        try:
            value = window.isAlive
            return bool(value() if callable(value) else value)
        except Exception:
            return False

    @staticmethod
    def _is_active(window: Any) -> bool:
        try:
            value = window.isActive
            return bool(value() if callable(value) else value)
        except Exception:
            return False

    @staticmethod
    def _window_title(window: Any) -> str:
        try:
            return str(window.title)
        except Exception:
            return ""

    def _resize_hint(self) -> str:
        return ""

    def _client_bounds(self, window: Any) -> WindowBounds:
        try:
            # getClientFrame() returns a Rect(left, top, right, bottom), not a Box —
            # width/height must be derived, they aren't attributes of the returned type.
            frame = window.getClientFrame()
            width = int(frame.right) - int(frame.left)
            height = int(frame.bottom) - int(frame.top)
            return WindowBounds(int(frame.left), int(frame.top), width, height)
        except Exception as exc:
            raise CaptureError(f"Cannot read client bounds: {exc}") from exc

    def _locate_and_resize(self) -> Any:
        with self._lock:
            window = self._find()
            self._window = window
            bounds = self._client_bounds(window)
            if (bounds.width, bounds.height) == (self._target_width, self._target_height):
                return window
            try:
                if getattr(window, "isMinimized", False) or getattr(window, "isMaximized", False):
                    window.restore(wait=False)
                outer = window.box
                frame_width = max(0, int(outer.width) - bounds.width)
                frame_height = max(0, int(outer.height) - bounds.height)
                window.resizeTo(
                    self._target_width + frame_width,
                    self._target_height + frame_height,
                    wait=False,
                )
            except Exception as exc:
                raise CaptureError(f"Window resize failed: {exc}") from exc

            for _ in range(10):
                actual = self._client_bounds(window)
                if (actual.width, actual.height) == (self._target_width, self._target_height):
                    return window
                if not self._wait(0.05):
                    raise CaptureError("Window resize interrupted")
            actual = self._client_bounds(window)
            raise CaptureError(
                f"Window client must be {self._target_width}x{self._target_height}; got "
                f"{actual.width}x{actual.height}.{self._resize_hint()}"
            )

    def ensure_window(self, resize: bool = False) -> None:
        with self._lock:
            if resize or self._window is None or not self._is_alive(self._window):
                self._locate_and_resize()

    def get_window_rect(self) -> WindowBounds:
        with self._lock:
            self.ensure_window()
            return self._client_bounds(self._window)

    def is_window_active(self) -> bool:
        with self._lock:
            try:
                self.ensure_window()
            except CaptureError:
                return False
            return self._window_title(self._window) == self._title and self._is_active(self._window)

    @abstractmethod
    def capture(self, max_y: int | None = None) -> np.ndarray: ...

    @abstractmethod
    def close(self) -> None: ...
