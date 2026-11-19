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


def _flag(window: Any, name: str) -> bool:
    """A pywinctl window state that is a property in some versions and a method in others."""
    try:
        value = getattr(window, name)
        return bool(value() if callable(value) else value)
    except Exception:
        return False


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
        self._window_pid: int | None = None

    def _wait(self, duration: float) -> bool:
        if self._stop_event is None:
            time.sleep(duration)
            return True
        return not self._stop_event.wait(duration)

    def _find(self) -> Any:
        assert pywinctl is not None
        try:
            matches = self._live_windows()
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

    @staticmethod
    def _process_id(window: Any) -> int | None:
        try:
            process_id = window.getPID()
            return int(process_id) if process_id is not None and int(process_id) > 0 else None
        except Exception:
            return None

    def mirror_windows(self) -> list[tuple[int, bool]]:
        try:
            return [(process_id, self._is_active(window))
                    for window in pywinctl.getAllWindows() or []
                    if self._is_alive(window) and (process_id := self._process_id(window))]
        except Exception as exc:
            raise CaptureError(f"Cannot discover scrcpy windows: {exc}") from exc

    def bind_mirror(self, process_id: int) -> None:
        if process_id <= 0:
            raise CaptureError("Invalid scrcpy process ID")
        with self._lock:
            self._window_pid = process_id
            self._window = None

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
            bounds = self._client_bounds(window)
            if (bounds.width, bounds.height) == (self._target_width, self._target_height):
                return window
            try:
                if _flag(window, "isMinimized") or _flag(window, "isMaximized"):
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
                self._window = None
                self._window = self._locate_and_resize()

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
            matches = (self._process_id(self._window) == self._window_pid
                       if self._window_pid is not None
                       else self._window_title(self._window) == self._title)
            return matches and self._is_active(self._window)

    def describe_environment(self) -> str:
        """One line of facts for a self-stop log: is the window there, at what size and state, and
        which app has the focus (the app name only, never a title: titles can carry tab names).
        Read-only and never raises. It asks pywinctl directly rather than going through
        ensure_window(): that would resize the window and, after a stop, fail on the set stop
        event inside _wait()."""
        try:
            with self._lock:
                windows = self._live_windows()
                facts = [f"windows={len(windows)}"]
                if windows:
                    window = windows[0]
                    try:
                        bounds = self._client_bounds(window)
                        facts.append(f"client={bounds.width}x{bounds.height}")
                    except CaptureError:
                        facts.append("client=?")
                    for name in ("isMaximized", "isMinimized", "isActive"):
                        facts.append(f"{name[2:].lower()}={_flag(window, name)}")
                active = pywinctl.getActiveWindow()
                facts.append(f"active_app={active.getAppName() if active is not None else 'none'}")
        except Exception as exc:
            return f"unavailable ({type(exc).__name__})"
        return " ".join(facts)

    def environment_ready(self) -> bool:
        """True only when start() would leave the window alone: exactly one live window, already at
        the target client size, neither maximized nor minimized, and already the active window.
        start() resizes (and pywinctl's restore() activates) a window that is not, which would move
        a person's window and take their focus. Read-only for the same reasons as
        describe_environment(); any doubt reads as not ready."""
        try:
            with self._lock:
                windows = self._live_windows()
                if len(windows) != 1:
                    return False
                window = windows[0]
                bounds = self._client_bounds(window)
                return (
                    (bounds.width, bounds.height) == (self._target_width, self._target_height)
                    and not _flag(window, "isMaximized")
                    and not _flag(window, "isMinimized")
                    and _flag(window, "isActive")
                )
        except Exception:
            return False

    def raise_window(self) -> bool:
        """Brings the target window to the front: the one deliberate focus change the bot makes on
        its own, used only by the auto-resume after nobody has touched the PC for
        resume.idle_seconds (everything else leaves the focus to the person). True when the window
        reports active afterwards. Best effort: any doubt, including a missing window, reads False
        and nothing is raised."""
        try:
            with self._lock:
                window = self._find()
                window.activate(wait=True)
                return self._is_active(window)
        except Exception:
            return False

    def _live_windows(self) -> list[Any]:
        windows = (pywinctl.getAllWindows() if self._window_pid is not None
                   else pywinctl.getWindowsWithTitle(self._title))
        return [
            window
            for window in windows or []
            if self._is_alive(window) and (
                self._process_id(window) == self._window_pid if self._window_pid is not None
                else self._window_title(window) == self._title
            )
        ]

    @abstractmethod
    def capture(self, max_y: int | None = None) -> np.ndarray: ...

    @abstractmethod
    def close(self) -> None: ...
