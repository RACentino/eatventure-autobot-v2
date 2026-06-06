import logging
import threading
import time
from typing import Any

import mss
import numpy as np
import pywinctl

from overlay_window import configure_overlay_canvas, configure_overlay_root, destroy_overlay_root, position_overlay_over_rect, set_overlay_visible_regions

logger = logging.getLogger(__name__)

WindowRect = tuple[int, int, int, int]
ForbiddenZone = tuple[int, int, int, int]
FORBIDDEN_AREA_OVERLAY_LOOP_ITERATION_LIMIT = 2_147_483_647


class WindowCaptureError(RuntimeError):
    pass


class WindowNotAvailableError(WindowCaptureError):
    pass


def _geometry_value(geometry: Any, names: tuple[str, ...], index: int) -> Any:
    for name in names:
        if hasattr(geometry, name):
            return getattr(geometry, name)
    if isinstance(geometry, dict):
        for name in names:
            if name in geometry:
                return geometry[name]
    return geometry[index]


def _bounds_from_geometry(geometry: Any) -> WindowRect:
    left = int(_geometry_value(geometry, ("left", "x"), 0))
    top = int(_geometry_value(geometry, ("top", "y"), 1))
    if hasattr(geometry, "right") or (isinstance(geometry, dict) and "right" in geometry):
        right = int(_geometry_value(geometry, ("right",), 2))
        bottom = int(_geometry_value(geometry, ("bottom",), 3))
        return left, top, right - left, bottom - top
    width = int(_geometry_value(geometry, ("width", "w"), 2))
    height = int(_geometry_value(geometry, ("height", "h"), 3))
    return left, top, width, height


class WindowCapture:
    def __init__(self, window_title: str, target_width: int = 800, target_height: int = 600) -> None:
        self.window_title = str(window_title)
        self.target_width = int(target_width)
        self.target_height = int(target_height)
        if self.target_width <= 0 or self.target_height <= 0:
            raise WindowCaptureError(f"Invalid target window size: {self.target_width}x{self.target_height}")
        self.hwnd = None
        self._window = None
        self._lock = threading.RLock()
        try:
            self._screenshotter = mss.mss()
        except Exception as exc:
            raise WindowCaptureError(f"Cannot initialize screenshot backend: {exc}") from exc
        try:
            self.ensure_window(resize=True)
        except WindowCaptureError as exc:
            logger.warning("%s", exc)

    @staticmethod
    def _alive(window: Any) -> bool:
        alive = getattr(window, "isAlive", None)
        try:
            return bool(alive() if callable(alive) else True if alive is None else alive)
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
        try:
            windows = pywinctl.getWindowsWithTitle(self.window_title) or []
        except Exception as exc:
            raise WindowCaptureError(f"Could not search for window '{self.window_title}': {exc}") from exc
        live_windows = [window for window in windows if self._alive(window)]
        for window in live_windows:
            if getattr(window, "title", None) == self.window_title:
                return window
        return live_windows[0] if live_windows else None

    def ensure_window(self, resize: bool = False) -> Any:
        with self._lock:
            if self._window is None or not self._alive(self._window):
                self._window = self._find_window()
                if self._window is None:
                    self.hwnd = None
                    raise WindowNotAvailableError(f"Window '{self.window_title}' not found")
                self.hwnd = self._handle(self._window)
                logger.info("Window found: %s (handle: %s)", self.window_title, self.hwnd)
            if resize:
                self._resize()
            return self._window

    def _resize(self) -> None:
        if self._window is None or not self._alive(self._window):
            return
        try:
            resized = self._window.resizeTo(self.target_width, self.target_height, wait=True)
        except Exception as exc:
            raise WindowCaptureError(f"Resizing window '{self.window_title}' failed: {exc}") from exc
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
            try:
                x, y, width, height = _bounds_from_geometry(window.getClientFrame())
            except Exception:
                try:
                    x, y, width, height = _bounds_from_geometry(window.box)
                except Exception as exc:
                    raise WindowCaptureError(f"Cannot read window bounds for '{self.window_title}': {exc}") from exc
            if width <= 0 or height <= 0:
                raise WindowCaptureError(f"Window '{self.window_title}' has invalid size: {width}x{height}")
            return x, y, width, height

    def capture(self, max_y: Any = None) -> np.ndarray:
        with self._lock:
            x, y, width, height = self.get_window_rect()
            if max_y is not None:
                height = min(height, int(max_y))
            if width <= 0 or height <= 0:
                raise WindowCaptureError(f"Window '{self.window_title}' cannot be captured with size {width}x{height}")
            try:
                screenshot = self._screenshotter.grab({"left": x, "top": y, "width": width, "height": height})
                image = np.asarray(screenshot)
            except Exception as exc:
                if not self.is_window_active():
                    raise WindowNotAvailableError(f"Window '{self.window_title}' is no longer available") from exc
                raise WindowCaptureError(f"Capturing window '{self.window_title}' failed: {exc}") from exc
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
            close_screenshotter()


class ForbiddenAreaOverlay:
    def __init__(self, window_capture: WindowCapture, forbidden_zones: list[ForbiddenZone]) -> None:
        self.window_capture = window_capture
        self.forbidden_zones = forbidden_zones
        self.running = False
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info("Forbidden area visualizer started")

    def stop(self) -> None:
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        logger.info("Forbidden area visualizer stopped")

    def _run(self) -> None:
        try:
            import tkinter as tk

            root = tk.Tk()
            background = configure_overlay_root(root, "Forbidden Area Visualizer")
            canvas = tk.Canvas(root, highlightthickness=0, background=background)
            configure_overlay_canvas(canvas, background)
            canvas.pack(fill="both", expand=True)
            self._overlay_loop(root, canvas)
        except Exception as exc:
            logger.error("Failed to create forbidden area visualizer: %s", exc)
        finally:
            self.running = False

    def _overlay_loop(self, root: Any, canvas: Any) -> None:
        for _ in range(FORBIDDEN_AREA_OVERLAY_LOOP_ITERATION_LIMIT):
            if not self.running:
                break
            if not self._draw(root, canvas):
                break
            root.update_idletasks()
            root.update()
            time.sleep(0.1)
        destroy_overlay_root(root)

    def _draw(self, root: Any, canvas: Any) -> bool:
        try:
            if not position_overlay_over_rect(root, canvas, self.window_capture.get_window_rect()):
                return False
            canvas.delete("zone")
            set_overlay_visible_regions(root, _zone_visible_regions(self.forbidden_zones))
            for x_min, x_max, y_min, y_max in self.forbidden_zones:
                canvas.create_rectangle(int(x_min), int(y_min), int(x_max), int(y_max), fill="#ff4040", stipple="gray25", outline="#ff4040", tags="zone")
            return True
        except Exception as exc:
            logger.error("Error in forbidden area visualizer loop: %s", exc)
            self.running = False
            return False


def _zone_visible_regions(zones: list[ForbiddenZone]) -> list[tuple[int, int, int, int]]:
    regions = []
    for x_min, x_max, y_min, y_max in zones:
        regions.append((int(x_min), int(y_min), max(1, int(x_max) - int(x_min)), max(1, int(y_max) - int(y_min))))
    return regions
