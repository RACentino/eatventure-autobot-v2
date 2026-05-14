import logging
import threading
from typing import Any

import numpy as np

from core.platform import mss, pywinctl, require_automation_backend

logger = logging.getLogger(__name__)
WindowRect = tuple[int, int, int, int]
ForbiddenZone = tuple[int, int, int, int]


class WindowCaptureError(RuntimeError):
    pass


class WindowNotAvailableError(WindowCaptureError):
    pass


class WindowCapture:
    def __init__(self, window_title: str, target_width: int = 800, target_height: int = 600) -> None:
        require_automation_backend("WindowCapture")
        self.window_title = window_title
        self.hwnd = None
        self._window = None
        try:
            self.target_width = int(target_width)
            self.target_height = int(target_height)
        except (TypeError, ValueError) as exc:
            raise WindowCaptureError(f"Invalid target window size: {target_width}x{target_height}") from exc
        if self.target_width <= 0 or self.target_height <= 0:
            raise WindowCaptureError(f"Invalid target window size: {self.target_width}x{self.target_height}")
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
    def _window_is_alive(window: Any) -> bool:
        alive = getattr(window, "isAlive", None)
        if alive is None:
            return True
        try:
            return bool(alive() if callable(alive) else alive)
        except Exception:
            return False

    @staticmethod
    def _window_handle(window: Any) -> Any:
        handle = getattr(window, "getHandle", None)
        if callable(handle):
            try:
                return handle()
            except Exception:
                return None
        return getattr(window, "handle", None)

    @staticmethod
    def _geometry_value(geometry: Any, names: tuple[str, ...], index: int) -> Any:
        for name in names:
            if hasattr(geometry, name):
                return getattr(geometry, name)
        if isinstance(geometry, dict):
            for name in names:
                if name in geometry:
                    return geometry[name]
        try:
            return geometry[index]
        except (TypeError, KeyError, IndexError) as exc:
            raise AttributeError(f"Geometry value not found for {names}") from exc

    @classmethod
    def _rect_bounds(cls: type["WindowCapture"], rect: Any) -> WindowRect:
        left = int(cls._geometry_value(rect, ("left", "x"), 0))
        top = int(cls._geometry_value(rect, ("top", "y"), 1))
        right = int(cls._geometry_value(rect, ("right",), 2))
        bottom = int(cls._geometry_value(rect, ("bottom",), 3))
        return left, top, right - left, bottom - top

    @classmethod
    def _box_bounds(cls: type["WindowCapture"], box: Any) -> WindowRect:
        left = int(cls._geometry_value(box, ("left", "x"), 0))
        top = int(cls._geometry_value(box, ("top", "y"), 1))
        width = int(cls._geometry_value(box, ("width", "w"), 2))
        height = int(cls._geometry_value(box, ("height", "h"), 3))
        return left, top, width, height

    def _find_window_object(self) -> Any | None:
        try:
            windows = pywinctl.getWindowsWithTitle(self.window_title) or []
        except Exception as exc:
            raise WindowCaptureError(f"Could not search for window '{self.window_title}': {exc}") from exc

        exact_matches = [
            window
            for window in windows
            if getattr(window, "title", None) == self.window_title and self._window_is_alive(window)
        ]
        if exact_matches:
            return exact_matches[0]

        for window in windows:
            if self._window_is_alive(window):
                return window
        return None

    def _invalidate_window(self) -> None:
        self.hwnd = None
        self._window = None

    def _resize_bound_window(self) -> None:
        if self._window is None or not self._window_is_alive(self._window):
            return
        try:
            resized = self._window.resizeTo(self.target_width, self.target_height, wait=True)
        except Exception as exc:
            raise WindowCaptureError(f"Resizing window '{self.window_title}' failed: {exc}") from exc
        if resized is False:
            raise WindowCaptureError(f"Resizing window '{self.window_title}' failed")
        logger.info("Window resized to %sx%s", self.target_width, self.target_height)

    def find_window(self) -> Any:
        with self._lock:
            window = self._find_window_object()
            if window is None:
                self._invalidate_window()
                raise WindowNotAvailableError(f"Window '{self.window_title}' not found")
            handle = self._window_handle(window)
            if handle != self.hwnd:
                logger.info("Window found: %s (handle: %s)", self.window_title, handle)
            self._window = window
            self.hwnd = handle
            return self._window

    def ensure_window(self, resize: bool = False) -> Any:
        with self._lock:
            if self._window is not None and self._window_is_alive(self._window):
                if resize:
                    self._resize_bound_window()
                return self._window

            previous_handle = self.hwnd
            window = self._find_window_object()
            if window is None:
                self._invalidate_window()
                if previous_handle:
                    logger.warning("Window handle %s is no longer valid", previous_handle)
                raise WindowNotAvailableError(f"Window '{self.window_title}' not found")

            handle = self._window_handle(window)
            self._window = window
            self.hwnd = handle
            if handle != previous_handle:
                logger.info("Window found: %s (handle: %s)", self.window_title, handle)
            if resize or handle != previous_handle:
                self._resize_bound_window()
            return self._window

    def get_hwnd(self) -> Any:
        self.ensure_window()
        return self.hwnd

    def resize_window(self) -> None:
        with self._lock:
            self.ensure_window()
            self._resize_bound_window()

    def _get_client_bounds(self, window: Any) -> WindowRect:
        try:
            frame = window.getClientFrame()
            x, y, width, height = self._rect_bounds(frame)
        except Exception:
            try:
                x, y, width, height = self._box_bounds(window.box)
            except Exception as exc:
                raise WindowCaptureError(f"Cannot read window bounds for '{self.window_title}': {exc}") from exc
        if width <= 0 or height <= 0:
            raise WindowCaptureError(
                f"Window '{self.window_title}' has an invalid client size: {width}x{height}"
            )
        return x, y, width, height

    def get_window_rect(self) -> WindowRect:
        with self._lock:
            window = self.ensure_window()
            return self._get_client_bounds(window)

    @staticmethod
    def _apply_capture_height_limit(height: int, max_y: Any) -> int:
        if max_y is None:
            return height
        try:
            return min(height, int(max_y))
        except (TypeError, ValueError) as exc:
            raise WindowCaptureError(f"Invalid capture height limit: {max_y}") from exc

    def _validate_capture_size(self, width: int, height: int) -> None:
        if width > 0 and height > 0:
            return
        raise WindowCaptureError(
            f"Window '{self.window_title}' cannot be captured with size {width}x{height}"
        )

    @staticmethod
    def _decode_screenshot(screenshot: Any, width: int, height: int) -> np.ndarray:
        screenshot_image = np.asarray(screenshot)
        if screenshot_image.dtype != np.uint8:
            screenshot_image = screenshot_image.astype(np.uint8, copy=False)
        if screenshot_image.shape[:2] != (height, width):
            raise WindowCaptureError(
                f"Captured image size mismatch: expected {width}x{height}, got "
                f"{screenshot_image.shape[1]}x{screenshot_image.shape[0]}"
            )
        if screenshot_image.ndim != 3 or screenshot_image.shape[2] < 3:
            raise WindowCaptureError(f"Captured image has invalid shape: {screenshot_image.shape}")
        decoded = screenshot_image[:, :, :3]
        if decoded.flags.c_contiguous:
            return decoded
        return np.ascontiguousarray(decoded)

    def capture(self, max_y: Any = None) -> np.ndarray:
        with self._lock:
            window = self.ensure_window()
            x, y, width, height = self._get_client_bounds(window)
            height = self._apply_capture_height_limit(height, max_y)
            self._validate_capture_size(width, height)
            try:
                screenshot = self._screenshotter.grab(
                    {"left": int(x), "top": int(y), "width": int(width), "height": int(height)}
                )
                return self._decode_screenshot(screenshot, width, height)
            except WindowCaptureError:
                raise
            except Exception as exc:
                if not self.is_window_active():
                    raise WindowNotAvailableError(
                        f"Window '{self.window_title}' is no longer available during capture"
                    ) from exc
                raise WindowCaptureError(f"Capturing window '{self.window_title}' failed: {exc}") from exc

    def is_window_active(self) -> bool:
        with self._lock:
            if self._window is not None and self._window_is_alive(self._window):
                return True
            self._window = self._find_window_object()
            self.hwnd = self._window_handle(self._window) if self._window is not None else None
            return self._window is not None

    def close(self) -> None:
        close = getattr(self._screenshotter, "close", None)
        if callable(close):
            close()


class ForbiddenAreaOverlay:
    def __init__(self, window_capture: WindowCapture, forbidden_zones: list[ForbiddenZone]) -> None:
        self.window_capture = window_capture
        self.forbidden_zones = forbidden_zones
        self.running = False
        self.thread = None
        self._root = None

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._create_overlay, daemon=True)
        self.thread.start()
        logger.info("Forbidden area visualizer started")

    def stop(self) -> None:
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        logger.info("Forbidden area visualizer stopped")

    @staticmethod
    def _preview_position(x: int, y: int, width: int) -> tuple[int, int]:
        return x + width + 16, max(0, y)

    def _draw_zones(self, canvas: Any, width: int, height: int) -> None:
        canvas.delete("zone")
        canvas.create_rectangle(0, 0, width, height, outline="#6f6f6f", width=1, tags="zone")
        for x_min, x_max, y_min, y_max in self.forbidden_zones:
            canvas.create_rectangle(
                int(x_min),
                int(y_min),
                int(x_max),
                int(y_max),
                fill="#ff4040",
                stipple="gray25",
                outline="#ff4040",
                tags="zone",
            )

    def _sync_preview_window(self, root: Any, canvas: Any) -> None:
        x, y, width, height = self.window_capture.get_window_rect()
        preview_x, preview_y = self._preview_position(x, y, width)
        root.geometry(f"{width}x{height}+{preview_x}+{preview_y}")
        canvas.config(width=width, height=height)
        self._draw_zones(canvas, width, height)

    def _create_overlay(self) -> None:
        try:
            import tkinter as tk

            root = tk.Tk()
            self._root = root
            root.title("Forbidden Area Visualizer")
            root.resizable(False, False)
            try:
                root.attributes("-topmost", True)
            except Exception:
                pass
            canvas = tk.Canvas(root, highlightthickness=0, background="#111111")
            canvas.pack(fill="both", expand=True)

            def tick() -> None:
                if not self.running:
                    root.destroy()
                    return
                try:
                    self._sync_preview_window(root, canvas)
                except Exception as exc:
                    logger.error("Error in forbidden area visualizer loop: %s", exc)
                    self.running = False
                    root.destroy()
                    return
                root.after(100, tick)

            tick()
            root.mainloop()
        except Exception as exc:
            logger.error("Failed to create forbidden area visualizer: %s", exc)
        finally:
            self.running = False
            self._root = None
