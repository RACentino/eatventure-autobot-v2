import ctypes
import ctypes.util
import logging
import multiprocessing as mp
import os
import platform
import queue
import threading
from typing import Any

import numpy as np

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
ForbiddenZone = tuple[int, int, int, int]
OverlayRegion = tuple[int, int, int, int]
FORBIDDEN_AREA_OVERLAY_LOOP_ITERATION_LIMIT = 2_147_483_647
FORBIDDEN_AREA_OVERLAY_REFRESH_MS = 100
OVERLAY_QUEUE_REPLACE_DRAIN_LIMIT = 8
OVERLAY_PROCESS_JOIN_TIMEOUT = 1.0
TRANSPARENT_OVERLAY_COLOR = "#010203"
WINDOWS_GWL_EXSTYLE = -20
WINDOWS_WS_EX_LAYERED = 0x00080000
WINDOWS_WS_EX_TRANSPARENT = 0x00000020
WINDOWS_WS_EX_TOOLWINDOW = 0x00000080
X11_SHAPE_BOUNDING = 0
X11_SHAPE_INPUT = 2


class _XRectangle(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_short),
        ("y", ctypes.c_short),
        ("width", ctypes.c_ushort),
        ("height", ctypes.c_ushort),
    ]


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
        if mss is None:
            raise WindowCaptureError(f"Cannot initialize screenshot backend: {_MSS_IMPORT_ERROR}")
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
        if pywinctl is None:
            raise WindowCaptureError(f"Could not initialize window backend: {_PYWINCTL_IMPORT_ERROR}")
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


def configure_overlay_root(root: Any, title: str) -> str:
    root.title(title)
    root.resizable(False, False)
    _call_if_supported(root, "overrideredirect", True)
    _set_window_attribute(root, "-topmost", True)
    root.configure(background=TRANSPARENT_OVERLAY_COLOR)
    if not _set_platform_transparency(root):
        _set_window_attribute(root, "-alpha", 0.85)
    enable_overlay_click_through(root)
    return TRANSPARENT_OVERLAY_COLOR


def configure_overlay_canvas(canvas: Any, background: str = TRANSPARENT_OVERLAY_COLOR) -> None:
    canvas.configure(background=background, highlightthickness=0, borderwidth=0)


def position_overlay_over_rect(root: Any, canvas: Any, rect: WindowRect) -> bool:
    x, y, width, height = rect
    if width <= 0 or height <= 0:
        return False
    root.geometry(f"{int(width)}x{int(height)}+{int(x)}+{int(y)}")
    canvas.config(width=int(width), height=int(height))
    return True


def create_overlay_queue(maxsize: int = 2) -> Any:
    return _overlay_process_context().Queue(maxsize=max(1, int(maxsize)))


def start_overlay_process(name: str, target: Any, *args: Any) -> tuple[Any, Any]:
    context = _overlay_process_context()
    stop_event = context.Event()
    process = context.Process(target=target, name=name, args=(*args, stop_event), daemon=True)
    process.start()
    return process, stop_event


def stop_overlay_process(process: Any | None, stop_event: Any | None, timeout: float = OVERLAY_PROCESS_JOIN_TIMEOUT) -> None:
    if stop_event is not None:
        stop_event.set()
    if process is None:
        return
    if not process.is_alive():
        process.join(timeout=0)
        return
    process.join(timeout=max(0.05, float(timeout)))
    if process.is_alive():
        process.terminate()
        process.join(timeout=max(0.05, float(timeout)))


def replace_queue_latest(payload_queue: Any, payload: Any) -> bool:
    for _ in range(OVERLAY_QUEUE_REPLACE_DRAIN_LIMIT):
        if not _discard_queue_item(payload_queue):
            break
    try:
        payload_queue.put_nowait(payload)
        return True
    except queue.Full:
        _discard_queue_item(payload_queue, timeout=0.02)
    try:
        payload_queue.put_nowait(payload)
        return True
    except queue.Full:
        return False


def enable_overlay_click_through(root: Any) -> bool:
    _call_if_supported(root, "update_idletasks")
    try:
        window_id = int(root.winfo_id())
    except Exception as exc:
        logger.debug("Overlay click-through unavailable without window id: %s", exc)
        return False
    enabled = _enable_platform_click_through(window_id)
    if not enabled:
        logger.debug("Overlay click-through is unsupported on this window manager")
    return enabled


def set_overlay_visible_regions(root: Any, regions: list[OverlayRegion]) -> bool:
    if platform.system() != "Linux":
        return False
    _call_if_supported(root, "update_idletasks")
    try:
        window_id = int(root.winfo_id())
    except Exception as exc:
        logger.debug("Overlay visible regions unavailable without window id: %s", exc)
        return False
    return _set_x11_shape_regions(window_id, X11_SHAPE_BOUNDING, regions)


def destroy_overlay_root(root: Any) -> None:
    try:
        root.destroy()
    except Exception:
        logger.debug("Overlay root was already closed")


def _set_platform_transparency(root: Any) -> bool:
    system_name = platform.system()
    if system_name == "Windows":
        return _set_window_attribute(root, "-transparentcolor", TRANSPARENT_OVERLAY_COLOR)
    if system_name == "Darwin":
        root.configure(background="systemTransparent")
        return _set_window_attribute(root, "-transparent", True)
    return False


def _enable_platform_click_through(window_id: int) -> bool:
    system_name = platform.system()
    if system_name == "Windows":
        return _enable_windows_click_through(window_id)
    if system_name == "Linux":
        return _enable_x11_click_through(window_id)
    return False


def _enable_windows_click_through(window_id: int) -> bool:
    windll = getattr(ctypes, "windll", None)
    if windll is None:
        return False
    user32 = windll.user32
    kernel32 = windll.kernel32
    style = user32.GetWindowLongW(window_id, WINDOWS_GWL_EXSTYLE)
    new_style = style | WINDOWS_WS_EX_LAYERED | WINDOWS_WS_EX_TRANSPARENT | WINDOWS_WS_EX_TOOLWINDOW
    kernel32.SetLastError(0)
    previous_style = user32.SetWindowLongW(window_id, WINDOWS_GWL_EXSTYLE, new_style)
    return bool(previous_style) or kernel32.GetLastError() == 0


def _enable_x11_click_through(window_id: int) -> bool:
    return _set_x11_shape_regions(window_id, X11_SHAPE_INPUT, [])


def _set_x11_shape_regions(window_id: int, shape_kind: int, regions: list[OverlayRegion]) -> bool:
    display_name = os.environ.get("DISPLAY")
    if not display_name:
        return False
    x11_path = ctypes.util.find_library("X11")
    xfixes_path = ctypes.util.find_library("Xfixes")
    if x11_path is None or xfixes_path is None:
        return False
    return _apply_x11_shape_regions(window_id, shape_kind, regions, display_name, x11_path, xfixes_path)


def _apply_x11_shape_regions(
    window_id: int,
    shape_kind: int,
    regions: list[OverlayRegion],
    display_name: str,
    x11_path: str,
    xfixes_path: str,
) -> bool:
    x11 = ctypes.CDLL(x11_path)
    xfixes = ctypes.CDLL(xfixes_path)
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    x11.XFlush.argtypes = [ctypes.c_void_p]
    xfixes.XFixesSetWindowShapeRegion.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    xfixes.XFixesCreateRegion.argtypes = [ctypes.c_void_p, ctypes.POINTER(_XRectangle), ctypes.c_int]
    xfixes.XFixesCreateRegion.restype = ctypes.c_ulong
    xfixes.XFixesDestroyRegion.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    display = x11.XOpenDisplay(display_name.encode())
    if not display:
        return False
    region = _create_x11_region(xfixes, display, regions)
    try:
        xfixes.XFixesSetWindowShapeRegion(display, ctypes.c_ulong(window_id), shape_kind, 0, 0, ctypes.c_void_p(region))
        x11.XFlush(display)
        return True
    finally:
        xfixes.XFixesDestroyRegion(display, region)
        x11.XCloseDisplay(display)


def _create_x11_region(xfixes: Any, display: Any, regions: list[OverlayRegion]) -> int:
    if not regions:
        return int(xfixes.XFixesCreateRegion(display, None, 0))
    rectangles = (_XRectangle * len(regions))(*[_x_rectangle(region) for region in regions])
    return int(xfixes.XFixesCreateRegion(display, rectangles, len(regions)))


def _x_rectangle(region: OverlayRegion) -> _XRectangle:
    x, y, width, height = region
    return _XRectangle(int(x), int(y), max(1, int(width)), max(1, int(height)))


def _set_window_attribute(root: Any, attribute_name: str, value: Any) -> bool:
    try:
        root.attributes(attribute_name, value)
        return True
    except Exception as exc:
        logger.debug("Overlay window attribute %s unsupported: %s", attribute_name, exc)
        return False


def _call_if_supported(target: Any, method_name: str, *args: Any) -> bool:
    method = getattr(target, method_name, None)
    if not callable(method):
        return False
    try:
        method(*args)
        return True
    except Exception as exc:
        logger.debug("Overlay method %s failed: %s", method_name, exc)
        return False


def _overlay_process_context() -> Any:
    return mp.get_context("spawn")


def _discard_queue_item(payload_queue: Any, timeout: float = 0.0) -> bool:
    try:
        if timeout <= 0:
            payload_queue.get_nowait()
        else:
            payload_queue.get(timeout=float(timeout))
        return True
    except queue.Empty:
        return False


def _find_overlay_window_rect(window_title: str) -> WindowRect | None:
    if pywinctl is None:
        return None
    try:
        windows = pywinctl.getWindowsWithTitle(str(window_title)) or []
    except Exception as exc:
        logger.debug("Overlay could not search for target window '%s': %s", window_title, exc)
        return None
    return _first_live_window_rect(str(window_title), windows)


def _first_live_window_rect(window_title: str, windows: list[Any]) -> WindowRect | None:
    live_windows = [window for window in windows if WindowCapture._alive(window)]
    for window in live_windows:
        if getattr(window, "title", None) == window_title:
            return _overlay_rect_from_window(window)
    if not live_windows:
        return None
    return _overlay_rect_from_window(live_windows[0])


def _overlay_rect_from_window(window: Any) -> WindowRect | None:
    try:
        return _bounds_from_geometry(window.getClientFrame())
    except Exception:
        try:
            return _bounds_from_geometry(window.box)
        except Exception as exc:
            logger.debug("Overlay could not read target window bounds: %s", exc)
            return None


def position_overlay_over_target(root: Any, canvas: Any, window_title: str) -> bool:
    rect = _find_overlay_window_rect(window_title)
    if rect is None:
        _call_if_supported(root, "withdraw")
        return False
    _call_if_supported(root, "deiconify")
    return position_overlay_over_rect(root, canvas, rect)


def _run_forbidden_area_overlay_process(window_title: str, forbidden_zones: tuple[ForbiddenZone, ...], stop_event: Any) -> None:
    try:
        import tkinter as tk

        root = tk.Tk()
        background = configure_overlay_root(root, "Forbidden Area Visualizer")
        canvas = tk.Canvas(root, highlightthickness=0, background=background)
        configure_overlay_canvas(canvas, background)
        canvas.pack(fill="both", expand=True)
        _schedule_forbidden_overlay_draw(root, canvas, window_title, forbidden_zones, stop_event)
        root.mainloop()
    except Exception as exc:
        logger.error("Failed to create forbidden area visualizer: %s", exc)
    finally:
        if "root" in locals():
            destroy_overlay_root(root)


def _schedule_forbidden_overlay_draw(root: Any, canvas: Any, window_title: str, forbidden_zones: tuple[ForbiddenZone, ...], stop_event: Any) -> None:
    if stop_event.is_set():
        _call_if_supported(root, "quit")
        return
    _draw_forbidden_overlay(root, canvas, window_title, forbidden_zones)
    root.after(FORBIDDEN_AREA_OVERLAY_REFRESH_MS, _schedule_forbidden_overlay_draw, root, canvas, window_title, forbidden_zones, stop_event)


def _draw_forbidden_overlay(root: Any, canvas: Any, window_title: str, forbidden_zones: tuple[ForbiddenZone, ...]) -> None:
    if not position_overlay_over_target(root, canvas, window_title):
        return
    canvas.delete("zone")
    zones = list(forbidden_zones)
    set_overlay_visible_regions(root, _zone_visible_regions(zones))
    for x_min, x_max, y_min, y_max in zones:
        canvas.create_rectangle(int(x_min), int(y_min), int(x_max), int(y_max), fill="#ff4040", stipple="gray25", outline="#ff4040", tags="zone")


class ForbiddenAreaOverlay:
    def __init__(self, window_capture: WindowCapture, forbidden_zones: list[ForbiddenZone]) -> None:
        self.window_capture = window_capture
        self.forbidden_zones = forbidden_zones
        self.running = False
        self.process: Any | None = None
        self.stop_event: Any | None = None

    def start(self) -> None:
        if self.running:
            return
        try:
            zones = tuple(self.forbidden_zones)
            self.process, self.stop_event = start_overlay_process("forbidden_area_overlay", _run_forbidden_area_overlay_process, self.window_capture.window_title, zones)
            self.running = True
        except Exception as exc:
            self.running = False
            logger.error("Failed to start forbidden area visualizer process: %s", exc)
            return
        logger.info("Forbidden area visualizer started")

    def stop(self) -> None:
        self.running = False
        stop_overlay_process(self.process, self.stop_event)
        self.process = None
        self.stop_event = None
        logger.info("Forbidden area visualizer stopped")

    def _run(self) -> None:
        if self.stop_event is None:
            return
        _ = (configure_overlay_root, configure_overlay_canvas)
        _run_forbidden_area_overlay_process(self.window_capture.window_title, tuple(self.forbidden_zones), self.stop_event)

    def _draw(self, root: Any, canvas: Any) -> bool:
        try:
            _ = (position_overlay_over_rect, set_overlay_visible_regions)
            _draw_forbidden_overlay(root, canvas, self.window_capture.window_title, tuple(self.forbidden_zones))
            return True
        except Exception as exc:
            logger.error("Error in forbidden area visualizer loop: %s", exc)
            return False


def _zone_visible_regions(zones: list[ForbiddenZone]) -> list[tuple[int, int, int, int]]:
    regions = []
    for x_min, x_max, y_min, y_max in zones:
        regions.append((int(x_min), int(y_min), max(1, int(x_max) - int(x_min)), max(1, int(y_max) - int(y_min))))
    return regions
