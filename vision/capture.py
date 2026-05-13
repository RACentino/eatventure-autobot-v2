import ctypes
import logging
import threading
import time
from typing import Any

import numpy as np

from core.platform import IS_WINDOWS, pywintypes, require_windows_backend, win32api, win32con, win32gui, win32ui

logger = logging.getLogger(__name__)
WindowRect = tuple[int, int, int, int]
ForbiddenZone = tuple[int, int, int, int]

if IS_WINDOWS:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        logger.debug("DPI awareness is already configured or unavailable")


class WindowCaptureError(RuntimeError):
    pass


class WindowNotAvailableError(WindowCaptureError):
    pass


class WindowCapture:
    def __init__(self, window_title: str, target_width: int = 800, target_height: int = 600) -> None:
        require_windows_backend("WindowCapture")
        self.window_title = window_title
        self.hwnd = None
        try:
            self.target_width = int(target_width)
            self.target_height = int(target_height)
        except (TypeError, ValueError) as exc:
            raise WindowCaptureError(f"Invalid target window size: {target_width}x{target_height}") from exc
        if self.target_width <= 0 or self.target_height <= 0:
            raise WindowCaptureError(f"Invalid target window size: {self.target_width}x{self.target_height}")
        self._lock = threading.RLock()
        try:
            self.ensure_window(resize=True)
        except WindowCaptureError as exc:
            logger.warning("%s", exc)

    def _find_window_handle(self) -> int | None:
        hwnd = win32gui.FindWindow(None, self.window_title)
        if hwnd and win32gui.IsWindow(hwnd):
            return hwnd
        return None

    def _invalidate_window(self) -> None:
        self.hwnd = None

    def _translate_win32_error(self, exc: pywintypes.error, action: str) -> WindowCaptureError:
        winerror = exc.args[0] if exc.args else None
        if winerror == 1400:
            self._invalidate_window()
            return WindowNotAvailableError(
                f"Window '{self.window_title}' is no longer available during {action}"
            )
        return WindowCaptureError(f"{action} failed for window '{self.window_title}': {exc}")

    def _resize_bound_window(self) -> None:
        if not self.hwnd or not win32gui.IsWindow(self.hwnd):
            return

        try:
            rect = win32gui.GetWindowRect(self.hwnd)
        except pywintypes.error as exc:
            raise self._translate_win32_error(exc, "resizing the window") from exc
        x, y = rect[0], rect[1]

        result = ctypes.windll.user32.SetWindowPos(
            self.hwnd,
            0,
            int(x),
            int(y),
            int(self.target_width),
            int(self.target_height),
            win32con.SWP_NOZORDER | win32con.SWP_SHOWWINDOW,
        )
        if not result:
            raise WindowCaptureError(f"SetWindowPos failed for '{self.window_title}'")
        logger.info("Window resized to %sx%s", self.target_width, self.target_height)

    def find_window(self) -> int:
        with self._lock:
            hwnd = self._find_window_handle()
            if not hwnd:
                self._invalidate_window()
                raise WindowNotAvailableError(f"Window '{self.window_title}' not found")
            if hwnd != self.hwnd:
                logger.info("Window found: %s (HWND: %s)", self.window_title, hwnd)
            self.hwnd = hwnd
            return self.hwnd

    def ensure_window(self, resize: bool = False) -> int:
        with self._lock:
            if self.hwnd and win32gui.IsWindow(self.hwnd):
                if resize:
                    self._resize_bound_window()
                return self.hwnd

            previous_hwnd = self.hwnd
            hwnd = self._find_window_handle()
            if not hwnd:
                self._invalidate_window()
                if previous_hwnd:
                    logger.warning("Window handle %s is no longer valid", previous_hwnd)
                raise WindowNotAvailableError(f"Window '{self.window_title}' not found")

            handle_changed = hwnd != previous_hwnd
            self.hwnd = hwnd
            if handle_changed:
                logger.info("Window found: %s (HWND: %s)", self.window_title, hwnd)
            if resize or handle_changed:
                self._resize_bound_window()
            return self.hwnd

    def get_hwnd(self) -> int:
        return self.ensure_window()

    def resize_window(self) -> None:
        with self._lock:
            self.ensure_window()
            self._resize_bound_window()

    def _get_client_size(self, hwnd: int) -> tuple[tuple[int, int, int, int], int, int]:
        try:
            rect = win32gui.GetClientRect(hwnd)
        except pywintypes.error as exc:
            raise self._translate_win32_error(exc, "reading the window bounds") from exc

        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        if width <= 0 or height <= 0:
            raise WindowCaptureError(
                f"Window '{self.window_title}' has an invalid client size: {width}x{height}"
            )
        return rect, width, height

    def get_window_rect(self) -> WindowRect:
        hwnd = self.ensure_window()
        rect, width, height = self._get_client_size(hwnd)
        try:
            x, y = win32gui.ClientToScreen(hwnd, (rect[0], rect[1]))
        except pywintypes.error as exc:
            raise self._translate_win32_error(exc, "reading the window bounds") from exc
        return x, y, width, height

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

    def _decode_bitmap(self, bitmap_bytes: bytes, width: int, height: int) -> np.ndarray:
        expected_size = height * width * 4
        if len(bitmap_bytes) != expected_size:
            raise WindowCaptureError(
                f"Captured bitmap size mismatch: expected {expected_size} bytes, got {len(bitmap_bytes)}"
            )
        bitmap_image = np.frombuffer(bitmap_bytes, dtype=np.uint8).reshape((height, width, 4))
        return np.ascontiguousarray(bitmap_image[:, :, :3])

    @staticmethod
    def _release_capture_resources(
        hwnd: int,
        hwnd_dc: Any,
        mfc_dc: Any,
        save_dc: Any,
        save_bitmap: Any,
        old_bitmap: Any,
    ) -> None:
        if save_dc is not None and old_bitmap is not None:
            try:
                save_dc.SelectObject(old_bitmap)
            except pywintypes.error as exc:
                logger.debug("Could not restore capture DC bitmap: %s", exc)
        if save_bitmap is not None:
            win32gui.DeleteObject(save_bitmap.GetHandle())
        if save_dc is not None:
            save_dc.DeleteDC()
        if mfc_dc is not None:
            mfc_dc.DeleteDC()
        if hwnd_dc is not None and hwnd:
            win32gui.ReleaseDC(hwnd, hwnd_dc)

    def capture(self, max_y: Any = None) -> np.ndarray:
        with self._lock:
            hwnd = self.ensure_window()
            _, width, height = self._get_client_size(hwnd)

            height = self._apply_capture_height_limit(height, max_y)
            self._validate_capture_size(width, height)

            hwnd_dc = None
            mfc_dc = None
            save_dc = None
            save_bitmap = None
            old_bitmap = None
            try:
                hwnd_dc = win32gui.GetWindowDC(hwnd)
                mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
                save_dc = mfc_dc.CreateCompatibleDC()

                save_bitmap = win32ui.CreateBitmap()
                save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
                old_bitmap = save_dc.SelectObject(save_bitmap)

                result = ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 3)
                if result != 1:
                    raise WindowCaptureError(f"PrintWindow failed for '{self.window_title}'")

                bitmap_bytes = save_bitmap.GetBitmapBits(True)
                return self._decode_bitmap(bitmap_bytes, width, height)
            except pywintypes.error as exc:
                raise self._translate_win32_error(exc, "capturing the window") from exc
            except ValueError as exc:
                raise WindowCaptureError(f"Captured bitmap could not be decoded: {exc}") from exc
            finally:
                self._release_capture_resources(hwnd, hwnd_dc, mfc_dc, save_dc, save_bitmap, old_bitmap)

    def is_window_active(self) -> bool:
        with self._lock:
            if self.hwnd and win32gui.IsWindow(self.hwnd):
                return True
            self.hwnd = self._find_window_handle()
            return bool(self.hwnd)


class ForbiddenAreaOverlay:
    def __init__(self, target_hwnd: int, forbidden_zones: list[ForbiddenZone]) -> None:
        require_windows_backend("ForbiddenAreaOverlay")
        self.target_hwnd = target_hwnd
        self.forbidden_zones = forbidden_zones
        self.overlay_hwnd = None
        self.running = False
        self.thread = None
        
    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._create_overlay, daemon=True)
        self.thread.start()
        logger.info("Forbidden area overlay started")
    
    def stop(self) -> None:
        self.running = False
        if self.overlay_hwnd:
            try:
                win32gui.DestroyWindow(self.overlay_hwnd)
            except pywintypes.error as exc:
                logger.debug("Overlay destroy failed: %s", exc)
            self.overlay_hwnd = None
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        logger.info("Forbidden area overlay stopped")

    def _build_window_class(self) -> Any:
        overlay_class = win32gui.WNDCLASS()
        overlay_class.lpfnWndProc = self._wnd_proc
        overlay_class.lpszClassName = "ForbiddenAreaOverlay"
        overlay_class.hCursor = win32gui.LoadCursor(0, win32con.IDC_ARROW)
        overlay_class.hbrBackground = win32gui.GetStockObject(win32con.NULL_BRUSH)
        return overlay_class

    def _register_window_class(self) -> None:
        try:
            win32gui.RegisterClass(self._build_window_class())
        except pywintypes.error:
            pass

    def _target_window_metrics(self) -> tuple[tuple[int, int], int, int]:
        target_rect = win32gui.GetClientRect(self.target_hwnd)
        target_position = win32gui.ClientToScreen(self.target_hwnd, (0, 0))
        width = target_rect[2] - target_rect[0]
        height = target_rect[3] - target_rect[1]
        return target_position, width, height

    def _create_overlay_window(self, target_position: tuple[int, int], width: int, height: int) -> int:
        return win32gui.CreateWindowEx(
            win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_TOPMOST | win32con.WS_EX_TOOLWINDOW,
            "ForbiddenAreaOverlay",
            "Forbidden Area Overlay",
            win32con.WS_POPUP,
            target_position[0], target_position[1],
            width, height,
            0, 0, 0, None
        )

    def _show_overlay_window(self) -> None:
        win32gui.SetLayeredWindowAttributes(
            self.overlay_hwnd,
            0,
            128,
            win32con.LWA_ALPHA
        )
        win32gui.ShowWindow(self.overlay_hwnd, win32con.SW_SHOW)
        win32gui.UpdateWindow(self.overlay_hwnd)

    def _sync_overlay_position(self, last_position: tuple[int, int], width: int, height: int) -> tuple[int, int]:
        new_position = win32gui.ClientToScreen(self.target_hwnd, (0, 0))
        if new_position == last_position:
            return last_position
        win32gui.SetWindowPos(
            self.overlay_hwnd,
            win32con.HWND_TOPMOST,
            new_position[0], new_position[1],
            width, height,
            win32con.SWP_SHOWWINDOW
        )
        self._draw_zones()
        return new_position
    
    def _create_overlay(self) -> None:
        try:
            self._register_window_class()
            target_position, width, height = self._target_window_metrics()
            self.overlay_hwnd = self._create_overlay_window(target_position, width, height)
            self._show_overlay_window()
            self._draw_zones()
            
            last_position = target_position
            while self.running:
                try:
                    last_position = self._sync_overlay_position(last_position, width, height)
                except pywintypes.error as exc:
                    logger.error("Error in overlay update loop: %s", exc)
                    break
                
                time.sleep(0.1)
                
        except pywintypes.error as exc:
            logger.error("Failed to create overlay window: %s", exc)
        finally:
            self.running = False

    def _draw_zone_rectangles(self, device_context: Any, brush: Any) -> None:
        for x_min, x_max, y_min, y_max in self.forbidden_zones:
            old_brush = win32gui.SelectObject(device_context, brush)
            win32gui.Rectangle(device_context, int(x_min), int(y_min), int(x_max), int(y_max))
            win32gui.SelectObject(device_context, old_brush)
    
    def _draw_zones(self) -> None:
        if not self.overlay_hwnd:
            return
            
        hdc = None
        red_brush = None
        try:
            hdc = win32gui.GetDC(self.overlay_hwnd)
            red_brush = win32gui.CreateSolidBrush(win32api.RGB(255, 0, 0))
            self._draw_zone_rectangles(hdc, red_brush)
        except pywintypes.error as exc:
            logger.error("Error drawing zones: %s", exc)
        finally:
            if red_brush is not None:
                win32gui.DeleteObject(red_brush)
            if hdc is not None:
                win32gui.ReleaseDC(self.overlay_hwnd, hdc)
    
    def _wnd_proc(self, hwnd: int, msg: int, wparam: int, lparam: int) -> int:
        if msg == win32con.WM_PAINT:
            hdc, ps = win32gui.BeginPaint(hwnd)
            
            red_brush = win32gui.CreateSolidBrush(win32api.RGB(255, 0, 0))
            self._draw_zone_rectangles(hdc, red_brush)
            win32gui.DeleteObject(red_brush)
            win32gui.EndPaint(hwnd, ps)
            return 0
        if msg == win32con.WM_DESTROY:
            win32gui.PostQuitMessage(0)
            return 0
        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)
