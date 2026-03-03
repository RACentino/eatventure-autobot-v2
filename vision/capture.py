"""Thread-safe screen capture using GDI and Win32 API."""

import ctypes
import logging
import threading
import numpy as np
import win32api
import win32con
import win32gui
import win32ui
from typing import Optional, Tuple, Dict

from core.exceptions import ScreenCaptureError
from core.logger import setup_logger

logger = setup_logger("vision.capture")

# Set DPI awareness for accurate coordinate mapping
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2) # PROCESS_PER_MONITOR_DPI_AWARE
except (AttributeError, OSError):
    logger.debug("DPI awareness API (shcore) unavailable; falling back to basic awareness.")
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        logger.warning("Failed to set DPI awareness. Coordinate mapping may be inaccurate.")

class WindowCapture:
    """
    Thread-safe GDI screen capturer.
    Uses a re-entrant lock to prevent GDI handle contention between main and monitor threads.
    """
    def __init__(self, window_title: str, target_width: int, target_height: int):
        self.window_title = window_title
        self.target_width = target_width
        self.target_height = target_height
        self.hwnd: Optional[int] = None
        self._lock = threading.RLock()
        
        # Caching
        self._cache: Dict[str, Tuple[float, np.ndarray]] = {}
        self._cache_ttl = 0.015 # Default TTL
        
        self._refresh_window_handle()
        self._ensure_window_size()

    def set_cache_ttl(self, ttl: float) -> None:
        """Sets the time-to-live for the capture cache."""
        self._cache_ttl = ttl

    def _refresh_window_handle(self) -> None:
        """Locates the window handle by title."""
        with self._lock:
            self.hwnd = win32gui.FindWindow(None, self.window_title)
            if not self.hwnd:
                raise ScreenCaptureError(f"Window '{self.window_title}' not found.")
            logger.debug(f"Window found: {self.window_title} (HWND: {self.hwnd})")

    def _ensure_window_size(self) -> None:
        """Resizes the window to the target dimensions if necessary."""
        with self._lock:
            if not self.hwnd:
                self._refresh_window_handle()
            
            # SWP_NOZORDER = 0x0004, SWP_SHOWWINDOW = 0x0040
            rect = win32gui.GetWindowRect(self.hwnd)
            x, y = rect[0], rect[1]
            
            ctypes.windll.user32.SetWindowPos(
                self.hwnd, 0, int(x), int(y), 
                int(self.target_width), int(self.target_height), 
                0x0004 | 0x0040
            )
            logger.info(f"Window synchronized to {self.target_width}x{self.target_height}")

    def get_client_rect(self) -> Tuple[int, int, int, int]:
        """Returns the (x, y, width, height) of the client area in screen coordinates."""
        with self._lock:
            if not self.hwnd or not win32gui.IsWindow(self.hwnd):
                self._refresh_window_handle()
            
            rect = win32gui.GetClientRect(self.hwnd)
            point = win32gui.ClientToScreen(self.hwnd, (rect[0], rect[1]))
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]
            return point[0], point[1], width, height

    def capture(self, max_y: Optional[int] = None, force: bool = False) -> np.ndarray:
        """
        Captures the client area of the window.
        
        Args:
            max_y: Optional vertical crop limit.
            force: If True, bypasses the cache.
            
        Returns:
            A BGR numpy array of the captured region.
        """
        import time
        cache_key = str(max_y) if max_y is not None else "full"
        
        with self._lock:
            now = time.monotonic()
            if not force and cache_key in self._cache:
                timestamp, frame = self._cache[cache_key]
                if now - timestamp <= self._cache_ttl:
                    return frame

            if not self.hwnd or not win32gui.IsWindow(self.hwnd):
                self._refresh_window_handle()

            x, y, width, height = self.get_client_rect()
            if max_y is not None:
                height = min(height, max_y)

            if width <= 0 or height <= 0:
                raise ScreenCaptureError(f"Invalid window dimensions: {width}x{height}")

            # GDI Resource Initialization
            hwnd_dc = win32gui.GetWindowDC(self.hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            
            save_bitmap = win32ui.CreateBitmap()
            save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(save_bitmap)
            
            try:
                # PW_RENDERFULLCONTENT = 3
                result = ctypes.windll.user32.PrintWindow(self.hwnd, save_dc.GetSafeHdc(), 3)
                if result != 1:
                    raise ScreenCaptureError(f"PrintWindow failed for HWND {self.hwnd}")

                bmp_info = save_bitmap.GetInfo()
                bmp_str = save_bitmap.GetBitmapBits(True)
                
                img = np.frombuffer(bmp_str, dtype=np.uint8)
                img.shape = (height, width, 4)
                
                # Drop alpha channel and return BGR
                frame = np.ascontiguousarray(img[:, :, :3])
                self._cache[cache_key] = (now, frame)
                return frame

            finally:
                # Strict GDI Resource Cleanup to prevent memory leaks
                win32gui.DeleteObject(save_bitmap.GetHandle())
                save_dc.DeleteDC()
                mfc_dc.DeleteDC()
                win32gui.ReleaseDC(self.hwnd, hwnd_dc)

    def is_active(self) -> bool:
        """Checks if the target window still exists and is visible."""
        with self._lock:
            return bool(self.hwnd and win32gui.IsWindow(self.hwnd) and win32gui.IsWindowVisible(self.hwnd))
