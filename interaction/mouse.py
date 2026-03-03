"""High-security, bounded mouse interaction controller."""

import logging
import time
import win32api
import win32con
import win32gui
from typing import Tuple, Optional, Callable, Dict, Any

from core import config
from core.exceptions import TargetOutOfBoundsError
from core.logger import setup_logger

logger = setup_logger("interaction.mouse")

class MouseController:
    """
    Defensive Mouse Controller.
    Implements a 'Paranoid' boundary validator that ensures all click coordinates
    reside strictly within the target window client rect.
    """
    def __init__(self, hwnd: int):
        self.hwnd = hwnd
        self._last_click_time = 0.0
        self._last_cursor_pos: Optional[Tuple[int, int]] = None
        
        # Injected safety hook for higher-level bot checks (e.g. state interrupts)
        self.interrupt_callback: Optional[Callable[[], bool]] = None

    def _get_window_origin(self) -> Tuple[int, int]:
        """Returns the screen coordinates of the window origin."""
        return win32gui.ClientToScreen(self.hwnd, (0, 0))

    def _get_client_size(self) -> Tuple[int, int]:
        """Returns the (width, height) of the client area."""
        rect = win32gui.GetClientRect(self.hwnd)
        return rect[2] - rect[0], rect[3] - rect[1]

    def _validate_bounds(self, x: int, y: int) -> None:
        """
        The Paranoid Validator.
        Ensures target coordinates are within the safe client bounds defined by WINDOW_WIDTH and WINDOW_HEIGHT.
        Also checks against user-defined FORBIDDEN_ZONES.
        """
        # 1. Screen-Space Check
        if x < 0 or x >= config.WINDOW_WIDTH or y < 0 or y >= config.WINDOW_HEIGHT:
            logger.error(f"Target coordinate ({x}, {y}) is outside client bounds ({config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT})")
            raise TargetOutOfBoundsError(f"Coordinate ({x}, {y}) out of bounds.")

        # 2. Forbidden Zone Check
        for zone in config.FORBIDDEN_ZONES:
            if (zone["x_min"] <= x <= zone["x_max"]) and (zone["y_min"] <= y <= zone["y_max"]):
                logger.warning(f"Target ({x}, {y}) resides within forbidden zone '{zone['name']}'")
                raise TargetOutOfBoundsError(f"Coordinate ({x}, {y}) is in a forbidden zone.")

    def _translate_to_screen(self, x: int, y: int) -> Tuple[int, int]:
        """Translates window-relative coordinates into absolute screen coordinates."""
        self._validate_bounds(x, y)
        origin_x, origin_y = self._get_window_origin()
        return origin_x + x, origin_y + y

    def _execute_click(self, x: int, y: int, wait_after: bool = True) -> None:
        """Executes the physical click via win32api."""
        screen_x, screen_y = self._translate_to_screen(x, y)
        
        # Move cursor to target
        win32api.SetCursorPos((screen_x, screen_y))
        time.sleep(config.MOUSE_MOVE_DELAY)
        
        # Dispatch MOUSEEVENTF_LEFTDOWN/UP
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, screen_x, screen_y, 0, 0)
        time.sleep(config.MOUSE_DOWN_UP_DELAY)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, screen_x, screen_y, 0, 0)
        
        if wait_after:
            time.sleep(config.CLICK_DELAY)
            
        self._last_click_time = time.monotonic()
        logger.debug(f"Click executed at screen ({screen_x}, {screen_y}) [rel: ({x}, {y})]")

    def is_in_forbidden_zone(self, x: int, y: int, relative: bool = True) -> bool:
        """Checks if a coordinate resides within any forbidden zone."""
        # Note: relative=True means x,y are in 'image' space (0 to WINDOW_WIDTH/HEIGHT)
        for zone in config.FORBIDDEN_ZONES:
            if (zone["x_min"] <= x <= zone["x_max"]) and (zone["y_min"] <= y <= zone["y_max"]):
                return True
        return False

    def click(self, x: int, y: int, wait_after: bool = True) -> bool:
        """
        Triggers a click at the specified relative coordinates.
        Returns True if successful, False if interrupted by callback or out of bounds.
        """
        if self.interrupt_callback and self.interrupt_callback():
            logger.info("Click action aborted by interrupt callback.")
            return False
            
        try:
            start_time = time.monotonic()
            self._execute_click(x, y, wait_after=wait_after)
            exec_time = time.monotonic() - start_time
            logger.debug(f"Click execution time: {exec_time:.4f}s")
            return True
        except TargetOutOfBoundsError as exc:
            logger.error(f"Click failed: {exc}")
            return False

    def hold(self, x: int, y: int, duration: float) -> bool:
        """Holds the mouse button down at (x, y) for a specified duration."""
        if self.interrupt_callback and self.interrupt_callback():
            return False
            
        try:
            screen_x, screen_y = self._translate_to_screen(x, y)
            win32api.SetCursorPos((screen_x, screen_y))
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, screen_x, screen_y, 0, 0)
            
            # Duration-based polling to allow interrupt detection
            start_time = time.monotonic()
            while time.monotonic() - start_time < duration:
                if self.interrupt_callback and self.interrupt_callback():
                    break
                time.sleep(0.05)
                
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, screen_x, screen_y, 0, 0)
            return True
        except TargetOutOfBoundsError as exc:
            logger.error(f"Hold failed: {exc}")
            return False

    def drag(self, from_x: int, from_y: int, to_x: int, to_y: int, duration: float = 0.3) -> bool:
        """Performs a smooth drag operation from start to end coordinates."""
        if self.interrupt_callback and self.interrupt_callback():
            return False
            
        try:
            start_sx, start_sy = self._translate_to_screen(from_x, from_y)
            end_sx, end_sy = self._translate_to_screen(to_x, to_y)
            
            win32api.SetCursorPos((start_sx, start_sy))
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, start_sx, start_sy, 0, 0)
            
            # Linear Interpolation for smooth movement
            steps = config.DRAG_STEPS
            for i in range(steps + 1):
                if self.interrupt_callback and self.interrupt_callback():
                    logger.info("Drag operation aborted by interrupt.")
                    break
                    
                t = i / float(steps)
                curr_x = int(start_sx + (end_sx - start_sx) * t)
                curr_y = int(start_sy + (end_sy - start_sy) * t)
                win32api.SetCursorPos((curr_x, curr_y))
                time.sleep(duration / float(steps))
                
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, end_sx, end_sy, 0, 0)
            return True
        except TargetOutOfBoundsError as exc:
            logger.error(f"Drag failed: {exc}")
            return False
