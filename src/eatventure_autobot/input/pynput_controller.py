import logging
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

from eatventure_autobot.domain.config import InputTimingConfig
from eatventure_autobot.domain.errors import InputError
from eatventure_autobot.domain.protocols import ScreenCapture
from eatventure_autobot.domain.types import Point, WindowBounds, Zone

logger = logging.getLogger(__name__)


class _MouseDevice(Protocol):
    """The slice of pynput.mouse.Controller this module depends on. A fake implementing
    this shape can be injected in place of the real device for tests that must not move the
    host's actual cursor."""

    position: Point

    def press(self, button: Any) -> None: ...

    def release(self, button: Any) -> None: ...


def _clamped_duration(value: float | None, default: float) -> float:
    if value is None:
        return max(0.0, default)
    return max(0.0, value)


class PynputInputController:
    def __init__(
        self,
        screen_capture: ScreenCapture,
        timing: InputTimingConfig,
        static_zones: tuple[Zone, ...] = (),
        stop_event: threading.Event | None = None,
        device: _MouseDevice | None = None,
    ) -> None:
        if device is None:
            from pynput import mouse

            device = mouse.Controller()
            self._left_button: Any = mouse.Button.left
        else:
            self._left_button = getattr(device, "left_button", "left")
        self._capture = screen_capture
        self._device = device
        self._timing = timing
        self._static_zones = static_zones
        self._event_zone: Zone | None = None
        self._stop_event = stop_event
        self._lock = threading.RLock()
        self._left_pressed = False

    def _wait(self, duration: float) -> bool:
        if self._stop_event is None:
            time.sleep(duration)
            return True
        return not self._stop_event.wait(duration)

    def is_target_foreground(self) -> bool:
        try:
            return bool(self._capture.is_window_active())
        except Exception:
            return False

    def _window_bounds(self) -> WindowBounds:
        try:
            return self._capture.get_window_rect()
        except Exception as exc:
            raise InputError(f"Cannot read target window bounds: {exc}") from exc

    def _zones(self) -> tuple[Zone, ...]:
        return (self._event_zone, *self._static_zones) if self._event_zone else self._static_zones

    def set_event_forbidden_zone(self, zone: Zone) -> None:
        with self._lock:
            self._event_zone = zone

    def is_in_forbidden_zone(self, x: int, y: int, relative: bool = True) -> bool:
        try:
            if not relative:
                bounds = self._window_bounds()
                x, y = x - bounds.left, y - bounds.top
            return any(zone.contains(x, y) for zone in self._zones())
        except InputError:
            return True

    def _resolve_screen_position(
        self, x: int, y: int, relative: bool = True, check_forbidden: bool = True
    ) -> Point | None:
        bounds = self._window_bounds()
        rel_x, rel_y = (x, y) if relative else (x - bounds.left, y - bounds.top)
        if not (0 <= rel_x < bounds.width and 0 <= rel_y < bounds.height):
            logger.warning("Rejected input outside target window: (%s, %s)", rel_x, rel_y)
            return None
        if check_forbidden and self.is_in_forbidden_zone(rel_x, rel_y):
            return None
        return bounds.left + rel_x, bounds.top + rel_y

    def _input_allowed(self) -> bool:
        if self._stop_event is not None and self._stop_event.is_set():
            return False
        if self.is_target_foreground():
            return True
        logger.warning("Rejected input because the target window is not foreground")
        return False

    def get_cursor_position(self) -> Point:
        with self._lock:
            x, y = self._device.position
            return int(x), int(y)

    def _set_cursor_pos(self, x: int, y: int) -> bool:
        if self._resolve_screen_position(x, y, relative=False) is None:
            return False
        retry_count = max(1, self._timing.retry_count)
        for attempt in range(retry_count):
            if not self._input_allowed():
                return False
            try:
                self._device.position = (x, y)
                if not self._wait(0.001):
                    return False
                if self.get_cursor_position() == (x, y):
                    return True
            except Exception as exc:
                logger.warning(
                    "Cursor move failed on attempt %s/%s: %s", attempt + 1, retry_count, exc
                )
            if attempt + 1 < retry_count and not self._wait(self._timing.retry_delay):
                return False
        return False

    def _best_effort_left_up(self) -> None:
        try:
            self._device.release(self._left_button)
        except Exception:
            logger.debug("Best-effort left-button release failed", exc_info=True)
        finally:
            self._left_pressed = False

    def release_left_button(self) -> bool:
        with self._lock:
            if not self._left_pressed:
                return True
            self._best_effort_left_up()
            return not self._left_pressed

    def _cursor_on_target(self, x: int, y: int) -> bool:
        tolerance = self._timing.cursor_drift_tolerance_px
        current = self.get_cursor_position()
        return abs(current[0] - x) <= tolerance and abs(current[1] - y) <= tolerance

    def _left_down_at(self, x: int, y: int, duration: float | None) -> bool:
        if not self._input_allowed():
            return False
        if not self._cursor_on_target(x, y):
            logger.warning("Cursor moved before mouse press at (%s, %s)", x, y)
            return False
        try:
            self._device.press(self._left_button)
            self._left_pressed = True
        except Exception as exc:
            self._best_effort_left_up()
            raise InputError(f"Mouse press failed at ({x}, {y}): {exc}") from exc
        wait_time = _clamped_duration(duration, self._timing.mouse_down_duration)
        if wait_time > 0 and not self._wait(wait_time):
            self._best_effort_left_up()
            return False
        return True

    def _left_up_at(self, x: int, y: int, duration: float | None) -> bool:
        try:
            self._device.release(self._left_button)
            self._left_pressed = False
        except Exception as exc:
            raise InputError(f"Mouse release failed at ({x}, {y}): {exc}") from exc
        wait_time = _clamped_duration(duration, self._timing.mouse_up_duration)
        return wait_time <= 0 or self._wait(wait_time)

    def _left_click_at(
        self, x: int, y: int, down_duration: float | None = None, up_duration: float | None = None
    ) -> bool:
        if not self._left_down_at(x, y, down_duration):
            return False
        released = False
        try:
            if not self._cursor_on_target(x, y):
                logger.warning("Cursor moved during click at (%s, %s)", x, y)
                return False
            released = self._left_up_at(x, y, up_duration)
            return released
        finally:
            if not released:
                self._best_effort_left_up()

    def _hover_before_click(self) -> bool:
        if not self._timing.hover_enabled:
            return True
        return self._wait(self._timing.hover_duration)

    def click(self, x: int, y: int, relative: bool = True, delay: float | None = None) -> bool:
        with self._lock:
            position = self._resolve_screen_position(x, y, relative)
            if position is None or not self._set_cursor_pos(*position):
                return False
            if not self._wait(self._timing.move_delay) or not self._hover_before_click():
                return False
            if not self._left_click_at(*position):
                return False
            return self._wait(self._timing.click_delay if delay is None else delay)

    def precise_click(
        self, x: int, y: int, relative: bool = True, delay: float | None = None
    ) -> bool:
        with self._lock:
            position = self._resolve_screen_position(x, y, relative)
            if position is None or not self._set_cursor_pos(*position):
                return False
            if not self._wait(self._timing.move_delay) or not self._hover_before_click():
                return False
            if not self._left_down_at(*position, None):
                return False
            released = False
            try:
                if self._set_cursor_pos(*position):
                    released = self._left_up_at(*position, None)
                return released
            finally:
                if not released:
                    self._best_effort_left_up()

    def spam_click_at(
        self,
        x: int,
        y: int,
        duration: float,
        click_delay: float,
        relative: bool = True,
        down_duration: float | None = None,
        up_duration: float | None = None,
    ) -> bool:
        with self._lock:
            position = self._resolve_screen_position(x, y, relative)
            if position is None or not self._set_cursor_pos(*position):
                return False
            if not self._wait(self._timing.move_delay):
                return False
            end_time = time.monotonic() + max(0.0, duration)
            click_count = 0
            while time.monotonic() < end_time:
                if not self._left_click_at(*position, down_duration, up_duration):
                    return False
                click_count += 1
                if not self._wait(max(0.001, click_delay)):
                    return False
            return click_count > 0 and self._wait(self._timing.click_delay)

    def hold_at(
        self,
        x: int,
        y: int,
        duration: float,
        check_interval: float,
        interrupt_check: Callable[[], bool],
        relative: bool = True,
    ) -> bool:
        """Press and hold at (x, y), polling interrupt_check() every check_interval until it
        returns True or duration elapses, then release. Bails (returning False) if the cursor
        drifts off-target or the target window loses foreground mid-hold — matches the
        upgrade-station hold behavior verified in both source repos' HOLD_UPGRADE_STATION."""
        with self._lock:
            position = self._resolve_screen_position(x, y, relative)
            if position is None or not self._set_cursor_pos(*position):
                return False
            if not self._wait(self._timing.move_delay) or not self._left_down_at(*position, 0.0):
                return False
            released = False
            try:
                deadline = time.monotonic() + max(0.0, duration)
                interval = max(0.001, check_interval)
                while time.monotonic() < deadline:
                    if not self._wait(min(interval, max(0.0, deadline - time.monotonic()))):
                        return False
                    if not self._cursor_on_target(*position):
                        logger.warning("Cursor moved during hold at %s; releasing", position)
                        return False
                    if not self.is_target_foreground():
                        return False
                    if interrupt_check():
                        break
                released = self._left_up_at(*position, None)
                return released
            finally:
                if not released:
                    self._best_effort_left_up()

    def drag(
        self,
        from_x: int,
        from_y: int,
        to_x: int,
        to_y: int,
        duration: float | None = None,
        relative: bool = True,
        steps: int = 20,
    ) -> bool:
        with self._lock:
            start = self._resolve_screen_position(from_x, from_y, relative)
            end = self._resolve_screen_position(to_x, to_y, relative)
            if start is None or end is None or not self._set_cursor_pos(*start):
                return False
            if not self._wait(self._timing.move_delay) or not self._left_down_at(*start, None):
                return False
            released = False
            try:
                total_duration = _clamped_duration(duration, self._timing.click_delay)
                started = time.monotonic()
                for step in range(1, steps + 1):
                    ratio = step / steps
                    point = (
                        round(start[0] + (end[0] - start[0]) * ratio),
                        round(start[1] + (end[1] - start[1]) * ratio),
                    )
                    if not self._set_cursor_pos(*point):
                        return False
                    deadline = started + total_duration * ratio
                    if not self._wait(max(0.0, deadline - time.monotonic())):
                        return False
                released = self._left_up_at(*end, None)
                return released and self._wait(self._timing.click_delay)
            finally:
                if not released:
                    self._best_effort_left_up()
