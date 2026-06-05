import logging
import math
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

from pynput import mouse as pynput_mouse

import config

logger = logging.getLogger(__name__)

Point = tuple[int, int]
WindowBounds = tuple[int, int, int, int]
ForbiddenZone = tuple[str, int, int, int, int | None]
MIN_SLEEP_SLICE = 0.001
MAX_SLEEP_ITERATIONS = 120_000
CURSOR_ATTEMPTS = 2
CURSOR_RETRY_DELAY = 0.08
MAX_DRAG_STEPS = 60
MAX_STATS_UPGRADE_CLICKS = 500


def _duration(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    if not math.isfinite(number):
        number = float(default)
    return max(0.0, number)


def precise_sleep(duration: Any) -> None:
    wait_event(None, duration)


def sleep_until(deadline: float, stop_event: threading.Event | None = None) -> bool:
    for _ in range(_sleep_iterations(deadline)):
        if stop_event is not None and stop_event.is_set():
            return False
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return stop_event is None or not stop_event.is_set()
        if stop_event is None:
            time.sleep(min(remaining, 0.05))
        elif stop_event.wait(min(remaining, 0.05)):
            return False
    return stop_event is None or not stop_event.is_set()


def wait_event(stop_event: threading.Event | None, duration: Any) -> bool:
    duration = _duration(duration)
    if duration <= 0:
        return stop_event is None or not stop_event.is_set()
    return sleep_until(time.perf_counter() + duration, stop_event)


def _sleep_iterations(deadline: float) -> int:
    remaining = max(0.0, deadline - time.perf_counter())
    return min(MAX_SLEEP_ITERATIONS, max(1, int(math.ceil(remaining / MIN_SLEEP_SLICE)) + 3))


def _as_bounds(bounds: Any) -> WindowBounds:
    if isinstance(bounds, (str, bytes, bytearray)) or not isinstance(bounds, Sequence):
        raise TypeError(f"expected a 4-item window bounds sequence, got {type(bounds).__name__}")
    if len(bounds) != 4:
        raise ValueError(f"expected 4 window bounds values, got {len(bounds)}")
    x, y, width, height = (int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3]))
    if width <= 0 or height <= 0:
        raise ValueError(f"Target window has invalid client size: {width}x{height}")
    return x, y, width, height


def _inside_forbidden_zone(x: int, y: int, zone: ForbiddenZone) -> bool:
    _, x_min, x_max, y_min, y_max = zone
    if y_max is None:
        return y >= y_min and x_min <= x <= x_max
    return x_min <= x <= x_max and y_min <= y <= y_max


class MouseController:
    def __init__(
        self,
        window_bounds_source: Any,
        click_delay: Any = None,
        move_delay: Any = None,
        hover_enabled: bool | None = None,
        hover_duration: Any = None,
        mouse_device: Any = None,
    ) -> None:
        self._mouse = mouse_device if mouse_device is not None else pynput_mouse.Controller()
        self._left_button = pynput_mouse.Button.left
        self._window_bounds_source = window_bounds_source
        self.click_delay = _duration(config.CLICK_DELAY if click_delay is None else click_delay, 0.08)
        self.move_delay = _duration(config.MOUSE_MOVE_DELAY if move_delay is None else move_delay, 0.025)
        self.hover_enabled = bool(config.HOVER_ENABLED if hover_enabled is None else hover_enabled)
        self.hover_duration = _duration(config.HOVER_DURATION if hover_duration is None else hover_duration)
        self._input_lock = threading.RLock()
        self._left_button_is_down = False
        self._forbidden_zones = self._configured_forbidden_zones()

    @staticmethod
    def _configured_forbidden_zones() -> list[ForbiddenZone]:
        zones: list[ForbiddenZone] = [
            ("FORBIDDEN_CLICK", config.FORBIDDEN_CLICK_X_MIN, config.FORBIDDEN_CLICK_X_MAX, config.FORBIDDEN_CLICK_Y_MIN, None),
        ]
        for index, (x_min, x_max, y_min, y_max) in enumerate(config.NUMBERED_FORBIDDEN_ZONE_BOUNDS, start=1):
            zones.append((f"FORBIDDEN_ZONE_{index}", x_min, x_max, y_min, y_max))
        return zones

    def get_cursor_position(self) -> Point:
        x, y = self._mouse.position
        return int(x), int(y)

    def get_window_bounds(self) -> WindowBounds:
        try:
            source = self._window_bounds_source() if callable(self._window_bounds_source) else self._window_bounds_source
            return _as_bounds(source)
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc

    def get_window_position(self) -> Point:
        x, y, _, _ = self.get_window_bounds()
        return x, y

    def _relative_position(self, x: Any, y: Any, relative: bool) -> tuple[int, int, int, int, int, int] | None:
        try:
            window_x, window_y, width, height = self.get_window_bounds()
            if relative:
                rel_x, rel_y = int(x), int(y)
            else:
                rel_x, rel_y = int(x) - window_x, int(y) - window_y
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.error("Cannot resolve input position: %s", exc)
            return None
        if not (0 <= rel_x < width and 0 <= rel_y < height):
            logger.warning("Rejected input outside target window: relative=(%s, %s), bounds=%sx%s", rel_x, rel_y, width, height)
            return None
        return rel_x, rel_y, window_x, window_y, width, height

    def _screen_position(self, x: Any, y: Any, relative: bool = True, check_forbidden: bool = True) -> Point | None:
        position = self._relative_position(x, y, relative)
        if position is None:
            return None
        rel_x, rel_y, window_x, window_y, _, _ = position
        if check_forbidden and self.is_in_forbidden_zone(rel_x, rel_y, relative=True):
            return None
        return window_x + rel_x, window_y + rel_y

    def _resolve_screen_position(self, x: Any, y: Any, relative: bool = True, check_forbidden: bool = True) -> Point | None:
        return self._screen_position(x, y, relative=relative, check_forbidden=check_forbidden)

    def is_in_forbidden_zone(self, x: Any, y: Any, relative: bool = True) -> bool:
        position = self._relative_position(x, y, relative)
        if position is None:
            return True
        rel_x, rel_y, _, _, _, _ = position
        for zone in self._forbidden_zones:
            if _inside_forbidden_zone(rel_x, rel_y, zone):
                logger.debug("Coordinates (%s, %s) blocked by %s", rel_x, rel_y, zone[0])
                return True
        return False

    def _set_cursor(self, screen_x: int, screen_y: int, verify: bool = True) -> bool:
        for attempt in range(1, CURSOR_ATTEMPTS + 1):
            try:
                self._mouse.position = (int(screen_x), int(screen_y))
                if not verify:
                    return True
                precise_sleep(0.001)
                current_x, current_y = self.get_cursor_position()
                if abs(current_x - screen_x) <= 1 and abs(current_y - screen_y) <= 1:
                    return True
            except Exception as exc:
                logger.warning("Cursor positioning failed on attempt %s/%s: %s", attempt, CURSOR_ATTEMPTS, exc)
            if attempt < CURSOR_ATTEMPTS:
                precise_sleep(CURSOR_RETRY_DELAY)
        logger.error("Cursor positioning failed at (%s, %s)", screen_x, screen_y)
        return False

    def _button(self, action_name: str, action: Callable[[], Any], x: int, y: int) -> bool:
        try:
            action()
            return True
        except Exception as exc:
            logger.error("%s failed at (%s, %s): %s", action_name, x, y, exc)
            return False

    def _press_left(self, x: int, y: int, duration: Any = None, interrupt_check: Callable[[], bool] | None = None) -> bool:
        if not self._button("left down", lambda: self._mouse.press(self._left_button), x, y):
            return False
        self._left_button_is_down = True
        wait_time = _duration(config.MOUSE_DOWN_DURATION if duration is None else duration, config.MOUSE_DOWN_DURATION)
        if wait_time > 0 and not self._interruptible_delay(wait_time, interrupt_check):
            self._release_left(x, y)
            return False
        return True

    def _release_left(self, x: int, y: int, duration: Any = None, interrupt_check: Callable[[], bool] | None = None) -> bool:
        if not self._button("left up", lambda: self._mouse.release(self._left_button), x, y):
            self._left_button_is_down = False
            return False
        self._left_button_is_down = False
        wait_time = _duration(config.MOUSE_UP_DURATION if duration is None else duration, config.MOUSE_UP_DURATION)
        return self._interruptible_delay(wait_time, interrupt_check)

    @staticmethod
    def _interruptible_delay(duration: Any, interrupt_check: Callable[[], bool] | None = None) -> bool:
        wait_time = _duration(duration)
        if interrupt_check is None:
            precise_sleep(wait_time)
            return True
        return wait_event(_InterruptAdapter(interrupt_check), wait_time)

    def _click_screen(self, screen_x: int, screen_y: int, down_duration: Any = None, up_duration: Any = None, interrupt_check: Callable[[], bool] | None = None) -> bool:
        if not self._press_left(screen_x, screen_y, down_duration, interrupt_check):
            return False
        if not self._release_left(screen_x, screen_y, up_duration, interrupt_check):
            return False
        return True

    def move_to(self, x: Any, y: Any, relative: bool = True) -> bool:
        with self._input_lock:
            screen_pos = self._screen_position(x, y, relative=relative)
            if screen_pos is None:
                return False
            if not self._set_cursor(*screen_pos):
                return False
            precise_sleep(self.move_delay)
            return True

    def click(self, x: Any, y: Any, relative: bool = True, delay: Any = None) -> bool:
        return self._click(x, y, relative, delay, precise=False)

    def precise_click(self, x: Any, y: Any, relative: bool = True, delay: Any = None) -> bool:
        return self._click(x, y, relative, delay, precise=True)

    def _click(self, x: Any, y: Any, relative: bool, delay: Any, precise: bool) -> bool:
        with self._input_lock:
            screen_pos = self._screen_position(x, y, relative=relative)
            if screen_pos is None:
                return False
            screen_x, screen_y = screen_pos
            if not self._set_cursor(screen_x, screen_y, verify=precise):
                return False
            precise_sleep(self.move_delay)
            if self.hover_enabled:
                precise_sleep(self.hover_duration)
            if not self._click_screen(screen_x, screen_y):
                return False
            precise_sleep(self.click_delay if delay is None else _duration(delay, self.click_delay))
            return True

    def hold_at(self, x: Any, y: Any, duration: Any = None, relative: bool = True, interrupt_check: Callable[[], bool] | None = None) -> bool:
        with self._input_lock:
            screen_pos = self._screen_position(x, y, relative=relative)
            if screen_pos is None:
                return False
            screen_x, screen_y = screen_pos
            if not self._set_cursor(screen_x, screen_y):
                return False
            precise_sleep(self.move_delay)
            if not self._press_left(screen_x, screen_y, interrupt_check=interrupt_check):
                return False
            if not self._interruptible_delay(_duration(4.0 if duration is None else duration, 4.0), interrupt_check):
                self._release_left(screen_x, screen_y)
                return False
            if not self._release_left(screen_x, screen_y, interrupt_check=interrupt_check):
                return False
            precise_sleep(self.click_delay)
            return True

    def click_stats_upgrade_at(self, x: Any, y: Any, duration: Any, click_delay: Any, relative: bool = True, interrupt_check: Callable[[], bool] | None = None) -> bool:
        duration = _duration(duration)
        click_delay = _duration(click_delay)
        if duration <= 0 or click_delay <= 0:
            logger.warning("Rejected stats upgrade click loop: duration=%.3f delay=%.3f", duration, click_delay)
            return False
        with self._input_lock:
            screen_pos = self._screen_position(x, y, relative=relative)
            if screen_pos is None or not self._set_cursor(*screen_pos):
                return False
            precise_sleep(self.move_delay)
            click_count = 0
            start = time.perf_counter()
            limit = min(MAX_STATS_UPGRADE_CLICKS, max(1, int(math.ceil(duration / click_delay))))
            for index in range(limit):
                if interrupt_check and interrupt_check():
                    return False
                target_time = start + index * click_delay
                if target_time >= start + duration:
                    break
                if not sleep_until(target_time):
                    return False
                if not self._click_screen(screen_pos[0], screen_pos[1], down_duration=0.0, up_duration=0.0, interrupt_check=interrupt_check):
                    return False
                click_count += 1
            logger.debug("Stats upgrade clicking complete: %s clicks", click_count)
            return click_count > 0

    def drag(self, from_x: Any, from_y: Any, to_x: Any, to_y: Any, duration: Any = None, relative: bool = True) -> bool:
        with self._input_lock:
            start_pos = self._screen_position(from_x, from_y, relative=relative)
            end_pos = self._screen_position(to_x, to_y, relative=relative)
            if start_pos is None or end_pos is None:
                return False
            duration = max(0.01, _duration(config.SCROLL_DURATION if duration is None else duration, config.SCROLL_DURATION))
            steps = min(MAX_DRAG_STEPS, 20)
            path = self._drag_path(start_pos, end_pos, steps)
            if path is None or not self._set_cursor(*start_pos):
                return False
            precise_sleep(self.move_delay)
            if not self._press_left(*start_pos):
                return False
            start_time = time.perf_counter()
            for index, (screen_x, screen_y) in enumerate(path):
                if not self._set_cursor(screen_x, screen_y, verify=index == steps):
                    self._release_left(screen_x, screen_y)
                    return False
                sleep_until(start_time + ((index + 1) * duration / steps))
            if not self._release_left(*end_pos):
                return False
            precise_sleep(self.click_delay)
            return True

    def _drag_path(self, start_pos: Point, end_pos: Point, steps: int) -> list[Point] | None:
        path = []
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        for index in range(steps + 1):
            ratio = index / steps
            x = int(start_x + (end_x - start_x) * ratio)
            y = int(start_y + (end_y - start_y) * ratio)
            if self.is_in_forbidden_zone(x, y, relative=False):
                return None
            path.append((x, y))
        return path


class _InterruptAdapter:
    def __init__(self, interrupt_check: Callable[[], bool]) -> None:
        self._interrupt_check = interrupt_check

    def is_set(self) -> bool:
        return bool(self._interrupt_check())

    def wait(self, timeout: float) -> bool:
        end_time = time.perf_counter() + max(0.0, timeout)
        for _ in range(_sleep_iterations(end_time)):
            if self.is_set():
                return True
            remaining = end_time - time.perf_counter()
            if remaining <= 0:
                return self.is_set()
            time.sleep(min(remaining, 0.01))
        return self.is_set()
