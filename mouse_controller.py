import logging
import math
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from pynput import mouse as pynput_mouse

import config
from forbidden_zones import (
    ForbiddenZone,
    configured_forbidden_zones,
    first_forbidden_zone_containing_point,
    point_inside_forbidden_zone,
)

logger = logging.getLogger(__name__)

Point = tuple[int, int]
WindowBounds = tuple[int, int, int, int]
MIN_SLEEP_SLICE = 0.001
SLEEP_POLL_INTERVAL_SECONDS = 0.05
INTERRUPT_WAIT_POLL_INTERVAL_SECONDS = 0.01
MAX_SLEEP_ITERATIONS = 120_000
CURSOR_ATTEMPTS = 2
CURSOR_RETRY_DELAY = 0.08
MAX_DRAG_STEPS = 60
MAX_STATS_UPGRADE_CLICKS = 500
STATS_UPGRADE_MOUSE_DOWN_DURATION = 0.0
STATS_UPGRADE_MOUSE_UP_DURATION = 0.0
MINIMUM_PHYSICAL_INPUT_EVENT_INTERVAL_SECONDS = config.SIXTY_FPS_FRAME_DURATION_SECONDS


class StopEventLike(Protocol):
    def is_set(self) -> bool: ...

    def wait(self, timeout: float) -> bool: ...


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


def sleep_until(deadline: float, stop_event: StopEventLike | None = None) -> bool:
    for _ in range(_sleep_iterations(deadline)):
        if stop_event is not None and stop_event.is_set():
            return False
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return stop_event is None or not stop_event.is_set()
        if stop_event is None:
            time.sleep(min(remaining, SLEEP_POLL_INTERVAL_SECONDS))
        elif stop_event.wait(min(remaining, SLEEP_POLL_INTERVAL_SECONDS)):
            return False
    return stop_event is None or not stop_event.is_set()


def wait_event(stop_event: StopEventLike | None, duration: Any) -> bool:
    duration = _duration(duration)
    if duration <= 0:
        return stop_event is None or not stop_event.is_set()
    return sleep_until(time.perf_counter() + duration, stop_event)


def _sleep_iterations(deadline: float) -> int:
    remaining = max(0.0, deadline - time.perf_counter())
    return min(MAX_SLEEP_ITERATIONS, max(1, math.ceil(remaining / MIN_SLEEP_SLICE) + 3))


def _as_bounds(bounds: Any) -> WindowBounds:
    if isinstance(bounds, (str, bytes, bytearray)) or not isinstance(bounds, Sequence):
        raise TypeError(
            f"expected a 4-item window bounds sequence, got {type(bounds).__name__}"
        )
    if len(bounds) != 4:
        raise ValueError(f"expected 4 window bounds values, got {len(bounds)}")
    x, y, width, height = (
        int(bounds[0]),
        int(bounds[1]),
        int(bounds[2]),
        int(bounds[3]),
    )
    if width <= 0 or height <= 0:
        raise ValueError(f"Target window has invalid client size: {width}x{height}")
    return x, y, width, height


def _inside_forbidden_zone(x: int, y: int, zone: ForbiddenZone) -> bool:
    return point_inside_forbidden_zone(x, y, zone)


class MouseController:
    def __init__(
        self,
        window_bounds_source: Any,
        click_delay: Any = None,
        move_delay: Any = None,
        hover_enabled: bool | None = None,
        hover_duration: Any = None,
        mouse_device: Any = None,
        stop_event: StopEventLike | None = None,
    ) -> None:
        self._mouse = (
            mouse_device if mouse_device is not None else pynput_mouse.Controller()
        )
        self._left_button = pynput_mouse.Button.left
        self._window_bounds_source = window_bounds_source
        self._stop_event = stop_event
        self.click_delay = _duration(
            config.CLICK_DELAY if click_delay is None else click_delay, 0.08
        )
        self.move_delay = _duration(
            config.MOUSE_MOVE_DELAY if move_delay is None else move_delay, 0.025
        )
        self.hover_enabled = bool(
            config.HOVER_ENABLED if hover_enabled is None else hover_enabled
        )
        self.hover_duration = _duration(
            config.HOVER_DURATION if hover_duration is None else hover_duration
        )
        self._input_lock = threading.RLock()
        self._physical_input_event_governor = _PhysicalInputEventGovernor()
        self._left_button_is_down = False
        self._forbidden_zones = self._configured_forbidden_zones()

    @staticmethod
    def _configured_forbidden_zones() -> list[ForbiddenZone]:
        return list(configured_forbidden_zones())

    def get_cursor_position(self) -> Point:
        x, y = self._mouse.position
        return int(x), int(y)

    def get_window_bounds(self) -> WindowBounds:
        try:
            source = (
                self._window_bounds_source()
                if callable(self._window_bounds_source)
                else self._window_bounds_source
            )
            return _as_bounds(source)
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc

    def get_window_position(self) -> Point:
        x, y, _, _ = self.get_window_bounds()
        return x, y

    def _stopped(self) -> bool:
        return self._stop_event is not None and self._stop_event.is_set()

    def _relative_position(
        self,
        x: Any,
        y: Any,
        relative: bool,
        bounds: WindowBounds | None = None,
    ) -> tuple[int, int, int, int, int, int] | None:
        try:
            window_x, window_y, width, height = (
                self.get_window_bounds() if bounds is None else bounds
            )
            if relative:
                rel_x, rel_y = int(x), int(y)
            else:
                rel_x, rel_y = int(x) - window_x, int(y) - window_y
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.error("Cannot resolve input position: %s", exc)
            return None
        if not (0 <= rel_x < width and 0 <= rel_y < height):
            logger.warning(
                "Rejected input outside target window: relative=(%s, %s), bounds=%sx%s",
                rel_x,
                rel_y,
                width,
                height,
            )
            return None
        return rel_x, rel_y, window_x, window_y, width, height

    def _screen_position(
        self,
        x: Any,
        y: Any,
        relative: bool = True,
        check_forbidden: bool = True,
        bounds: WindowBounds | None = None,
    ) -> Point | None:
        if self._stopped():
            return None
        position = self._relative_position(x, y, relative, bounds)
        if position is None:
            return None
        rel_x, rel_y, window_x, window_y, _, _ = position
        if check_forbidden:
            zone = first_forbidden_zone_containing_point(
                rel_x, rel_y, self._forbidden_zones
            )
            if zone is not None:
                logger.debug(
                    "Coordinates (%s, %s) blocked by %s", rel_x, rel_y, zone.name
                )
                return None
        return window_x + rel_x, window_y + rel_y

    def _resolve_screen_position(
        self, x: Any, y: Any, relative: bool = True, check_forbidden: bool = True
    ) -> Point | None:
        return self._screen_position(
            x, y, relative=relative, check_forbidden=check_forbidden
        )

    def is_in_forbidden_zone(self, x: Any, y: Any, relative: bool = True) -> bool:
        position = self._relative_position(x, y, relative)
        if position is None:
            return True
        rel_x, rel_y, _, _, _, _ = position
        zone = first_forbidden_zone_containing_point(
            rel_x, rel_y, self._forbidden_zones
        )
        if zone is None:
            return False
        logger.debug("Coordinates (%s, %s) blocked by %s", rel_x, rel_y, zone.name)
        return True

    def _set_cursor(self, screen_x: int, screen_y: int, verify: bool = True) -> bool:
        for attempt in range(1, CURSOR_ATTEMPTS + 1):
            if self._stopped():
                return False
            try:
                self._physical_input_event_governor.wait_for_next_dispatch()
                self._mouse.position = (int(screen_x), int(screen_y))
                if not verify:
                    return True
                precise_sleep(0.001)
                current_x, current_y = self.get_cursor_position()
                if abs(current_x - screen_x) <= 1 and abs(current_y - screen_y) <= 1:
                    return True
            except Exception as exc:
                logger.warning(
                    "Cursor positioning failed on attempt %s/%s: %s",
                    attempt,
                    CURSOR_ATTEMPTS,
                    exc,
                )
            if attempt < CURSOR_ATTEMPTS:
                precise_sleep(CURSOR_RETRY_DELAY)
        logger.error("Cursor positioning failed at (%s, %s)", screen_x, screen_y)
        return False

    def _cursor_matches_safe_position(self, screen_x: int, screen_y: int) -> bool:
        current_x, current_y = self.get_cursor_position()
        safe_position = self._screen_position(current_x, current_y, relative=False)
        if safe_position == (screen_x, screen_y):
            return True
        logger.warning(
            "Rejected left down: expected cursor=(%s, %s), actual=(%s, %s)",
            screen_x,
            screen_y,
            current_x,
            current_y,
        )
        return False

    def _button(
        self,
        action_name: str,
        action: Callable[[], Any],
        x: int,
        y: int,
        require_safe_cursor: bool = False,
    ) -> bool | None:
        try:
            self._physical_input_event_governor.wait_for_next_dispatch()
            safe_cursor = not require_safe_cursor or self._cursor_matches_safe_position(
                x, y
            )
        except Exception as exc:
            logger.error(
                "%s pre-dispatch validation failed at (%s, %s): %s",
                action_name,
                x,
                y,
                exc,
            )
            return None
        if not safe_cursor:
            return None
        try:
            action()
            return True
        except Exception as exc:
            logger.error("%s failed at (%s, %s): %s", action_name, x, y, exc)
            return False

    def _dispatch_left_down(self) -> None:
        self._left_button_is_down = True
        self._mouse.press(self._left_button)

    def _press_left(
        self,
        x: int,
        y: int,
        duration: Any = None,
        interrupt_check: Callable[[], bool] | None = None,
    ) -> bool:
        if self._stopped():
            return False
        press_completed = self._button(
            "left down",
            self._dispatch_left_down,
            x,
            y,
            require_safe_cursor=True,
        )
        if press_completed is None:
            return False
        if not press_completed:
            self._release_after_failed_sequence(x, y)
            return False
        wait_time = _duration(
            config.MOUSE_DOWN_DURATION if duration is None else duration,
            config.MOUSE_DOWN_DURATION,
        )
        if wait_time > 0 and not self._interruptible_delay(wait_time, interrupt_check):
            self._release_after_failed_sequence(x, y)
            return False
        return True

    def _release_left(
        self,
        x: int,
        y: int,
        duration: Any = None,
        interrupt_check: Callable[[], bool] | None = None,
    ) -> bool:
        if not self._button(
            "left up", lambda: self._mouse.release(self._left_button), x, y
        ):
            return False
        self._left_button_is_down = False
        wait_time = _duration(
            config.MOUSE_UP_DURATION if duration is None else duration,
            config.MOUSE_UP_DURATION,
        )
        return self._interruptible_delay(wait_time, interrupt_check)

    def _release_after_failed_sequence(self, x: int, y: int) -> None:
        if not self._release_left(x, y):
            logger.error(
                "Could not release left button after interrupted sequence at (%s, %s)",
                x,
                y,
            )

    def release_left_button(self) -> bool:
        with self._input_lock:
            if not self._left_button_is_down:
                return True
            try:
                screen_x, screen_y = self.get_cursor_position()
            except Exception:
                screen_x, screen_y = 0, 0
            return self._release_left(screen_x, screen_y, duration=0.0)

    @staticmethod
    def _interruptible_delay(
        duration: Any, interrupt_check: Callable[[], bool] | None = None
    ) -> bool:
        wait_time = _duration(duration)
        if interrupt_check is None:
            precise_sleep(wait_time)
            return True
        return wait_event(_InterruptAdapter(interrupt_check), wait_time)

    def _click_screen(
        self,
        screen_x: int,
        screen_y: int,
        down_duration: Any = None,
        up_duration: Any = None,
        interrupt_check: Callable[[], bool] | None = None,
    ) -> bool:
        if not self._press_left(screen_x, screen_y, down_duration, interrupt_check):
            return False
        release_completed = False
        try:
            release_completed = self._release_left(
                screen_x, screen_y, up_duration, interrupt_check
            )
            return release_completed
        finally:
            if not release_completed and self._left_button_is_down:
                self._release_after_failed_sequence(screen_x, screen_y)

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

    def precise_click(
        self, x: Any, y: Any, relative: bool = True, delay: Any = None
    ) -> bool:
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
            precise_sleep(
                self.click_delay
                if delay is None
                else _duration(delay, self.click_delay)
            )
            return True

    def hold_at(
        self,
        x: Any,
        y: Any,
        duration: Any = None,
        relative: bool = True,
        interrupt_check: Callable[[], bool] | None = None,
    ) -> bool:
        with self._input_lock:
            screen_pos = self._screen_position(x, y, relative=relative)
            if screen_pos is None:
                return False
            screen_x, screen_y = screen_pos
            if not self._set_cursor(screen_x, screen_y):
                return False
            precise_sleep(self.move_delay)
            if not self._press_left(
                screen_x, screen_y, duration=0.0, interrupt_check=interrupt_check
            ):
                return False
            release_completed = False
            try:
                if not self._interruptible_delay(
                    _duration(4.0 if duration is None else duration, 4.0),
                    interrupt_check,
                ):
                    self._release_after_failed_sequence(screen_x, screen_y)
                    return False
                release_completed = self._release_left(
                    screen_x, screen_y, interrupt_check=interrupt_check
                )
                if not release_completed:
                    return False
                precise_sleep(self.click_delay)
                return True
            finally:
                if not release_completed and self._left_button_is_down:
                    self._release_after_failed_sequence(screen_x, screen_y)

    def click_stats_upgrade_at(
        self,
        x: Any,
        y: Any,
        duration: Any,
        click_delay: Any,
        relative: bool = True,
        interrupt_check: Callable[[], bool] | None = None,
    ) -> bool:
        duration = _duration(duration)
        click_delay = _duration(click_delay)
        if duration <= 0 or click_delay <= 0:
            logger.warning(
                "Rejected stats upgrade click loop: duration=%.3f delay=%.3f",
                duration,
                click_delay,
            )
            return False
        with self._input_lock:
            screen_pos = self._screen_position(x, y, relative=relative)
            if screen_pos is None or not self._set_cursor(*screen_pos):
                return False
            precise_sleep(self.move_delay)
            click_count = self._stats_upgrade_click_loop(
                screen_pos, duration, click_delay, interrupt_check
            )
            if click_count is None:
                return False
            logger.debug("Stats upgrade clicking complete: %s clicks", click_count)
            return click_count > 0

    def _stats_upgrade_click_loop(
        self,
        screen_pos: Point,
        duration: float,
        click_delay: float,
        interrupt_check: Callable[[], bool] | None,
    ) -> int | None:
        click_count = 0
        start = time.perf_counter()
        limit = min(MAX_STATS_UPGRADE_CLICKS, max(1, math.ceil(duration / click_delay)))
        for index in range(limit):
            if interrupt_check and interrupt_check():
                return None
            target_time = start + index * click_delay
            if (
                target_time >= start + duration
                or time.perf_counter() >= start + duration
            ):
                break
            if not sleep_until(target_time):
                return None
            if not self._click_screen(
                screen_pos[0],
                screen_pos[1],
                down_duration=STATS_UPGRADE_MOUSE_DOWN_DURATION,
                up_duration=STATS_UPGRADE_MOUSE_UP_DURATION,
                interrupt_check=interrupt_check,
            ):
                return None
            click_count += 1
        return click_count

    def drag(
        self,
        from_x: Any,
        from_y: Any,
        to_x: Any,
        to_y: Any,
        duration: Any = None,
        relative: bool = True,
    ) -> bool:
        with self._input_lock:
            try:
                bounds = self.get_window_bounds()
            except RuntimeError as exc:
                logger.error("Cannot resolve drag bounds: %s", exc)
                return False
            start_pos = self._screen_position(
                from_x, from_y, relative=relative, bounds=bounds
            )
            end_pos = self._screen_position(
                to_x, to_y, relative=relative, bounds=bounds
            )
            if start_pos is None or end_pos is None:
                return False
            duration = max(
                0.01,
                _duration(
                    config.SCROLL_DURATION if duration is None else duration,
                    config.SCROLL_DURATION,
                ),
            )
            steps = min(MAX_DRAG_STEPS, 20)
            path = self._drag_path(start_pos, end_pos, steps, bounds)
            if path is None or not self._set_cursor(*start_pos):
                return False
            precise_sleep(self.move_delay)
            if not self._press_left(*start_pos):
                return False
            release_completed = False
            try:
                start_time = time.perf_counter()
                if not self._move_along_drag_path(path, duration, steps, start_time):
                    return False
                release_completed = self._release_left(*end_pos)
                if not release_completed:
                    return False
                precise_sleep(self.click_delay)
                return True
            finally:
                if not release_completed and self._left_button_is_down:
                    self._release_after_failed_sequence(*end_pos)

    def _move_along_drag_path(
        self, path: list[Point], duration: float, steps: int, start_time: float
    ) -> bool:
        for index, (screen_x, screen_y) in enumerate(path):
            if self._stopped():
                return False
            if not self._set_cursor(screen_x, screen_y, verify=index == steps):
                return False
            if not sleep_until(
                start_time + ((index + 1) * duration / steps), self._stop_event
            ):
                return False
        return True

    def _drag_path(
        self,
        start_pos: Point,
        end_pos: Point,
        steps: int,
        bounds: WindowBounds,
    ) -> list[Point] | None:
        path = []
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        for index in range(steps + 1):
            ratio = index / steps
            x = int(start_x + (end_x - start_x) * ratio)
            y = int(start_y + (end_y - start_y) * ratio)
            position = self._relative_position(x, y, relative=False, bounds=bounds)
            if position is None:
                return None
            rel_x, rel_y, _, _, _, _ = position
            if (
                first_forbidden_zone_containing_point(
                    rel_x, rel_y, self._forbidden_zones
                )
                is not None
            ):
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
            time.sleep(min(remaining, INTERRUPT_WAIT_POLL_INTERVAL_SECONDS))
        return self.is_set()


class _PhysicalInputEventGovernor:
    def __init__(self) -> None:
        self._minimum_interval_seconds = MINIMUM_PHYSICAL_INPUT_EVENT_INTERVAL_SECONDS
        self._next_dispatch_time = 0.0
        self._dispatch_lock = threading.Lock()

    def wait_for_next_dispatch(self) -> None:
        with self._dispatch_lock:
            remaining = self._next_dispatch_time - time.perf_counter()
            if remaining > 0:
                precise_sleep(remaining)
            dispatch_time = time.perf_counter()
            self._next_dispatch_time = dispatch_time + self._minimum_interval_seconds

    def get_minimum_interval_seconds(self) -> float:
        return self._minimum_interval_seconds
