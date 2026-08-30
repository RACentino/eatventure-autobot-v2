import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any

import config

logger = logging.getLogger(__name__)

Point = tuple[int, int]
Bounds = tuple[int, int, int, int]
Zone = tuple[int, int, int, int]


def _duration(value: Any, default: float = 0.0) -> float:
    try:
        default = float(default)
    except (TypeError, ValueError):
        default = 0.0
    if not math.isfinite(default):
        default = 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return max(0.0, value if math.isfinite(value) else default)


class MouseController:
    def __init__(
        self,
        window_capture: Any,
        click_delay: Any = None,
        move_delay: Any = None,
        stop_event: threading.Event | None = None,
        controller: Any = None,
    ) -> None:
        if controller is None:
            from pynput import mouse

            controller = mouse.Controller()
            self._left_button = mouse.Button.left
        else:
            self._left_button = getattr(controller, "left_button", "left")
        self._window = window_capture
        self._mouse = controller
        self._stop_event = stop_event
        self.click_delay = _duration(
            config.CLICK_DELAY if click_delay is None else click_delay,
            config.CLICK_DELAY,
        )
        self.move_delay = _duration(
            config.MOUSE_MOVE_DELAY if move_delay is None else move_delay,
            config.MOUSE_MOVE_DELAY,
        )
        self._input_lock = threading.RLock()
        self._left_pressed = False
        self._event_zone: Zone | None = None

    def _wait(self, duration: Any) -> bool:
        delay = _duration(duration)
        if self._stop_event is None:
            time.sleep(delay)
            return True
        return not self._stop_event.wait(delay)

    def is_target_foreground(self) -> bool:
        try:
            return bool(self._window.is_window_active())
        except Exception:
            return False

    def get_window_bounds(self) -> Bounds:
        try:
            bounds = tuple(map(int, self._window.get_input_window_rect()))
            if len(bounds) != 4 or bounds[2] <= 0 or bounds[3] <= 0:
                raise ValueError(f"invalid bounds: {bounds!r}")
            return bounds
        except Exception as exc:
            raise RuntimeError(f"Cannot read active target bounds: {exc}") from exc

    def get_cursor_position(self) -> Point:
        x, y = self._mouse.position
        return int(x), int(y)

    @staticmethod
    def _inside(x: int, y: int, zone: Zone) -> bool:
        x_min, x_max, y_min, y_max = zone
        return x_min <= x <= x_max and y_min <= y <= y_max

    def _zones(self) -> tuple[Zone, ...]:
        configured = tuple(config.NUMBERED_FORBIDDEN_ZONE_BOUNDS) + (
            (
                config.FORBIDDEN_CLICK_X_MIN,
                config.FORBIDDEN_CLICK_X_MAX,
                config.FORBIDDEN_CLICK_Y_MIN,
                config.WINDOW_HEIGHT - 1,
            ),
        )
        return ((self._event_zone,) + configured) if self._event_zone else configured

    def set_event_forbidden_zone(self, bounds: Zone) -> None:
        if len(bounds) != 4:
            raise ValueError("Event forbidden zone must contain four coordinates")
        x_min, x_max, y_min, y_max = map(int, bounds)
        zone: Zone = (x_min, x_max, y_min, y_max)
        if zone[0] > zone[1] or zone[2] > zone[3]:
            raise ValueError("Event forbidden zone has reversed bounds")
        self._event_zone = zone

    def is_in_forbidden_zone(self, x: Any, y: Any, relative: bool = True) -> bool:
        try:
            x, y = int(x), int(y)
            if not relative:
                left, top, _, _ = self.get_window_bounds()
                x, y = x - left, y - top
            return any(self._inside(x, y, zone) for zone in self._zones())
        except (TypeError, ValueError, RuntimeError):
            return True

    def _resolve_screen_position(
        self,
        x: Any,
        y: Any,
        relative: bool = True,
        check_forbidden: bool = True,
    ) -> Point | None:
        try:
            left, top, width, height = self.get_window_bounds()
            x, y = int(x), int(y)
            rel_x, rel_y = (x, y) if relative else (x - left, y - top)
            if not (0 <= rel_x < width and 0 <= rel_y < height):
                logger.warning(
                    "Rejected input outside target window: (%s, %s)", rel_x, rel_y
                )
                return None
            if check_forbidden and self.is_in_forbidden_zone(rel_x, rel_y):
                return None
            return left + rel_x, top + rel_y
        except (TypeError, ValueError, RuntimeError) as exc:
            logger.error("Cannot resolve mouse position: %s", exc)
            return None

    def _input_allowed(self) -> bool:
        return (
            not (self._stop_event and self._stop_event.is_set())
            and self.is_target_foreground()
        )

    def _set_cursor_pos(self, x: Any, y: Any) -> bool:
        screen_x, screen_y = int(x), int(y)
        if self._resolve_screen_position(screen_x, screen_y, relative=False) is None:
            return False
        last_error: Exception | None = None
        for attempt in range(max(1, int(config.INPUT_RETRY_COUNT))):
            if not self._input_allowed():
                return False
            try:
                self._mouse.position = (screen_x, screen_y)
                if not self._wait(0.001):
                    return False
                if self.get_cursor_position() == (screen_x, screen_y):
                    return True
            except Exception as exc:
                last_error = exc
                logger.warning("Cursor move failed: %s", exc)
            if attempt + 1 < int(config.INPUT_RETRY_COUNT) and not self._wait(
                config.INPUT_RETRY_DELAY
            ):
                return False
        raise RuntimeError(
            f"Cursor move failed after {max(1, int(config.INPUT_RETRY_COUNT))} attempts"
        ) from last_error

    def _left_down_at_screen(self, x: Any, y: Any, duration: Any = None) -> bool:
        x, y = int(x), int(y)
        if not self._input_allowed():
            return False
        if self.get_cursor_position() != (x, y):
            raise RuntimeError("Cursor moved before mouse press")
        try:
            self._mouse.press(self._left_button)
            self._left_pressed = True
        except Exception as exc:
            self._best_effort_left_up(x, y)
            raise RuntimeError("Mouse press failed") from exc
        wait = config.MOUSE_DOWN_DURATION if duration is None else duration
        if _duration(wait) and not self._wait(wait):
            self._best_effort_left_up(x, y)
            return False
        return True

    def _left_up_at_screen(self, x: Any, y: Any, duration: Any = None) -> bool:
        try:
            self._mouse.release(self._left_button)
            self._left_pressed = False
        except Exception as exc:
            raise RuntimeError("Mouse release failed") from exc
        wait = config.MOUSE_UP_DURATION if duration is None else duration
        return not _duration(wait) or self._wait(wait)

    def _best_effort_left_up(self, _x: Any = 0, _y: Any = 0) -> bool:
        try:
            self._mouse.release(self._left_button)
            self._left_pressed = False
            return True
        except Exception:
            return False

    def release_left_button(self) -> bool:
        with self._input_lock:
            return not self._left_pressed or self._best_effort_left_up()

    def _left_click_at_screen(
        self, x: int, y: int, down_duration: Any = None, up_duration: Any = None
    ) -> bool:
        if not self._left_down_at_screen(x, y, down_duration):
            return False
        released = False
        try:
            if self.get_cursor_position() != (x, y):
                logger.warning(
                    "Mouse moved during click; releasing without crediting success"
                )
                return False
            released = self._left_up_at_screen(x, y, up_duration)
            return released
        finally:
            if not released:
                self._best_effort_left_up(x, y)

    def click(self, x: Any, y: Any, relative: bool = True, delay: Any = None) -> bool:
        with self._input_lock:
            position = self._resolve_screen_position(x, y, relative)
            if position is None or not self._set_cursor_pos(*position):
                return False
            if not self._wait(self.move_delay) or not self._left_click_at_screen(
                *position
            ):
                return False
            return self._wait(self.click_delay if delay is None else delay)

    precise_click = click

    def spam_click_at(
        self,
        x: Any,
        y: Any,
        duration: Any,
        click_delay: Any,
        relative: bool = True,
        mouse_down_duration: Any = None,
        mouse_up_duration: Any = None,
    ) -> bool:
        with self._input_lock:
            position = self._resolve_screen_position(x, y, relative)
            if (
                position is None
                or not self._set_cursor_pos(*position)
                or not self._wait(self.move_delay)
            ):
                return False
            end = time.monotonic() + _duration(duration)
            clicks = 0
            while time.monotonic() < end:
                if not self._left_click_at_screen(
                    *position, mouse_down_duration, mouse_up_duration
                ):
                    return False
                clicks += 1
                if not self._wait(click_delay):
                    return False
            return clicks > 0 and self._wait(self.click_delay)

    def hold_at(
        self,
        x: Any,
        y: Any,
        duration: Any,
        check_interval: Any,
        interrupt_check: Callable[[], bool],
        relative: bool = True,
    ) -> bool:
        with self._input_lock:
            position = self._resolve_screen_position(x, y, relative)
            if position is None or not self._set_cursor_pos(*position):
                return False
            if not self._wait(self.move_delay) or not self._left_down_at_screen(
                *position, duration=0
            ):
                return False
            released = False
            try:
                deadline = time.monotonic() + _duration(duration)
                interval = max(0.001, _duration(check_interval, 0.1))
                while time.monotonic() < deadline:
                    if not self._wait(min(interval, deadline - time.monotonic())):
                        return False
                    if self.get_cursor_position() != position:
                        logger.warning("Mouse moved during hold; releasing")
                        return False
                    if not self.is_target_foreground():
                        return False
                    if interrupt_check():
                        break
                released = self._left_up_at_screen(*position)
                return released
            finally:
                if not released:
                    self._best_effort_left_up(*position)

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
            start = self._resolve_screen_position(from_x, from_y, relative)
            end = self._resolve_screen_position(to_x, to_y, relative)
            if start is None or end is None or not self._set_cursor_pos(*start):
                return False
            if not self._wait(self.move_delay) or not self._left_down_at_screen(*start):
                return False
            released = False
            try:
                started = time.monotonic()
                for step in range(1, 21):
                    ratio = step / 20
                    point = (
                        round(start[0] + (end[0] - start[0]) * ratio),
                        round(start[1] + (end[1] - start[1]) * ratio),
                    )
                    if not self._set_cursor_pos(*point):
                        return False
                    if not self._wait(
                        max(
                            0.0,
                            started
                            + _duration(duration, config.SCROLL_DURATION) * ratio
                            - time.monotonic(),
                        )
                    ):
                        return False
                released = self._left_up_at_screen(*end)
                return released and self._wait(self.click_delay)
            finally:
                if not released:
                    self._best_effort_left_up(*end)
