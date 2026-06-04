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
RelativeScreenPosition = tuple[int, int, int, int, int, int]
ForbiddenZone = tuple[str, int, int, int, int | None]
MIN_DEADLINE_POLL_SLICE = 0.001
MAX_DEADLINE_POLL_ITERATIONS = 120_000
CURSOR_POSITION_ATTEMPTS = 2
CURSOR_POSITION_RETRY_DELAY = 0.08
MAX_DRAG_STEPS = 60
MAX_STATS_UPGRADE_CLICKS = 500


def _point_within_client_area(
    relative_x_coordinate: int,
    relative_y_coordinate: int,
    client_width: int,
    client_height: int,
) -> bool:
    return (
        0 <= relative_x_coordinate < client_width
        and 0 <= relative_y_coordinate < client_height
    )


def _point_in_forbidden_zone(
    relative_x_coordinate: int,
    relative_y_coordinate: int,
    forbidden_zone: ForbiddenZone,
) -> bool:
    _, x_minimum, x_maximum, y_minimum, y_maximum = forbidden_zone
    if y_maximum is None:
        return (
            relative_y_coordinate >= y_minimum
            and x_minimum <= relative_x_coordinate <= x_maximum
        )
    return (
        y_minimum <= relative_y_coordinate <= y_maximum
        and x_minimum <= relative_x_coordinate <= x_maximum
    )


def _matching_forbidden_zone_name(
    relative_x_coordinate: int,
    relative_y_coordinate: int,
    forbidden_zones: list[ForbiddenZone],
) -> str | None:
    for forbidden_zone in forbidden_zones:
        if _point_in_forbidden_zone(
            relative_x_coordinate,
            relative_y_coordinate,
            forbidden_zone,
        ):
            return forbidden_zone[0]
    return None


def _screen_to_relative_position(
    screen_x_coordinate: int,
    screen_y_coordinate: int,
    window_bounds: WindowBounds,
) -> RelativeScreenPosition:
    (
        window_x_coordinate,
        window_y_coordinate,
        client_width,
        client_height,
    ) = window_bounds
    return (
        screen_x_coordinate - window_x_coordinate,
        screen_y_coordinate - window_y_coordinate,
        window_x_coordinate,
        window_y_coordinate,
        client_width,
        client_height,
    )


def _coerce_duration(duration: Any, default: float = 0.0) -> float:
    try:
        value = float(duration)
    except (TypeError, ValueError):
        value = float(default)
    if not math.isfinite(value):
        value = float(default)
    return max(0.0, value)


def precise_sleep(duration: Any) -> None:
    duration = _coerce_duration(duration)
    if duration <= 0:
        return
    sleep_until(time.perf_counter() + duration)


def _wait_until_next_deadline_slice(remaining: float, stop_event: threading.Event | None) -> bool:
    if remaining > 0.004:
        wait_time = min(remaining - 0.002, 0.05)
        if stop_event is None:
            time.sleep(wait_time)
            return True
        return not stop_event.wait(wait_time)
    if remaining > 0.001:
        time.sleep(0)
    return True


def _deadline_poll_iterations(deadline: float, poll_slice: float = MIN_DEADLINE_POLL_SLICE) -> int:
    remaining = max(0.0, float(deadline) - time.perf_counter())
    bounded_slice = max(MIN_DEADLINE_POLL_SLICE, float(poll_slice))
    return min(MAX_DEADLINE_POLL_ITERATIONS, max(1, int(math.ceil(remaining / bounded_slice)) + 3))


def sleep_until(deadline: float, stop_event: threading.Event | None = None) -> bool:
    for _ in range(_deadline_poll_iterations(deadline)):
        if stop_event is not None and stop_event.is_set():
            return False
        remaining = float(deadline) - time.perf_counter()
        if remaining <= 0:
            return stop_event is None or not stop_event.is_set()
        if not _wait_until_next_deadline_slice(remaining, stop_event):
            return False
    return stop_event is None or not stop_event.is_set()


def wait_event(stop_event: threading.Event | None, duration: Any) -> bool:
    duration = _coerce_duration(duration)
    if stop_event is None:
        precise_sleep(duration)
        return True
    if duration <= 0:
        return not stop_event.is_set()
    return sleep_until(time.perf_counter() + duration, stop_event)


def _window_bounds_sequence(bounds: Any) -> Sequence[Any]:
    if isinstance(bounds, (str, bytes, bytearray)):
        raise TypeError(f"expected a 4-item window bounds sequence, got {type(bounds).__name__}")
    if not isinstance(bounds, Sequence):
        raise TypeError(f"expected a 4-item window bounds sequence, got {type(bounds).__name__}")
    if len(bounds) != 4:
        raise ValueError(f"expected 4 window bounds values, got {len(bounds)}")
    return bounds


def _coerce_window_bounds(bounds: Any) -> WindowBounds:
    values = _window_bounds_sequence(bounds)
    try:
        window_x_coordinate = int(values[0])
        window_y_coordinate = int(values[1])
        client_width = int(values[2])
        client_height = int(values[3])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"window bounds must contain integer-compatible values: {bounds!r}") from exc
    return window_x_coordinate, window_y_coordinate, client_width, client_height


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
        if mouse_device is None:
            self._mouse = pynput_mouse.Controller()
        else:
            self._mouse = mouse_device
        self._left_button = pynput_mouse.Button.left
        self._window_bounds_source = window_bounds_source
        self.click_delay = self._coerce_non_negative_float(
            config.CLICK_DELAY if click_delay is None else click_delay,
            0.08,
        )
        self.move_delay = self._coerce_non_negative_float(
            config.MOUSE_MOVE_DELAY if move_delay is None else move_delay,
            0.025,
        )
        self.hover_enabled = bool(config.HOVER_ENABLED if hover_enabled is None else hover_enabled)
        self.hover_duration = self._coerce_non_negative_float(
            config.HOVER_DURATION if hover_duration is None else hover_duration,
            0.0,
        )
        self._input_lock = threading.RLock()
        self._forbidden_zones = self._configured_forbidden_zones()
        self._left_button_is_down = False

    @staticmethod
    def _coerce_non_negative_float(value: Any, default: float = 0.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return max(0.0, float(default))
        if not math.isfinite(number):
            return max(0.0, float(default))
        return max(0.0, number)

    def get_cursor_position(self) -> Point:
        current_x, current_y = self._mouse.position
        return int(current_x), int(current_y)

    def get_window_position(self) -> Point:
        win_x, win_y, _, _ = self.get_window_bounds()
        return win_x, win_y

    def _read_window_bounds_source(self) -> Any:
        if callable(self._window_bounds_source):
            return self._window_bounds_source()
        return self._window_bounds_source

    def get_window_bounds(self) -> WindowBounds:
        try:
            win_x, win_y, width, height = _coerce_window_bounds(
                self._read_window_bounds_source()
            )
        except Exception as exc:
            raise RuntimeError(f"Cannot read target window bounds: {exc}") from exc
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Target window has invalid client size: {width}x{height}")
        return win_x, win_y, width, height

    def _relative_from_screen(self, x: Any, y: Any) -> RelativeScreenPosition:
        return _screen_to_relative_position(int(x), int(y), self.get_window_bounds())

    @staticmethod
    def _within_client(x: Any, y: Any, width: Any, height: Any) -> bool:
        return _point_within_client_area(int(x), int(y), int(width), int(height))

    def _target_position_components(
        self,
        x: Any,
        y: Any,
        relative: bool,
    ) -> RelativeScreenPosition:
        if relative:
            (
                window_x_coordinate,
                window_y_coordinate,
                client_width,
                client_height,
            ) = self.get_window_bounds()
            return (
                int(x),
                int(y),
                window_x_coordinate,
                window_y_coordinate,
                client_width,
                client_height,
            )
        return self._relative_from_screen(x, y)

    @staticmethod
    def _position_within_client_bounds(position: RelativeScreenPosition) -> bool:
        relative_x_coordinate, relative_y_coordinate, _, _, client_width, client_height = position
        return _point_within_client_area(
            relative_x_coordinate,
            relative_y_coordinate,
            client_width,
            client_height,
        )

    def _matching_target_forbidden_zone(
        self,
        position: RelativeScreenPosition,
    ) -> str | None:
        relative_x_coordinate, relative_y_coordinate, _, _, _, _ = position
        return _matching_forbidden_zone_name(
            relative_x_coordinate,
            relative_y_coordinate,
            self._forbidden_zones,
        )

    def _cursor_matches_position(self, screen_x: int, screen_y: int) -> tuple[bool, int, int]:
        current_x, current_y = self.get_cursor_position()
        matches = abs(current_x - screen_x) <= 1 and abs(current_y - screen_y) <= 1
        return matches, current_x, current_y

    def _cursor_matches_safe_position(self, screen_x: int, screen_y: int) -> tuple[bool, int, int]:
        matches, current_x, current_y = self._cursor_matches_position(screen_x, screen_y)
        if not matches:
            return False, current_x, current_y
        if self._validated_screen_input_position(current_x, current_y, "settled cursor") is None:
            return False, current_x, current_y
        return True, current_x, current_y

    @staticmethod
    def _sleep_before_cursor_retry(attempt: int) -> None:
        if attempt < CURSOR_POSITION_ATTEMPTS:
            precise_sleep(CURSOR_POSITION_RETRY_DELAY)

    def _set_cursor_pos(self, x: Any, y: Any, verify_position: bool = True) -> bool:
        validated_screen_position = self._validated_screen_input_position(x, y, "cursor move")
        if validated_screen_position is None:
            return False
        screen_x, screen_y = validated_screen_position
        last_exc = None
        for attempt in range(1, CURSOR_POSITION_ATTEMPTS + 1):
            try:
                self._mouse.position = (screen_x, screen_y)
                if not verify_position:
                    return True
                precise_sleep(0.001)
                matches, current_x, current_y = self._cursor_matches_safe_position(screen_x, screen_y)
                if matches:
                    return True
                last_exc = RuntimeError(f"cursor settled at ({current_x}, {current_y})")
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Cursor positioning failed at (%s, %s) on attempt %s/%s: %s",
                    screen_x,
                    screen_y,
                    attempt,
                    CURSOR_POSITION_ATTEMPTS,
                    exc,
                )
            self._sleep_before_cursor_retry(attempt)
        logger.error("Cursor positioning failed at (%s, %s): %s", screen_x, screen_y, last_exc)
        return False

    def _mouse_button_action(self, action_name: str, action: Callable[[], Any], x: Any, y: Any) -> bool:
        screen_x = int(x)
        screen_y = int(y)
        try:
            action()
            return True
        except Exception as exc:
            logger.error("%s failed at (%s, %s): %s", action_name, screen_x, screen_y, exc)
            return False

    def _best_effort_left_up(self, x: Any, y: Any) -> bool:
        if not self._left_button_is_down:
            return True
        try:
            self._mouse.release(self._left_button)
        except Exception:
            return False
        self._left_button_is_down = False
        return True

    @staticmethod
    def _configured_forbidden_zones() -> list[ForbiddenZone]:
        zones: list[ForbiddenZone] = [
            (
                "FORBIDDEN_CLICK zone",
                config.FORBIDDEN_CLICK_X_MIN,
                config.FORBIDDEN_CLICK_X_MAX,
                config.FORBIDDEN_CLICK_Y_MIN,
                None,
            ),
        ]
        for index, (x_min, x_max, y_min, y_max) in enumerate(
            config.NUMBERED_FORBIDDEN_ZONE_BOUNDS, start=1,
        ):
            zones.append((f"FORBIDDEN_ZONE_{index}", x_min, x_max, y_min, y_max))
        return zones

    @staticmethod
    def _position_in_forbidden_zone(x: int, y: int, forbidden_zone: ForbiddenZone) -> bool:
        return _point_in_forbidden_zone(int(x), int(y), forbidden_zone)

    def is_in_forbidden_zone(self, x: Any, y: Any, relative: bool = True) -> bool:
        try:
            if not relative:
                x, y, _, _, _, _ = self._relative_from_screen(x, y)
            relative_x = int(x)
            relative_y = int(y)
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.error("Cannot evaluate forbidden zone: %s", exc)
            return True

        forbidden_zone_name = _matching_forbidden_zone_name(
            relative_x,
            relative_y,
            self._forbidden_zones,
        )
        if forbidden_zone_name is not None:
            logger.debug(
                "Coordinates (%s, %s) blocked - %s",
                relative_x,
                relative_y,
                forbidden_zone_name,
            )
            return True

        return False

    def _validated_target_position(
        self,
        x: Any,
        y: Any,
        relative: bool = True,
        check_forbidden: bool = True,
        action_name: str = "input",
    ) -> RelativeScreenPosition | None:
        try:
            position = self._target_position_components(x, y, relative)
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.error("Cannot validate %s position: %s", action_name, exc)
            return None

        (
            relative_x_coordinate,
            relative_y_coordinate,
            window_x_coordinate,
            window_y_coordinate,
            client_width,
            client_height,
        ) = position
        if not self._position_within_client_bounds(position):
            logger.warning(
                "Rejected %s outside target window: relative=(%s, %s), bounds=%sx%s",
                action_name,
                relative_x_coordinate,
                relative_y_coordinate,
                client_width,
                client_height,
            )
            return None

        if check_forbidden:
            forbidden_zone_name = self._matching_target_forbidden_zone(position)
            if forbidden_zone_name is not None:
                logger.warning(
                    "Rejected %s inside forbidden zone %s: relative=(%s, %s)",
                    action_name,
                    forbidden_zone_name,
                    relative_x_coordinate,
                    relative_y_coordinate,
                )
                return None

        return (
            relative_x_coordinate,
            relative_y_coordinate,
            window_x_coordinate,
            window_y_coordinate,
            client_width,
            client_height,
        )

    def _validated_screen_input_position(
        self,
        x: Any,
        y: Any,
        action_name: str,
    ) -> Point | None:
        validated_position = self._validated_target_position(
            x,
            y,
            relative=False,
            action_name=action_name,
        )
        if validated_position is None:
            return None
        (
            relative_x_coordinate,
            relative_y_coordinate,
            window_x_coordinate,
            window_y_coordinate,
            _,
            _,
        ) = validated_position
        return (
            int(window_x_coordinate + relative_x_coordinate),
            int(window_y_coordinate + relative_y_coordinate),
        )

    def _resolve_screen_position(
        self,
        x: Any,
        y: Any,
        relative: bool = True,
        check_forbidden: bool = True,
    ) -> Point | None:
        validated_position = self._validated_target_position(
            x,
            y,
            relative=relative,
            check_forbidden=check_forbidden,
            action_name="input",
        )
        if validated_position is None:
            return None
        (
            relative_x_coordinate,
            relative_y_coordinate,
            window_x_coordinate,
            window_y_coordinate,
            _,
            _,
        ) = validated_position
        return (
            int(window_x_coordinate + relative_x_coordinate),
            int(window_y_coordinate + relative_y_coordinate),
        )

    def move_to(self, x: Any, y: Any, relative: bool = True) -> bool:
        with self._input_lock:
            screen_pos = self._resolve_screen_position(x, y, relative=relative)
            if screen_pos is None:
                return False

            screen_x, screen_y = screen_pos
            if not self._set_cursor_pos(screen_x, screen_y):
                return False
            if self.move_delay > 0:
                precise_sleep(self.move_delay)
            logger.debug("Cursor moved to (%s, %s)", screen_x, screen_y)
            return True

    def _hover_before_click(self) -> None:
        if not self.hover_enabled:
            return
        duration = self._coerce_non_negative_float(self.hover_duration, 0.0)
        if duration > 0:
            precise_sleep(duration)

    @staticmethod
    def _configured_mouse_down_duration() -> float:
        return MouseController._coerce_non_negative_float(config.MOUSE_DOWN_DURATION, config.MOUSE_DOWN_DURATION)

    @staticmethod
    def _configured_mouse_up_duration() -> float:
        return MouseController._coerce_non_negative_float(config.MOUSE_UP_DURATION, config.MOUSE_UP_DURATION)

    @property
    def mouse_down_duration(self) -> float:
        return self._configured_mouse_down_duration()

    @property
    def mouse_up_duration(self) -> float:
        return self._configured_mouse_up_duration()

    def _click_down_up_delay(self) -> float:
        return self._get_mouse_down_duration()

    def _get_mouse_down_duration(self) -> float:
        return self._configured_mouse_down_duration()

    def _get_mouse_up_duration(self) -> float:
        return self._configured_mouse_up_duration()

    @staticmethod
    def _interruptible_delay(duration: Any, interrupt_check: Callable[[], bool] | None = None) -> bool:
        wait_time = MouseController._coerce_non_negative_float(duration, 0.0)
        return MouseController._interruptible_sleep_until(
            time.perf_counter() + wait_time,
            interrupt_check,
        )

    @staticmethod
    def _interruptible_sleep_until(deadline: float, interrupt_check: Callable[[], bool] | None = None) -> bool:
        for _ in range(_deadline_poll_iterations(deadline)):
            if interrupt_check and interrupt_check():
                return False
            remaining = float(deadline) - time.perf_counter()
            if remaining <= 0:
                return True
            precise_sleep(min(remaining, 0.005))
        return True

    def _left_down_at_screen(
        self,
        screen_x: Any,
        screen_y: Any,
        duration: Any = None,
        interrupt_check: Callable[[], bool] | None = None,
    ) -> bool:
        validated_screen_position = self._validated_screen_input_position(
            screen_x,
            screen_y,
            "left down",
        )
        if validated_screen_position is None:
            return False
        screen_x, screen_y = validated_screen_position
        if not self._mouse_button_action(
            "left down",
            lambda: self._mouse.press(self._left_button),
            screen_x,
            screen_y,
        ):
            self._best_effort_left_up(screen_x, screen_y)
            return False
        self._left_button_is_down = True
        wait_time = (
            self._get_mouse_down_duration()
            if duration is None
            else self._coerce_non_negative_float(duration, self._get_mouse_down_duration())
        )
        if wait_time > 0:
            if interrupt_check:
                if not self._interruptible_delay(wait_time, interrupt_check):
                    self._best_effort_left_up(screen_x, screen_y)
                    return False
            else:
                precise_sleep(wait_time)
        return True

    def _left_up_at_screen(
        self,
        screen_x: Any,
        screen_y: Any,
        duration: Any = None,
        interrupt_check: Callable[[], bool] | None = None,
    ) -> bool:
        validated_screen_position = self._validated_screen_input_position(
            screen_x,
            screen_y,
            "left up",
        )
        if validated_screen_position is None:
            self._best_effort_left_up(screen_x, screen_y)
            return False
        screen_x, screen_y = validated_screen_position
        if not self._mouse_button_action(
            "left up",
            lambda: self._mouse.release(self._left_button),
            screen_x,
            screen_y,
        ):
            self._best_effort_left_up(screen_x, screen_y)
            return False
        self._left_button_is_down = False
        wait_time = (
            self._get_mouse_up_duration()
            if duration is None
            else self._coerce_non_negative_float(duration, self._get_mouse_up_duration())
        )
        if wait_time > 0:
            if interrupt_check:
                return self._interruptible_delay(wait_time, interrupt_check)
            precise_sleep(wait_time)
        return True

    def _left_click_at_screen(
        self,
        screen_x: Any,
        screen_y: Any,
        down_duration: Any = None,
        up_duration: Any = None,
        interrupt_check: Callable[[], bool] | None = None,
    ) -> bool:
        if not self._left_down_at_screen(screen_x, screen_y, down_duration, interrupt_check):
            return False
        if not self._left_up_at_screen(screen_x, screen_y, up_duration, interrupt_check):
            self._best_effort_left_up(screen_x, screen_y)
            return False
        return True

    def _wait_after_click(self, delay: Any = None) -> None:
        wait_time = self.click_delay if delay is None else delay
        wait_time = self._coerce_non_negative_float(wait_time, self.click_delay)
        if wait_time > 0:
            precise_sleep(wait_time)

    def _position_cursor_for_click_attempt(self, x: Any, y: Any, relative: bool) -> tuple[Point | None, bool]:
        screen_pos = self._resolve_screen_position(x, y, relative=relative)
        if screen_pos is None:
            return None, False

        screen_x, screen_y = screen_pos
        if not self._set_cursor_pos(screen_x, screen_y):
            return screen_pos, False
        if self.move_delay > 0:
            precise_sleep(self.move_delay)
        self._hover_before_click()
        return screen_pos, True

    @staticmethod
    def _log_failed_click_attempt(action_name: str, screen_pos: Point | None) -> None:
        if screen_pos is not None:
            logger.error("%s failed at (%s, %s)", action_name, screen_pos[0], screen_pos[1])

    def _perform_precise_click_at_screen(self, screen_x: int, screen_y: int) -> bool:
        matches, _, _ = self._cursor_matches_safe_position(screen_x, screen_y)
        if not matches:
            if not self._set_cursor_pos(screen_x, screen_y):
                return False
            if self.move_delay > 0:
                precise_sleep(self.move_delay)
            self._hover_before_click()
        if not self._left_down_at_screen(screen_x, screen_y):
            return False
        matches, _, _ = self._cursor_matches_safe_position(screen_x, screen_y)
        if not matches:
            self._best_effort_left_up(screen_x, screen_y)
            return False
        if not self._left_up_at_screen(screen_x, screen_y):
            self._best_effort_left_up(screen_x, screen_y)
            return False
        return True

    def _execute_click_action(
        self,
        action_label: str,
        click_fn: Callable[[int, int], bool],
        x: Any,
        y: Any,
        relative: bool,
        delay: Any,
    ) -> bool:
        with self._input_lock:
            screen_pos, cursor_positioned = self._position_cursor_for_click_attempt(x, y, relative)
            if screen_pos is None:
                return False
            if not cursor_positioned:
                self._log_failed_click_attempt(action_label, screen_pos)
                return False

            screen_x, screen_y = screen_pos
            if not click_fn(screen_x, screen_y):
                self._log_failed_click_attempt(action_label, screen_pos)
                return False

            logger.debug("%s at (%s, %s)", action_label, screen_x, screen_y)
            self._wait_after_click(delay)
            return True

    def click(self, x: Any, y: Any, relative: bool = True, delay: Any = None) -> bool:
        return self._execute_click_action("Clicked", self._left_click_at_screen, x, y, relative, delay)

    def precise_click(self, x: Any, y: Any, relative: bool = True, delay: Any = None) -> bool:
        return self._execute_click_action("Precise-clicked", self._perform_precise_click_at_screen, x, y, relative, delay)

    def hold_at(
        self,
        x: Any,
        y: Any,
        duration: Any = None,
        relative: bool = True,
        interrupt_check: Callable[[], bool] | None = None,
    ) -> bool:
        with self._input_lock:
            if duration is None:
                duration = 4.0
            duration = self._coerce_non_negative_float(duration, 4.0)

            screen_pos = self._resolve_screen_position(x, y, relative=relative)
            if screen_pos is None:
                return False

            screen_x, screen_y = screen_pos
            if not self._set_cursor_pos(screen_x, screen_y):
                return False
            if self.move_delay > 0:
                precise_sleep(self.move_delay)

            if not self._left_down_at_screen(screen_x, screen_y, interrupt_check=interrupt_check):
                return False
            logger.debug("Holding at (%s, %s) for %.2fs", screen_x, screen_y, duration)

            if not self._interruptible_delay(duration, interrupt_check):
                if not self._left_up_at_screen(screen_x, screen_y):
                    self._best_effort_left_up(screen_x, screen_y)
                return False

            if not self._left_up_at_screen(screen_x, screen_y, interrupt_check=interrupt_check):
                self._best_effort_left_up(screen_x, screen_y)
                return False
            if self.click_delay > 0:
                precise_sleep(self.click_delay)
            return True

    def _run_stats_upgrade_click_loop(
        self,
        screen_x: int,
        screen_y: int,
        duration: float,
        click_delay: float,
        interrupt_check: Callable[[], bool] | None,
    ) -> int | None:
        start_time = time.perf_counter()
        end_time = start_time + duration
        click_count = 0

        logger.debug(
            "Stats upgrade clicking at (%s, %s) for %.2fs (delay=%.3fs)",
            screen_x,
            screen_y,
            duration,
            click_delay,
        )

        for click_index in range(self._stats_upgrade_click_limit(duration, click_delay)):
            if interrupt_check and interrupt_check():
                logger.debug("Stats upgrade clicking interrupted after %s clicks", click_count)
                return None

            next_click_at = start_time + (click_index * click_delay)
            if next_click_at >= end_time:
                return click_count
            if time.perf_counter() < next_click_at:
                if not self._interruptible_sleep_until(next_click_at, interrupt_check):
                    return None
            if time.perf_counter() >= end_time:
                return click_count

            if not self._left_click_at_screen(
                screen_x,
                screen_y,
                down_duration=0.0,
                up_duration=0.0,
                interrupt_check=interrupt_check,
            ):
                return None

            click_count += 1
        return click_count

    @staticmethod
    def _stats_upgrade_click_limit(duration: float, click_delay: float) -> int:
        return min(MAX_STATS_UPGRADE_CLICKS, max(1, int(math.ceil(duration / click_delay))))

    def click_stats_upgrade_at(
        self,
        x: Any,
        y: Any,
        duration: Any,
        click_delay: Any,
        relative: bool = True,
        interrupt_check: Callable[[], bool] | None = None,
    ) -> bool:
        with self._input_lock:
            duration = self._coerce_non_negative_float(duration, 0.0)
            click_delay = self._coerce_non_negative_float(click_delay, 0.0)
            if duration <= 0:
                logger.warning("Rejected stats upgrade click with non-positive duration: %.3f", duration)
                return False
            if click_delay <= 0:
                logger.warning("Rejected stats upgrade click with non-positive delay: %.3f", click_delay)
                return False

            screen_pos = self._resolve_screen_position(x, y, relative=relative)
            if screen_pos is None:
                return False

            base_x, base_y = screen_pos
            if not self._set_cursor_pos(base_x, base_y):
                return False
            if self.move_delay > 0:
                precise_sleep(self.move_delay)
            self._hover_before_click()

            click_count = self._run_stats_upgrade_click_loop(
                base_x,
                base_y,
                duration,
                click_delay,
                interrupt_check,
            )
            if click_count is None:
                return False

            logger.debug("Stats upgrade clicking complete: %s clicks", click_count)
            return click_count > 0

    def _validated_drag_path(
        self,
        screen_from_x: int,
        screen_from_y: int,
        screen_to_x: int,
        screen_to_y: int,
        steps: int,
    ) -> list[Point] | None:
        steps = min(MAX_DRAG_STEPS, max(1, int(steps)))
        drag_path = []
        for index in range(steps + 1):
            position_ratio = index / steps
            current_x_coordinate = int(
                screen_from_x + (screen_to_x - screen_from_x) * position_ratio
            )
            current_y_coordinate = int(
                screen_from_y + (screen_to_y - screen_from_y) * position_ratio
            )
            validated_position = self._validated_screen_input_position(
                current_x_coordinate,
                current_y_coordinate,
                "drag path",
            )
            if validated_position is None:
                return None
            drag_path.append(validated_position)
        return drag_path

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
            if duration is None:
                duration = config.SCROLL_DURATION
            duration = max(0.01, self._coerce_non_negative_float(duration, config.SCROLL_DURATION))

            from_pos = self._resolve_screen_position(from_x, from_y, relative=relative)
            to_pos = self._resolve_screen_position(to_x, to_y, relative=relative)
            if from_pos is None or to_pos is None:
                return False
            screen_from_x, screen_from_y = from_pos
            screen_to_x, screen_to_y = to_pos
            steps = 20
            drag_path = self._validated_drag_path(
                screen_from_x,
                screen_from_y,
                screen_to_x,
                screen_to_y,
                steps,
            )
            if drag_path is None:
                return False

            if not self._set_cursor_pos(screen_from_x, screen_from_y):
                return False
            if self.move_delay > 0:
                precise_sleep(self.move_delay)

            if not self._left_down_at_screen(screen_from_x, screen_from_y):
                return False

            start_time = time.perf_counter()
            for index, (current_x, current_y) in enumerate(drag_path):
                if not self._set_cursor_pos(current_x, current_y, verify_position=index == steps):
                    self._best_effort_left_up(current_x, current_y)
                    return False
                sleep_until(start_time + ((index + 1) * (duration / steps)))

            if not self._left_up_at_screen(screen_to_x, screen_to_y):
                self._best_effort_left_up(screen_to_x, screen_to_y)
                return False
            logger.debug("Dragged from (%s, %s) to (%s, %s)", from_x, from_y, to_x, to_y)
            if self.click_delay > 0:
                precise_sleep(self.click_delay)
            return True
