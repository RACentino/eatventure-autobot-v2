import logging
import math
import random
import threading
import time
from collections.abc import Callable
from typing import Any

from core import config
from core.platform import pynput_mouse, require_automation_backend

logger = logging.getLogger(__name__)

Point = tuple[int, int]
WindowBounds = tuple[int, int, int, int]
RelativeScreenPosition = tuple[int, int, int, int, int, int]
ForbiddenZone = tuple[str, int, int, int, int | None]
MINIMUM_SLEEP_SLICE_SECONDS = 0.001
MAXIMUM_SLEEP_SLICE_SECONDS = 0.050
INTERRUPTIBLE_SLEEP_SLICE_SECONDS = 0.005


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


def _deadline_remaining_seconds(deadline: float) -> float:
    return float(deadline) - time.perf_counter()


def _deadline_wait_attempt_count(deadline: float, slice_seconds: float) -> int:
    remaining_seconds = max(0.0, _deadline_remaining_seconds(deadline))
    bounded_slice_seconds = max(float(slice_seconds), MINIMUM_SLEEP_SLICE_SECONDS)
    return max(1, int(math.ceil(remaining_seconds / bounded_slice_seconds)) + 2)


def _stop_event_is_clear(stop_event: threading.Event | None) -> bool:
    return stop_event is None or not stop_event.is_set()


def _interrupt_check_is_clear(interrupt_check: Callable[[], bool] | None) -> bool:
    return interrupt_check is None or not interrupt_check()


def _finish_stop_event_deadline_wait(deadline: float, stop_event: threading.Event | None) -> bool:
    if not _stop_event_is_clear(stop_event):
        return False
    remaining_seconds = _deadline_remaining_seconds(deadline)
    if remaining_seconds <= 0:
        return _stop_event_is_clear(stop_event)
    if stop_event is None:
        time.sleep(remaining_seconds)
        return True
    return not stop_event.wait(remaining_seconds)


def _wait_for_interruptible_deadline(
    deadline: float,
    interrupt_check: Callable[[], bool] | None,
) -> bool:
    attempt_count = _deadline_wait_attempt_count(deadline, INTERRUPTIBLE_SLEEP_SLICE_SECONDS)
    for _ in range(attempt_count):
        if not _interrupt_check_is_clear(interrupt_check):
            return False
        remaining_seconds = _deadline_remaining_seconds(deadline)
        if remaining_seconds <= 0:
            return True
        time.sleep(min(remaining_seconds, INTERRUPTIBLE_SLEEP_SLICE_SECONDS))

    if not _interrupt_check_is_clear(interrupt_check):
        return False
    remaining_seconds = _deadline_remaining_seconds(deadline)
    if remaining_seconds > 0:
        time.sleep(remaining_seconds)
    return _interrupt_check_is_clear(interrupt_check)


def precise_sleep(duration: Any) -> None:
    duration = _coerce_duration(duration)
    if duration <= 0:
        return
    sleep_until(time.perf_counter() + duration)


def _wait_until_next_deadline_slice(remaining: float, stop_event: threading.Event | None) -> bool:
    if remaining > 0.004:
        wait_time = min(remaining - 0.002, MAXIMUM_SLEEP_SLICE_SECONDS)
        if stop_event is None:
            time.sleep(wait_time)
            return True
        return not stop_event.wait(wait_time)
    if remaining > 0.001:
        time.sleep(0)
    return True


def sleep_until(deadline: float, stop_event: threading.Event | None = None) -> bool:
    attempt_count = _deadline_wait_attempt_count(deadline, MINIMUM_SLEEP_SLICE_SECONDS)
    for _ in range(attempt_count):
        if stop_event is not None and stop_event.is_set():
            return False
        remaining = float(deadline) - time.perf_counter()
        if remaining <= 0:
            return stop_event is None or not stop_event.is_set()
        if not _wait_until_next_deadline_slice(remaining, stop_event):
            return False
    return _finish_stop_event_deadline_wait(deadline, stop_event)


def wait_event(stop_event: threading.Event | None, duration: Any) -> bool:
    duration = _coerce_duration(duration)
    if stop_event is None:
        precise_sleep(duration)
        return True
    if duration <= 0:
        return not stop_event.is_set()
    return sleep_until(time.perf_counter() + duration, stop_event)


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
            require_automation_backend("MouseController")
            self._mouse = pynput_mouse.Controller()
        else:
            self._mouse = mouse_device
        self._left_button = pynput_mouse.Button.left if pynput_mouse is not None else "left"
        self._window_bounds_source = window_bounds_source
        self.click_delay = self._coerce_non_negative_float(
            config.CLICK_DELAY if click_delay is None else click_delay,
            float(config.CLICK_DELAY),
        )
        self.move_delay = self._coerce_non_negative_float(
            config.MOUSE_MOVE_DELAY if move_delay is None else move_delay,
            float(config.MOUSE_MOVE_DELAY),
        )
        self.hover_enabled = bool(config.HOVER_ENABLED if hover_enabled is None else hover_enabled)
        self.hover_duration = self._coerce_non_negative_float(
            config.HOVER_DURATION if hover_duration is None else hover_duration,
            0.0,
        )
        self.input_retry_count = max(1, int(config.INPUT_RETRY_COUNT))
        self.input_retry_delay = max(0.0, float(config.INPUT_RETRY_DELAY))
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

    @staticmethod
    def _coerce_non_negative_int(value: Any, default: int = 0) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            try:
                number = int(default)
            except (TypeError, ValueError):
                number = 0
        return max(0, number)

    def get_cursor_position(self) -> Point:
        current_x, current_y = self._mouse.position
        return int(current_x), int(current_y)

    def get_window_position(self) -> Point:
        win_x, win_y, _, _ = self.get_window_bounds()
        return win_x, win_y

    def get_window_bounds(self) -> WindowBounds:
        try:
            bounds = self._window_bounds_source() if callable(self._window_bounds_source) else self._window_bounds_source
            win_x, win_y, width, height = bounds
            win_x = int(win_x)
            win_y = int(win_y)
            width = int(width)
            height = int(height)
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

    def _cursor_matches_position(self, screen_x: int, screen_y: int) -> tuple[bool, int, int]:
        current_x, current_y = self.get_cursor_position()
        matches = abs(current_x - screen_x) <= 1 and abs(current_y - screen_y) <= 1
        return matches, current_x, current_y

    def _sleep_before_input_retry(self, attempt: int) -> None:
        if attempt < self.input_retry_count:
            precise_sleep(self.input_retry_delay)

    def _set_cursor_pos(self, x: Any, y: Any, verify_position: bool = True) -> bool:
        validated_screen_position = self._validated_screen_input_position(x, y, "cursor move")
        if validated_screen_position is None:
            return False
        screen_x, screen_y = validated_screen_position
        last_exc = None
        for attempt in range(1, self.input_retry_count + 1):
            try:
                self._mouse.position = (screen_x, screen_y)
                if not verify_position:
                    return True
                precise_sleep(0.001)
                matches, current_x, current_y = self._cursor_matches_position(screen_x, screen_y)
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
                    self.input_retry_count,
                    exc,
                )
            self._sleep_before_input_retry(attempt)
        logger.error("Cursor positioning failed at (%s, %s): %s", screen_x, screen_y, last_exc)
        return False

    def _mouse_button_action(self, action_name: str, action: Callable[[], Any], x: Any, y: Any) -> bool:
        screen_x = int(x)
        screen_y = int(y)
        last_exc = None
        for attempt in range(1, self.input_retry_count + 1):
            try:
                action()
                return True
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "%s failed at (%s, %s) on attempt %s/%s: %s",
                    action_name,
                    screen_x,
                    screen_y,
                    attempt,
                    self.input_retry_count,
                    exc,
                )
                if attempt < self.input_retry_count:
                    precise_sleep(self.input_retry_delay)
        logger.error("%s failed at (%s, %s): %s", action_name, screen_x, screen_y, last_exc)
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
            config.numbered_forbidden_zone_bounds(), start=1,
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
            target_position = self._target_position_from_input(x, y, relative)
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.error("Cannot validate %s position: %s", action_name, exc)
            return None

        if self._target_position_is_rejected(target_position, check_forbidden, action_name):
            return None
        return target_position

    def _target_position_from_input(
        self,
        x: Any,
        y: Any,
        relative: bool,
    ) -> RelativeScreenPosition:
        if not relative:
            return self._relative_from_screen(x, y)

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

    def _target_position_is_rejected(
        self,
        target_position: RelativeScreenPosition,
        check_forbidden: bool,
        action_name: str,
    ) -> bool:
        (
            relative_x_coordinate,
            relative_y_coordinate,
            _,
            _,
            client_width,
            client_height,
        ) = target_position
        if not _point_within_client_area(
            relative_x_coordinate,
            relative_y_coordinate,
            client_width,
            client_height,
        ):
            logger.warning(
                "Rejected %s outside target window: relative=(%s, %s), bounds=%sx%s",
                action_name,
                relative_x_coordinate,
                relative_y_coordinate,
                client_width,
                client_height,
            )
            return True

        if check_forbidden:
            forbidden_zone_name = _matching_forbidden_zone_name(
                relative_x_coordinate,
                relative_y_coordinate,
                self._forbidden_zones,
            )
            if forbidden_zone_name is not None:
                logger.warning(
                    "Rejected %s inside forbidden zone %s: relative=(%s, %s)",
                    action_name,
                    forbidden_zone_name,
                    relative_x_coordinate,
                    relative_y_coordinate,
                )
                return True
        return False

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
        deadline = time.perf_counter() + wait_time
        return _wait_for_interruptible_deadline(deadline, interrupt_check)

    @staticmethod
    def _interruptible_sleep_until(deadline: float, interrupt_check: Callable[[], bool] | None = None) -> bool:
        return _wait_for_interruptible_deadline(deadline, interrupt_check)

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
        matches, _, _ = self._cursor_matches_position(screen_x, screen_y)
        if not matches:
            if not self._set_cursor_pos(screen_x, screen_y):
                return False
            if self.move_delay > 0:
                precise_sleep(self.move_delay)
            self._hover_before_click()
        if not self._left_down_at_screen(screen_x, screen_y):
            return False
        matches, _, _ = self._cursor_matches_position(screen_x, screen_y)
        if not matches:
            self._best_effort_left_up(screen_x, screen_y)
            return False
        if not self._left_up_at_screen(screen_x, screen_y):
            self._best_effort_left_up(screen_x, screen_y)
            return False
        return True

    def click(self, x: Any, y: Any, relative: bool = True, delay: Any = None) -> bool:
        with self._input_lock:
            last_screen_pos = None
            for _ in range(self.input_retry_count):
                screen_pos, cursor_positioned = self._position_cursor_for_click_attempt(x, y, relative)
                if screen_pos is None:
                    return False

                screen_x, screen_y = screen_pos
                last_screen_pos = (screen_x, screen_y)
                if not cursor_positioned:
                    continue

                if not self._left_click_at_screen(screen_x, screen_y):
                    continue

                logger.debug("Clicked at (%s, %s)", screen_x, screen_y)
                self._wait_after_click(delay)
                return True

            self._log_failed_click_attempt("Click", last_screen_pos)
            return False

    def precise_click(self, x: Any, y: Any, relative: bool = True, delay: Any = None) -> bool:
        with self._input_lock:
            last_screen_pos = None
            for _ in range(self.input_retry_count):
                screen_pos, cursor_positioned = self._position_cursor_for_click_attempt(x, y, relative)
                if screen_pos is None:
                    return False

                screen_x, screen_y = screen_pos
                last_screen_pos = (screen_x, screen_y)
                if not cursor_positioned:
                    continue
                if not self._perform_precise_click_at_screen(screen_x, screen_y):
                    continue

                logger.debug("Precise-clicked at (%s, %s)", screen_x, screen_y)
                self._wait_after_click(delay)
                return True

            self._log_failed_click_attempt("Precise click", last_screen_pos)
            return False

    def double_click(self, x: Any, y: Any, relative: bool = True) -> bool:
        if not self.click(x, y, relative=relative, delay=config.DOUBLE_CLICK_INTER_DELAY):
            return False
        return self.click(x, y, relative=relative)

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

    def click_sequence(
        self,
        x: Any,
        y: Any,
        count: Any,
        interval: Any,
        relative: bool = True,
        interrupt_check: Callable[[], bool] | None = None,
    ) -> bool:
        with self._input_lock:
            count = self._coerce_non_negative_int(count)
            interval = max(0.0, self._coerce_non_negative_float(interval, 0.0))
            if count <= 0:
                return True

            screen_pos = self._resolve_screen_position(x, y, relative=relative)
            if screen_pos is None:
                return False

            screen_x, screen_y = screen_pos
            if not self._set_cursor_pos(screen_x, screen_y):
                return False
            if self.move_delay > 0:
                precise_sleep(self.move_delay)
            self._hover_before_click()

            first_click_at = time.perf_counter()
            for index in range(count):
                if not self._interruptible_sleep_until(first_click_at + (index * interval), interrupt_check):
                    return False
                if not self._left_click_at_screen(
                    screen_x,
                    screen_y,
                    interrupt_check=interrupt_check,
                ):
                    return False

            logger.debug("Click sequence complete at (%s, %s): %s clicks", screen_x, screen_y, count)
            return True

    def _resolve_jittered_screen_position(
        self,
        base_x: int,
        base_y: int,
        jitter: int,
    ) -> Point | None:
        if jitter <= 0:
            return self._validated_screen_input_position(base_x, base_y, "spam click target")
        target_x = base_x + random.randint(-jitter, jitter)
        target_y = base_y + random.randint(-jitter, jitter)
        jittered_position = self._resolve_screen_position(target_x, target_y, relative=False)
        if jittered_position is None:
            return None
        if not self._set_cursor_pos(jittered_position[0], jittered_position[1]):
            return None
        return jittered_position

    def _run_spam_click_loop(
        self,
        base_x: int,
        base_y: int,
        duration: float,
        click_delay: float,
        jitter: int,
        interrupt_check: Callable[[], bool] | None,
    ) -> int | None:
        start_time = time.perf_counter()
        end_time = start_time + duration
        next_click_at = start_time
        click_count = 0

        logger.debug(
            "Spam-clicking at (%s, %s) for %.2fs (delay=%.3fs, jitter=%s)",
            base_x,
            base_y,
            duration,
            click_delay,
            jitter,
        )

        maximum_click_attempts = max(1, int(math.ceil(duration / click_delay)) + 1)
        for _ in range(maximum_click_attempts):
            if interrupt_check and interrupt_check():
                logger.debug("Spam-click interrupted after %s clicks", click_count)
                return None

            now = time.perf_counter()
            if now >= end_time:
                return click_count

            if now < next_click_at:
                if next_click_at >= end_time:
                    return click_count
                if not self._interruptible_sleep_until(next_click_at, interrupt_check):
                    return None
                continue

            target_position = self._resolve_jittered_screen_position(base_x, base_y, jitter)
            if target_position is None:
                return None
            target_x, target_y = target_position

            if not self._left_click_at_screen(
                target_x,
                target_y,
                down_duration=0.0,
                up_duration=0.0,
                interrupt_check=interrupt_check,
            ):
                return None

            click_count += 1
            next_click_at = time.perf_counter() + click_delay
        return click_count

    def spam_click_at(
        self,
        x: Any,
        y: Any,
        duration: Any = None,
        click_delay: Any = None,
        jitter: Any = 0,
        relative: bool = True,
        interrupt_check: Callable[[], bool] | None = None,
    ) -> bool:
        with self._input_lock:
            if duration is None:
                duration = config.SPAM_CLICK_DURATION
            if click_delay is None:
                click_delay = config.SPAM_CLICK_DELAY

            duration = self._coerce_non_negative_float(duration, config.SPAM_CLICK_DURATION)
            click_delay = max(0.001, self._coerce_non_negative_float(click_delay, config.SPAM_CLICK_DELAY))
            jitter = self._coerce_non_negative_int(jitter)

            screen_pos = self._resolve_screen_position(x, y, relative=relative)
            if screen_pos is None:
                return False

            base_x, base_y = screen_pos
            if not self._set_cursor_pos(base_x, base_y):
                return False
            if self.move_delay > 0:
                precise_sleep(self.move_delay)
            self._hover_before_click()

            click_count = self._run_spam_click_loop(
                base_x,
                base_y,
                duration,
                click_delay,
                jitter,
                interrupt_check,
            )
            if click_count is None:
                return False

            logger.debug("Spam-click complete: %s clicks", click_count)
            return True

    def _validated_drag_path(
        self,
        screen_from_x: int,
        screen_from_y: int,
        screen_to_x: int,
        screen_to_y: int,
        steps: int,
    ) -> list[Point] | None:
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
