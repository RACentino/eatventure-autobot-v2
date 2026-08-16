import logging
import math
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from pynput import keyboard as pynput_keyboard

import config
from bot import EatventureBot
from domain import (
    BOX_TEMPLATE_NAMES,
    MAX_UPGRADE_SEARCH_ATTEMPTS,
    RED_ICON_TEMPLATE_NAMES,
)
from image_matcher import HSV_REGION_RANGE_LIMIT

bot_instance: EatventureBot | None = None
exit_requested = threading.Event()
MAX_NUMBERED_FORBIDDEN_ZONES = 32
HSV_CHANNEL_MAXIMUMS = (179, 255, 255)
POSITIVE_UNIT_INTERVAL_CONFIGURATION_NAMES = (
    "MATCH_THRESHOLD",
    "RED_ICON_THRESHOLD",
    "NEW_LEVEL_RED_ICON_THRESHOLD",
    "STATS_RED_ICON_THRESHOLD",
    "UPGRADE_STATION_THRESHOLD",
    "BOX_THRESHOLD",
    "UNLOCK_THRESHOLD",
    "NEW_LEVEL_THRESHOLD",
    "UPGRADE_STATION_HSV_MIN_MATCH_RATIO",
    "BOX_HSV_MIN_MATCH_RATIO",
    "RED_ICON_HSV_MIN_MATCH_RATIO",
)
NMS_CONFIGURATION_NAMES = (
    "SUPERVISION_BOX_NMS_IOU_THRESHOLD",
    "SUPERVISION_RED_ICON_NMS_IOU_THRESHOLD",
    "SUPERVISION_UPGRADE_STATION_NMS_IOU_THRESHOLD",
)
NONNEGATIVE_CONFIGURATION_NAMES = (
    "SCRCPY_RED_ICON_MISS_RECOVERY_DELAY",
    "SCRCPY_BOX_MISS_RECOVERY_DELAY",
    "CLICK_DELAY",
    "MOUSE_MOVE_DELAY",
    "MOUSE_DOWN_DURATION",
    "MOUSE_UP_DURATION",
    "HOVER_DURATION",
    "UPGRADE_SEARCH_INTERVAL",
    "UPGRADE_STATION_VERIFY_SETTLE_DELAY",
    "UPGRADE_STATION_VERIFY_SEARCH_INTERVAL",
    "SCROLL_INTERVAL_PAUSE",
    "POST_SCROLL_SETTLE",
)
POSITIVE_CONFIGURATION_NAMES = (
    "SIXTY_FPS_FRAME_DURATION_SECONDS",
    "CLICK_HOLD_MAX_DURATION",
    "STATS_UPGRADE_CLICK_DURATION",
    "STATS_UPGRADE_CLICK_DELAY",
    "SCROLL_DURATION",
    "SCROLL_DISTANCE_RATIO",
)
BOOLEAN_CONFIGURATION_NAMES = (
    "DEBUG",
    "SCRCPY_MISS_RECOVERY_ENABLED",
    "HOVER_ENABLED",
    "RED_ICON_FAST_MODE_ENABLED",
    "TELEGRAM_ENABLED",
)


def _validate_finite_number(
    configuration_name: str,
    configuration_value: Any,
    minimum_value: float,
    maximum_value: float | None = None,
) -> str | None:
    if isinstance(configuration_value, bool):
        return f"{configuration_name} must be numeric"
    try:
        numeric_value = float(configuration_value)
    except (TypeError, ValueError):
        return f"{configuration_name} must be numeric"
    if not math.isfinite(numeric_value) or numeric_value < minimum_value:
        return f"{configuration_name} must be finite and at least {minimum_value}"
    if maximum_value is not None and numeric_value > maximum_value:
        return f"{configuration_name} must not exceed {maximum_value}"
    return None


def _validate_integer(
    configuration_name: str,
    configuration_value: Any,
    minimum_value: int | None = None,
    maximum_value: int | None = None,
) -> str | None:
    if isinstance(configuration_value, bool) or not isinstance(
        configuration_value, int
    ):
        return f"{configuration_name} must be an integer"
    if minimum_value is not None and configuration_value < minimum_value:
        return f"{configuration_name} must be at least {minimum_value}"
    if maximum_value is not None and configuration_value > maximum_value:
        return f"{configuration_name} must not exceed {maximum_value}"
    return None


def _validate_position(configuration_name: str, position: object) -> str | None:
    if not isinstance(position, tuple) or len(position) != 2:
        return f"{configuration_name} must be a two-item tuple"
    if any(
        isinstance(coordinate, bool) or not isinstance(coordinate, int)
        for coordinate in position
    ):
        return f"{configuration_name} coordinates must be integers"
    position_x, position_y = position
    if not (
        0 <= position_x < config.WINDOW_WIDTH and 0 <= position_y < config.WINDOW_HEIGHT
    ):
        return f"{configuration_name} must be inside the configured window"
    return None


def _number_configuration_errors(
    names: tuple[str, ...], minimum: float, maximum: float | None = None
) -> list[str]:
    validation_errors = []
    for name in names:
        validation_error = _validate_finite_number(
            name, getattr(config, name), minimum, maximum
        )
        if validation_error is not None:
            validation_errors.append(validation_error)
    return validation_errors


def _integer_configuration_errors(
    bounds: tuple[tuple[str, Any, int | None, int | None], ...],
) -> list[str]:
    validation_errors = []
    for name, value, minimum, maximum in bounds:
        validation_error = _validate_integer(name, value, minimum, maximum)
        if validation_error is not None:
            validation_errors.append(validation_error)
    return validation_errors


def _configuration_dimension_limits() -> tuple[int | None, int | None]:
    limits = []
    for name, value in (
        ("WINDOW_WIDTH", config.WINDOW_WIDTH),
        ("WINDOW_HEIGHT", config.WINDOW_HEIGHT),
    ):
        limits.append(value if _validate_integer(name, value, 1) is None else None)
    return limits[0], limits[1]


def _integer_configuration_bounds() -> tuple[
    tuple[str, Any, int | None, int | None], ...
]:
    window_width_limit, window_height_limit = _configuration_dimension_limits()
    return (
        ("WINDOW_WIDTH", config.WINDOW_WIDTH, 1, None),
        ("WINDOW_HEIGHT", config.WINDOW_HEIGHT, 1, None),
        ("MAX_SEARCH_Y", config.MAX_SEARCH_Y, 1, window_height_limit),
        ("EXTENDED_SEARCH_Y", config.EXTENDED_SEARCH_Y, 1, window_height_limit),
        (
            "UPGRADE_STATION_SEARCH_Y",
            config.UPGRADE_STATION_SEARCH_Y,
            1,
            window_height_limit,
        ),
        ("BOX_SEARCH_Y", config.BOX_SEARCH_Y, 1, window_height_limit),
        (
            "BOXES_MIN_MATCHES",
            config.BOXES_MIN_MATCHES,
            1,
            len(BOX_TEMPLATE_NAMES),
        ),
        (
            "RED_ICON_MIN_MATCHES",
            config.RED_ICON_MIN_MATCHES,
            1,
            len(RED_ICON_TEMPLATE_NAMES),
        ),
        ("RED_ICON_FAST_MIN_DISTANCE", config.RED_ICON_FAST_MIN_DISTANCE, 1, None),
        (
            "UPGRADE_STATION_VERIFY_SEARCH_ATTEMPTS",
            config.UPGRADE_STATION_VERIFY_SEARCH_ATTEMPTS,
            1,
            MAX_UPGRADE_SEARCH_ATTEMPTS,
        ),
        ("SCROLL_PIXEL_STEP", config.SCROLL_PIXEL_STEP, 1, window_height_limit),
        (
            "MAX_SCROLL_CYCLES",
            config.MAX_SCROLL_CYCLES,
            1,
            None,
        ),
        (
            "SCROLL_INCREMENT_STEP",
            config.SCROLL_INCREMENT_STEP,
            1,
            None,
        ),
        (
            "RED_ICON_OFFSET_X",
            config.RED_ICON_OFFSET_X,
            -window_width_limit if window_width_limit is not None else None,
            window_width_limit,
        ),
        (
            "RED_ICON_OFFSET_Y",
            config.RED_ICON_OFFSET_Y,
            -window_height_limit if window_height_limit is not None else None,
            window_height_limit,
        ),
    )


def _numeric_configuration_errors() -> list[str]:
    errors = _integer_configuration_errors(_integer_configuration_bounds())
    errors.extend(
        _number_configuration_errors(
            POSITIVE_UNIT_INTERVAL_CONFIGURATION_NAMES, 0.000001, 1
        )
    )
    errors.extend(_number_configuration_errors(NMS_CONFIGURATION_NAMES, 0, 1))
    errors.extend(_number_configuration_errors(NONNEGATIVE_CONFIGURATION_NAMES, 0))
    errors.extend(_number_configuration_errors(POSITIVE_CONFIGURATION_NAMES, 0.000001))
    errors.extend(
        _number_configuration_errors(
            ("TELEGRAM_REQUEST_TIMEOUT", "TELEGRAM_SHUTDOWN_TIMEOUT"), 0.001
        )
    )
    return errors


def _position_configuration_errors() -> list[str]:
    validation_errors: list[str] = []
    for configuration_name, position in (
        ("IDLE_CLICK_POS", config.IDLE_CLICK_POS),
        ("STATS_UPGRADE_BUTTON_POS", config.STATS_UPGRADE_BUTTON_POS),
        ("STATS_UPGRADE_POS", config.STATS_UPGRADE_POS),
        ("SCROLL_START_POS", config.SCROLL_START_POS),
        ("NEW_LEVEL_BUTTON_POS", config.NEW_LEVEL_BUTTON_POS),
        ("LEVEL_TRANSITION_POS", config.LEVEL_TRANSITION_POS),
    ):
        validation_error = _validate_position(configuration_name, position)
        if validation_error is not None:
            validation_errors.append(validation_error)
    return validation_errors


def _validate_rectangle(configuration_name: str, bounds: object) -> str | None:
    if not isinstance(bounds, tuple) or len(bounds) != 4:
        return f"{configuration_name} must be a four-item tuple"
    if any(
        isinstance(coordinate, bool) or not isinstance(coordinate, int)
        for coordinate in bounds
    ):
        return f"{configuration_name} coordinates must be integers"
    x_minimum, x_maximum, y_minimum, y_maximum = bounds
    if not (
        0 <= x_minimum <= x_maximum < config.WINDOW_WIDTH
        and 0 <= y_minimum <= y_maximum < config.WINDOW_HEIGHT
    ):
        return f"{configuration_name} must be ordered inside the configured window"
    return None


def _rectangle_configuration_errors() -> list[str]:
    rectangles = (
        (
            "NEW_LEVEL_RED_ICON_BOUNDS",
            (
                config.NEW_LEVEL_RED_ICON_X_MIN,
                config.NEW_LEVEL_RED_ICON_X_MAX,
                config.NEW_LEVEL_RED_ICON_Y_MIN,
                config.NEW_LEVEL_RED_ICON_Y_MAX,
            ),
        ),
        (
            "UPGRADE_RED_ICON_BOUNDS",
            (
                config.UPGRADE_RED_ICON_X_MIN,
                config.UPGRADE_RED_ICON_X_MAX,
                config.UPGRADE_RED_ICON_Y_MIN,
                config.UPGRADE_RED_ICON_Y_MAX,
            ),
        ),
        (
            "FORBIDDEN_CLICK_BOUNDS",
            (
                config.FORBIDDEN_CLICK_X_MIN,
                config.FORBIDDEN_CLICK_X_MAX,
                config.FORBIDDEN_CLICK_Y_MIN,
                config.WINDOW_HEIGHT - 1,
            ),
        ),
    )
    errors = [
        error
        for name, bounds in rectangles
        if (error := _validate_rectangle(name, bounds)) is not None
    ]
    zones = config.NUMBERED_FORBIDDEN_ZONE_BOUNDS
    if (
        not isinstance(zones, tuple)
        or not zones
        or len(zones) > MAX_NUMBERED_FORBIDDEN_ZONES
    ):
        errors.append(
            "NUMBERED_FORBIDDEN_ZONE_BOUNDS must be a bounded tuple of rectangles"
        )
        return errors
    for zone_index, bounds in enumerate(zones, start=1):
        error = _validate_rectangle(f"FORBIDDEN_ZONE_{zone_index}", bounds)
        if error is not None:
            errors.append(error)
    return errors


def _validate_hsv_bound(
    configuration_name: str, range_index: int, bound: object
) -> str | None:
    if not isinstance(bound, tuple) or len(bound) != 3:
        return f"{configuration_name}[{range_index}] bounds must have three channels"
    for channel_index, value in enumerate(bound):
        if isinstance(value, bool) or not isinstance(value, int):
            return f"{configuration_name}[{range_index}] channels must be integers"
        if not 0 <= value <= HSV_CHANNEL_MAXIMUMS[channel_index]:
            return f"{configuration_name}[{range_index}] channel is out of range"
    return None


def _validate_hsv_range(
    configuration_name: str, range_index: int, hsv_range: object
) -> str | None:
    if not isinstance(hsv_range, tuple) or len(hsv_range) != 2:
        return f"{configuration_name}[{range_index}] must contain lower and upper"
    lower, upper = hsv_range
    for bound in (lower, upper):
        validation_error = _validate_hsv_bound(configuration_name, range_index, bound)
        if validation_error is not None:
            return validation_error
    if lower[1] > upper[1] or lower[2] > upper[2]:
        return f"{configuration_name}[{range_index}] saturation/value is reversed"
    return None


def _validate_hsv_ranges(configuration_name: str, ranges: object) -> str | None:
    if not isinstance(ranges, tuple) or not 1 <= len(ranges) <= HSV_REGION_RANGE_LIMIT:
        return f"{configuration_name} must contain 1..{HSV_REGION_RANGE_LIMIT} ranges"
    for range_index, hsv_range in enumerate(ranges, start=1):
        validation_error = _validate_hsv_range(
            configuration_name, range_index, hsv_range
        )
        if validation_error is not None:
            return validation_error
    return None


def _hsv_configuration_errors() -> list[str]:
    errors = []
    for name, ranges in (
        ("RED_ICON_HSV_RANGES", config.RED_ICON_HSV_RANGES),
        ("UPGRADE_STATION_HSV_RANGES", config.UPGRADE_STATION_HSV_RANGES),
        ("BOX_HSV_RANGES", config.BOX_HSV_RANGES),
    ):
        error = _validate_hsv_ranges(name, ranges)
        if error is not None:
            errors.append(error)
    return errors


def _general_configuration_errors() -> list[str]:
    errors = []
    for name, value in (
        ("WINDOW_TITLE", config.WINDOW_TITLE),
        ("ASSETS_DIR", config.ASSETS_DIR),
        ("LOGS_DIR", config.LOGS_DIR),
    ):
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{name} must be a non-empty string")
    for name in BOOLEAN_CONFIGURATION_NAMES:
        value = getattr(config, name)
        if not isinstance(value, bool):
            errors.append(f"{name} must be a boolean")
    return errors


def _red_icon_template_configuration_errors() -> list[str]:
    names = config.RED_ICON_FAST_TEMPLATE_NAMES
    if (
        not isinstance(names, tuple)
        or not names
        or len(names) > len(RED_ICON_TEMPLATE_NAMES)
    ):
        return ["RED_ICON_FAST_TEMPLATE_NAMES must be a non-empty bounded tuple"]
    if any(not isinstance(name, str) or not name.strip() for name in names):
        return ["RED_ICON_FAST_TEMPLATE_NAMES must contain non-empty strings"]
    if len(set(names)) != len(names):
        return ["RED_ICON_FAST_TEMPLATE_NAMES must not contain duplicates"]
    unknown_names = tuple(name for name in names if name not in RED_ICON_TEMPLATE_NAMES)
    if unknown_names:
        return [
            "RED_ICON_FAST_TEMPLATE_NAMES contains unknown names: "
            + ", ".join(unknown_names)
        ]
    return []


def _validate_configuration() -> None:
    validation_errors = _general_configuration_errors()
    validation_errors.extend(_red_icon_template_configuration_errors())
    validation_errors.extend(_numeric_configuration_errors())
    window_width_limit, window_height_limit = _configuration_dimension_limits()
    if window_width_limit is not None and window_height_limit is not None:
        validation_errors.extend(_position_configuration_errors())
        validation_errors.extend(_rectangle_configuration_errors())
    validation_errors.extend(_hsv_configuration_errors())
    credentials_are_valid = all(
        isinstance(value, str) and bool(value.strip())
        for value in (config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
    )
    if config.TELEGRAM_ENABLED and not credentials_are_valid:
        validation_errors.append(
            "Telegram requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID"
        )
    if validation_errors:
        raise ValueError("Invalid configuration: " + "; ".join(validation_errors))


def _get_key_character(key: Any) -> str | None:
    character = getattr(key, "char", None)
    if character is None:
        return None
    return str(character).lower()


def _log_window_relative_cursor_position(logger: logging.Logger) -> None:
    if bot_instance is None or not bot_instance.window_capture.is_window_active():
        logger.info("[X pressed] Bot window is not available")
        return
    screen_x, screen_y = bot_instance.mouse_controller.get_cursor_position()
    window_x, window_y, _, _ = bot_instance.window_capture.get_window_rect()
    logger.info(
        "[X pressed] Window position: (%s, %s)",
        screen_x - window_x,
        screen_y - window_y,
    )


def _toggle_bot_running(logger: logging.Logger) -> None:
    if bot_instance is None:
        return
    if bot_instance.running:
        stopped = bot_instance.stop()
        bot_instance.telegram.notify_bot_stopped()
        if stopped:
            logger.info("[Z pressed] Bot STOPPED")
        else:
            logger.error("[Z pressed] Bot stopped with cleanup failures")
        return
    if bot_instance.start():
        bot_instance.telegram.notify_bot_started()
        logger.info("[Z pressed] Bot STARTED")
        return
    logger.warning("[Z pressed] Bot START failed")


def _request_program_exit(logger: logging.Logger) -> None:
    logger.info("[P pressed] Exiting program")
    exit_requested.set()
    if bot_instance is not None:
        bot_instance.stop()


def on_press(key: Any) -> None:
    try:
        character = _get_key_character(key)
        if character is None:
            return
        logger = logging.getLogger(__name__)
        key_handlers = {
            "x": _log_window_relative_cursor_position,
            "z": _toggle_bot_running,
            "p": _request_program_exit,
        }
        handler = key_handlers.get(character)
        if handler is not None:
            handler(logger)
    except Exception as exc:
        logging.getLogger(__name__).error("Keyboard listener error: %s", exc)


def _create_keyboard_listener() -> Any:
    return pynput_keyboard.Listener(on_press=on_press)


def setup_logging() -> None:
    logs_dir = Path(config.LOGS_DIR)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_level = logging.DEBUG if config.DEBUG else logging.INFO

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))

    file_handler = RotatingFileHandler(
        logs_dir / "bot.log",
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    for existing_handler in root_logger.handlers:
        existing_handler.close()
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def _print_startup_banner() -> None:
    print("=" * 60)
    print("Eatventure Bot - Screen Automation Tool")
    print("=" * 60)
    print(f"Window Title: {config.WINDOW_TITLE}")
    print(f"Match Threshold: {config.MATCH_THRESHOLD * 100}%")
    print(f"Assets Directory: {config.ASSETS_DIR}")
    print("=" * 60)


def _run_bot_event_loop() -> None:
    while not exit_requested.is_set():
        if bot_instance is not None and bot_instance.running:
            bot_instance.step()
        else:
            exit_requested.wait(0.1)


def _cleanup_runtime(listener: Any | None) -> None:
    logger = logging.getLogger(__name__)
    if bot_instance is not None and not bot_instance.close():
        logger.error("Bot cleanup completed with resource or input release failures")
    if listener is not None:
        listener.stop()
        listener.join(timeout=1.0)


def main() -> int:
    global bot_instance
    listener = None

    try:
        _validate_configuration()
        setup_logging()
        _print_startup_banner()
        exit_requested.clear()
        listener = _create_keyboard_listener()
        listener.start()

        bot_instance = EatventureBot()
        if not bot_instance.ready:
            raise RuntimeError(
                "Bot initialization failed: required templates unavailable"
            )
        logger = logging.getLogger(__name__)
        logger.info("Bot initialized and ready")
        logger.info("Press Z to START/STOP the bot")
        logger.info("Press X to see window-relative cursor position")
        logger.info("Press P to EXIT the program")

        _run_bot_event_loop()
        logger.info("Program exiting")
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Bot stopped by user (Ctrl+C)")
        return 0
    except Exception:
        logging.getLogger(__name__).exception("Fatal error")
        return 1
    finally:
        _cleanup_runtime(listener)

    return 0


if __name__ == "__main__":
    sys.exit(main())
