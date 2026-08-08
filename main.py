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
from mouse_controller import precise_sleep

bot_instance: EatventureBot | None = None
exit_requested = threading.Event()
BOT_EVENT_LOOP_ITERATION_LIMIT = 2_147_483_647


def _validate_finite_number(
    configuration_name: str,
    configuration_value: Any,
    minimum_value: float,
    maximum_value: float | None = None,
) -> str | None:
    try:
        numeric_value = float(configuration_value)
    except (TypeError, ValueError):
        return f"{configuration_name} must be numeric"
    if not math.isfinite(numeric_value) or numeric_value < minimum_value:
        return f"{configuration_name} must be finite and at least {minimum_value}"
    if maximum_value is not None and numeric_value > maximum_value:
        return f"{configuration_name} must not exceed {maximum_value}"
    return None


def _validate_position(configuration_name: str, position: object) -> str | None:
    if not isinstance(position, tuple) or len(position) != 2:
        return f"{configuration_name} must be a two-item tuple"
    try:
        position_x, position_y = int(position[0]), int(position[1])
    except (TypeError, ValueError):
        return f"{configuration_name} coordinates must be integers"
    if not (
        0 <= position_x < config.WINDOW_WIDTH and 0 <= position_y < config.WINDOW_HEIGHT
    ):
        return f"{configuration_name} must be inside the configured window"
    return None


def _numeric_configuration_errors() -> list[str]:
    numeric_bounds = (
        ("WINDOW_WIDTH", config.WINDOW_WIDTH, 1, None),
        ("WINDOW_HEIGHT", config.WINDOW_HEIGHT, 1, None),
        ("MAX_SEARCH_Y", config.MAX_SEARCH_Y, 1, config.WINDOW_HEIGHT),
        ("EXTENDED_SEARCH_Y", config.EXTENDED_SEARCH_Y, 1, config.WINDOW_HEIGHT),
        (
            "UPGRADE_STATION_SEARCH_Y",
            config.UPGRADE_STATION_SEARCH_Y,
            1,
            config.WINDOW_HEIGHT,
        ),
        ("BOX_SEARCH_Y", config.BOX_SEARCH_Y, 1, config.WINDOW_HEIGHT),
        ("MATCH_THRESHOLD", config.MATCH_THRESHOLD, 0, 1),
        ("RED_ICON_THRESHOLD", config.RED_ICON_THRESHOLD, 0, 1),
        ("UPGRADE_STATION_THRESHOLD", config.UPGRADE_STATION_THRESHOLD, 0, 1),
        ("BOX_THRESHOLD", config.BOX_THRESHOLD, 0, 1),
        ("UNLOCK_THRESHOLD", config.UNLOCK_THRESHOLD, 0, 1),
        ("NEW_LEVEL_THRESHOLD", config.NEW_LEVEL_THRESHOLD, 0, 1),
        ("CLICK_DELAY", config.CLICK_DELAY, 0, None),
        ("MOUSE_MOVE_DELAY", config.MOUSE_MOVE_DELAY, 0, None),
        (
            "ASSET_TRACKING_MAX_SNAPSHOT_AGE",
            config.ASSET_TRACKING_MAX_SNAPSHOT_AGE,
            0,
            None,
        ),
        ("AI_LEARNING_RECORDS_LIMIT", config.AI_LEARNING_RECORDS_LIMIT, 1, None),
    )
    validation_errors: list[str] = []
    for (
        configuration_name,
        configuration_value,
        minimum_value,
        maximum_value,
    ) in numeric_bounds:
        validation_error = _validate_finite_number(
            configuration_name, configuration_value, minimum_value, maximum_value
        )
        if validation_error is not None:
            validation_errors.append(validation_error)
    return validation_errors


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


def _validate_configuration() -> None:
    validation_errors: list[str] = []
    if not isinstance(config.WINDOW_TITLE, str) or not config.WINDOW_TITLE.strip():
        validation_errors.append("WINDOW_TITLE must be a non-empty string")
    validation_errors.extend(_numeric_configuration_errors())
    validation_errors.extend(_position_configuration_errors())
    if config.TELEGRAM_ENABLED and (
        not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID
    ):
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


def _wipe_bot_memory(logger: logging.Logger) -> None:
    if bot_instance is None:
        return
    if bot_instance.wipe_memory():
        logger.info("[C pressed] AI memory wiped")
    else:
        logger.warning("[C pressed] Stop the bot before wiping AI memory")


def _request_program_exit(logger: logging.Logger) -> None:
    logger.info("[P pressed] Exiting program")
    exit_requested.set()
    if bot_instance is not None:
        bot_instance.request_stop()


def on_press(key: Any) -> None:
    try:
        character = _get_key_character(key)
        if character is None:
            return
        logger = logging.getLogger(__name__)
        key_handlers = {
            "x": _log_window_relative_cursor_position,
            "z": _toggle_bot_running,
            "c": _wipe_bot_memory,
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
    logger = logging.getLogger(__name__)
    for _ in range(BOT_EVENT_LOOP_ITERATION_LIMIT):
        if exit_requested.is_set():
            return
        if bot_instance is not None and bot_instance.running:
            bot_instance.step()
        precise_sleep(0.1)
    logger.error("Bot event loop reached iteration limit")
    exit_requested.set()


def _cleanup_runtime(listener: Any | None) -> None:
    logger = logging.getLogger(__name__)
    if bot_instance is not None and not bot_instance.stop():
        logger.error("Bot cleanup completed with worker or input release failures")
    if listener is not None:
        listener.stop()
        listener.join(timeout=1.0)
    if bot_instance is not None:
        bot_instance.telegram.close()
        try:
            bot_instance.window_capture.close()
        except Exception as exc:
            logger.debug("Window capture close failed: %s", exc)


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
        logger = logging.getLogger(__name__)
        logger.info("Bot initialized and ready")
        logger.info("Press Z to START/STOP the bot")
        logger.info("Press X to see window-relative cursor position")
        logger.info("Press C to wipe AI memory")
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
