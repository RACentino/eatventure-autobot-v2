import logging
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
        config.validate_config()
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
