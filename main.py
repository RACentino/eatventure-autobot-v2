import logging
import queue
import sys
import threading
from enum import Enum, auto
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from typing import Any

from core import config
from core.platform import BackendDependencyError, pynput_keyboard, require_keyboard_backend
from bot import EatventureBot
from interaction.mouse import precise_sleep

bot_instance: EatventureBot | None = None
log_listener: QueueListener | None = None
exit_requested = threading.Event()
bot_command_queue: queue.SimpleQueue["BotCommand"] = queue.SimpleQueue()


class BotCommand(Enum):
    LOG_CURSOR_POSITION = auto()
    TOGGLE_RUNNING = auto()
    WIPE_MEMORY = auto()
    EXIT_PROGRAM = auto()


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
    logger.info("[X pressed] Window position: (%s, %s)", screen_x - window_x, screen_y - window_y)


def _toggle_bot_running(logger: logging.Logger) -> None:
    if bot_instance is None:
        return
    if bot_instance.running:
        bot_instance.stop()
        bot_instance.telegram.notify_bot_stopped()
        logger.info("[Z pressed] Bot STOPPED")
        return

    started = bot_instance.start()
    if started:
        bot_instance.telegram.notify_bot_started()
        logger.info("[Z pressed] Bot STARTED")
        return
    logger.warning("[Z pressed] Bot START failed")


def _wipe_bot_memory(logger: logging.Logger) -> None:
    if bot_instance is None:
        return
    bot_instance.wipe_memory()
    logger.info("[C pressed] AI memory wiped")


def _request_program_exit(logger: logging.Logger) -> None:
    logger.info("[P pressed] Exiting program")
    exit_requested.set()


def _enqueue_bot_command(command: BotCommand) -> None:
    if command in {BotCommand.TOGGLE_RUNNING, BotCommand.EXIT_PROGRAM} and bot_instance is not None:
        bot_instance.request_stop()
    bot_command_queue.put(command)


def _process_bot_command(command: BotCommand, logger: logging.Logger) -> None:
    command_handlers = {
        BotCommand.LOG_CURSOR_POSITION: _log_window_relative_cursor_position,
        BotCommand.TOGGLE_RUNNING: _toggle_bot_running,
        BotCommand.WIPE_MEMORY: _wipe_bot_memory,
        BotCommand.EXIT_PROGRAM: _request_program_exit,
    }
    handler = command_handlers.get(command)
    if handler is not None:
        handler(logger)


def _drain_bot_commands(logger: logging.Logger) -> None:
    while True:
        try:
            command = bot_command_queue.get_nowait()
        except queue.Empty:
            return
        _process_bot_command(command, logger)


def on_press(key: Any) -> None:
    try:
        character = _get_key_character(key)
        if character is None:
            return

        key_commands = {
            "x": BotCommand.LOG_CURSOR_POSITION,
            "z": BotCommand.TOGGLE_RUNNING,
            "c": BotCommand.WIPE_MEMORY,
            "p": BotCommand.EXIT_PROGRAM,
        }
        command = key_commands.get(character)
        if command is not None:
            _enqueue_bot_command(command)
    except Exception as exc:
        logging.getLogger(__name__).error("Keyboard listener error: %s", exc)


def _create_keyboard_listener() -> Any:
    require_keyboard_backend("Keyboard listener")
    if pynput_keyboard is None:
        raise BackendDependencyError("Keyboard listener backend is unavailable")
    return pynput_keyboard.Listener(on_press=on_press)


def setup_logging() -> None:
    global log_listener
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
    root_logger.handlers.clear()

    if log_listener is not None:
        log_listener.stop()

    log_queue = queue.SimpleQueue()
    queue_handler = QueueHandler(log_queue)
    root_logger.addHandler(queue_handler)

    log_listener = QueueListener(
        log_queue,
        console_handler,
        file_handler,
        respect_handler_level=True,
    )
    log_listener.start()


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
    while not exit_requested.is_set():
        _drain_bot_commands(logger)
        if bot_instance is not None and bot_instance.running:
            bot_instance.step()
        _drain_bot_commands(logger)
        precise_sleep(0.1)


def _cleanup_runtime(listener: Any | None) -> None:
    global log_listener
    logger = logging.getLogger(__name__)
    if bot_instance is not None and bot_instance.running:
        bot_instance.stop()
    if listener is not None:
        listener.stop()
        listener.join(timeout=1.0)
    if bot_instance is not None:
        bot_instance.telegram.close()
        try:
            bot_instance.window_capture.close()
        except Exception as exc:
            logger.debug("Window capture close failed: %s", exc)
    if log_listener is not None:
        log_listener.stop()
        log_listener = None


def main() -> int:
    global bot_instance
    listener = None

    _print_startup_banner()

    try:
        setup_logging()
        exit_requested.clear()

        try:
            listener = _create_keyboard_listener()
        except BackendDependencyError as exc:
            logging.getLogger(__name__).error("Keyboard listener unavailable: %s", exc)
            return 1
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
    except Exception as exc:
        logging.getLogger(__name__).error("Fatal error: %s", exc, exc_info=True)
        return 1
    finally:
        _cleanup_runtime(listener)

    return 0


if __name__ == "__main__":
    sys.exit(main())
