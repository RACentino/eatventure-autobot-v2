import logging
import queue
import sys
import threading
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from typing import Any

import config
from bot import EatventureBot
from pynput import keyboard

bot_instance: EatventureBot | None = None
should_exit = threading.Event()
toggle_requested = threading.Event()
log_listener: QueueListener | None = None
primed_event_selection: tuple[int, tuple[int, int, int, int]] | None = None


def _key_character(key: Any) -> str | None:
    character = getattr(key, "char", None)
    return str(character).lower() if character is not None else None


def _event_options() -> list[tuple[int, tuple[int, int, int, int]]]:
    options = []
    for count, bounds in config.EVENT_FORBIDDEN_ZONE_OPTIONS.items():
        if not isinstance(count, int) or count < 1 or len(bounds) != 4:
            raise ValueError("event options must map positive integers to rectangles")
        x_min, x_max, y_min, y_max = bounds
        if not (0 <= x_min <= x_max < config.WINDOW_WIDTH):
            raise ValueError(f"event option {count} has invalid horizontal bounds")
        if not (0 <= y_min <= y_max < config.WINDOW_HEIGHT):
            raise ValueError(f"event option {count} has invalid vertical bounds")
        options.append((count, tuple(map(int, bounds))))
    if not options:
        raise ValueError("at least one event option is required")
    return sorted(options)


def _select_event_zone() -> tuple[int, tuple[int, int, int, int]] | None:
    try:
        options = _event_options()
    except ValueError as exc:
        print(f"\nCannot prime bot: {exc}")
        return None
    print("\nSelect the number of active Eatventure events:")
    for count, bounds in options:
        print(f"  {count}: protect x={bounds[0]}-{bounds[1]}, y={bounds[2]}-{bounds[3]}")
    choices = dict(options)
    while not should_exit.is_set():
        try:
            count = int(input("Selection: ").strip())
        except EOFError:
            return None
        except ValueError:
            count = 0
        if count in choices:
            print(
                f"Bot primed for {count} event(s). Focus '{config.WINDOW_TITLE}' "
                "and press Z again to start."
            )
            return count, choices[count]
        print(f"Enter one of: {', '.join(map(str, choices))}.")
    return None


def _toggle_bot() -> None:
    global primed_event_selection
    if bot_instance is None:
        return
    if bot_instance.running:
        bot_instance.stop()
        bot_instance.telegram.notify_bot_stopped()
        primed_event_selection = None
        logging.info("Bot stopped; press Z to prime the next run")
        return
    if primed_event_selection is None:
        selection = _select_event_zone()
        toggle_requested.clear()
        if selection is None:
            logging.warning("Bot priming cancelled")
            return
        bot_instance.set_event_forbidden_zone(selection[1])
        primed_event_selection = selection
        logging.info("Bot primed for %s active event(s)", selection[0])
        return
    if bot_instance.start():
        bot_instance.telegram.notify_bot_started()
        logging.info("Bot started")
    else:
        logging.warning(
            "Start failed; selection remains primed. Focus '%s' and press Z to retry",
            config.WINDOW_TITLE,
        )


def _log_cursor() -> None:
    if bot_instance is None or not bot_instance.window_capture.is_window_active():
        logging.info("[X] Bot window is unavailable")
        return
    screen_x, screen_y = bot_instance.mouse_controller.get_cursor_position()
    left, top, _, _ = bot_instance.window_capture.get_window_rect()
    logging.info("[X] Window position: (%s, %s)", screen_x - left, screen_y - top)


def on_press(key: Any) -> None:
    character = _key_character(key)
    if character == "z":
        if bot_instance is not None and bot_instance.running:
            bot_instance.request_stop()
        toggle_requested.set()
    elif character == "x":
        _log_cursor()
    elif character == "p":
        if bot_instance is not None:
            bot_instance.request_stop()
        should_exit.set()


def setup_logging() -> None:
    global log_listener
    logs_dir = Path(config.LOGS_DIR)
    logs_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if config.DEBUG else logging.INFO
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console = logging.StreamHandler(sys.stdout)
    logfile = RotatingFileHandler(
        logs_dir / "bot.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    for handler in (console, logfile):
        handler.setLevel(level)
        handler.setFormatter(formatter)
    records: queue.SimpleQueue[logging.LogRecord] = queue.SimpleQueue()
    root = logging.getLogger()
    root.handlers[:] = [QueueHandler(records)]
    root.setLevel(level)
    log_listener = QueueListener(records, console, logfile, respect_handler_level=True)
    log_listener.start()


def _run() -> None:
    while not should_exit.is_set():
        if toggle_requested.is_set():
            toggle_requested.clear()
            _toggle_bot()
        elif bot_instance is not None and bot_instance.running:
            bot_instance.step()
        should_exit.wait(config.EVENT_LOOP_INTERVAL)


def main() -> int:
    global bot_instance, log_listener
    listener = None
    print(f"Eatventure Bot — target: {config.WINDOW_TITLE} ({config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT})")
    try:
        setup_logging()
        should_exit.clear()
        toggle_requested.clear()
        bot_instance = EatventureBot()
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        logging.info("Z: prime/start/stop | X: cursor position | P: exit")
        _run()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception:
        logging.exception("Fatal error")
        return 1
    finally:
        if listener is not None:
            listener.stop()
            listener.join(timeout=1)
        if bot_instance is not None:
            bot_instance.close()
        if log_listener is not None:
            log_listener.stop()
            log_listener = None


if __name__ == "__main__":
    sys.exit(main())
