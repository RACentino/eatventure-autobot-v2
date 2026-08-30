import logging
import queue
import signal
import sys
import threading
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from typing import Any

import config
from bot import EatventureBot

bot_instance: EatventureBot | None = None
should_exit = threading.Event()
toggle_requested = threading.Event()
mode_toggle_requested = threading.Event()
log_listener: QueueListener | None = None
log_output_handlers: list[logging.Handler] = []
primed_event_selection: tuple[int, tuple[int, int, int, int]] | None = None
pressed_keys: set[str] = set()
pressed_keys_lock = threading.Lock()


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
        print(
            f"  {count}: protect x={bounds[0]}-{bounds[1]}, y={bounds[2]}-{bounds[3]}"
        )
    choices = dict(options)
    selection: queue.SimpleQueue[tuple[int, tuple[int, int, int, int]] | None] = (
        queue.SimpleQueue()
    )

    def read_selection() -> None:
        while not should_exit.is_set():
            try:
                count = int(input("Selection: ").strip())
            except EOFError:
                selection.put(None)
                return
            except ValueError:
                count = 0
            if count in choices:
                selection.put((count, choices[count]))
                return
            print(f"Enter one of: {', '.join(map(str, choices))}.")

    threading.Thread(target=read_selection, name="event_selection", daemon=True).start()
    while not should_exit.wait(config.EVENT_LOOP_INTERVAL):
        try:
            selected = selection.get_nowait()
        except queue.Empty:
            continue
        if selected is None:
            return None
        print(
            f"Bot primed for {selected[0]} event(s). Focus '{config.WINDOW_TITLE}' "
            "and press Z again to start."
        )
        return selected
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


def _toggle_red_icon_mode() -> None:
    if bot_instance is None:
        return
    if bot_instance.running:
        logging.warning("Red-icon mode can only be changed while stopped")
        return
    logging.info("Red-icon mode: %s", bot_instance.toggle_red_icon_mode())


def on_press(key: Any) -> None:
    character = _key_character(key)
    if character not in {"z", "x", "m", "p"}:
        return
    with pressed_keys_lock:
        if character in pressed_keys:
            return
        pressed_keys.add(character)
    if character == "z":
        if bot_instance is not None and bot_instance.running:
            bot_instance.request_stop()
        toggle_requested.set()
    elif character == "x":
        _log_cursor()
    elif character == "m":
        mode_toggle_requested.set()
    elif character == "p":
        if bot_instance is not None:
            bot_instance.request_stop()
        should_exit.set()


def on_release(key: Any) -> None:
    character = _key_character(key)
    if character is not None:
        with pressed_keys_lock:
            pressed_keys.discard(character)


def setup_logging() -> None:
    global log_listener, log_output_handlers
    logs_dir = Path(config.LOGS_DIR)
    level = logging.DEBUG if config.DEBUG else logging.INFO
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console = logging.StreamHandler(sys.stdout)
    outputs: list[logging.Handler] = [console]
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        outputs.append(
            RotatingFileHandler(
                logs_dir / "bot.log",
                maxBytes=5_000_000,
                backupCount=3,
                encoding="utf-8",
            )
        )
    except OSError as exc:
        print(f"Log file unavailable; using console only: {exc}", file=sys.stderr)
    for handler in outputs:
        handler.setLevel(level)
        handler.setFormatter(formatter)
    records: queue.SimpleQueue[logging.LogRecord] = queue.SimpleQueue()
    root = logging.getLogger()
    root.handlers[:] = [QueueHandler(records)]
    root.setLevel(level)
    logging.raiseExceptions = False
    log_output_handlers = outputs
    log_listener = QueueListener(records, *outputs, respect_handler_level=True)
    log_listener.start()


def shutdown_logging() -> None:
    global log_listener, log_output_handlers
    logging.getLogger().handlers.clear()
    if log_listener is not None:
        log_listener.stop()
    for handler in log_output_handlers:
        handler.close()
    log_listener = None
    log_output_handlers = []


def _run() -> None:
    while not should_exit.is_set():
        if toggle_requested.is_set():
            toggle_requested.clear()
            _toggle_bot()
        elif mode_toggle_requested.is_set():
            mode_toggle_requested.clear()
            _toggle_red_icon_mode()
        elif bot_instance is not None and bot_instance.running:
            bot_instance.step()
        should_exit.wait(config.EVENT_LOOP_INTERVAL)


def _request_exit(_signum: int, _frame: Any) -> None:
    if bot_instance is not None:
        bot_instance.request_stop()
    should_exit.set()


def main() -> int:
    global bot_instance
    listener = None
    previous_signal_handlers: dict[int, Any] = {}
    print(
        f"Eatventure Bot — target: {config.WINDOW_TITLE} ({config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT})"
    )
    try:
        setup_logging()
        should_exit.clear()
        toggle_requested.clear()
        mode_toggle_requested.clear()
        with pressed_keys_lock:
            pressed_keys.clear()
        for signal_number in (signal.SIGINT, signal.SIGTERM):
            previous_signal_handlers[signal_number] = signal.getsignal(signal_number)
            signal.signal(signal_number, _request_exit)
        try:
            from pynput import keyboard
        except Exception as exc:
            raise RuntimeError(f"Cannot initialize keyboard listener: {exc}") from exc
        bot_instance = EatventureBot()
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        logging.info(
            "Z: prime/start/stop | M: Fast/Normal (stopped) | X: cursor position | P: exit"
        )
        _run()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception:
        logging.exception("Fatal error")
        return 1
    finally:
        for signal_number, previous_handler in previous_signal_handlers.items():
            signal.signal(signal_number, previous_handler)
        if listener is not None:
            listener.stop()
            listener.join(timeout=1)
        if bot_instance is not None:
            try:
                bot_instance.close()
            except Exception:
                logging.exception("Bot cleanup failed")
            bot_instance = None
        shutdown_logging()


if __name__ == "__main__":
    sys.exit(main())
