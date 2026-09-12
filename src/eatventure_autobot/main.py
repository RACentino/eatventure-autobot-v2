"""Entrypoint: logging, hotkeys, signal handling, and the composition root that wires every
module together. Hotkeys match both source repos: Z primes/starts/retries/stops, M toggles
Fast/Normal red-icon matching while stopped, X logs the cursor position, P exits.
"""

import logging
import queue
import signal
import sys
import threading
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from typing import Any

from eatventure_autobot.capture import create_screen_capture
from eatventure_autobot.detection import create_template_matcher
from eatventure_autobot.domain.config import BotConfig
from eatventure_autobot.domain.types import Zone
from eatventure_autobot.input import create_input_controller
from eatventure_autobot.notifier import create_notifier
from eatventure_autobot.runtime.bot import EatventureBot
from eatventure_autobot.runtime.config_factory import build_default_config
from eatventure_autobot.runtime.vision import GameVision

logger = logging.getLogger(__name__)

HOTKEYS = {"z", "x", "m", "p"}

bot_instance: EatventureBot | None = None
should_exit = threading.Event()
toggle_requested = threading.Event()
mode_toggle_requested = threading.Event()
primed_event_count: int | None = None
_log_listener: QueueListener | None = None
_log_handlers: list[logging.Handler] = []
_pressed_keys: set[str] = set()
_pressed_keys_lock = threading.Lock()


# --- composition root ---------------------------------------------------------------------


def build_bot(config: BotConfig) -> EatventureBot:
    # One stop event shared by the bot and by every blocking wait underneath it, so a Z-stop
    # interrupts an in-flight sleep/hold immediately. Deliberately NOT the program-exit event.
    stop_event = threading.Event()
    capture = create_screen_capture(
        config.window.title, config.window.width, config.window.height, stop_event
    )
    vision = GameVision(capture, create_template_matcher(), config)
    failed = vision.load_templates()
    if failed:
        logger.warning("Templates that failed to load: %s", ", ".join(failed))

    zones = config.forbidden_zones
    static_zones = (*zones.numbered_zones, zones.click_zone)
    input_controller = create_input_controller(
        capture, config.input_timing, static_zones, stop_event
    )
    return EatventureBot(
        config, vision, input_controller, create_notifier(config.telegram), stop_event
    )


# --- hotkeys ------------------------------------------------------------------------------


def _key_character(key: Any) -> str | None:
    character = getattr(key, "char", None)
    return str(character).lower() if character is not None else None


def on_press(key: Any) -> None:
    character = _key_character(key)
    if character not in HOTKEYS:
        return
    with _pressed_keys_lock:
        if character in _pressed_keys:
            return
        _pressed_keys.add(character)
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
        with _pressed_keys_lock:
            _pressed_keys.discard(character)


def _log_cursor() -> None:
    if bot_instance is None:
        return
    position = bot_instance.cursor_position_in_window()
    if position is None:
        logger.info("[X] Target window is unavailable")
        return
    logger.info("[X] Window position: (%s, %s)", position[0], position[1])


def _toggle_red_icon_mode() -> None:
    if bot_instance is None:
        return
    if bot_instance.running:
        logger.warning("Red-icon mode can only be changed while stopped")
        return
    logger.info("Red-icon mode: %s", bot_instance.toggle_red_icon_mode())


# --- priming and start/stop ----------------------------------------------------------------


def _select_event_count(config: BotConfig) -> int | None:
    """Asks how many in-game events are active, which selects the forbidden zone protecting the
    event banner. Runs the blocking input() on a worker so P/Ctrl-C still exit promptly."""
    options = dict(sorted(config.forbidden_zones.event_zone_options.items()))
    if not options:
        print("\nCannot prime: no event zone options are configured.")
        return None
    print("\nSelect the number of active Eatventure events:")
    for count, zone in options.items():
        print(f"  {count}: protect x={zone.x_min}-{zone.x_max}, y={zone.y_min}-{zone.y_max}")

    selection: queue.SimpleQueue[int | None] = queue.SimpleQueue()

    def read_selection() -> None:
        while not should_exit.is_set():
            try:
                raw = input("Selection: ").strip()
            except EOFError:
                selection.put(None)
                return
            try:
                count = int(raw)
            except ValueError:
                count = 0
            if count in options:
                selection.put(count)
                return
            print(f"Enter one of: {', '.join(map(str, options))}.")

    threading.Thread(target=read_selection, name="event_selection", daemon=True).start()
    while not should_exit.wait(config.flow_timing.event_loop_interval):
        try:
            return selection.get_nowait()
        except queue.Empty:
            continue
    return None


def _toggle_bot(config: BotConfig) -> None:
    global primed_event_count
    if bot_instance is None:
        return
    if bot_instance.running:
        bot_instance.stop()
        # v1's exact (asymmetric) behavior: priming is cleared only on this manual-stop path.
        # An internal auto-stop (watchdog escalation, foreground loss, an unhandled exception)
        # leaves the prior event-zone selection primed, so the next Z press restarts instantly
        # without re-prompting — including v1's own unclosed race where on_press("z")'s
        # request_stop() and toggle_requested.set() are two separate statements, so step() can
        # self-stop in the gap before this function ever runs for that press.
        primed_event_count = None
        logger.info("Bot stopped; press Z to prime the next run")
        return

    if primed_event_count is None:
        count = _select_event_count(config)
        toggle_requested.clear()
        if count is None:
            logger.warning("Bot priming cancelled")
            return
        zone: Zone = config.forbidden_zones.event_zone_options[count]
        bot_instance.set_event_forbidden_zone(zone)
        primed_event_count = count
        print(
            f"Bot primed for {count} event(s). Focus '{config.window.title}' "
            "and press Z again to start."
        )
        return

    if bot_instance.start():
        logger.info("Bot started")
    else:
        logger.warning(
            "Start failed; selection stays primed. Focus '%s' and press Z to retry",
            config.window.title,
        )


# --- logging -------------------------------------------------------------------------------


def setup_logging(config: BotConfig) -> None:
    global _log_listener, _log_handlers
    level = logging.DEBUG if config.window.debug else logging.INFO
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    outputs: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        logs_dir = Path(config.paths.logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        outputs.append(
            RotatingFileHandler(
                logs_dir / "bot.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
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
    _log_handlers = outputs
    _log_listener = QueueListener(records, *outputs, respect_handler_level=True)
    _log_listener.start()


def shutdown_logging() -> None:
    global _log_listener, _log_handlers
    logging.getLogger().handlers.clear()
    if _log_listener is not None:
        _log_listener.stop()
    for handler in _log_handlers:
        handler.close()
    _log_listener = None
    _log_handlers = []


# --- run loop ------------------------------------------------------------------------------


def _run(config: BotConfig) -> None:
    while not should_exit.is_set():
        if toggle_requested.is_set():
            toggle_requested.clear()
            _toggle_bot(config)
        elif mode_toggle_requested.is_set():
            mode_toggle_requested.clear()
            _toggle_red_icon_mode()
        elif bot_instance is not None and bot_instance.running:
            bot_instance.step()
        should_exit.wait(config.flow_timing.event_loop_interval)


def _request_exit(_signum: int, _frame: Any) -> None:
    if bot_instance is not None:
        bot_instance.request_stop()
    should_exit.set()


def main() -> int:
    global bot_instance
    config = build_default_config(Path(__file__).resolve().parent.parent.parent)
    listener = None
    previous_handlers: dict[int, Any] = {}
    print(
        f"Eatventure Autobot — target: {config.window.title} "
        f"({config.window.width}x{config.window.height})"
    )
    try:
        setup_logging(config)
        should_exit.clear()
        toggle_requested.clear()
        mode_toggle_requested.clear()
        with _pressed_keys_lock:
            _pressed_keys.clear()
        for signal_number in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signal_number] = signal.getsignal(signal_number)
            signal.signal(signal_number, _request_exit)

        try:
            from pynput import keyboard
        except Exception as exc:
            raise RuntimeError(f"Cannot initialize keyboard listener: {exc}") from exc

        bot_instance = build_bot(config)
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        logger.info("Z: prime/start/stop | M: Fast/Normal (stopped) | X: cursor | P: exit")
        _run(config)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception:
        logging.exception("Fatal error")
        return 1
    finally:
        for restored_signal, previous_handler in previous_handlers.items():
            signal.signal(restored_signal, previous_handler)
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
