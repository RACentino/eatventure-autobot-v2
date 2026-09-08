# Eatventure Autobot

A deterministic OpenCV screen-automation bot for the mobile game *Eatventure*, driving an Android
device mirrored through [scrcpy](https://github.com/Genymobile/scrcpy). It runs an eleven-state
finite state machine over masked template matching with an HSV colour gate, and sends only guarded
mouse input into a focused, exactly-sized target window.

This is a greenfield rewrite that merges the behaviour of two earlier versions. See
`GREENFIELD_PLAN.md` for the decisions, the verification trail, and what is and isn't proven.

## Requirements

- **Python 3.11+** (developed and tested on 3.14)
- **scrcpy** and **adb** on `PATH`
- An Android device with USB debugging enabled
- **Windows**, **Linux/X11**, or **Linux/Wayland via XWayland**

Native Wayland is deliberately out of scope: capture and global input there need portal/PipeWire
and libei paths that have not been shown to work reliably for pixel-exact template matching. On a
Wayland session, run the target under XWayland (below).

## Install

```bash
pip install -e .            # add [linux] or [windows] for the platform capture backend
pip install -e ".[linux]"   # PyWinCtl + python-xlib
pip install -e ".[windows]" # mss
```

## Running

The bot attaches to exactly one window titled `EatventureAuto` and resizes its client area to
360×780. Launch scrcpy with that title, then start the bot.

**Windows**
```powershell
scrcpy --window-title "EatventureAuto"
eatventure-autobot
```

**Linux (X11)**
```bash
scrcpy --window-title "EatventureAuto"
eatventure-autobot
```

**Linux (Wayland, through XWayland)**
```bash
SDL_VIDEODRIVER=x11 scrcpy --window-title "EatventureAuto"
eatventure-autobot
```

On a tiling compositor, make the scrcpy window floating so the 360×780 client size can be applied.

### Controls

| Key | Action |
| --- | --- |
| `Z` | Prime → start → stop. First press asks how many in-game events are active (this selects a forbidden zone protecting the event banner); focus the scrcpy window and press `Z` again to start. |
| `M` | Toggle Fast/Normal red-icon matching (only while stopped) |
| `X` | Log the cursor position relative to the target window |
| `P` | Exit cleanly |

Hotkeys are global — they work regardless of which window has focus. The bot refuses to start, and
stops itself, whenever the target window is not in the foreground.

Fast mode matches a single red-icon template; Normal mode matches several and requires multiple
templates to agree on a spot. A Fast-mode miss automatically retries the next frame in Normal mode.

## Configuration

`BotConfig` (`src/eatventure_autobot/domain/config.py`) holds all calibration as validated, frozen
dataclasses: thresholds, HSV gates, timings, click targets, capture regions, scroll behaviour and
forbidden zones. Values are validated on construction, so a bad threshold fails loudly at startup
rather than silently degrading detection.

Telegram notifications (start/stop/level-complete) are optional and credentials come only from the
environment — never commit a token:

```bash
export EATVENTURE_TELEGRAM_ENABLED=true
export EATVENTURE_TELEGRAM_BOT_TOKEN=...
export EATVENTURE_TELEGRAM_CHAT_ID=...
```

Incomplete credentials disable Telegram with a warning. Logs rotate under `logs/` and fall back to
console-only if the log file cannot be opened.

## Architecture

Built bottom-up, each layer depending only on the one beneath it:

| Layer | Package | Responsibility |
| --- | --- | --- |
| Foundation | `domain/` | `State`, types, validated `BotConfig`, error taxonomy, and the `Protocol` contracts every backend implements |
| Modules | `capture/`, `input/`, `detection/`, `notifier/`, `resilience/`, `state/` | Isolated engines: per-platform capture, `pynput` input, the OpenCV matcher, Telegram, the stall watchdog, and the pure per-state decision functions |
| Orchestration | `runtime/`, `main.py` | `GameVision` (semantic screen reads), `EatventureBot` (the eleven handlers and step loop), and the composition root |

The state machine's decision logic is deliberately pure: handlers in `runtime/bot.py` perform all
I/O, then hand an observation to a function in `state/transitions.py` that decides the next state
and updates flow counters. That split is what makes the FSM testable without a screen or a mouse.

Failure handling is bounded self-healing, then fail-closed: a state that stalls past
`state_stall_timeout_seconds` resets the search flow, but repeated resets with no real progress stop
the bot rather than looping forever.

## Development

```bash
pytest          # unit, integration and fixture-replay tests
ruff check .    # lint
ruff format .   # format
mypy            # strict type checking
```

`tests/fixtures/red_icons/` holds real 360×780 scrcpy-resolution frames replayed through the actual
detection pipeline against a verified baseline, so any change that moves a detection fails loudly.
`tests/test_capture_linux.py` exercises the real X11 backend against a live window and skips itself
where no display is available.

> **Note for external drives:** exFAT and similar filesystems do not support symlinks, so
> `python -m venv` fails inside a checkout on one. Create the virtualenv elsewhere (for example
> under `~/.venvs/`) and point it at the repo with an editable install.

## Disclaimer

For educational purposes. Automating a game may violate its terms of service and could result in
account restrictions. Use at your own risk.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
