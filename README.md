# Eatventure Autobot

A deterministic OpenCV screen-automation bot for the mobile game *Eatventure*, driving an Android
device mirrored through [scrcpy](https://github.com/Genymobile/scrcpy). It runs a finite state
machine over masked template matching with an HSV colour gate, and sends only guarded
mouse input into a focused, exactly-sized target window.

This rewrite merges the behaviour of two earlier versions. See [validation](VALIDATION.md)
for the accepted results, verification evidence and remaining limitations.
For future AI-assisted development, start with [the project handover](Handover.md).

## Requirements

- **Python 3.12+** (developed and tested on 3.14)
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

The bot uses a 360×780 client area. On Linux, starting the bot adopts an existing scrcpy window
for the selected phone, including one with a custom title or launched with `--select-usb`.
It verifies the live server's device serial and binds by process ID. If several user mirrors
match, focus the one to use. If none matches, the bot launches its managed `EatventureAuto`
window and waits for it to open. Set `GameRecoveryConfig.device_serial` when more
than one phone is connected. The selected serial and ADB transport identify the session; reconnecting
requires manual resume. Only the bot's own scrcpy process can be restarted, and its command never
uses `--kill-adb-on-close`. The bot leaves your external scrcpy process running on shutdown or
recovery. Existing-window device verification currently uses Linux `/proc`; Windows retains the
managed-launch path.

Start the bot with the commands below; it selects the device before adopting or launching scrcpy.
See [candidate validation](VALIDATION.md) for the calibration and throughput gates. The default strategy
maxes the first station, then the last, then the intermediate stations in reverse order. Only a
visible MAX observation advances a phase; missing or unaffordable buttons do not. Normal runs save confirmed progress and recovery events. Set
`EATVENTURE_DETAILED=1` to enable per-state timings, boost sampling, candidate accounting, and
diagnostic scans. The progress report labels essential-mode records as having partial timing.

Station searches combine stable detections and rank eligible station centers by screen height.
FIRST chooses the highest center, with the leftmost winning a tie. LAST chooses the lowest
center, with the rightmost winning a tie. A stable one-pixel difference takes priority.
If a marker wobbles, stations whose measured height ranges overlap at the endpoint tie vertically.
FIRST can use a stable visible target. LAST verifies the bottom camera edge before selecting.
Two identical nonempty target sets finish localization; differing sets use a third view.
The observation interval includes detection work rather
than adding an extra delay after it. Searches retain the original down-first 3/3/6/6 sweep and
check for the station after every settled drag, even at a stationary edge. A responsive
restaurant keeps scrolling through repeated passes while servicing stats and boxes. A saved
station that cannot be located across a pass is reacquired from its phase's camera edge.
`StrategyConfig.camera_edge_checks_first=True` also requires an
edge check before FIRST.

Managed scrcpy uses the recorded test profile: always-on-top, screen-off, 360×780, no audio,
stay-awake, H.264, zero video buffering, 12M bitrate, maximum 144 FPS and maximum dimension 1024.
It binds the selected serial explicitly and keeps the shared ADB server alive on close.

**Windows**
```powershell
eatventure-autobot
```

**Linux (X11)**
```bash
eatventure-autobot
```

**Linux (Wayland, through XWayland)**
```bash
SDL_VIDEODRIVER=x11 eatventure-autobot
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
templates to agree on a spot. Station search retries the same observation in Normal mode when
Fast mode finds no usable world arrow, including when it sees only protected HUD icons.

Automatic mirror/environment recovery retains the current MAX-proven phase and discards old
coordinates. Manual resume and process restart reacquire FIRST. Reconstructing an already-MAX
FIRST after personal gameplay or a successful game relaunch remains a limitation when its built
station cannot be recognized;
the loot-box counter is never used as a restaurant identifier. For predictable ordering, start
at a restaurant Open or city Welcome screen.

The November 19 timings were accepted for this implementation: five confirmed restaurants took
46.08, 76.55, 129.21, 135.70 and 121.35 seconds (101.78-second average). A sixth used another
352.71 seconds before manual takeover, for 14 minutes 21.59 seconds overall; its completion was
not confirmed. These results do not establish a complete-city average or 48/72-hour reliability.

## Configuration

`BotConfig` (`src/eatventure_autobot/domain/config.py`) holds all calibration as validated, frozen
dataclasses: thresholds, HSV gates, timings, click targets, capture regions, scroll behaviour and
forbidden zones. The application validates these settings in its configuration factory at startup.
Call `validate_bot_config` after constructing a custom configuration.

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
| Orchestration | `runtime/`, `main.py` | `GameVision` (semantic screen reads), `EatventureBot` (handlers and step loop), and the composition root |

`runtime/bot.py` manages the bot lifecycle and dispatches states. Handlers in
`runtime/_handlers.py` perform capture and input work. The decision functions in
`state/transitions.py` accept observations and update flow state without hardware.

Ordinary state stalls reset the search flow; restaurant-arrival confirmation has its own loading
deadline. Known screen-observation failures (including persistent UNKNOWN station panels)
release input and retry localization through the same recovery path as connected-device
game/window failures, after the desktop has been idle for
60 seconds, retrying with delays capped at 15 minutes. Manual pause and phone disconnection latch
a manual-resume requirement. Programming errors stay visible and never enter automatic retries.

## Development

```bash
pytest                              # includes the live X11 window test when DISPLAY is set
pytest --ignore=tests/test_capture_linux.py  # offline, without desktop interaction
ruff check src tests tools           # lint
mypy                                # strict source type checking
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
