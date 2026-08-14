# Eatventure Autobot V2

A small OpenCV-driven desktop bot for Eatventure running in a dedicated `scrcpy` window. It captures the target window, detects current UI targets, and advances one verified action at a time.

## Runtime flow

The bot uses direct state dispatch with these handlers:

1. Find and click one red icon from the current frame.
2. Click an unlock button when present.
3. Find, verify, and hold the upgrade station.
4. Open one box from a fresh frame.
5. Periodically upgrade stats.
6. Scroll after two complete empty search cycles.
7. Verify level completion and wait for the next unlock.

Targets are matched at their native template size with transparent masks and HSV color gates. One HSV mask is reused for each asset class per frame, and overlapping matches are removed with deterministic IoU suppression. Red-icon detection can use fast single-template matching or full multi-template consensus. There are no background trackers, learning workers, or persisted runtime state.

Recoverable window, capture, scroll, and level-transition failures are retried indefinitely with incident-level Telegram alerts. Explicit stop requests, invalid startup configuration, and unexpected state-machine failures still stop the bot safely.

## Safety

- The configured title must identify exactly one live window.
- The target window must remain active for input to be accepted.
- Every click and drag is checked against the configured window bounds and forbidden zones.
- Stop requests prevent new input and release a held left mouse button.
- Templates and log paths resolve from the project directory.

## Requirements

- Python supported by the pinned packages in `requirements.txt`
- Windows or Linux desktop supported by PyWinCtl, MSS, and pynput
- Android device connected through ADB
- `scrcpy` available on `PATH`

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start scrcpy with the configured exact title:

```bash
scrcpy --window-title "EatventureAuto"
```

Run the bot:

```bash
python main.py
```

Hotkeys:

- `Z`: start or stop automation
- `X`: log the cursor position relative to the target window
- `P`: exit

## Configuration

Runtime thresholds, timings, coordinates, and forbidden zones live in `config.py`. The bot loads these templates from `assets/`:

- `RedIcon.png`, `RedIcon2.png` through `RedIcon15.png`, and `RedIconNoBG.png`
- `upgradeStation.png`
- `newLevel.png`
- `unlock.png`
- `box1.png` through `box4.png`

Set `RED_ICON_FAST_MODE_ENABLED = True` to match only the filename stems in `RED_ICON_FAST_TEMPLATE_NAMES` (by default, `("RedIcon5",)`). Set it to `False` to use all red-icon templates and require `RED_ICON_MIN_MATCHES` distinct templates to agree on a detection.

Telegram is optional. Set `TELEGRAM_ENABLED = True` in `config.py`, then provide `EATVENTURE_TELEGRAM_BOT_TOKEN` and `EATVENTURE_TELEGRAM_CHAT_ID` through the process environment. Notifications are queued so network requests do not block state processing.

## Disclaimer

This project is for educational use. Game automation may violate the game's terms and may lead to account restrictions. Use it at your own risk.

## License

See `LICENSE`.
