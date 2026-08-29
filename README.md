# Eatventure Autobot V2

A compact OpenCV desktop bot for Eatventure in a dedicated 360×780 `scrcpy` window. Its behavior and state order mirror `eatventure-autobot-v1`, while capture, matching, input, and dispatch stay in a smaller direct runtime.

## Main flow

The bot runs eleven directly dispatched states:

1. Scan for a level transition or red icons.
2. Click one prioritized red icon and check for an unlock.
3. Search for, verify, and hold the selected upgrade station until that same station disappears.
4. Detect boxes in confidence order, then recapture and reverify each target immediately before clicking it.
5. Upgrade stats after every two station upgrades.
6. Oscillate the restaurant view after an empty pass or repeated station misses.
7. Confirm either level-transition path, wait a bounded number of times for unlock, and record the completed level.

Runtime window/capture/input faults recover in-process with capped exponential backoff; they do not clear the selected event or stop the bot. Input is released and frozen whenever the exact-title target is missing, duplicated, or unfocused. The bot never steals focus: refocus the one `EatventureAuto` window and recovery continues automatically. Transition recovery stays on its known checkpoint, and unlock waiting only retries a still-visible new-level action.

Invalid startup configuration, an unsupported platform, and any missing, corrupt, or oversized required asset are fatal. Native crashes, operating-system termination, and a permanently stuck native library call remain outside what an in-process recovery loop can handle.

## Install

```bash
python3.14 -m pip install -r requirements.txt
```

The runtime supports Windows and Linux X11. Hyprland/Wayland is supported only by running `scrcpy` through XWayland; native Wayland capture and input are intentionally not implemented.

### Windows or native X11

```bash
scrcpy --window-title "EatventureAuto"
python3.14 main.py
```

The bot automatically resizes the client area to `WINDOW_WIDTH` × `WINDOW_HEIGHT` when it attaches, starts, and before every active step. Resize checks are finite and interruptible; input remains frozen until the exact client size is available.

### Hyprland

Add this one-time window rule to the Hyprland configuration:

```lua
hl.window_rule({
  match = { class = "^scrcpy$", title = "^EatventureAuto$", xwayland = true },
  float = true,
})
```

Reload Hyprland, then launch and keep the floating `scrcpy` window focused:

```bash
SDL_VIDEODRIVER=x11 scrcpy --window-title "EatventureAuto"
python3.14 main.py
```

## Controls

`Z` uses the same two-stage start flow as v1:

- First press: choose the number of active events and prime its forbidden zone.
- Second press while the target is focused: start.
- Press while running: stop and clear the selection.

`M` switches between Fast red-icon matching and Normal two-template consensus while stopped. Fast mode automatically retries a second missed frame in Normal mode.

`X` logs the cursor position relative to the target. `P` exits without waiting for Enter at the event-selection prompt.

## Configuration

`config.py` contains runtime calibration: thresholds, HSV ranges, timings, positions, scrolling, forbidden zones, recovery backoff, heartbeat interval, and the bounded log-queue size. Startup validates these values as one aggregate report. All 23 named PNG templates in `assets/` are required; unexpected PNGs are ignored with a warning. Logs rotate under `logs/`, fall back to the console if the file cannot be opened, and emit a local health heartbeat every 300 seconds.

Telegram notifications are optional and queued. Incomplete enabled credentials warn and disable Telegram without stopping the bot. Recovery sends one incident-start message and one recovered message rather than alerting on every retry. Set these environment variables to enable notifications:

```bash
export EATVENTURE_TELEGRAM_ENABLED=true
export EATVENTURE_TELEGRAM_BOT_TOKEN=...
export EATVENTURE_TELEGRAM_CHAT_ID=...
```

## Offline reliability checks

The standard suite includes a 100,000-step mocked recovery/state soak. Local image fixtures under the ignored `test/` directory can be replayed 25 times against the fixed box, red-icon, and upgrade-station oracles:

```bash
python3.14 -m unittest -v test_reliability.py
EATVENTURE_FIXTURE_PASSES=25 python3.14 -m unittest -v test_reliability.py
```

GitHub Actions runs the offline suite on Windows and Linux with Python 3.14. Live gameplay is intentionally not part of automated verification.

## Disclaimer

Game automation may violate the game's terms and can lead to account restrictions. Use it at your own risk.

## License

See `LICENSE`.
