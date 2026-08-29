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

Window loss, focus loss, unexpected handler errors, and exhausted bounded retries safety-pause the bot. Press `Z` to resume from a fresh scan with the current event selection.

## Install

```bash
python -m pip install -r requirements.txt
```

The runtime supports Windows and Linux X11. Hyprland/Wayland is supported only by running `scrcpy` through XWayland; native Wayland capture and input are intentionally not implemented.

### Windows or native X11

```bash
scrcpy --window-title "EatventureAuto"
python main.py
```

The bot automatically resizes the client area to `WINDOW_WIDTH` × `WINDOW_HEIGHT` when it attaches, starts, and before every active step. If the window manager refuses the exact size, the bot stops before capture or input.

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
python main.py
```

## Controls

`Z` uses the same two-stage start flow as v1:

- First press: choose the number of active events and prime its forbidden zone.
- Second press while the target is focused: start.
- Press while running: stop and clear the selection.

`M` switches between Fast red-icon matching and Normal two-template consensus while stopped. Fast mode automatically retries a second missed frame in Normal mode.

`X` logs the cursor position relative to the target. `P` exits without waiting for Enter at the event-selection prompt.

## Configuration

`config.py` contains only runtime calibration: thresholds, HSV ranges, timings, positions, scrolling, and forbidden zones. `WINDOW_WIDTH` and `WINDOW_HEIGHT` are the required positive client-area dimensions. Templates live in `assets/`; logs rotate under `logs/`.

Telegram notifications are optional and queued. Set these environment variables to enable start, stop, safety-pause, and completed-level messages:

```bash
export EATVENTURE_TELEGRAM_ENABLED=true
export EATVENTURE_TELEGRAM_BOT_TOKEN=...
export EATVENTURE_TELEGRAM_CHAT_ID=...
```

## Disclaimer

Game automation may violate the game's terms and can lead to account restrictions. Use it at your own risk.

## License

See `LICENSE`.
