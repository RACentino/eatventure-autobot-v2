# Eatventure Autobot V2

A compact OpenCV desktop bot for Eatventure in a dedicated 360×780 `scrcpy`
window. Its eleven-state behavior and action order mirror v1, while capture and
input remain compatible with Windows, Linux X11, and XWayland.

## Main flow

The bot directly dispatches these states:

1. `FIND_RED_ICONS`
2. `CLICK_RED_ICON`
3. `CHECK_UNLOCK`
4. `SEARCH_UPGRADE_STATION`
5. `HOLD_UPGRADE_STATION`
6. `OPEN_BOXES`
7. `UPGRADE_STATS`
8. `SCROLL`
9. `CHECK_NEW_LEVEL`
10. `TRANSITION_LEVEL`
11. `WAIT_FOR_UNLOCK`

Normal progress is red-icon scan and click, unlock check, one verified station
click and hold, then one-pass box collection. Stats are upgraded after every
two stations. Empty passes and repeated station misses use the same oscillating
scroll flow as v1. Level-complete detections preempt normal work and unlock
waiting is bounded before the bot resets to `FIND_RED_ICONS`.

The bot does not recover indefinitely. A missing, duplicated, unfocused, or
unreadable target stops the current run, releases mouse input, and resets the
state flow. The selected event count remains primed, so fix the target and
press `Z` to restart. An intentional `Z` stop clears the selection.

## Install

```bash
python3.14 -m pip install -r requirements.txt
```

### Windows

```powershell
scrcpy --window-title "EatventureAuto"
python main.py
```

### Linux X11

```bash
scrcpy --window-title "EatventureAuto"
python3 main.py
```

### Linux Wayland through XWayland

Native Wayland capture and input are not supported. Launch the target through
XWayland and keep it focused:

```bash
SDL_VIDEODRIVER=x11 scrcpy --window-title "EatventureAuto"
python3 main.py
```

The bot requires exactly one live `EatventureAuto` window and resizes its
client area to 360×780. On a tiling compositor, configure the scrcpy window as
floating so the requested client size can be applied.

## Controls

- `Z`: select active events, prime, start, retry, or stop.
- `M`: switch Fast/Normal red-icon matching while stopped.
- `X`: log the cursor position relative to the target.
- `P`: exit cleanly.

Fast mode scans `RedIcon5`. Normal mode scans `RedIcon4`, `RedIcon5`,
`RedIcon6`, `RedIcon8`, and `RedIcon14` with two-template consensus. A Fast
miss automatically retries the second frame in Normal mode. The other eleven
red-icon PNGs remain in `assets/` for manual recalibration but are not loaded.

Assets are best-effort. Missing, corrupt, or oversized runtime templates are
reported once and only the affected detection becomes unavailable. Normal-mode
consensus falls to one if only one selected red template loads; with none, red
detection returns no matches.

## Configuration and notifications

`config.py` contains only live calibration values: thresholds, HSV ranges,
timings, coordinates, scrolling, and forbidden zones. Telegram notifications
for bot start, stop, and completed levels are optional:

```bash
export EATVENTURE_TELEGRAM_ENABLED=true
export EATVENTURE_TELEGRAM_BOT_TOKEN=...
export EATVENTURE_TELEGRAM_CHAT_ID=...
```

Incomplete credentials disable Telegram with a warning. Logs rotate under
`logs/` and fall back to the console when the log file cannot be opened.

## Disclaimer

Game automation may violate the game's terms and can lead to account
restrictions. Use it at your own risk.

## License

See `LICENSE`.
