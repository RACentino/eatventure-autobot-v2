"""Bot configuration and operator timing constraints.

Timing reset 2026-11-19: every timing was re-derived from scratch on this rig (scrcpy 5.0.1, h264,
12M, buffer 0, max-fps 144, max-size 1024, client resized to 360x780 first), not from earlier
values.
Measured inputs (live scrcpy unless noted; no purchases were made while measuring):
  F     changed-frame gap: p50 33, p95 61, p99 80, max 123 ms (n=2,563 idle; drags: max 111)
  Tpan  station tap -> panel settled, from click() return, 33 ms press: 168-265 ms (n=36);
        dismissal tap -> panel gone: 102-178 ms (n=30)
  Tstat stats button -> panel ready: 203-247 ms (n=16); red-X tap -> X gone: 130-176 ms (n=16)
  Press 0/16/33/66/100 ms all registered (60/60 station taps); press start -> panel open was
        lowest at 16-33 ms (~202 ms) vs 0 ms (~223) and 100 ms (~271)
  Drag  180 px: 0.0 s moved 0/4, 0.1 s 3/4, 0.15 s 12/12, 0.2 s 28/28; camera at rest
        172-602 ms after release. The strategy post-scroll sampler was replayed offline on 12
        recorded 0.2 s drags for each (settle, pause, interval) choice.
  Work  minimum cost between two captures: red-X check 15, Open-button try 26, CONFIRM poll 36,
        hold miss-check 40, unlock try 46, box scan 53, verify round 86, target view 90,
        sampler step 16 ms
  Logs  restaurant load, attempt -> confirmed: max 12.14 s (n=213, 0 unconfirmed); cold start
        after an adb relaunch: last popup X <= 21.6 s (n=17 relaunches)
  adb   devices <= 10, dumpsys <= 92, force-stop <= 160, screencap <= 802, monkey <= 1,341 ms
Rules: a gap between two observations that must see different frames = F max - the minimum work
between them; a gap whose repeated frame only repeats a harmless scan = 0. A blind wait before a
capture = the slowest measured settle of that UI change; before a dependent blind tap, + F max.
A deadline that ends the moment the awaited event arrives = 2 x the longest observed duration (it
costs time only when the event never comes); a window that always runs in full, and the stall
watchdog, = 1.5 x the longest legitimate duration. The active flow is strategy mode
(strategy.enabled); legacy-only caveats are noted at the fields they affect.

Operator pins: state_delay 0; hover disabled, hover_duration 0; hold cap 15 s; stats burst 1.5 s
with a 16 ms inter-click pause.
"""

from dataclasses import dataclass, field
from pathlib import Path

from eatventure_autobot.domain.types import HsvGate, HsvRange, Point, Zone


@dataclass(frozen=True, slots=True)
class PathsConfig:
    project_root: Path
    assets_dir: Path
    logs_dir: Path


@dataclass(frozen=True, slots=True)
class WindowConfig:
    title: str = "EatventureAuto"
    width: int = 360
    height: int = 780
    debug: bool = False
    # v1 gated a Windows-GDI debug overlay (ForbiddenAreaOverlay) on this flag: semi-transparent
    # red rectangles drawn over the configured forbidden click zones, for visually verifying them
    # against the live window. Deliberately not ported — it's cosmetic, off by default in v1, and
    # would require new per-platform compositing code in the capture layer for a debug-only tool.
    # The field itself is kept as a config value; no behavior is currently wired to it.
    show_forbidden_area: bool = False


@dataclass(frozen=True, slots=True)
class ScrcpyRecoveryConfig:
    enabled: bool = True
    # Rescans after a miss. 0: the scan itself (box >= 53 ms) already spans the median frame gap
    # (33 ms), and a repeated frame only repeats a harmless scan; red icons and boxes persist, so
    # the next pass sees what this one missed. A fresh-frame guarantee (F max - 53 = 70 ms) would
    # cost every empty box pass ~1.5% of run time. red_icon_delay is legacy-mode only.
    red_icon_delay: float = 0.0
    box_delay: float = 0.0
    # The second look before a hold is released must be a different frame (a false release costs
    # a full re-acquire): F max 123 - hold miss-check minimum 40 = 83 -> 0.084.
    upgrade_delay: float = 0.084
    # Station tap -> panel settled: max 265 ms after click() return (n=36, 33 ms press), so the
    # CHECK_UNLOCK read never meets a panel still animating in (an UNKNOWN read wastes a cycle).
    # The red-X pre-check before it (>= 15 ms) is extra margin. Also covers the dismissal close
    # (max 178 ms) in _dismiss_station_panel.
    action_settle_delay: float = 0.27
    # 0 disables the probe: each probe scan costs ~98 ms (station match), an early exit needs three
    # scans, and the last scan can overrun the deadline, so a 0.27 s plain sleep is faster.
    popup_probe_interval: float = 0.0
    popup_early_exit: bool = True


@dataclass(frozen=True, slots=True)
class ThresholdConfig:
    match: float = 0.98
    red_icon: float = 0.93
    new_level_red_icon: float = 0.942
    stats_red_icon: float = 0.943
    upgrade_station: float = 0.910
    # Reverted attempted 0.880 — live-frame evidence showed loosening this let shape-only matches
    # through on unrelated flat UI surfaces (counter/icons); the HSV gate below now carries
    # accuracy instead.
    box: float = 0.930
    unlock: float = 0.905
    new_level: float = 0.965
    # assets/redX.png masks the white X glyph plus 2 px of red face (249 px): a full-silhouette mask
    # let a flat red square (0.948) and a red "!" button (0.944) through. With the glyph mask the
    # true X scores >= 0.986 (captured frames, JPEG q40, noise, brightness), the best lookalike
    # 0.887 and the best real-frame false match 0.882, so 0.940 would have left >= 0.046 / 0.053.
    # The value set now is 0.980: 0.006 under the weakest true X, 0.093 over the best lookalike.
    close_button: float = 0.980


@dataclass(frozen=True, slots=True)
class RedIconDetectionConfig:
    hsv: HsvGate = HsvGate(
        ranges=(
            HsvRange((0, 45, 120), (12, 255, 255)),
            HsvRange((166, 45, 120), (179, 255, 255)),
        ),
        min_match_ratio=0.65,
    )
    min_matches: int = 2
    fast_mode_enabled: bool = True
    fast_template_names: tuple[str, ...] = ("RedIcon5",)
    normal_template_names: tuple[str, ...] = (
        "RedIcon4",
        "RedIcon5",
        "RedIcon6",
        "RedIcon8",
        "RedIcon14",
    )
    fast_min_distance: int = 30
    nms_iou_threshold: float = 0.20
    offset: Point = (10, 10)


@dataclass(frozen=True, slots=True)
class BoxDetectionConfig:
    # Reset from box1-4.png's actual alpha-masked pixel colors (hue ~10-24, orange/gold, S~100-190,
    # V~135-255), cross-checked against live-captured crate pixels (see tools/box_heatmap_test.py).
    # An earlier, wider attempt (hue 8-28 + a low-saturation highlight range) collided with other
    # warm-toned UI (order icons, counter surfaces) and produced false positives on a live frame —
    # this single consolidated range was verified against a live capture to land true-positive
    # crates at a 0.99-1.00 HSV match ratio (well clear of the 0.65 floor) while matching 0
    # candidates on the false-positive locations from the earlier attempt.
    hsv: HsvGate = HsvGate(
        ranges=(
            HsvRange((10, 65, 180), (13, 105, 255)),
            HsvRange((13, 90, 120), (15, 190, 245)),
            HsvRange((18, 90, 120), (18, 129, 245)),
            HsvRange((18, 130, 120), (18, 130, 229)),
            HsvRange((18, 130, 235), (18, 130, 245)),
            HsvRange((18, 131, 120), (18, 190, 245)),
            HsvRange((20, 115, 220), (20, 125, 245)),
            HsvRange((23, 65, 140), (30, 115, 255)),),
        min_match_ratio=0.390,
    )
    template_names: tuple[str, ...] = ("box1", "box2", "box3", "box4")
    # v1 parity (eatventure-autobot-v1/bot.py:_collect_box_candidates/handle_open_boxes): a single
    # template match is sufficient (min_matches=1, so _gather() takes the plain suppress_overlaps
    # path, not the cross-template filter_by_template_consensus gate v1 has no equivalent of),
    # merged via v1's exact suppress_overlaps(box_candidates, 0.20) IoU threshold, with v1's exact
    # hardcoded min_distance=12 per-template local-minima clustering window. The 2026-09-19 "Config
    # Changes" commit tightened all three (0.190->0.144, 1->2, unrelated but also stale at 15) with
    # no v1 basis and is the confirmed cause of reduced box-open frequency.
    nms_iou_threshold: float = 0.144
    min_matches: int = 1
    min_distance: int = 12
    # Threads for matching the box templates side by side (cv2.matchTemplate releases the GIL; the
    # four templates took ~137 ms one after another and ~50 ms on four threads, with identical
    # candidates, 2026-10-26). Scheduling only: thresholds, HSV, NMS and the frame are untouched.
    # 1 = the old sequential scan. The pool belongs to the matcher, so the only other multi-template
    # scan (red-icon normal mode, off by default: fast mode matches one template) is threaded too,
    # with the same results.
    match_threads: int = 4
    # Most boxes one scan may click; 0 = no cap. A scan with MORE clickable matches than this is not
    # a crate scene: nothing is clicked, the pass counts as idle (so the scroll, the stall probes
    # and the dead-game check see it) and a "Box flood" line is logged. Evidence, 2026-11-09: in
    # 32,138 normal passes over 13 days none clicked more than 10 boxes, while a "box storm"
    # (01:34-06:20, 0 restaurants, 35,821 clicks) had passes of 13-90 matches scoring 0.93-0.94,
    # the same as real crates, so no confidence threshold can tell them apart. Ceiling: a real
    # crate rain of more than this would be skipped too.
    max_clicks_per_pass: int = 12


@dataclass(frozen=True, slots=True)
class UpgradeStationConfig:
    hsv: HsvGate = HsvGate(
        ranges=(
            HsvRange((12, 88, 185), (29, 199, 252)),
            HsvRange((100, 135, 204), (103, 191, 255)),
        ),
        min_match_ratio=0.50,
    )
    # Retry gap (legacy-mode search): a capture + station scan takes >= 86 ms, past the p99 frame
    # gap (80 ms), so no added wait is needed for the next attempt to see a new frame.
    search_interval: float = 0.0
    search_attempts: int = 5
    failed_searches_before_scroll: int = 3
    # Dead-loop guard: after this many box-opening passes with no scroll and no completed upgrade
    # hold in between, OPEN_BOXES forces a SCROLL. Live logs (Sep 8-25) held 9 streaks of 393-5,850
    # consecutive box passes (~11 h, zero level completions) where a stuck "box" was re-clicked
    # forever and the search never scrolled. Replaying those logs (purchases approximated by
    # pauses), only ~6 of 6,344 other box-runs reached 8 and 3 of those were shorter dead loops
    # (96-224 passes), so a spurious trip costs one spare scroll, roughly 4 per 100 running hours.
    max_box_only_passes: int = 8
    # Multi-candidate scan geometry so SEARCH_UPGRADE_STATION can skip a forbidden-zone best
    # match in favor of the next candidate, matching v1's _find_upgrade_station_match exactly:
    # its own template-match dedup distance and find_all_templates()'s hardcoded NMS threshold.
    candidate_min_distance: int = 15
    candidate_nms_iou_threshold: float = 0.20
    # Relaxation applied to the match threshold on later search/verify attempts (both repos'
    # verified behavior). The two call sites that use this trigger it at different attempt
    # counts on purpose — see the comments at each site — only the value is shared here.
    threshold_relaxation: float = 0.05
    verify_search_attempts: int = 4
    # Each verify attempt (capture + station match) takes >= 86 ms, past the p99 frame gap, so
    # retries need no added wait. This 0 also clamps the hold poll below to its 1 ms floor.
    verify_search_interval: float = 0.0
    # 0: in strategy mode the panel was already read settled (CHECK_UNLOCK / SEARCH) before the
    # hold, and the two rounds are spaced by strategy.target_observation_interval. Legacy mode
    # also uses this after its priming click; there the 4-attempt round (>= 344 ms) outlasts the
    # 265 ms panel settle.
    verify_settle_delay: float = 0.0
    # Consecutive misses required before a hold treats the station as gone; debounces a single
    # flaky/transient miss so a real hold isn't cut short by one bad frame. Each tick now does a
    # single capture+check (see EatventureBot._station_disappeared), so this cross-tick count is
    # the only debounce mechanism left.
    # NOTE: v1's value (1) provides zero actual debounce — a single miss already ends the hold.
    # Reverted to 1 anyway as a deliberate, confirmed tradeoff: this codebase's state-handler flow
    # now follows v1's verbatim, including this known flakiness. Raise to >= 2 to restore real
    # debouncing if it causes trouble in practice.
    disappear_confirmation_count: int = 1
    # Live-verified (a real hold on the real device): once the held station's popup reaches its
    # "MAX" state, the relaxed-threshold match can false-positive onto an unrelated button
    # elsewhere on screen at a confidence at/above the base threshold — too high for threshold
    # tuning alone to exclude. A match farther than this from the position actually being held is
    # rejected as not the same station, regardless of where on screen it lands.
    disappear_position_tolerance_px: int = 60
    # Hold poll = clamp(verify_search_interval, min, max) = 0 -> hold_at's 1 ms floor: the release
    # follows the station's disappearance by one frame + one check. Unchanged frames short-circuit
    # (strategy mode keeps the verified view), so faster polling adds no extra matches.
    hold_check_interval_min: float = 0.0
    hold_check_interval_max: float = 0.0
    # Operator anchor: exactly 15.0 s, the longest one hold may press. It is one handler call, not
    # a repeated state, so the stall watchdog never counts it.
    click_hold_max_duration: float = 15.0


@dataclass(frozen=True, slots=True)
class InputTimingConfig:
    # click_delay 0: every click's consumer waits its own measured settle from click() return
    # (action/close/unlock settles, the stats-panel poll, the scroll sampler), so a global
    # post-click wait would only stack on top of them. It also ends drags and the stats burst.
    # move_delay 0: _set_cursor_pos reads the warp back (<= 1.55 ms, n=40) and X orders the warp
    # before the press; all 152 live bench taps ran with it at 0.
    # mouse_down_duration 0.033: one median frame. Every press length registered (60/60 station
    # taps: 6 each at 0/16/66/100 ms, 36 at 33 ms), but the
    # game answers a 0 ms press ~1.5 frames later after release (216 vs 164 ms), so press start ->
    # panel open is lowest at 16-33 ms (~202 ms) and rises at 66 ms (240) and 100 ms (271). 33 ms
    # had 36/36 taps + 30/30 dismissals register. mouse_up_duration 0: nothing needs a wait after
    # the release that its consumer does not already have. retry_delay: the warp read-back max
    # (1.55 ms) -> 0.002 before a re-warp.
    click_delay: float = 0.0
    move_delay: float = 0.0
    mouse_down_duration: float = 0.033
    mouse_up_duration: float = 0.0
    retry_count: int = 3
    retry_delay: float = 0.002
    # Operator pin: hover inactive and 0.
    hover_enabled: bool = False
    hover_duration: float = 0.0
    # Default 0 preserves the exact-equality cursor-drift check every click/press/hold path uses
    # (PynputInputController._cursor_on_target). A nonzero value absorbs DPI-scaling/driver
    # rounding noise or transient compositor jitter without silently changing verified behavior
    # for anyone who doesn't opt in. Live-observed need for this on Linux/X11 too, not just the
    # not-yet-verified Windows backend: an unattended run logged repeated "Cursor moved before
    # mouse press"/"during click" 1px-scale mismatches that aborted otherwise-good box clicks —
    # raise this if you see the same in your own logs/bot.log.
    cursor_drift_tolerance_px: int = 0


@dataclass(frozen=True, slots=True)
class FlowTimingConfig:
    upgrades_before_stats: int = 2
    # 1.5 x (the longest legitimate same-state chain + one red-X storm), rounded up to 0.5 s. The
    # longest chain is strategy FIND_RED_ICONS escalating over upgrade_station.search_attempts (5)
    # passes, each with one panel dismissal: X check 31 + preamble 46 + dismissal 971 (panel 107,
    # idle point 139, tap 44, 2 x action settle 270, 2 x re-read 124) + anchor 145 + scroll settle
    # 300 = 1.49 s -> 7.47 s (WAIT_FOR_UNLOCK, 4 tries: ~1.0 s). Storm: 6 x (X check 31 + tap 44
    # + close settle 180) = 1.53 s. (7.47 + 1.53) x 1.5 = 13.5. A trip only matters for a
    # loop that is really stuck; a hold (click_hold_max_duration) is one handler call, not a
    # repeated state, so it does not count.
    # v1 and v2 both reset to FIND_RED_ICONS on every watchdog stall, forever, with no escalation,
    # no cap, no counting — a genuinely stuck bot self-heal-loops indefinitely rather than stopping
    # and waiting for a human. resilience/watchdog.py matches that mechanism exactly, and only that
    # mechanism: an earlier greenfield-only "no progress across changing states" addition was
    # removed after live evidence showed it resetting the oscillating scroll search before it could
    # complete a widening sweep, actively preventing the progress it was meant to detect.
    state_stall_timeout_seconds: float = 13.5
    # Idle-loop wake period after every step (also the hotkey poll and the stats-panel poll
    # floor): the smallest nonzero wait. At ~2.4 steps/s it adds ~0.24% to run time; 0 would
    # busy-spin a core while the bot is stopped or waiting for the event-count prompt.
    event_loop_interval: float = 0.001
    # 0: in strategy mode CHECK_NEW_LEVEL's idle click is a no-op, so this would be pure dead time,
    # and WAIT_FOR_UNLOCK's dialog settle is carried by level_transition.settle_delay (Open click)
    # while its retries are spaced by unlock_search_interval. Legacy caveat: legacy mode's real
    # idle click is then followed by the verification drag with no settle.
    focus_settle_delay: float = 0.0
    # v1's STATE_DELAY: the settle after a completed upgrade-station hold's idle click and after
    # the stats-panel button click. Operator anchor: exactly 0.0 (as in v1; still stop-aware).
    state_delay: float = 0.0
    # How often GameVision.ensure_target_ready() is allowed to run the expensive window
    # relocate/resize query (a full window enumeration). A cheap liveness check still runs every
    # step; only the full relocate is throttled to this cadence instead of running unconditionally
    # on every single step. Measured: relocate 2.5 ms p50 / 5.2 p99 (13 windows), cheap path ~0.
    # 5.2 ms / 2.0 s = 0.26% of wall time; a window resized by hand is corrected within 2 s
    # (captures already follow a moved window on every grab).
    window_relocate_interval: float = 2.0
    # Period of the INFO "metrics" line (config fingerprint, per-state time share, levels, holds,
    # boxes) that makes levels/hour measurable from bot.log; 0 disables it, including the closing
    # line logged when the bot stops. 300 s: a crash loses at most 5 min of tallies, ~12 lines/hour,
    # and the line costs ~1 ms, so it has no bearing on speed.
    metrics_log_interval: float = 300.0


@dataclass(frozen=True, slots=True)
class CaptureRegionConfig:
    max_search_y: int = 660
    extended_search_y: int = 780
    upgrade_station_search_y: int = 760
    box_search_y: int = 780


@dataclass(frozen=True, slots=True)
class ClickTargetConfig:
    idle_click_pos: Point = (0, 725)
    stats_upgrade_button_pos: Point = (330, 750)
    stats_upgrade_pos: Point = (290, 310)
    stats_upgrade_close_pos: Point = (308, 250)
    scroll_start_pos: Point = (200, 380)
    new_level_button_pos: Point = (30, 740)
    level_transition_pos: Point = (180, 550)

@dataclass(frozen=True, slots=True)
class StatsUpgradeConfig:
    # Operator pins: a 1.5 s burst with a 16 ms pause between taps.
    click_duration: float = 1.5
    click_delay: float = 0.016
    # Added dwell only: native input still sends separate ordered press/release events. 0: a 0 ms
    # press registered (6/6 live taps), and throughput (taps per burst), not response latency, is
    # what counts here, so no dwell gets the most taps into the pinned 1.5 s.
    mouse_down_duration: float = 0.0
    mouse_up_duration: float = 0.0
    # A deadline, not a sleep: the poll returns the moment the panel is ready (203-247 ms, n=16).
    # 2 x the measured max: only a panel that never opens waits it out, and a short deadline would
    # skip a whole burst.
    panel_ready_timeout: float = 0.5


@dataclass(frozen=True, slots=True)
class RedIconZoneConfig:
    new_level_zone: Zone = Zone(40, 60, 700, 722)
    upgrade_zone: Zone = Zone(280, 315, 700, 722)


@dataclass(frozen=True, slots=True)
class LevelTransitionConfig:
    # Restaurant load, from the attempt to the 2nd confirmation: max 12.14 s (bot.log, n=213, none
    # unconfirmed). A deadline: 2 x 12.14 = 24.3 -> 24.5. It only waits this long when a transition
    # fails; giving up early on a slow load forfeits the level's credit.
    confirmation_timeout: float = 24.5
    search_attempts: int = 2
    # Retry gap for the Open button (2nd attempt, no click before it), so it reads a new frame:
    # F max 123 - minimum try 26 (capture + new-level match) = 97 -> 0.098.
    search_interval: float = 0.098
    # These dialogs could not be triggered on demand, so their settles use the slowest measured UI
    # transition on this rig (station panel 265 ms; the stats panel 247 and red X 176 are faster).
    # settle_delay: after the Open click (and the city "Okay"); it carries WAIT_FOR_UNLOCK's dialog
    # settle so the first unlock read is not an animating dialog that looks greyed (a false greyed
    # early-out ends the attempt). confirmation_delay: the footer New Level button opens the
    # dialog that the next, unverified tap at level_transition_pos needs; a miss costs a failed
    # transition (confirmation_timeout), so it gets one extra worst frame gap: 265 + 123 -> 0.39.
    # secondary_settle_delay 0: CONFIRM_RESTAURANT polls for the new restaurant itself.
    settle_delay: float = 0.27
    confirmation_delay: float = 0.39
    secondary_settle_delay: float = 0.0
    # v1's real, practical give-up point for WAIT_FOR_UNLOCK: reverted from the greenfield
    # redesign's 1000 (which relied on the 9s same-state watchdog as the sole backstop instead) per
    # the decision to follow v1's state-handler flow verbatim, including its pacing.
    unlock_search_attempts: int = 4
    # Gap between WAIT_FOR_UNLOCK tries and between CONFIRM_RESTAURANT polls, so each read is a new
    # frame (CONFIRM needs two separate sightings): F max 123 - the cheaper minimum of the two
    # (CONFIRM poll 36; unlock try 46) = 87 -> 0.088.
    unlock_search_interval: float = 0.088
    # After an Unlock click (CHECK_UNLOCK's station unlock, WAIT_FOR_UNLOCK's dialog): the slowest
    # measured UI transition, 265 ms -> 0.27 (cannot be timed without spending on an unlock).
    unlock_settle_delay: float = 0.27
    # A greyed-out Unlock button (cash or a requirement is short) reads 0.85-0.90 against the 0.905
    # gate: 83% of 6,454 misses, and the last misses before a completion read the same as the
    # earliest, so retrying cannot turn it on within the ~4 s the retries last. With this on, the
    # first try whose raw best match is at least greyed_unlock_min_conf ends the sequence, as if the
    # attempts had run out; lower matches (0.70-0.85: the dialog still animating in) keep every
    # retry. False is v1's plain unlock_search_attempts tries again.
    greyed_unlock_early_out: bool = True
    greyed_unlock_min_conf: float = 0.85
    # Restored from v1 (73f5db0/eccd810; removed in 12e397a as an incidental bundle alongside an
    # unrelated upgrade-station fix): one down-drag "verification scroll" performed before the
    # very first new-level red-icon rescan, forcing a fresh render before trusting a miss. Kept
    # here rather than on ScrollConfig by deliberate decision, despite overlapping defaults —
    # ScrollConfig is independently used by the oscillating FIND_RED_ICONS/OPEN_BOXES/
    # UPGRADE_STATS scroll and must not be conflated with this one-shot verification step. Origin
    # point reuses click_targets.scroll_start_pos; only distance/duration/settle timing are here.
    verification_scroll_distance: int = 200
    # Same reliable drag as ScrollConfig.duration (0.2 s moved 28/28; 0.0 s 0/4, 0.1 s 3/4; a drag
    # that does not move can read as a tap). What follows only reads the HUD footer badge and the
    # Open dialog, which do not depend on the camera having stopped, so the settle only has to
    # guarantee a frame rendered after the release: F max 123 -> 0.123. The pause adds nothing
    # on top of that: 0.
    verification_scroll_duration: float = 0.2
    verification_scroll_settle_delay: float = 0.123
    verification_scroll_interval_pause: float = 0.0


@dataclass(frozen=True, slots=True)
class ScrollConfig:
    pixel_step: int = 180
    distance_ratio: float = 1.0
    max_cycles: int = 2
    increment_step: int = 3
    max_idle_pass_attempts: int = 1
    # duration: the shortest tested drag >= 1.5 x F max (185 ms) that always moved the camera:
    # 0.2 s moved 28/28 (0.15 s 12/12, 0.1 s 3/4, 0.0 s 0/4). A drag that does not move reads as a
    # camera edge to the strategy sweep. The camera comes to rest 172-602 ms after the release.
    # Strategy mode (_strategy_scroll) waits post_scroll_settle + interval_pause blind, then samples
    # every strategy.target_observation_interval until two consecutive still pairs, within
    # max(12, ceil(12 x settle / interval)) samples, so the settle also sets the give-up allowance
    # (~12 x settle); running out raises the viewport error that stops the bot. Replayed on 12
    # recorded 0.2 s drags at interval 0.05, every setting below exited on a frame at rest: settle
    # 0.15 + pause 0.05 at 646 ms mean / 770 max after the release; settle 0.25-0.35 + pause 0.05
    # at 663-665 mean / 784-808 max. Settle 0 leaves only 12 samples, which ran out on 1 of 12
    # drags (9 of 12 at interval 0.034). Settle 0.3 costs ~20 ms per scroll over 0.15 and doubles
    # the allowance to 72 samples (3.6 s of waits, 6 x the slowest rest), since the camera is rarely
    # at rest before 0.5 s anyway. Pause 0.05 >= interval, so the first still pair starts at the
    # pause. The settle is also the liveness/anchor re-check pause in strategy FIND_RED_ICONS.
    # Legacy caveat (strategy.enabled False): the legacy scroll waits only settle + pause (0.35 s)
    # before the box scan, while the camera may move for up to 0.6 s.
    interval_pause: float = 0.05
    post_scroll_settle: float = 0.3
    duration: float = 0.2
    # Stall alert: after this many completed scrolls in a row with no box opened, upgrade held,
    # stats upgrade or level completed, log a WARNING and probe box detection on the next scan.
    # 150 idle passes is ~7 min in the no-work loop (FIND, UPGRADE_STATS, OPEN, SCROLL, OPEN:
    # ~3.0 s per scroll with these timings); the Sep 8-25 logs held 12 idle-scroll stalls
    # of >= 5 min (10 of them >= 9 min, longest 5.3 h) in which the bot logged nothing.
    # Repeats every multiple. 0 disables.
    stall_scrolls_before_alert: int = 150
    # Early, log-only probe: once per idle streak, at this many completed scrolls (~1.5 min), one
    # INFO line says what box AND red-icon detection saw on the next scan. It never alerts and never
    # triggers the dead-game relaunch (that stays at stall_scrolls_before_alert). The 150-scroll
    # alert can't name a stall that is stopped by hand sooner (Oct 9: stopped at ~110 scrolls, so no
    # probe data existed); healthy streaks of this length are ~3 a day. 0 disables.
    stall_probe_scrolls: int = 30


@dataclass(frozen=True, slots=True)
class GameRecoveryConfig:
    """Revive a dead game over adb. The 09-30 logs held two stalls (39 and 21 min) where the stall
    probe saw a static frame (dark, or bright and frozen; mean-abs-diff 0.06-0.86 over minutes) and
    the bot scrolled for nothing until a human stopped it."""

    enabled: bool = True
    adb_path: str = "adb"
    device_serial: str | None = None
    scrcpy_path: str = "scrcpy"
    # Strategy liveness: no thumbnail change >= static_recent_delta across FIND passes for this
    # long queues a relaunch (force-stop + cold start). static_recent_delta (3.0, unchanged) is
    # validated only at 5-minute spacing: bot.log frame_delta_5m, n=1,997, healthy p1 4.7. One busy
    # farm crossed 3.0 within 0.35 s (149/149 live references), but a calm fresh restaurant was not
    # measurable, and a false trip force-stops the game while a late trip costs minutes on a rare
    # real freeze, so the window stays at the validated 300 s. That is also above the minimum the
    # flow needs (an ignored red X: suppress 29.5 + one storm 1.53, x 1.5 = 46.5 s).
    stale_after_seconds: float = 300.0
    # The game's package is found once at start: the single installed package whose id contains
    # this (case-insensitive). Zero or several matches disable relaunch rather than guess.
    # Checked on the real phone (09-30): the game is com.hwqgrhhjfd.idlefastfood. "eatventure"
    # matches nothing there, and "food" also matches com.global.foodpanda.android.
    package_hint: str = "idlefastfood"
    # Mean-abs-diff on a 64x36 gray thumbnail between the first scan of an idle streak and the
    # scan at the stall alert (~7.5 min later). A live farm animates; measured static frames 0.86.
    static_frame_delta: float = 1.5
    # The same measure against a RECENT scan of the streak (the early-probe scan, then the previous
    # alert's scan) instead of its first one. The first scan can predate the freeze: on 2026-10-16
    # a New Level panel dimmed and stopped the screen after it (delta vs streak start 78-82, "alive"
    # on all 48 alerts, 8.7 h lost) while consecutive scans differed by 0.4-1.5. Healthy farms
    # measured >= 5.4 (n=1,663 five-minute samples, median ~37), so 3.0 keeps ~2x margin each side.
    static_recent_delta: float = 3.0
    # Length of the recovery window after the launch command: the main flow stays suspended for
    # all of it, and only the red-X rule runs (cold start + any popups); see _recover_popups.
    # From the 17 logged relaunches: the last cold-start popup X came <= 21.6 s after the relaunch
    # line (every window ended "screen changing"). 1.5 x 21.6 + close settle 0.18 + the 5 s
    # _RECOVERY_TAIL_SECONDS verdict = 37.6 -> 38.0. Runs only after a dead-game relaunch.
    relaunch_settle_seconds: float = 38.0
    # Pause between X scans when no X is on screen (recovery window), also the scrcpy-window poll
    # at startup: F max 123 - minimum X check 15 = 108 -> 0.108, so each scan reads a new frame.
    recovery_poll_seconds: float = 0.108
    # Per adb call, and the wait for a launched scrcpy window (_prepare_mirror polls for
    # command_timeout) or for one to exit. Slowest measured adb call: monkey 1,341 ms (n=20,
    # no-op package); screencap 802 ms. scrcpy's cold start could not be timed without stopping
    # the running mirror, and bot.log (2026-11-18 14:55) shows two startups in a row timing out,
    # each relaunching scrcpy, so a too-short value loops. A timeout on the per-second
    # `adb devices` check reads as a disconnect and needs a manual restart. Only a failing call
    # waits this long: 10.0 (7.5 x the slowest adb call). UNMEASURED part: scrcpy cold start.
    command_timeout: float = 10.0


@dataclass(frozen=True, slots=True)
class ResumeConfig:
    """Auto-resume after the bot stopped itself on a window or focus error. Over Sep 28-Oct 6,
    44.5 of 205 hours went to such stops followed by nothing until a person came back. Off by
    default until 2026-10-27: the check only LOGGED what it would do (shadow mode). That log showed
    that live resume alone recovers almost nothing: in 7 of 8 stops another app had the focus, and
    start() never takes the focus. So it is on now, and `raise_window` lets it bring the game
    window up once nobody has touched the PC for idle_seconds."""

    enabled: bool = True
    # Seconds between checks; also the first backoff step after a failed live attempt (then x2 per
    # failure up to 900 s). Checks only run on a cursor sample, every 1 s (_RESUME_SAMPLE_SECONDS),
    # so 1.0 is the fastest that does anything; a check's window query costs ~3 ms (0.3% duty).
    check_seconds: float = 1.0
    # No key press and no cursor movement for this long before the bot may start itself: it never
    # takes the mouse from someone who is at the PC. Not measurable: a human-factors choice. It
    # costs nothing unattended (the last input is long past, so the first check resumes); it only
    # delays a resume while someone recently used the PC. 60 s counts reading as present.
    idle_seconds: float = 60.0
    # Deprecated compatibility field; recoverable failures now retry indefinitely.
    max_attempts: int = 0
    # Bring the game window to the front when the PC has been idle for idle_seconds and the window
    # is not already ready (another app has the focus). Skipped while the session is locked.
    # False keeps the old rule: only start a window that is already in front.
    raise_window: bool = True


@dataclass(frozen=True, slots=True)
class CloseButtonConfig:
    """The red X that closes boot/event popups (the Offline Earnings dialog after a relaunch), which
    takes priority over every handler except the hard-locked New Level flow."""

    enabled: bool = True
    template_names: tuple[str, ...] = ("redX",)
    # Built from assets/redX.png's own masked pixels (coral red face hue 1-2, S 105-190, V 207-255;
    # white glyph S <= 66, V >= 238), the way the box ranges were built. On the true X the ratio is
    # >= 0.944 under stress; real-frame false matches reach 0.21, so 0.85 leaves 0.09 / 0.64.
    hsv: HsvGate = HsvGate(
        ranges=(
            HsvRange((0, 90, 180), (12, 255, 255)),
            HsvRange((166, 90, 180), (179, 255, 255)),
            HsvRange((0, 0, 225), (179, 70, 255)),
        ),
        min_match_ratio=0.85,
    )
    min_distance: int = 20
    nms_iou_threshold: float = 0.20
    # Cheap prefilter so the rule can run every step: the full masked match costs ~85 ms on a
    # 780x360 frame, a red-block scan ~4 ms. The X button is a solid red block of block_window px
    # (its on-screen size, 38): 0.93 of the window on the real X under JPEG q40 / noise / brightness
    # stress, at most 0.31 on real farm frames, so 0.60 leaves room on both sides. Only a frame
    # with such a block is handed to the template match, and only the area around the block.
    block_ranges: tuple[HsvRange, ...] = (
        HsvRange((0, 90, 180), (12, 255, 255)),
        HsvRange((166, 90, 180), (179, 255, 255)),
    )
    block_window: int = 38
    block_min_fraction: float = 0.60
    # Wait after the tap, so the X is gone before the next capture (a second tap would land on
    # the scene under it). Live: the stats panel's X at (308, 250) stopped matching 130-176 ms
    # after the click's return (n=16) -> 0.18.
    settle_delay: float = 0.18
    # Guard for absolute priority: the same spot tapped this many times in a row means the tap is
    # not closing it (or it is not an X), so the rule stands down for suppress_seconds instead of
    # starving every other behaviour. One stuck storm is 6 x (X check 31 + tap 44 + settle 180) =
    # 1.53 s; 19 x that = 29.1 -> 29.5 s keeps the rule at <= 5% of the time if it never closes.
    max_consecutive_clicks: int = 6
    suppress_seconds: float = 29.5
    # Hard-lock layer 2: after any step in a New Level state, the X rule stays off this long, so
    # whatever dialog is still open runs exactly as it does without the rule: one full UI
    # transition, the slowest measured (265 ms) -> 0.27.
    new_level_lock_seconds: float = 0.27


@dataclass(frozen=True, slots=True)
class TelegramConfig:
    enabled: bool = False
    bot_token: str = field(default="", repr=False)
    chat_id: str = ""
    queue_maxsize: int = 100
    # TelegramNotifier._send's 5 s HTTP timeout applies to the connect and to each read separately,
    # so one post can take up to 2 x 5 s; 10.0 never abandons an in-flight request.
    close_timeout: float = 10.0


@dataclass(frozen=True, slots=True)
class ForbiddenZoneConfig:
    click_zone: Zone = Zone(60, 260, 668, None)
    event_zone_options: dict[int, Zone] = field(
        default_factory=lambda: {
            1: Zone(310, 360, 70, 260),
            2: Zone(310, 360, 70, 320),
            3: Zone(310, 360, 70, 370),
            4: Zone(310, 360, 70, 340),
        }
    )
    numbered_zones: tuple[Zone, ...] = (
        Zone(0, 55, 50, 255),
        Zone(0, 56, 642, 705),
        Zone(157, 203, 48, 91),
        Zone(60, 300, 710, 766),
    )


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    # MAX/non-MAX and station positions validated with the recorded 360x780 fixtures.
    enabled: bool = True
    calibrated: bool = True
    camera_edge_checks_first: bool = False
    max_templates: tuple[str, ...] = ("stationMax", "stationMaxSmall", "stationMaxMedium")
    locked_template: str = "stationLocked"
    locked_threshold: float = 0.975  # native cash-gap controls score as low as0.97576
    locked_price_hsv: HsvGate = HsvGate((
        HsvRange((0, 180, 150), (10, 255, 255)),
        HsvRange((170, 180, 150), (179, 255, 255)),
    ), 0.01)
    panel_template: str = "stationPanel"
    unlock_panel_template: str = "stationUnlockPanel"
    marker_templates: tuple[str, ...] = (
        "stationMarkerWide", "stationMarkerTall", "stationMarkerSquare", "stationMarkerFlat",
        # ponytail: Coffee is the calibrated built landmark; add other foods after capture.
        "stationBuiltCoffee", "stationBuiltCoffeeMachine",
    )
    marker_threshold: float = 0.83
    restaurant_template: str = "restaurantStart"
    city_template: str = "cityArrival"
    city_continue_template: str = "cityOkay"
    boost_templates: tuple[str, str] = ("boost48", "boost480")
    threshold: float = 0.98
    panel_threshold: float = 0.96
    max_threshold: float = 0.90
    max_text_min_value: int = 230
    # Original 3/3/6/6 sweep, then one directed endpoint pass (up to 6 travel + 2 checks).
    # Stop as soon as the required edge/target is verified; no third oscillation cycle.
    viewport_limit: int = 26
    # Spacing of the post-scroll stillness samples, of the two pre-hold verify rounds and the
    # minimum spacing of the three target views. 0.05: two sampler steps (2 x (0.05 + 0.016)) =
    # 132 ms > F max 123, so two "still" readings in a row are never the same frame twice; the
    # verify rounds are >= 86 + 50 = 136 ms apart (always a new frame); a target view itself takes
    # >= 90 ms, so the views get no added wait. Replay result: see ScrollConfig.
    target_observation_interval: float = 0.05


@dataclass(frozen=True, slots=True)
class MeasurementConfig:
    loadout: str = "unknown"
    candidate: bool = True


@dataclass(frozen=True, slots=True)
class BotConfig:
    paths: PathsConfig
    window: WindowConfig = field(default_factory=WindowConfig)
    scrcpy_recovery: ScrcpyRecoveryConfig = field(default_factory=ScrcpyRecoveryConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    red_icon: RedIconDetectionConfig = field(default_factory=RedIconDetectionConfig)
    box: BoxDetectionConfig = field(default_factory=BoxDetectionConfig)
    upgrade_station: UpgradeStationConfig = field(default_factory=UpgradeStationConfig)
    input_timing: InputTimingConfig = field(default_factory=InputTimingConfig)
    flow_timing: FlowTimingConfig = field(default_factory=FlowTimingConfig)
    capture_regions: CaptureRegionConfig = field(default_factory=CaptureRegionConfig)
    click_targets: ClickTargetConfig = field(default_factory=ClickTargetConfig)
    stats_upgrade: StatsUpgradeConfig = field(default_factory=StatsUpgradeConfig)
    red_icon_zones: RedIconZoneConfig = field(default_factory=RedIconZoneConfig)
    level_transition: LevelTransitionConfig = field(default_factory=LevelTransitionConfig)
    scroll: ScrollConfig = field(default_factory=ScrollConfig)
    telegram: TelegramConfig = TelegramConfig()
    forbidden_zones: ForbiddenZoneConfig = field(default_factory=ForbiddenZoneConfig)
    game_recovery: GameRecoveryConfig = field(default_factory=GameRecoveryConfig)
    close_button: CloseButtonConfig = field(default_factory=CloseButtonConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
