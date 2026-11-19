"""Bot configuration and operator timing constraints.

Timing calibration evidence: analytics/hardware-2026-11-18/timing-retune-12.
At 360x780 with scrcpy 5.0.1/h264/12M/buffer0/max-fps144/max-size1024,
changed-frame gaps were 34/79/109 ms (median/p95/max, 15 s observation).
Eight station opening trials settled within 155 ms of click return. The opening
wait adds one median frame; repeated target observations wait an observed maximum
frame gap. Pre-hold verification and all input guards remain active. These are measured bounds,
not guarantees about every future frame or a proven throughput improvement.

Operator pins: state/hover delay 0; hover disabled; ordinary press/release 100-150 ms;
ordinary scroll duration/settle/pause 300 ms; hold cap 15 s; stats burst 1.5 s,
inter-click pause 16 ms. Other historical measurement notes below refer to the
Nov 10 pipeline and remain reference evidence for settings not yet reassessed.
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
    red_icon_delay: float = 0.0
    box_delay: float = 0.0
    # A missing station gets a second observation after another observed frame can arrive.
    upgrade_delay: float = 0.0
    # 155 ms observed station-opening maximum + 34 ms median changed-frame gap, rounded up.
    action_settle_delay: float = 0.0
    # A full-world probe costs more than this wait. Mandatory verification stays enabled.
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
    # Poll gap of the search loop: F95 68.2 - (capture 6.4 + station candidates 24.8) = 37.1 ->
    # 0.038, so consecutive attempts see distinct frames at 95% confidence.
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
    # Capture and matching already span a rendered-frame period on the calibrated rig.
    verify_search_interval: float = 0.0
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
    hold_check_interval_min: float = 0.0
    hold_check_interval_max: float = 0.0
    # Operator anchor: exactly 15.0 s, the longest one hold may press. It is one handler call, not
    # a repeated state, so the stall watchdog never counts it.
    click_hold_max_duration: float = 0.0


@dataclass(frozen=True, slots=True)
class InputTimingConfig:
    # Operator anchors: press/release durations are in the inclusive 0.100-0.150 s band.
    # click_delay 0.0 (owner's ruling): the waits after each click are sized from its return
    # (release + 0.101), so they carry the screen's response. A tap and the scan right after it can
    # see a closing popup (idle-tap dismissal 177 ms p50 / 199 p95 after the release); every chain
    # that ends in a scroll drag or a hold spans >= 383 ms (max dismissal 213). A delay sized to the
    # p95 dismissal (0.098) would cost ~12% of run time at ~4,600 clicks/h.
    # move_delay 0.0: _set_cursor_pos reads the cursor back and X orders the warp before the press;
    # 110/110 bench taps and a 45 min bot run registered with no drift abort. retry_delay = the
    # warp read-back p99 (4.4 ms) -> 0.005. A click costs 5.3 ms (max 7.3) beyond these waits.
    click_delay: float = 0.0
    move_delay: float = 0.0
    mouse_down_duration: float = 0.0
    mouse_up_duration: float = 0.0
    retry_count: int = 3
    retry_delay: float = 0.0
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
    upgrades_before_stats: int = 1
    # 1.5 x (the longest legitimate same-state chain + one full red-X storm), rounded up to 0.5 s,
    # with worst-case costs. CHECK_NEW_LEVEL try: idle click 0.209 + focus 0.077 + verification
    # drag 0.308 + its settle 0.625 + scan 0.089 + 2 clicks 0.418 + confirmation 0.221 + secondary
    # 0.332 = 2.28 s (WAIT_FOR_UNLOCK, 4 tries: 1.66 s); X storm: 6 x (scan 0.107 + click 0.209 +
    # settle 0.062) = 2.27 s; 4.55 s x 1.5 = 6.82 -> 7.0. Waiting costs nothing on the hot path; a
    # false trip aborts a New Level attempt. A hold (click_hold_max_duration) is one handler call,
    # not a repeated state, so it does not count.
    # v1 and v2 both reset to FIND_RED_ICONS on every watchdog stall, forever, with no escalation,
    # no cap, no counting — a genuinely stuck bot self-heal-loops indefinitely rather than stopping
    # and waiting for a human. resilience/watchdog.py matches that mechanism exactly, and only that
    # mechanism: an earlier greenfield-only "no progress across changing states" addition was
    # removed after live evidence showed it resetting the oscillating scroll search before it could
    # complete a widening sweep, actively preventing the progress it was meant to detect.
    state_stall_timeout_seconds: float = 0.0
    # Idle-loop wake period after every step (also the hotkey poll), owner's ruling for speed:
    # 1 ms adds ~0.15% to run time at ~1.5 steps/s and costs 2.3% of one core while stopped
    # (measured; 5 ms: 0.8% of a core, ~0.75% of run time). 0 would busy-spin a core.
    event_loop_interval: float = 0.0
    # After the idle click, before capturing, in the New Level flow (CHECK_NEW_LEVEL and
    # WAIT_FOR_UNLOCK). WAIT_FOR_UNLOCK's first try sees the Unlock dialog <= 46 ms after the
    # click's return (n=15, p50 39); CHECK_NEW_LEVEL's idle click changed nothing on screen
    # (21/21). 46 + F50 30 = 0.077. A dialog slower than that costs one retry (~0.4 s), not a tap.
    focus_settle_delay: float = 0.0
    # v1's STATE_DELAY: the settle after a completed upgrade-station hold's idle click and after
    # the stats-panel button click. Operator anchor: exactly 0.0 (as in v1; still stop-aware).
    state_delay: float = 0.0
    # How often GameVision.ensure_target_ready() is allowed to run the expensive window
    # relocate/resize query (a full window enumeration). A cheap liveness check still runs every
    # step; only the full relocate is throttled to this cadence instead of running unconditionally
    # on every single step. A relocate costs 3.7 ms (6.1 ms p95): 6.07 / 0.005 = 1.214 s keeps it
    # at or under 0.5% of wall time.
    window_relocate_interval: float = 0.0
    # Period of the INFO "metrics" line (config fingerprint, per-state time share, levels, holds,
    # boxes) that makes levels/hour measurable from bot.log; 0 disables it, including the closing
    # line logged when the bot stops. 300 s: a crash loses at most 5 min of tallies, ~12 lines/hour.
    metrics_log_interval: float = 0.0


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
    click_duration: float = 0.0
    click_delay: float = 0.0
    # Added dwell only: native input still sends separate ordered press/release events.
    # Real purchases registered with zero dwell; production burst delivered 62 taps in 1.504 s.
    mouse_down_duration: float = 0.0
    mouse_up_duration: float = 0.0
    # Readiness returns immediately when observed. Covers 400 ms opening + 79 ms frame gap;
    # unlike a fixed sleep, this deadline does not slow the usual 119-157 ms opening.
    panel_ready_timeout: float = 0.0


@dataclass(frozen=True, slots=True)
class RedIconZoneConfig:
    new_level_zone: Zone = Zone(40, 60, 700, 722)
    upgrade_zone: Zone = Zone(280, 315, 700, 722)


@dataclass(frozen=True, slots=True)
class LevelTransitionConfig:
    confirmation_timeout: float = 0.0
    search_attempts: int = 2
    # Poll gap (the 2nd attempt has no click before it): F95 68.2 - (capture 6.4 + new level
    # 17.0) = 44.8 -> 0.045.
    search_interval: float = 0.0
    # Post-click settles of this flow, timed in a 45 min bot run from the click's return until the
    # screen is calm again (2 frame-to-frame changes within the farm's own animation noise) + F50:
    # after the Open click 256 ms (n=10) -> 0.287; after the footer New Level button 190 ms (n=10)
    # -> 0.221; after the level-transition tap 301 ms (n=10) -> 0.332.
    settle_delay: float = 0.0
    confirmation_delay: float = 0.0
    secondary_settle_delay: float = 0.0
    # v1's real, practical give-up point for WAIT_FOR_UNLOCK: reverted from the greenfield
    # redesign's 1000 (which relied on the 9s same-state watchdog as the sole backstop instead) per
    # the decision to follow v1's state-handler flow verbatim, including its pacing.
    unlock_search_attempts: int = 4
    # Derived 0: every retry already repeats an idle click (0.209 s) plus focus_settle_delay
    # (0.077 s), longer than Fmax (195 ms), so a gap here would only add dead time.
    unlock_search_interval: float = 0.0
    # After the Unlock click (CHECK_UNLOCK) and after a completed level. The clicked button is gone
    # <= 100 ms after the click's return, with no large screen change (n=133) + F50 = 0.131. The
    # level-complete animation got no sample (no level completed in the run).
    unlock_settle_delay: float = 0.0
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
    # The 200 px drag that minimises drag + settle (scene calm after the release, max): 0.200 +
    # 0.695 beats 0.300 + 0.671 (n=6 each, every drag moved the scene). Settle: 695 - return 101
    # + F50 30 = 0.625. The pause is derived 0: the settle already ends on a calm frame.
    verification_scroll_duration: float = 0.0
    verification_scroll_settle_delay: float = 0.0
    verification_scroll_interval_pause: float = 0.0


@dataclass(frozen=True, slots=True)
class ScrollConfig:
    pixel_step: int = 180
    distance_ratio: float = 1.0
    max_cycles: int = 2
    increment_step: int = 3
    max_idle_pass_attempts: int = 1
    # Operator anchors: oscillating scroll drag, settle and pause are exactly 0.300 s each.
    # Measured: the scene is calm <= 638 ms after the drag's release (n=10); the drag returns 101 ms
    # after it, so settle + pause end at 701 ms.
    interval_pause: float = 0.0
    post_scroll_settle: float = 0.0
    duration: float = 0.0
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
    stale_after_seconds: float = 0.0
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
    # UNMEASURED: a cold start can only be timed by force-stopping the game, which the owner left
    # out of scope (the device log holds no game cold start either). Declared bound: 60 s, well
    # above the _RECOVERY_TAIL_SECONDS (5 s) verdict window plus a cold start. It runs only after
    # a dead-game relaunch, so it does not touch restaurants per hour.
    relaunch_settle_seconds: float = 0.0
    # Pause between X scans inside the recovery window when no X is on screen: F95 68.2 -
    # (capture 6.4 + red-block prefilter 3.9) = 58.0 -> 0.058.
    recovery_poll_seconds: float = 0.0
    # Per adb call. Measured: echo 50 ms (max 99), pm list packages 122 (max 139), am force-stop
    # 106 (max 134), monkey 487 (max 511; no-op package names). 5 x 0.511 s, rounded up to 0.5 s:
    # only a failing call waits this long; the margin covers a force-stop of a frozen app.
    command_timeout: float = 0.0


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
    # failure up to 900 s). A check's window query costs 4.2 ms (p95 6.6), so 100 x 6.6 ms keeps it
    # under 1% duty; the 1 s cursor-sampling period is the floor: 1.0.
    check_seconds: float = 0.0
    # No key press and no cursor movement for this long before the bot may start itself: it never
    # takes the mouse from someone who is at the PC. Not measurable: a human-factors choice (0 would
    # let the bot take the mouse and the window while a person is still working); 60 s (owner's
    # ruling) counts reading as present and does not idle for minutes after they leave.
    idle_seconds: float = 0.0
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
    # Wait after the tap, so the popup is gone before the next capture (a second tap would land on
    # the scene under the X). Measured on 111 live taps (all the stats panel's X at (308, 250)):
    # the X was already gone at the first scan, <= 31 ms after the click's return, with no large
    # screen change after it; 31 + F50 30 = 0.062. Boot/event popups did not occur in the run.
    settle_delay: float = 0.0
    # Guard for absolute priority: the same spot tapped this many times in a row means the tap is
    # not closing it (or it is not an X), so the rule stands down for suppress_seconds instead of
    # starving every other behaviour. One stuck cycle is 6 x (scan 0.107 + click 0.209 + settle
    # 0.062) = 2.27 s; 19 x that = 43.5 s keeps the rule at <= 5% of the time if it never closes.
    max_consecutive_clicks: int = 6
    suppress_seconds: float = 0.0
    # Hard-lock layer 2: after any step in a New Level state, the X rule stays off this long, so
    # whatever dialog is still open runs exactly as it does without the rule. Covers the next
    # step's idle click (press 0.101 + 7.3 ms) and a panel it closed (gone by the first scan,
    # <= 126 ms after its release, n=12) + F95 68 = 0.303. In 19 of 31 cases the panel stayed open
    # on a static screen and no X ever matched on it (all 111 X taps were the stats panel's).
    new_level_lock_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class TelegramConfig:
    enabled: bool = False
    bot_token: str = field(default="", repr=False)
    chat_id: str = ""
    queue_maxsize: int = 100
    # TelegramNotifier._send's 5 s HTTP timeout applies to the connect and to each read separately,
    # so one post can take up to 2 x 5 s; 10.0 never abandons an in-flight request.
    close_timeout: float = 0.0


@dataclass(frozen=True, slots=True)
class ForbiddenZoneConfig:
    click_zone: Zone = Zone(60, 260, 668, None)
    event_zone_options: dict[int, Zone] = field(
        default_factory=lambda: {
            1: Zone(310, 360, 70, 260),
            2: Zone(310, 360, 70, 320),
            3: Zone(310, 360, 70, 370),
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
    # Non-scroll observations: observed maximum changed-frame gap 109.4 ms, rounded up.
    # Ordinary post-scroll settle/pause remain the operator's separate 300 ms pins.
    target_observation_interval: float = 0.0


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
