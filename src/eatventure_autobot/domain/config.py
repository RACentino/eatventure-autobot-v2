"""BotConfig schema. State-gating values (attempt caps, click/zone positions, capture Y-bounds) are
ported verbatim from v1's config.py by deliberate decision: this codebase's state-handler flow and
sequence follows v1's exactly. Detection-internal tuning (HSV ranges, NMS thresholds, template
offsets) stays independently calibrated from live-frame empirical testing.

TIMINGS were re-derived from scratch on 2026-10-28 (a total reset, no earlier value reused) from
measurements of the live rig: scrcpy 5.0.1, h264, --video-buffer=0, 12M, --max-fps=75, the bot's
own capture/match/input classes. Measured there:
  * a new visible frame every 33 ms (p95 58, p99 83, max 116; 40 s, 1,181 frames);
  * capture 4 ms; match 22 ms (station), 33 (red icons), 54 (boxes, 4 threads), 18 (unlock),
    16 (new level); a window relocate 3 ms (p95 5); adb round trip 38-95 ms (p95 170);
  * a tapped red icon's station popup is found 123 ms after the release (p50; p95 144, max 156,
    n=70) and has stopped moving 165 ms after it (p95 210, max 216, n=30); an idle tap closes it
    in 103-109 ms (p50; max 151); one click costs 6-10 ms (max 33) beyond its configured waits.
Each field's comment gives its formula and inputs. Operator anchors (fixed by the owner, not
derived): state_delay and hover_duration 0.0; input mouse_down/up_duration in 0.10-0.15 (0.100);
scroll duration / post_scroll_settle / interval_pause 0.300; verify_settle_delay in 0.1-0.2;
click_hold_max_duration 15.0; stats click_duration 2.0 and click_delay 0.032. Waits that follow a
click are sized from the click's return (release + mouse_up_duration + click_delay = 133 ms), not
from the release. New Level dialogs, the X-popup close and the unlock animation cannot be measured
without a purchase-adjacent tap, so those waits use the operator's 0.300 s constant instead of the
measured 0.150 s popup settle."""

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
    # Re-scan delays after a first miss: p95 visible-frame gap (58 ms) minus the cheapest match of
    # that scan (red icons 33 ms, boxes 54 ms, station 22 ms), so the second capture sees a new
    # frame at 95% confidence (a stale one only repeats the first scan). A box scan already takes
    # about one gap, so its delay is a few ms.
    red_icon_delay: float = 0.025
    box_delay: float = 0.004
    upgrade_delay: float = 0.036
    # Blind settle after a red-icon click, before the one-shot unlock check. The popup has stopped
    # moving <= 216 ms after the release (max, n=30) and the click returns 133 ms after it
    # (mouse_up 0.100 + click_delay 0.033): 216 - 133 + one visible frame (33 ms) = 116 ms, plus one
    # more frame because the locked-station Unlock dialog was measured only for its first sighting
    # (<= 143 ms), not its stillness: 0.150. The SEARCH poll that follows absorbs any tail.
    action_settle_delay: float = 0.150
    # Look during that settle (EatventureBot._settle_and_watch): how often to scan for the popup
    # the click opened. A scan period is one station match (22 ms) + this = one visible frame
    # (33 ms), so consecutive scans read different frames. Must stay > 0: 0 turns the probe and its
    # early exit off and the settle is the plain blind sleep again.
    popup_probe_interval: float = 0.011
    # Measured on that current pipeline (7,577 settles, 2026-10-23..26): 82% showed the popup, first
    # seen p50 85-87 ms / p95 92-106 ms, none later than 264 ms; the other 18% never showed it
    # (a locked station), so the full settle is not needed once a popup that appeared AFTER the click has
    # held still for two scans. True ends the settle there (EatventureBot._settle_and_watch); a
    # popup already on screen at the first scan, or one still moving, keeps the full
    # action_settle_delay. False is the log-only probe of 2026-10-23 again.
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
    # Poll gap for every retry loop: p95 visible-frame gap (58 ms) minus one station capture+match
    # (22 ms), so consecutive attempts see distinct frames at 95% confidence.
    search_interval: float = 0.036
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
    # Same poll gap as search_interval; it is also the hold's poll gap (see hold_check_interval_*).
    verify_search_interval: float = 0.036
    # Settle delay used before each of the two steps in the pre-hold verification pass: once
    # before the priming click on the stored position, and again before the single
    # verify_upgrade_station_round() check that follows it (see
    # EatventureBot._verify_upgrade_station). Ports v1's UPGRADE_STATION_VERIFY_SETTLE_DELAY.
    # This field previously existed and was removed as unread/dead config before this gap was
    # found — it is genuinely read now, so keep it wired up. Two needs: before round 1 a frame newer
    # than the search frame (worst visible-frame gap 116 ms); after the priming tap (precise_click
    # returns 100 ms after the release) a popup that has stopped moving (<= 216 ms after the
    # release): 216 - 100 + one visible frame (33 ms) = 149 ms, + 11 ms margin = 0.160. Operator
    # band 0.1-0.2 s.
    verify_settle_delay: float = 0.160
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
    # The hold's poll gap is verify_search_interval clamped into this band: never faster than one
    # visible frame (33 ms median, faster only re-reads the same frame), never slower than the p95
    # gap (58 ms). The 36 ms poll gap sits inside it, so the clamp never changes it; a poll period
    # (36 ms + one 22 ms station match) is about one p95 gap.
    hold_check_interval_min: float = 0.033
    hold_check_interval_max: float = 0.058
    # Operator anchor: exactly 15.0 s, the longest one hold may press. It is one handler call, not
    # a repeated state, so the stall watchdog never counts it.
    click_hold_max_duration: float = 15.0


@dataclass(frozen=True, slots=True)
class InputTimingConfig:
    # Operator anchors: mouse_down/up_duration in 0.10-0.15 s, at the 0.100 floor (the shortest
    # click and the shortest exposure to a physical-mouse drift abort; 100+ taps at this value all
    # registered); hover_duration exactly 0.0 (a zero-length wait, so hover is inactive).
    # click_delay = one visible frame (median gap 33 ms): a click returns 133 ms after its release.
    # move_delay / retry_delay = one game frame (17 ms): _set_cursor_pos already reads the cursor
    # back, so this only covers one event-loop turn (0 failures in 100+ taps). A click costs 6-10 ms
    # (max 33) beyond these waits.
    click_delay: float = 0.033
    move_delay: float = 0.017
    mouse_down_duration: float = 0.100
    mouse_up_duration: float = 0.100
    retry_count: int = 3
    retry_delay: float = 0.017
    hover_enabled: bool = True
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
    # 1.35 x (the longest legitimate same-state chain + one full red-X storm). WAIT_FOR_UNLOCK: 3
    # repeated tries (the 4th ends the state) x (idle click 0.26 s + focus_settle_delay + capture and
    # match 0.02 s = 0.58 s) = 1.7 s; X storm: 6 taps x (0.26 s click + 0.09 s scan + close_button
    # settle 0.30 s) = 3.9 s; 5.6 s x 1.35 = 7.6 -> 8.0. A hold (click_hold_max_duration) is one
    # handler call, not a repeated state, so it does not count.
    # v1 and v2 both reset to FIND_RED_ICONS on every watchdog stall, forever, with no escalation,
    # no cap, no counting — a genuinely stuck bot self-heal-loops indefinitely rather than stopping
    # and waiting for a human. resilience/watchdog.py matches that mechanism exactly, and only that
    # mechanism: an earlier greenfield-only "no progress across changing states" addition was
    # removed after live evidence showed it resetting the oscillating scroll search before it could
    # complete a widening sweep, actively preventing the progress it was meant to detect.
    state_stall_timeout_seconds: float = 8.0
    # Idle-loop wake period after every step (also the hotkey poll): the smallest non-zero value,
    # 1 ms. 0 would busy-spin a core while the bot is stopped; 1 ms is ~0.4% of the shortest step.
    event_loop_interval: float = 0.001
    # After the idle click, before capturing, in the New Level flow (CHECK_NEW_LEVEL and
    # WAIT_FOR_UNLOCK). A popup dismissal measured <= 151 ms after the release, but the New Level
    # dialogs animate longer and cannot be measured without a purchase-adjacent tap, and a short
    # wait here costs a whole attempt, so it keeps the operator's 0.300 s constant (~0.2 s extra per
    # attempt, well under 0.5% of run time).
    focus_settle_delay: float = 0.300
    # v1's STATE_DELAY: the settle after a completed upgrade-station hold's idle click and after
    # the stats-panel button click. Operator anchor: exactly 0.0 (as in v1; still stop-aware).
    state_delay: float = 0.0
    # How often GameVision.ensure_target_ready() is allowed to run the expensive window
    # relocate/resize query (a full window enumeration). A cheap liveness check still runs every
    # step; only the full relocate is throttled to this cadence instead of running unconditionally
    # on every single step. A relocate costs 3.0 ms (5.2 ms p95): 1.0 s keeps it under 0.5% overhead.
    window_relocate_interval: float = 1.0
    # Period of the INFO "metrics" line (config fingerprint, per-state time share, levels, holds,
    # boxes) that makes levels/hour measurable from bot.log; 0 disables it, including the closing
    # line logged when the bot stops. 300 s keeps it to ~12 lines/hour.
    metrics_log_interval: float = 300.0


@dataclass(frozen=True, slots=True)
class CaptureRegionConfig:
    max_search_y: int = 660
    extended_search_y: int = 780
    upgrade_station_search_y: int = 760
    box_search_y: int = 780


@dataclass(frozen=True, slots=True)
class ClickTargetConfig:
    idle_click_pos: Point = (0, 390)
    stats_upgrade_button_pos: Point = (330, 750)
    stats_upgrade_pos: Point = (290, 310)
    scroll_start_pos: Point = (200, 380)
    new_level_button_pos: Point = (30, 740)
    level_transition_pos: Point = (180, 550)

@dataclass(frozen=True, slots=True)
class StatsUpgradeConfig:
    # Operator anchors: spam for exactly 2.0 s, 0.032 s between taps.
    click_duration: float = 2.0
    click_delay: float = 0.032
    # Independent of input_timing.mouse_down_duration/mouse_up_duration by design (the operator's
    # 0.10-0.15 band covers input_timing only): this is the only click path that needs to be fast
    # enough to register many clicks inside click_duration, and it must not affect every other click
    # the bot makes. One game frame (17 ms) each keeps press and release in separate frames: a tap
    # cycle is 0.017 + 0.017 + 0.032 = 0.066 s (~27 taps in 2.0 s). Measured: 40 of 40 single 17 ms
    # presses on a real red icon registered (press to release 18-25 ms).
    mouse_down_duration: float = 0.017
    mouse_up_duration: float = 0.017


@dataclass(frozen=True, slots=True)
class RedIconZoneConfig:
    new_level_zone: Zone = Zone(40, 60, 700, 722)
    upgrade_zone: Zone = Zone(280, 315, 700, 722)


@dataclass(frozen=True, slots=True)
class LevelTransitionConfig:
    search_attempts: int = 2
    # The same poll gap as the upgrade-station loops (the 2nd attempt has no click before it).
    search_interval: float = 0.036
    # Blind post-click settles in this flow cannot be observed (no template), and their dialogs
    # (the New Level panel, the Unlock dialog, the level-complete animation) cannot be measured
    # without a purchase-adjacent tap. They use the operator's 0.300 s constant, not the measured
    # 0.150 s station-popup settle: a short wait here costs a whole attempt, and 0.15 s extra over
    # ~10 attempts per hour is well under 0.5% of run time.
    settle_delay: float = 0.300
    confirmation_delay: float = 0.300
    secondary_settle_delay: float = 0.300
    # v1's real, practical give-up point for WAIT_FOR_UNLOCK: reverted from the greenfield
    # redesign's 1000 (which relied on the 9s same-state watchdog as the sole backstop instead) per
    # the decision to follow v1's state-handler flow verbatim, including its pacing.
    unlock_search_attempts: int = 4
    # Deliberately 0: every retry already repeats an idle click (0.26 s) plus focus_settle_delay
    # (0.30 s), which is longer than the worst visible-frame gap (116 ms), so a gap here would only
    # add dead time.
    unlock_search_interval: float = 0.0
    # After the Unlock click, and after a completed level: the purchase / level-complete animation
    # is unmeasured (see settle_delay), so the operator's 0.300 s.
    unlock_settle_delay: float = 0.300
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
    # Gesture, settle and pause all mirror ScrollConfig (the same physical drag, 0.300 s each): the
    # capture that follows must show the post-scroll frame, and this step runs about once per New
    # Level attempt (under 0.5% of run time), so there is nothing to gain from a shorter one.
    verification_scroll_duration: float = 0.300
    verification_scroll_settle_delay: float = 0.300
    verification_scroll_interval_pause: float = 0.300


@dataclass(frozen=True, slots=True)
class ScrollConfig:
    pixel_step: int = 180
    distance_ratio: float = 1.0
    max_cycles: int = 6
    increment_step: int = 1
    max_idle_pass_attempts: int = 1
    # Operator anchors: the oscillating scroll's drag duration, settle and pause are exactly 0.300 s.
    interval_pause: float = 0.300
    post_scroll_settle: float = 0.300
    duration: float = 0.300
    # Stall alert: after this many completed scrolls in a row with no box opened, upgrade held,
    # stats upgrade or level completed, log a WARNING and probe box detection on the next scan.
    # 150 idle passes is ~10.5 min in the no-work loop (FIND, UPGRADE_STATS, OPEN, SCROLL, OPEN:
    # ~4.2 s per scroll; it was ~7.5 min at 3.0 s); the Sep 8-25 logs held 12 idle-scroll stalls
    # of >= 5 min (10 of them >= 9 min, longest 5.3 h) in which the bot logged nothing.
    # Repeats every multiple. 0 disables.
    stall_scrolls_before_alert: int = 150
    # Early, log-only probe: once per idle streak, at this many completed scrolls (~2 min), one
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
    # 60 s: not re-measured (that needs a force-stop of the game); the dead-game relaunches logged
    # on 10-14 ended their 60 s window with the screen changing and 0 X taps. It runs only after a
    # relaunch, so it does not touch restaurants per hour.
    relaunch_settle_seconds: float = 60.0
    # Pause between X scans inside the recovery window when no X is on screen: the p95 visible-frame
    # gap (58 ms); a scan with no red block costs ~4 ms.
    recovery_poll_seconds: float = 0.058
    # Per adb call. Measured round trips: 38 ms (echo) to 95 ms (pm list packages, p95 170 ms); 5.0 s
    # is 29x the slowest, which leaves room for a force-stop of a frozen app and a cold launch.
    command_timeout: float = 5.0


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
    # failure up to 900 s). A check is a few ms of X queries, so 10 s keeps the wait after the PC
    # goes quiet short; it stays above the 1 s cursor-sampling period.
    check_seconds: float = 10.0
    # No key press and no cursor movement for this long before the bot may start itself: it never
    # takes the mouse from someone who is at the PC. Not measurable: a human-factors choice (0 would
    # let the bot take the mouse and the window while a person is still working); 60 s is long
    # enough that reading counts as present and short enough not to idle for minutes after they leave.
    idle_seconds: float = 60.0
    # Failed live attempts in one stop before the bot gives up and waits for a person.
    max_attempts: int = 20
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
    # Blind wait after the tap, so the popup is gone before the next capture (a second tap would land
    # on the scene under the X). Boot/event popups are bigger dialogs than the measured station
    # popup (dismissal <= 151 ms after the release), and their close animation is unmeasured, so this
    # is the operator's 0.300 s constant.
    settle_delay: float = 0.300
    # Guard for absolute priority: the same spot tapped this many times in a row means the tap is
    # not closing it (or it is not an X), so the rule stands down for suppress_seconds instead of
    # starving every other behaviour. One stuck cycle costs ~3.9 s (6 x 0.65 s), so 60 s keeps the
    # rule under 6% of the time even if it never closes.
    max_consecutive_clicks: int = 6
    suppress_seconds: float = 60.0
    # Hard-lock layer 2: after any step in a New Level state, the X rule stays off this long, so
    # whatever dialog is still open runs exactly as it does without the rule. The next step's idle
    # click dismisses it: release at ~0.12 s + a measured dismissal <= 0.15 s = ~0.27 s; 1.0 s covers
    # that nearly 4x, for the New Level dialogs that were not measured.
    new_level_lock_seconds: float = 1.0


@dataclass(frozen=True, slots=True)
class TelegramConfig:
    enabled: bool = False
    bot_token: str = field(default="", repr=False)
    chat_id: str = ""
    queue_maxsize: int = 100
    # >= the 5 s HTTP timeout in TelegramNotifier._send, so an in-flight request is never abandoned.
    close_timeout: float = 5.0


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
