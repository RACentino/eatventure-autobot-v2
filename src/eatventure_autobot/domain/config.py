"""BotConfig schema. State-gating values (attempt caps, click/zone positions, capture Y-bounds) are
ported verbatim from v1's config.py by deliberate decision: this codebase's state-handler flow and
sequence follows v1's exactly. Detection-internal tuning (HSV ranges, NMS thresholds, template
offsets) stays independently calibrated from live-frame empirical testing.

Timing values are derived from the live scrcpy rig, not ported. Measured there: a new visible frame
every 35 ms (p95 94 ms, worst 152 ms); a tapped popup is visible <= 0.94 s after release and gone
<= 0.91 s after the dismissing tap; one click costs ~5 ms (12 ms p95) beyond its configured waits;
one capture+match costs 22 ms (station) to 167 ms (boxes). Operator-fixed anchors: state_delay 0,
hover/down/up bands, scroll 0.300 s x3, verify-settle band, hold max 9.0 s, stats 2.0 s/0.032 s."""

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
    # Re-scan delays after a first miss: (worst visible-frame gap) - (cheapest match of that scan),
    # so the second capture is guaranteed a fresh frame. Boxes' match already outlasts the gap.
    red_icon_delay: float = 0.115
    box_delay: float = 0.035
    upgrade_delay: float = 0.140
    # Blind settle after a red-icon click, before the one-shot unlock check: popup visible <= 0.94 s
    # after release, minus the click's own 0.16 s post-release wait (0.78 s), plus a visible-frame
    # margin, rounded to 0.800; the SEARCH poll that follows absorbs any tail beyond it.
    action_settle_delay: float = 0.800


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


@dataclass(frozen=True, slots=True)
class UpgradeStationConfig:
    hsv: HsvGate = HsvGate(
        ranges=(
            HsvRange((12, 88, 185), (29, 199, 252)),
            HsvRange((100, 135, 204), (103, 191, 255)),
        ),
        min_match_ratio=0.50,
    )
    # Poll gap for every retry loop: p95 visible-frame gap (94 ms) minus one capture+match (22 ms),
    # so consecutive attempts see distinct frames.
    search_interval: float = 0.075
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
    verify_search_interval: float = 0.075
    # Settle delay used before each of the two steps in the pre-hold verification pass: once
    # before the priming click on the stored position, and again before the single
    # verify_upgrade_station_round() check that follows it (see
    # EatventureBot._verify_upgrade_station). Ports v1's UPGRADE_STATION_VERIFY_SETTLE_DELAY.
    # This field previously existed and was removed as unread/dead config before this gap was
    # found — it is genuinely read now, so keep it wired up. Above the worst visible-frame gap
    # (152 ms) so a fresh frame always separates observation from verification; inside 0.1-0.2 s.
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
    # visible frame (35 ms, faster only re-reads the same frame), never slower than the p95 gap.
    hold_check_interval_min: float = 0.035
    hold_check_interval_max: float = 0.100
    click_hold_max_duration: float = 9.0


@dataclass(frozen=True, slots=True)
class InputTimingConfig:
    # hover/down/up sit just inside their operator bands (0.1-0.2 / 0.1-0.15 / 0.1-0.15 s), at one
    # game frame (17 ms) above the floor: the game itself registers taps down to a 0 ms press, so
    # the bands, not the game, set these. click_delay (one visible frame) + mouse_up_duration puts
    # a click's return 0.16 s after release; move_delay/retry_delay are one game frame (16.7 ms).
    click_delay: float = 0.032
    move_delay: float = 0.017
    mouse_down_duration: float = 0.117
    mouse_up_duration: float = 0.117
    retry_count: int = 3
    retry_delay: float = 0.017
    hover_enabled: bool = True
    hover_duration: float = 0.117
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
    # Must exceed the longest legitimate same-state chain (WAIT_FOR_UNLOCK: unlock_search_attempts x
    # (idle click + focus_settle_delay + capture + unlock_search_interval) ~ 5.3 s) and the longest
    # single hold (click_hold_max_duration, 9.0 s); 9.0 s clears the former by 1.7x.
    # v1 and v2 both reset to FIND_RED_ICONS on every watchdog stall, forever, with no escalation,
    # no cap, no counting — a genuinely stuck bot self-heal-loops indefinitely rather than stopping
    # and waiting for a human. resilience/watchdog.py matches that mechanism exactly, and only that
    # mechanism: an earlier greenfield-only "no progress across changing states" addition was
    # removed after live evidence showed it resetting the oscillating scroll search before it could
    # complete a widening sweep, actively preventing the progress it was meant to detect.
    state_stall_timeout_seconds: float = 9.0
    # Idle-loop wake period (also the hotkey poll): 200 Hz, so an idle bot does not spin a core.
    event_loop_interval: float = 0.005
    # After the idle click, before capturing: a dismissed popup is gone <= 0.91 s after the tap.
    focus_settle_delay: float = 0.800
    # v1's STATE_DELAY: the settle after a completed upgrade-station hold's idle click and after
    # the stats-panel button click. 0.0 in v1 (still stop-aware), separate from the settles above.
    state_delay: float = 0.0
    # How often GameVision.ensure_target_ready() is allowed to run the expensive window
    # relocate/resize query (a full window enumeration). A cheap liveness check still runs every
    # step; only the full relocate is throttled to this cadence instead of running unconditionally
    # on every single step. A relocate costs 5.6 ms (10.7 ms p95): 1.0 s keeps it near 1% overhead.
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
    click_duration: float = 2.0
    click_delay: float = 0.032
    # Independent of input_timing.mouse_down_duration/mouse_up_duration by design: this is the
    # only click path that needs to be fast enough to register many clicks inside click_duration,
    # and it must not affect every other click the bot makes. Starting point, not a measured
    # minimum. Single taps were live-verified to register down to a 0 ms press; one game frame
    # (17 ms) each keeps press and release in separate frames, so a tap cycle is ~4 frames.
    mouse_down_duration: float = 0.017
    mouse_up_duration: float = 0.017


@dataclass(frozen=True, slots=True)
class RedIconZoneConfig:
    new_level_zone: Zone = Zone(40, 60, 700, 722)
    upgrade_zone: Zone = Zone(280, 315, 700, 722)


@dataclass(frozen=True, slots=True)
class LevelTransitionConfig:
    search_attempts: int = 2
    search_interval: float = 0.075
    # Every blind post-click settle in this flow is the same modal-transition constant as
    # action_settle_delay: nothing here can be observed (no template), so each waits out the
    # measured worst-case visible latency instead of polling for it.
    settle_delay: float = 0.800
    confirmation_delay: float = 0.800
    secondary_settle_delay: float = 0.800
    # v1's real, practical give-up point for WAIT_FOR_UNLOCK: reverted from the greenfield
    # redesign's 1000 (which relied on the 9s same-state watchdog as the sole backstop instead) per
    # the decision to follow v1's state-handler flow verbatim, including its pacing.
    unlock_search_attempts: int = 4
    unlock_search_interval: float = 0.075
    unlock_settle_delay: float = 0.800
    # Restored from v1 (73f5db0/eccd810; removed in 12e397a as an incidental bundle alongside an
    # unrelated upgrade-station fix): one down-drag "verification scroll" performed before the
    # very first new-level red-icon rescan, forcing a fresh render before trusting a miss. Kept
    # here rather than on ScrollConfig by deliberate decision, despite overlapping defaults —
    # ScrollConfig is independently used by the oscillating FIND_RED_ICONS/OPEN_BOXES/
    # UPGRADE_STATS scroll and must not be conflated with this one-shot verification step. Origin
    # point reuses click_targets.scroll_start_pos; only distance/duration/settle timing are here.
    verification_scroll_distance: int = 200
    # Gesture and pause mirror ScrollConfig (same physical drag); the settle is the modal constant
    # because the capture that follows must show the post-scroll frame (visible <= 0.94 s).
    verification_scroll_duration: float = 0.300
    verification_scroll_settle_delay: float = 0.800
    verification_scroll_interval_pause: float = 0.300


@dataclass(frozen=True, slots=True)
class ScrollConfig:
    pixel_step: int = 180
    distance_ratio: float = 1.0
    max_cycles: int = 6
    increment_step: int = 1
    max_idle_pass_attempts: int = 1
    interval_pause: float = 0.300
    post_scroll_settle: float = 0.300
    duration: float = 0.300
    # Stall alert: after this many completed scrolls in a row with no box opened, upgrade held,
    # stats upgrade or level completed, log a WARNING and probe box detection on the next scan.
    # 150 idle passes is ~8-9 min (an idle pass is 3.0-3.5 s); the Sep 8-25 logs held 12 idle-scroll
    # stalls of >= 5 min (10 of them >= 9 min, longest 5.3 h) in which the bot logged nothing.
    # Repeats every multiple. 0 disables.
    stall_scrolls_before_alert: int = 150


@dataclass(frozen=True, slots=True)
class TelegramConfig:
    enabled: bool = False
    bot_token: str = ""
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
