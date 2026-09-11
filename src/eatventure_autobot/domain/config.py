"""BotConfig schema. Defaults are ported from the verified v1/v2 config.py calibration values
(v1 is the fuller, more field-tested set; v2-only keys are layered in where v1 lacked them).
Detection thresholds/HSV ranges/NMS values are provisional pending the Stage 2 empirical
detection spike (see GREENFIELD_PLAN.md decision 6) and click-target pixel positions are
provisional pending a live recalibration pass, since v1 and v2 disagree on footer layout
coordinates (different game UI snapshots)."""

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
    show_forbidden_area: bool = False


@dataclass(frozen=True, slots=True)
class ScrcpyRecoveryConfig:
    enabled: bool = True
    red_icon_delay: float = 0.1
    box_delay: float = 0.1
    action_settle_delay: float = 0.15


@dataclass(frozen=True, slots=True)
class ThresholdConfig:
    match: float = 0.98
    red_icon: float = 0.93
    new_level_red_icon: float = 0.942
    stats_red_icon: float = 0.943
    upgrade_station: float = 0.910
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
    hsv: HsvGate = HsvGate(
        ranges=(
            HsvRange((22, 69, 170), (23, 69, 172)),
            HsvRange((21, 112, 255), (24, 114, 255)),
            HsvRange((0, 0, 254), (28, 107, 255)),
            HsvRange((16, 140, 165), (19, 149, 187)),
            HsvRange((16, 124, 187), (19, 129, 205)),
            HsvRange((23, 108, 137), (26, 120, 234)),
            HsvRange((29, 0, 254), (91, 107, 255)),
            HsvRange((16, 121, 232), (19, 127, 247)),
            HsvRange((23, 55, 210), (27, 55, 212)),
            HsvRange((14, 149, 165), (16, 158, 179)),
            HsvRange((20, 115, 235), (21, 120, 242)),
            HsvRange((23, 69, 174), (24, 70, 174)),
            HsvRange((0, 108, 0), (12, 162, 255)),
            HsvRange((23, 53, 218), (25, 54, 225)),
            HsvRange((31, 0, 240), (179, 107, 255)),
            HsvRange((23, 47, 0), (30, 51, 232)),
            HsvRange((22, 121, 137), (24, 128, 203)),
            HsvRange((10, 64, 208), (13, 79, 223)),
            HsvRange((10, 80, 192), (13, 95, 207)),
            HsvRange((10, 128, 160), (13, 143, 175)),
            HsvRange((14, 96, 192), (17, 111, 207)),
            HsvRange((14, 144, 128), (17, 159, 143)),
            HsvRange((14, 144, 144), (17, 159, 159)),
            HsvRange((14, 160, 128), (17, 175, 143)),
            HsvRange((18, 112, 208), (21, 127, 223)),
            HsvRange((18, 112, 240), (21, 127, 255)),
            HsvRange((18, 128, 240), (21, 143, 255)),
            HsvRange((22, 96, 240), (25, 111, 255)),
            HsvRange((26, 48, 176), (29, 63, 191)),
            HsvRange((26, 96, 128), (29, 111, 143)),
            HsvRange((30, 64, 240), (33, 79, 255)),
            HsvRange((46, 176, 240), (49, 191, 255)),
        ),
        min_match_ratio=0.5115,
    )
    template_names: tuple[str, ...] = ("box1", "box2", "box3", "box4")
    nms_iou_threshold: float = 0.190
    min_matches: int = 1
    min_distance: int = 15


@dataclass(frozen=True, slots=True)
class UpgradeStationConfig:
    hsv: HsvGate = HsvGate(
        ranges=(
            HsvRange((12, 88, 185), (29, 199, 252)),
            HsvRange((100, 135, 204), (103, 191, 255)),
        ),
        min_match_ratio=0.50,
    )
    search_interval: float = 0.1
    search_attempts: int = 5
    failed_searches_before_scroll: int = 3
    # Relaxation applied to the match threshold on later search/verify attempts (both repos'
    # verified behavior). The two call sites that use this trigger it at different attempt
    # counts on purpose — see the comments at each site — only the value is shared here.
    threshold_relaxation: float = 0.05
    verify_search_attempts: int = 4
    verify_search_interval: float = 0.1
    # Settle delay used before each of the two steps in the pre-hold verification pass: once
    # before the priming click on the stored position, and again before the single
    # verify_upgrade_station_round() check that follows it (see
    # EatventureBot._verify_upgrade_station). Ports v1's UPGRADE_STATION_VERIFY_SETTLE_DELAY.
    # This field previously existed and was removed as unread/dead config before this gap was
    # found — it is genuinely read now, so keep it wired up.
    verify_settle_delay: float = 0.15
    # Consecutive misses required before a hold treats the station as gone; debounces a single
    # flaky/transient miss so a real hold isn't cut short by one bad frame. Each tick now does a
    # single capture+check (see EatventureBot._station_disappeared), so this cross-tick count is
    # the only debounce mechanism left — must be >= 2 to actually debounce anything.
    disappear_confirmation_count: int = 2
    # Live-verified (a real hold on the real device): once the held station's popup reaches its
    # "MAX" state, the relaxed-threshold match can false-positive onto an unrelated button
    # elsewhere on screen at a confidence at/above the base threshold — too high for threshold
    # tuning alone to exclude. A match farther than this from the position actually being held is
    # rejected as not the same station, regardless of where on screen it lands.
    disappear_position_tolerance_px: int = 60
    hold_check_interval_min: float = 0.05
    hold_check_interval_max: float = 0.15
    click_hold_max_duration: float = 9.0


@dataclass(frozen=True, slots=True)
class InputTimingConfig:
    click_delay: float = 0.05
    move_delay: float = 0.033
    mouse_down_duration: float = 0.14
    mouse_up_duration: float = 0.11
    retry_count: int = 3
    retry_delay: float = 0.033
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
    # Verified v1 value (STATE_STALL_TIMEOUT_SECONDS in ../eatventure-autobot-v1/config.py).
    # v1 and v2 both reset to FIND_RED_ICONS on every watchdog stall, forever, with no escalation,
    # no cap, no counting — a genuinely stuck bot self-heal-loops indefinitely rather than stopping
    # and waiting for a human. resilience/watchdog.py matches that mechanism exactly, and only that
    # mechanism: an earlier greenfield-only "no progress across changing states" addition was
    # removed after live evidence showed it resetting the oscillating scroll search before it could
    # complete a widening sweep, actively preventing the progress it was meant to detect.
    state_stall_timeout_seconds: float = 9.0
    event_loop_interval: float = 0.033
    focus_settle_delay: float = 0.05
    # How often GameVision.ensure_target_ready() is allowed to run the expensive window
    # relocate/resize query (a full window enumeration). A cheap liveness check still runs every
    # step; only the full relocate is throttled to this cadence instead of running unconditionally
    # on every single step.
    window_relocate_interval: float = 1.0


@dataclass(frozen=True, slots=True)
class CaptureRegionConfig:
    max_search_y: int = 660
    extended_search_y: int = 780
    upgrade_station_search_y: int = 760
    box_search_y: int = 780


@dataclass(frozen=True, slots=True)
class ClickTargetConfig:
    idle_click_pos: Point = (3, 390)
    stats_upgrade_button_pos: Point = (330, 750)
    stats_upgrade_pos: Point = (290, 310)
    scroll_start_pos: Point = (170, 380)
    new_level_button_pos: Point = (30, 740)
    level_transition_pos: Point = (180, 550)


@dataclass(frozen=True, slots=True)
class StatsUpgradeConfig:
    click_duration: float = 1.5
    click_delay: float = 0.016
    search_max_attempts: int = 2
    # Independent of input_timing.mouse_down_duration/mouse_up_duration by design: this is the
    # only click path that needs to be fast enough to register many clicks inside click_duration,
    # and it must not affect every other click the bot makes. Starting point, not a measured
    # minimum — live-verify clicks still register in-game before trusting this blindly.
    mouse_down_duration: float = 0.03
    mouse_up_duration: float = 0.03


@dataclass(frozen=True, slots=True)
class RedIconZoneConfig:
    new_level_zone: Zone = Zone(40, 60, 700, 722)
    upgrade_zone: Zone = Zone(280, 315, 700, 722)


@dataclass(frozen=True, slots=True)
class LevelTransitionConfig:
    search_attempts: int = 5
    search_interval: float = 0.1
    settle_delay: float = 0.3
    confirmation_delay: float = 0.15
    secondary_settle_delay: float = 0.3
    # Deliberately not a tight cap: WAIT_FOR_UNLOCK should not be the thing that gives up early.
    # The same-state watchdog (flow_timing.state_stall_timeout_seconds) is the real backstop for
    # this state and will already force a reset long before this many attempts is ever reached —
    # this value exists as a distant ceiling, not the practical "give up" mechanism. Excluded from
    # validate_bot_config's bounded-retry-budget check for exactly this reason.
    unlock_search_attempts: int = 1000
    unlock_search_interval: float = 0.1
    unlock_settle_delay: float = 0.15
    new_level_verify_max_attempts: int = 2
    # Caps for the two retry branches that previously had no attempt limit of their own and relied
    # entirely on the watchdog's same-state timeout to end a failure loop.
    check_unlock_click_max_attempts: int = 3
    new_level_click_max_attempts: int = 5


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
    # Cap for the drag-fail retry branch, which previously had no attempt limit of its own and
    # relied entirely on the watchdog's same-state timeout to end a failure loop.
    drag_max_attempts: int = 3


@dataclass(frozen=True, slots=True)
class TelegramConfig:
    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""
    queue_maxsize: int = 100
    close_timeout: float = 5.0


@dataclass(frozen=True, slots=True)
class ForbiddenZoneConfig:
    click_zone: Zone = Zone(60, 260, 668, None)
    event_zone_options: dict[int, Zone] = field(
        default_factory=lambda: {
            1: Zone(290, 350, 93, 260),
            2: Zone(290, 350, 93, 320),
            3: Zone(290, 350, 93, 370),
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
