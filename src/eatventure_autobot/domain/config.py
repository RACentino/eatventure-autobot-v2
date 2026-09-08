"""BotConfig schema. Defaults are ported from the verified v1/v2 config.py calibration values
(v1 is the fuller, more field-tested set; v2-only keys are layered in where v1 lacked them).
Detection thresholds/HSV ranges/NMS values are provisional pending the Stage 2 empirical
detection spike (see GREENFIELD_PLAN.md decision 6) and click-target pixel positions are
provisional pending a live recalibration pass, since v1 and v2 disagree on footer layout
coordinates (different game UI snapshots)."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from eatventure_autobot.domain.errors import ConfigError
from eatventure_autobot.domain.types import HsvGate, HsvRange, Point, Zone


def _fraction(name: str, value: float) -> float:
    if not (0.0 < value <= 1.0):
        raise ConfigError(f"{name} must be in (0, 1], got {value}")
    return value


def _positive(name: str, value: float) -> float:
    if value < 0:
        raise ConfigError(f"{name} must be >= 0, got {value}")
    return value


@dataclass(frozen=True, slots=True)
class PathsConfig:
    project_root: Path
    assets_dir: Path
    logs_dir: Path

    @classmethod
    def under(cls, project_root: Path) -> "PathsConfig":
        return cls(
            project_root=project_root,
            assets_dir=project_root / "assets",
            logs_dir=project_root / "logs",
        )


@dataclass(frozen=True, slots=True)
class WindowConfig:
    title: str = "EatventureAuto"
    width: int = 360
    height: int = 780
    debug: bool = False
    show_forbidden_area: bool = False

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ConfigError(f"WindowConfig size must be positive, got {self.width}x{self.height}")


@dataclass(frozen=True, slots=True)
class ScrcpyRecoveryConfig:
    enabled: bool = True
    red_icon_delay: float = 0.144
    box_delay: float = 0.144
    upgrade_delay: float = 0.144
    action_settle_delay: float = 0.016

    def __post_init__(self) -> None:
        for name in ("red_icon_delay", "box_delay", "upgrade_delay", "action_settle_delay"):
            _positive(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class ThresholdConfig:
    match: float = 0.98
    red_icon: float = 0.94
    new_level_red_icon: float = 0.942
    stats_red_icon: float = 0.943
    upgrade_station: float = 0.910
    box: float = 0.930
    unlock: float = 0.905
    new_level: float = 0.965

    def __post_init__(self) -> None:
        for name in (
            "match",
            "red_icon",
            "new_level_red_icon",
            "stats_red_icon",
            "upgrade_station",
            "box",
            "unlock",
            "new_level",
        ):
            _fraction(name, getattr(self, name))


def _red_icon_hsv() -> HsvGate:
    return HsvGate(
        ranges=(
            HsvRange((0, 45, 120), (12, 255, 255)),
            HsvRange((166, 45, 120), (179, 255, 255)),
        ),
        min_match_ratio=0.65,
    )


@dataclass(frozen=True, slots=True)
class RedIconDetectionConfig:
    hsv: HsvGate = field(default_factory=_red_icon_hsv)
    min_matches: int = 4
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
    offset: Point = (10, 10)

    def __post_init__(self) -> None:
        if self.min_matches < 1:
            raise ConfigError(f"min_matches must be >= 1, got {self.min_matches}")
        if not self.fast_template_names:
            raise ConfigError("fast_template_names must not be empty")


def _box_hsv() -> HsvGate:
    raw_ranges = (
        ((22, 69, 170), (23, 69, 172)),
        ((21, 112, 255), (24, 114, 255)),
        ((0, 0, 254), (28, 107, 255)),
        ((16, 140, 165), (19, 149, 187)),
        ((16, 124, 187), (19, 129, 205)),
        ((23, 108, 137), (26, 120, 234)),
        ((29, 0, 254), (91, 107, 255)),
        ((16, 121, 232), (19, 127, 247)),
        ((23, 55, 210), (27, 55, 212)),
        ((14, 149, 165), (16, 158, 179)),
        ((20, 115, 235), (21, 120, 242)),
        ((23, 69, 174), (24, 70, 174)),
        ((0, 108, 0), (12, 162, 255)),
        ((23, 53, 218), (25, 54, 225)),
        ((31, 0, 240), (179, 107, 255)),
        ((23, 47, 0), (30, 51, 232)),
        ((22, 121, 137), (24, 128, 203)),
        ((10, 64, 208), (13, 79, 223)),
        ((10, 80, 192), (13, 95, 207)),
        ((10, 128, 160), (13, 143, 175)),
        ((14, 96, 192), (17, 111, 207)),
        ((14, 144, 128), (17, 159, 143)),
        ((14, 144, 144), (17, 159, 159)),
        ((14, 160, 128), (17, 175, 143)),
        ((18, 112, 208), (21, 127, 223)),
        ((18, 112, 240), (21, 127, 255)),
        ((18, 128, 240), (21, 143, 255)),
        ((22, 96, 240), (25, 111, 255)),
        ((26, 48, 176), (29, 63, 191)),
        ((26, 96, 128), (29, 111, 143)),
        ((30, 64, 240), (33, 79, 255)),
        ((46, 176, 240), (49, 191, 255)),
    )
    return HsvGate(
        ranges=tuple(HsvRange(lower, upper) for lower, upper in raw_ranges),
        min_match_ratio=0.5115,
    )


@dataclass(frozen=True, slots=True)
class BoxDetectionConfig:
    hsv: HsvGate = field(default_factory=_box_hsv)
    template_names: tuple[str, ...] = ("box1", "box2", "box3", "box4")
    nms_iou_threshold: float = 0.190
    min_matches: int = 1

    def __post_init__(self) -> None:
        _fraction("nms_iou_threshold", self.nms_iou_threshold)
        if self.min_matches < 1:
            raise ConfigError(f"min_matches must be >= 1, got {self.min_matches}")


def _upgrade_station_hsv() -> HsvGate:
    return HsvGate(
        ranges=(
            HsvRange((12, 88, 185), (29, 199, 252)),
            HsvRange((100, 135, 204), (103, 191, 255)),
        ),
        min_match_ratio=0.50,
    )


@dataclass(frozen=True, slots=True)
class UpgradeStationConfig:
    hsv: HsvGate = field(default_factory=_upgrade_station_hsv)
    search_interval: float = 0.080
    search_attempts: int = 5
    failed_searches_before_scroll: int = 3
    verify_settle_delay: float = 0.144
    verify_search_attempts: int = 4
    verify_search_interval: float = 0.080
    disappear_confirmation_count: int = 1
    hold_check_interval_min: float = 0.080
    hold_check_interval_max: float = 0.144
    hold_max_checks: int = 400
    click_hold_max_duration: float = 9.0

    def __post_init__(self) -> None:
        if self.hold_check_interval_min > self.hold_check_interval_max:
            raise ConfigError(
                "hold_check_interval_min must be <= hold_check_interval_max: "
                f"{self.hold_check_interval_min} > {self.hold_check_interval_max}"
            )
        for name in (
            "search_attempts",
            "verify_search_attempts",
            "disappear_confirmation_count",
            "hold_max_checks",
            "failed_searches_before_scroll",
        ):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be >= 1, got {getattr(self, name)}")


@dataclass(frozen=True, slots=True)
class InputTimingConfig:
    click_delay: float = 0.175
    move_delay: float = 0.016
    mouse_down_duration: float = 0.125
    mouse_up_duration: float = 0.125
    retry_count: int = 3
    retry_delay: float = 0.016
    hover_enabled: bool = False
    hover_duration: float = 0.0

    def __post_init__(self) -> None:
        if self.retry_count < 1:
            raise ConfigError(f"retry_count must be >= 1, got {self.retry_count}")
        for name in (
            "click_delay",
            "move_delay",
            "mouse_down_duration",
            "mouse_up_duration",
            "retry_delay",
            "hover_duration",
        ):
            _positive(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class FlowTimingConfig:
    upgrades_before_stats: int = 2
    state_stall_timeout_seconds: float = 9.0
    # v1 and v2 both reset to FIND_RED_ICONS on every watchdog stall, forever, with no escalation —
    # a genuinely stuck bot would self-heal-loop indefinitely. This cap is a greenfield addition
    # (decision 3): after this many CONSECUTIVE watchdog-forced resets with no real progress
    # between them, resilience/ escalates to a fail-closed stop instead of resetting again.
    max_consecutive_watchdog_resets: int = 3
    event_loop_interval: float = 0.016
    state_delay: float = 0.0
    focus_settle_delay: float = 0.016
    spam_click_duration: float = 1.5
    spam_click_delay: float = 0.016

    def __post_init__(self) -> None:
        if self.upgrades_before_stats < 1:
            raise ConfigError(
                f"upgrades_before_stats must be >= 1, got {self.upgrades_before_stats}"
            )
        if self.state_stall_timeout_seconds <= 0:
            raise ConfigError(
                f"state_stall_timeout_seconds must be > 0, got {self.state_stall_timeout_seconds}"
            )
        if self.max_consecutive_watchdog_resets < 1:
            raise ConfigError(
                "max_consecutive_watchdog_resets must be >= 1, got "
                f"{self.max_consecutive_watchdog_resets}"
            )


@dataclass(frozen=True, slots=True)
class CaptureRegionConfig:
    max_search_y: int = 660
    extended_search_y: int = 710
    upgrade_station_search_y: int = 760
    box_search_y: int = 780

    def __post_init__(self) -> None:
        names = ("max_search_y", "extended_search_y", "upgrade_station_search_y", "box_search_y")
        for name in names:
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be > 0, got {getattr(self, name)}")


@dataclass(frozen=True, slots=True)
class ClickTargetConfig:
    idle_click_pos: Point = (8, 390)
    stats_upgrade_button_pos: Point = (310, 698)
    stats_upgrade_pos: Point = (270, 304)
    scroll_start_pos: Point = (170, 380)
    new_level_button_pos: Point = (30, 692)
    level_transition_pos: Point = (174, 520)


@dataclass(frozen=True, slots=True)
class StatsUpgradeConfig:
    click_duration: float = 1.5
    click_delay: float = 0.016

    def __post_init__(self) -> None:
        _positive("click_duration", self.click_duration)
        _positive("click_delay", self.click_delay)


@dataclass(frozen=True, slots=True)
class RedIconZoneConfig:
    new_level_zone: Zone = field(default_factory=lambda: Zone(40, 60, 665, 680))
    upgrade_zone: Zone = field(default_factory=lambda: Zone(280, 310, 665, 680))


@dataclass(frozen=True, slots=True)
class LevelTransitionConfig:
    search_attempts: int = 5
    search_interval: float = 0.080
    settle_delay: float = 0.300
    confirmation_delay: float = 0.300
    secondary_settle_delay: float = 0.300
    unlock_search_attempts: int = 4
    unlock_search_interval: float = 0.300
    unlock_settle_delay: float = 0.016

    def __post_init__(self) -> None:
        for name in ("search_attempts", "unlock_search_attempts"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be >= 1, got {getattr(self, name)}")


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

    def __post_init__(self) -> None:
        if self.pixel_step <= 0:
            raise ConfigError(f"pixel_step must be > 0, got {self.pixel_step}")
        if self.max_cycles < 1:
            raise ConfigError(f"max_cycles must be >= 1, got {self.max_cycles}")


@dataclass(frozen=True, slots=True)
class TelegramConfig:
    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""
    queue_maxsize: int = 100
    close_timeout: float = 5.5

    def __post_init__(self) -> None:
        if self.enabled and not (self.bot_token and self.chat_id):
            raise ConfigError("Telegram is enabled but bot_token/chat_id are incomplete")

    @classmethod
    def from_env(cls) -> "TelegramConfig":
        enabled = os.environ.get("EATVENTURE_TELEGRAM_ENABLED", "").strip().casefold() in {
            "1",
            "true",
            "yes",
            "on",
        }
        bot_token = os.environ.get("EATVENTURE_TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.environ.get("EATVENTURE_TELEGRAM_CHAT_ID", "").strip()
        if enabled and not (bot_token and chat_id):
            return cls(enabled=False, bot_token=bot_token, chat_id=chat_id)
        return cls(enabled=enabled, bot_token=bot_token, chat_id=chat_id)


@dataclass(frozen=True, slots=True)
class ForbiddenZoneConfig:
    click_zone: Zone = field(default_factory=lambda: Zone(60, 260, 668, None))
    event_zone_options: dict[int, Zone] = field(
        default_factory=lambda: {
            1: Zone(290, 350, 93, 260),
            2: Zone(290, 350, 93, 320),
            3: Zone(290, 350, 93, 370),
        }
    )
    numbered_zones: tuple[Zone, ...] = field(
        default_factory=lambda: (
            Zone(0, 55, 50, 255),
            Zone(0, 56, 610, 667),
            Zone(150, 193, 70, 109),
            Zone(55, 285, 670, 725),
        )
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
    telegram: TelegramConfig = field(default_factory=TelegramConfig.from_env)
    forbidden_zones: ForbiddenZoneConfig = field(default_factory=ForbiddenZoneConfig)

    @classmethod
    def default(cls, project_root: Path) -> "BotConfig":
        return cls(paths=PathsConfig.under(project_root))
