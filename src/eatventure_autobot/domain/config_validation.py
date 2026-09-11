"""Validation for BotConfig and its sections. config.py holds only literal defaults; every check
that used to live in a dataclass __post_init__ lives here instead, called explicitly by whoever
constructs a config (see runtime/config_factory.py for the app's own construction path)."""

from eatventure_autobot.domain.config import (
    BotConfig,
    BoxDetectionConfig,
    CaptureRegionConfig,
    FlowTimingConfig,
    InputTimingConfig,
    LevelTransitionConfig,
    RedIconDetectionConfig,
    ScrcpyRecoveryConfig,
    ScrollConfig,
    StatsUpgradeConfig,
    TelegramConfig,
    ThresholdConfig,
    UpgradeStationConfig,
    WindowConfig,
)
from eatventure_autobot.domain.errors import ConfigError


def _fraction(name: str, value: float) -> None:
    if not (0.0 < value <= 1.0):
        raise ConfigError(f"{name} must be in (0, 1], got {value}")


def _positive(name: str, value: float) -> None:
    if value < 0:
        raise ConfigError(f"{name} must be >= 0, got {value}")


def validate_window(config: WindowConfig) -> None:
    if config.width <= 0 or config.height <= 0:
        raise ConfigError(f"WindowConfig size must be positive, got {config.width}x{config.height}")


def validate_scrcpy_recovery(config: ScrcpyRecoveryConfig) -> None:
    for name in ("red_icon_delay", "box_delay", "upgrade_delay", "action_settle_delay"):
        _positive(name, getattr(config, name))


def validate_thresholds(config: ThresholdConfig) -> None:
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
        _fraction(name, getattr(config, name))


def validate_red_icon_detection(config: RedIconDetectionConfig) -> None:
    if config.min_matches < 1:
        raise ConfigError(f"min_matches must be >= 1, got {config.min_matches}")
    if not config.fast_template_names:
        raise ConfigError("fast_template_names must not be empty")
    _fraction("nms_iou_threshold", config.nms_iou_threshold)


def validate_box_detection(config: BoxDetectionConfig) -> None:
    _fraction("nms_iou_threshold", config.nms_iou_threshold)
    if config.min_matches < 1:
        raise ConfigError(f"min_matches must be >= 1, got {config.min_matches}")
    if config.min_distance < 1:
        raise ConfigError(f"min_distance must be >= 1, got {config.min_distance}")


def validate_upgrade_station(config: UpgradeStationConfig) -> None:
    if config.hold_check_interval_min > config.hold_check_interval_max:
        raise ConfigError(
            "hold_check_interval_min must be <= hold_check_interval_max: "
            f"{config.hold_check_interval_min} > {config.hold_check_interval_max}"
        )
    _fraction("threshold_relaxation", config.threshold_relaxation)
    _positive("verify_settle_delay", config.verify_settle_delay)
    if config.disappear_position_tolerance_px < 0:
        raise ConfigError(
            "disappear_position_tolerance_px must be >= 0, got "
            f"{config.disappear_position_tolerance_px}"
        )
    for name in (
        "search_attempts",
        "verify_search_attempts",
        "disappear_confirmation_count",
        "failed_searches_before_scroll",
    ):
        if getattr(config, name) < 1:
            raise ConfigError(f"{name} must be >= 1, got {getattr(config, name)}")


def validate_input_timing(config: InputTimingConfig) -> None:
    if config.retry_count < 1:
        raise ConfigError(f"retry_count must be >= 1, got {config.retry_count}")
    if config.cursor_drift_tolerance_px < 0:
        raise ConfigError(
            f"cursor_drift_tolerance_px must be >= 0, got {config.cursor_drift_tolerance_px}"
        )
    for name in (
        "click_delay",
        "move_delay",
        "mouse_down_duration",
        "mouse_up_duration",
        "retry_delay",
        "hover_duration",
    ):
        _positive(name, getattr(config, name))


def validate_flow_timing(config: FlowTimingConfig) -> None:
    if config.upgrades_before_stats < 1:
        raise ConfigError(
            f"upgrades_before_stats must be >= 1, got {config.upgrades_before_stats}"
        )
    if config.state_stall_timeout_seconds < 0:
        raise ConfigError(
            f"state_stall_timeout_seconds must be >= 0, got {config.state_stall_timeout_seconds}"
        )
    _positive("window_relocate_interval", config.window_relocate_interval)


def validate_capture_regions(config: CaptureRegionConfig) -> None:
    names = ("max_search_y", "extended_search_y", "upgrade_station_search_y", "box_search_y")
    for name in names:
        if getattr(config, name) <= 0:
            raise ConfigError(f"{name} must be > 0, got {getattr(config, name)}")


def validate_stats_upgrade(config: StatsUpgradeConfig) -> None:
    _positive("click_duration", config.click_duration)
    _positive("click_delay", config.click_delay)
    _positive("mouse_down_duration", config.mouse_down_duration)
    _positive("mouse_up_duration", config.mouse_up_duration)


def validate_level_transition(config: LevelTransitionConfig) -> None:
    for name in ("search_attempts", "unlock_search_attempts"):
        if getattr(config, name) < 1:
            raise ConfigError(f"{name} must be >= 1, got {getattr(config, name)}")


def validate_scroll(config: ScrollConfig) -> None:
    if config.pixel_step <= 0:
        raise ConfigError(f"pixel_step must be > 0, got {config.pixel_step}")
    if config.max_cycles < 1:
        raise ConfigError(f"max_cycles must be >= 1, got {config.max_cycles}")


def validate_telegram(config: TelegramConfig) -> None:
    if config.enabled and not (config.bot_token and config.chat_id):
        raise ConfigError("Telegram is enabled but bot_token/chat_id are incomplete")


def validate_bot_config(config: BotConfig) -> None:
    """Validates every section of config, then the cross-section checks a single section's own
    validator can't make: a config typo here wouldn't fail loudly at startup, it would just look
    like erratic bot behavior later."""
    validate_window(config.window)
    validate_scrcpy_recovery(config.scrcpy_recovery)
    validate_thresholds(config.thresholds)
    validate_red_icon_detection(config.red_icon)
    validate_box_detection(config.box)
    validate_upgrade_station(config.upgrade_station)
    validate_input_timing(config.input_timing)
    validate_flow_timing(config.flow_timing)
    validate_capture_regions(config.capture_regions)
    validate_stats_upgrade(config.stats_upgrade)
    validate_level_transition(config.level_transition)
    validate_scroll(config.scroll)
    validate_telegram(config.telegram)

    stall_timeout = config.flow_timing.state_stall_timeout_seconds
    bounded_retry_budgets = (
        (
            "upgrade_station.search_attempts/search_interval",
            config.upgrade_station.search_attempts * config.upgrade_station.search_interval,
        ),
        (
            "upgrade_station.verify_search_attempts/verify_search_interval",
            config.upgrade_station.verify_search_attempts
            * config.upgrade_station.verify_search_interval,
        ),
        (
            "level_transition.search_attempts/search_interval",
            config.level_transition.search_attempts * config.level_transition.search_interval,
        ),
        # level_transition.unlock_search_attempts/unlock_search_interval is deliberately excluded
        # here: WAIT_FOR_UNLOCK's attempt cap is intentionally far larger than the stall timeout
        # (see the field's own comment) — the watchdog, not this cap, is meant to be what ends a
        # stuck wait, so this budget is allowed to exceed state_stall_timeout_seconds on purpose.
    )
    for name, worst_case in bounded_retry_budgets:
        # The budget guard is meaningful only when there is a real stall budget to
        # protect. In an all-zero timing regime (mandated zeroing), stall_timeout is 0
        # and every worst_case is 0, so the comparison `0 >= 0` would falsely trip —
        # the guard would forbid the very config it is supposed to sanity-check.
        if stall_timeout > 0 and worst_case >= stall_timeout:
            raise ConfigError(
                f"{name}'s worst-case retry duration ({worst_case:.3f}s) must be less than "
                f"flow_timing.state_stall_timeout_seconds ({stall_timeout}s), or the watchdog "
                "can force-reset a state that is still legitimately retrying"
            )
    if config.capture_regions.max_search_y > config.capture_regions.extended_search_y:
        raise ConfigError(
            "capture_regions.max_search_y "
            f"({config.capture_regions.max_search_y}) must be <= extended_search_y "
            f"({config.capture_regions.extended_search_y}), or the red-icon scan area silently "
            "shrinks"
        )
