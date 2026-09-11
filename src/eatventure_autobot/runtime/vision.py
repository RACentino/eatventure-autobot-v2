"""Semantic reads of the game screen: turns the raw capture + template matcher into questions the
state handlers actually ask ("is the new-level button up?", "where are the red icons?"). Keeps
threshold/HSV/region lookups out of the handlers, so a handler reads as flow, not calibration.
"""

import logging
import time
from pathlib import Path

import numpy as np

from eatventure_autobot.detection.opencv_matcher import filter_by_template_consensus
from eatventure_autobot.domain.config import BotConfig
from eatventure_autobot.domain.errors import DetectionError
from eatventure_autobot.domain.protocols import ScreenCapture, TemplateMatcher
from eatventure_autobot.domain.types import MatchCandidate, MatchResult, Point, Zone

logger = logging.getLogger(__name__)

NEW_LEVEL_TEMPLATE = "newLevel"
UNLOCK_TEMPLATE = "unlock"
UPGRADE_STATION_TEMPLATE = "upgradeStation"


class GameVision:
    def __init__(
        self,
        capture: ScreenCapture,
        matcher: TemplateMatcher,
        config: BotConfig,
        fast_red_icon_mode: bool | None = None,
    ) -> None:
        self._capture = capture
        self._matcher = matcher
        self._config = config
        self.fast_red_icon_mode = (
            config.red_icon.fast_mode_enabled if fast_red_icon_mode is None else fast_red_icon_mode
        )
        self._loaded: set[str] = set()
        self._last_relocate_at = 0.0

    # --- template lifecycle -------------------------------------------------------------

    def load_templates(self) -> list[str]:
        """Loads every template the configured flow can ask for. Returns the names that failed to
        load; loading is best-effort so one bad asset degrades a single detection rather than
        preventing startup outright. validate_required_templates() is the actual start gate."""
        failed: list[str] = []
        for name in self._required_template_names():
            path = self._config.paths.assets_dir / f"{name}.png"
            try:
                self._matcher.load_template(path)
                self._loaded.add(name)
            except DetectionError as exc:
                logger.error("Template '%s' could not be loaded: %s", name, exc)
                failed.append(name)
        return failed

    def _required_template_names(self) -> tuple[str, ...]:
        red_icon_names = set(self._config.red_icon.fast_template_names) | set(
            self._config.red_icon.normal_template_names
        )
        return (
            NEW_LEVEL_TEMPLATE,
            UNLOCK_TEMPLATE,
            UPGRADE_STATION_TEMPLATE,
            *sorted(red_icon_names),
            *self._config.box.template_names,
        )

    def validate_required_templates(self) -> list[str]:
        """v1's start gate, which v2 dropped: refuse to run with assets missing rather than
        silently degrading into a bot that can never see the thing it is waiting for."""
        required = [NEW_LEVEL_TEMPLATE, UNLOCK_TEMPLATE, UPGRADE_STATION_TEMPLATE]
        active_red_icons = self._active_red_icon_template_names()
        missing = [name for name in required if name not in self._loaded]
        if not any(name in self._loaded for name in active_red_icons):
            missing.append("RedIcon*")
        return missing

    def _active_red_icon_template_names(self) -> tuple[str, ...]:
        names = (
            self._config.red_icon.fast_template_names
            if self.fast_red_icon_mode
            else self._config.red_icon.normal_template_names
        )
        return tuple(name for name in names if name in self._loaded)

    def toggle_red_icon_mode(self) -> str:
        self.fast_red_icon_mode = not self.fast_red_icon_mode
        return "Fast" if self.fast_red_icon_mode else "Normal"

    # --- captures ------------------------------------------------------------------------

    def capture(self, max_y: int | None = None) -> np.ndarray:
        return self._capture.capture(max_y=max_y)

    def ensure_target_ready(self) -> None:
        """Confirms the target window is ready to capture from. The expensive relocate/resize
        query (a full window enumeration) only actually needs to run when the window might have
        moved or changed size, not on every single call — ensure_window(resize=False) still runs
        a cheap liveness check every time and only skips the expensive path."""
        now = time.monotonic()
        interval = self._config.flow_timing.window_relocate_interval
        should_relocate = now - self._last_relocate_at >= interval
        if should_relocate:
            self._last_relocate_at = now
        self._capture.ensure_window(resize=should_relocate)

    def cursor_position_in_window(self, input_controller: object) -> Point | None:
        """Window-relative cursor readout for the 'X' debug hotkey."""
        getter = getattr(input_controller, "get_cursor_position", None)
        if not callable(getter):
            return None
        try:
            screen_x, screen_y = getter()
            bounds = self._capture.get_window_rect()
        except Exception:
            return None
        return screen_x - bounds.left, screen_y - bounds.top

    def close(self) -> None:
        self._capture.close()

    # --- single-template reads -----------------------------------------------------------

    def find_new_level_button(self, frame: np.ndarray) -> MatchResult:
        return self._find_one(frame, NEW_LEVEL_TEMPLATE, self._config.thresholds.new_level)

    def find_unlock_button(self, frame: np.ndarray) -> MatchResult:
        return self._find_one(frame, UNLOCK_TEMPLATE, self._config.thresholds.unlock)

    def find_upgrade_station(
        self, frame: np.ndarray, threshold: float | None = None
    ) -> MatchResult:
        return self._find_one(
            frame,
            UPGRADE_STATION_TEMPLATE,
            self._config.thresholds.upgrade_station if threshold is None else threshold,
            hsv_gated=True,
        )

    def _find_one(
        self, frame: np.ndarray, template_name: str, threshold: float, hsv_gated: bool = False
    ) -> MatchResult:
        if template_name not in self._loaded:
            return MatchResult(found=False, best=None)
        gate = self._config.upgrade_station.hsv if hsv_gated else None
        return self._matcher.find_template(frame, template_name, threshold, hsv_gate=gate)

    # --- multi-template reads ------------------------------------------------------------

    def find_red_icons(
        self, frame: np.ndarray, force_normal_mode: bool = False
    ) -> list[MatchCandidate]:
        use_fast = self.fast_red_icon_mode and not force_normal_mode
        names = (
            self._config.red_icon.fast_template_names
            if use_fast
            else self._config.red_icon.normal_template_names
        )
        loaded = tuple(name for name in names if name in self._loaded)
        if not loaded:
            return []
        # Fast mode scans a single template, so consensus is trivially satisfied; normal mode
        # requires several template variants to agree on the same spot.
        min_matches = 1 if use_fast else min(self._config.red_icon.min_matches, len(loaded))
        return self._gather(
            frame,
            loaded,
            self._config.thresholds.red_icon,
            self._config.red_icon.fast_min_distance,
            self._config.red_icon.hsv,
            min_matches,
            iou_threshold=self._config.red_icon.nms_iou_threshold,
        )

    def find_boxes(self, frame: np.ndarray) -> list[MatchCandidate]:
        loaded = tuple(name for name in self._config.box.template_names if name in self._loaded)
        if not loaded:
            return []
        return self._gather(
            frame,
            loaded,
            self._config.thresholds.box,
            self._config.box.min_distance,
            self._config.box.hsv,
            min(self._config.box.min_matches, len(loaded)),
            iou_threshold=self._config.box.nms_iou_threshold,
        )

    def _gather(
        self,
        frame: np.ndarray,
        template_names: tuple[str, ...],
        threshold: float,
        min_distance: int,
        hsv_gate: object,
        min_matches: int,
        iou_threshold: float,
    ) -> list[MatchCandidate]:
        raw = self._matcher.find_candidates_across_templates(
            frame,
            template_names,
            threshold,
            min_distance,
            hsv_gate,  # type: ignore[arg-type]
        )
        if min_matches > 1:
            return filter_by_template_consensus(raw, min_matches, iou_threshold)
        return self._matcher.suppress_overlaps(raw, iou_threshold)

    # --- zone classification --------------------------------------------------------------

    def in_zone(self, candidate: MatchCandidate, zone: Zone) -> bool:
        return zone.contains(*candidate.center)

    def split_red_icons(
        self, candidates: list[MatchCandidate]
    ) -> tuple[list[MatchCandidate], bool, bool]:
        """Partitions a red-icon scan into (actionable icons, new-level footer icon seen, stats
        icon seen). Footer icons live in dedicated zones and are signals, not click targets."""
        zones = self._config.red_icon_zones
        actionable: list[MatchCandidate] = []
        new_level_seen = False
        stats_seen = False
        for candidate in candidates:
            if self.in_zone(candidate, zones.new_level_zone):
                new_level_seen = True
            elif self.in_zone(candidate, zones.upgrade_zone):
                stats_seen = True
            elif candidate.center[1] < self._config.capture_regions.max_search_y:
                actionable.append(candidate)
        return actionable, new_level_seen, stats_seen

    def template_path(self, name: str) -> Path:
        return self._config.paths.assets_dir / f"{name}.png"
