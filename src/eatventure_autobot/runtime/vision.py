"""Semantic reads of the game screen: turns the raw capture + template matcher into questions the
state handlers actually ask ("is the new-level button up?", "where are the red icons?"). Keeps
threshold/HSV/region lookups out of the handlers, so a handler reads as flow, not calibration.
"""

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from eatventure_autobot.detection.opencv_matcher import (
    _hsv_range_mask,
    filter_by_template_consensus,
)
from eatventure_autobot.domain.config import BotConfig
from eatventure_autobot.domain.errors import DetectionError
from eatventure_autobot.domain.protocols import InputController, ScreenCapture, TemplateMatcher
from eatventure_autobot.domain.types import (
    BoundingBox,
    MatchCandidate,
    MatchResult,
    Point,
    TemplateExplanation,
    Zone,
)

logger = logging.getLogger(__name__)

NEW_LEVEL_TEMPLATE = "newLevel"
UNLOCK_TEMPLATE = "unlock"
UPGRADE_STATION_TEMPLATE = "upgradeStation"


def _shifted(candidate: MatchCandidate, dx: int, dy: int) -> MatchCandidate:
    """A candidate found inside a cropped area, expressed in full-frame coordinates."""
    box = candidate.bounding_box
    return MatchCandidate(
        candidate.template_name,
        candidate.confidence,
        (candidate.center[0] + dx, candidate.center[1] + dy),
        BoundingBox(box.x_min + dx, box.y_min + dy, box.x_max + dx, box.y_max + dy),
    )


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
            *self._config.close_button.template_names,
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
        self._capture.ensure_window(resize=should_relocate)
        if should_relocate:
            self._last_relocate_at = now

    def describe_environment(self) -> str:
        """Facts about the window for a self-stop log line. Diagnostics only: never raises."""
        try:
            return self._capture.describe_environment()
        except Exception as exc:
            return f"unavailable ({type(exc).__name__})"

    def environment_ready(self) -> bool:
        """True when starting would not touch the window (see the capture layer). Never raises."""
        try:
            return self._capture.environment_ready()
        except Exception:
            return False

    def cursor_position_in_window(self, input_controller: InputController) -> Point | None:
        """Window-relative cursor readout for the 'X' debug hotkey."""
        try:
            screen_x, screen_y = input_controller.get_cursor_position()
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

    def find_upgrade_station_candidates(
        self, frame: np.ndarray, threshold: float
    ) -> list[MatchCandidate]:
        """Multi-candidate scan so a handler can skip a forbidden-zone best match in favor of
        the next one — matches v1's _find_upgrade_station_match, which iterates every detected
        candidate rather than only ever seeing a single best. Only one template exists for this
        target, so there is no cross-template consensus to require (min_matches=1)."""
        if UPGRADE_STATION_TEMPLATE not in self._loaded:
            return []
        station = self._config.upgrade_station
        return self._gather(
            frame,
            (UPGRADE_STATION_TEMPLATE,),
            threshold,
            station.candidate_min_distance,
            station.hsv,
            1,
            station.candidate_nms_iou_threshold,
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

    def find_close_buttons(self, frame: np.ndarray) -> list[MatchCandidate]:
        """The red X of a popup, found the same way boxes are: masked template match plus the
        per-asset HSV gate. Not in validate_required_templates: a missing asset turns the rule off
        (load_templates already logs it) instead of blocking start."""
        config = self._config.close_button
        loaded = tuple(name for name in config.template_names if name in self._loaded)
        if not loaded:
            return []
        area = self._red_block_area(frame)
        if area is None:
            return []
        left, top, right, bottom = area
        found = self._gather(
            frame[top:bottom, left:right],
            loaded,
            self._config.thresholds.close_button,
            config.min_distance,
            config.hsv,
            1,
            iou_threshold=config.nms_iou_threshold,
        )
        return [_shifted(candidate, left, top) for candidate in found]

    def _red_block_area(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        """Bounding area (left, top, right, bottom) around every solid red block the size of the X
        button, or None when the frame has none: the ~4 ms gate in front of the ~85 ms match."""
        config = self._config.close_button
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red: np.ndarray = np.zeros(frame.shape[:2], dtype=np.uint8)
        for hsv_range in config.block_ranges:
            red = cv2.bitwise_or(red, _hsv_range_mask(hsv, hsv_range.lower, hsv_range.upper))
        window = config.block_window
        share = cv2.boxFilter(
            (red > 0).astype(np.float32),
            -1,
            (window, window),
            normalize=True,
            borderType=cv2.BORDER_CONSTANT,
        )
        rows, cols = np.where(share >= config.block_min_fraction)
        if rows.size == 0:
            return None
        reach = window // 2 + window // 2 + 6  # block centre -> far edge of a template placed on it
        height, width = frame.shape[:2]
        return (
            max(0, int(cols.min()) - reach),
            max(0, int(rows.min()) - reach),
            min(width, int(cols.max()) + reach),
            min(height, int(rows.max()) + reach),
        )

    def box_near_misses(self, frame: np.ndarray) -> list[TemplateExplanation]:
        """Best raw match per loaded box template with the threshold and HSV gate NOT applied: what
        the detector saw when find_boxes reported nothing. Diagnostics only (the stall probe)."""
        explained = (
            self._matcher.explain_template(frame, name, self._config.box.hsv)
            for name in self._config.box.template_names
            if name in self._loaded
        )
        return [item for item in explained if item is not None]

    def red_icon_near_misses(self, frame: np.ndarray) -> list[TemplateExplanation]:
        """box_near_misses for the red-icon templates the current mode really matches: threshold,
        HSV gate and zones NOT applied. Set against box_near_misses it tells a crate-only miss from
        a miss of everything on screen. Diagnostics only (the early stall probe)."""
        explained = (
            self._matcher.explain_template(frame, name, self._config.red_icon.hsv)
            for name in self._active_red_icon_template_names()
        )
        return [item for item in explained if item is not None]

    def unlock_near_miss(self, frame: np.ndarray) -> TemplateExplanation | None:
        """Best raw `unlock` match with the threshold NOT applied: a greyed-out button reads low, a
        near-miss reads close to the threshold. Diagnostics only (the unlock-miss log line)."""
        if UNLOCK_TEMPLATE not in self._loaded:
            return None
        return self._matcher.explain_template(frame, UNLOCK_TEMPLATE)

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
        thresholds = self._config.thresholds
        actionable: list[MatchCandidate] = []
        new_level_seen = False
        stats_seen = False
        for candidate in candidates:
            # Verified v1: a footer badge must clear its own, stricter confidence floor; one below
            # it is dropped outright, never reclassified as an actionable icon.
            if self.in_zone(candidate, zones.new_level_zone):
                new_level_seen |= candidate.confidence >= thresholds.new_level_red_icon
            elif self.in_zone(candidate, zones.upgrade_zone):
                stats_seen |= candidate.confidence >= thresholds.stats_red_icon
            elif candidate.center[1] < self._config.capture_regions.max_search_y:
                actionable.append(candidate)
        return actionable, new_level_seen, stats_seen

    def template_path(self, name: str) -> Path:
        return self._config.paths.assets_dir / f"{name}.png"
