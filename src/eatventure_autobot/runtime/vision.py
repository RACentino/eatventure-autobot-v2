"""Semantic reads of the game screen: turns the raw capture + template matcher into questions the
state handlers actually ask ("is the new-level button up?", "where are the red icons?"). Keeps
threshold/HSV/region lookups out of the handlers, so a handler reads as flow, not calibration.
"""

import logging
import math
import time
from itertools import groupby

import cv2
import numpy as np

from eatventure_autobot.detection.opencv_matcher import (
    filter_by_template_consensus,
    hsv_ranges_mask,
)
from eatventure_autobot.domain.config import BotConfig, StrategyConfig
from eatventure_autobot.domain.errors import CaptureError, DetectionError
from eatventure_autobot.domain.protocols import InputController, ScreenCapture, TemplateMatcher
from eatventure_autobot.domain.state import BoostLevel, StationPanel
from eatventure_autobot.domain.types import (
    BoundingBox,
    HsvGate,
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
_WORLD_SEARCH_TOP = 90
_BUILT_STATION_MIN_CONFIDENCE = 0.95
_VIEWPORT_MIN_RESPONSE = 0.25
_VIEWPORT_MAX_DRIFT = 1.0
_IDLE_WORLD_REGION = BoundingBox(56, _WORLD_SEARCH_TOP, 310, 640)
_IDLE_STATION_CLEARANCE_PX = 60
_IDLE_COLOR_BIN_SIZE = 16
_IDLE_COLOR_BIN_TOLERANCE = 1
_IDLE_SECONDARY_MIN_FRACTION = 0.5
_IDLE_PALETTE_MERGE_DISTANCE = 2
_IDLE_MEDIAN_KERNEL = 5
_IDLE_MIN_CLEAR_RADIUS_PX = 30
_POSITIONAL_FOOTPRINT_MIN_SIZE = (30, 20)
_POSITIONAL_FOOTPRINT_MAX_SIZE = (68, 64)
_POSITIONAL_FOOTPRINT_EDGE_WIDTH = 5
_POSITIONAL_FOOTPRINT_INSET = 6
_POSITIONAL_FOOTPRINT_MIN_EDGE_FRACTION = 0.05
_POSITIONAL_FOOTPRINT_MAX_INTERIOR_FRACTION = 0.06
_POSITIONAL_FOOTPRINT_MAX_INTERIOR_DEVIATION = 10
_POSITIONAL_FOOTPRINT_BORDER_EXPANSION = 4
_POSITIONAL_FOOTPRINT_MIN_DASHED_SIDES = 3
_POSITIONAL_ARROW_SIZE = (18, 30)
_POSITIONAL_ARROW_MIN_ROUNDNESS = 0.8
_POSITIONAL_GLYPH_WIDTH = (6, 12)
_POSITIONAL_GLYPH_HEIGHT = (8, 14)
_POSITIONAL_GLYPH_MIN_AREA = 18
_POSITIONAL_GLYPH_CENTER_TOLERANCE = 3
_POSITIONAL_GLYPH_MIN_STEM_FRACTION = 0.7
_POSITIONAL_GLYPH_MAX_WING_FRACTION = 0.2
_POSITIONAL_PARTIAL_ROW_TOLERANCE = 4
_POSITIONAL_PARTIAL_RED_FRACTION = (0.1, 0.65)
_POSITIONAL_PARTIAL_INTERIOR_SCALE = 0.8


def _station_outline(frame: np.ndarray) -> np.ndarray:
    return np.asarray(cv2.adaptiveThreshold(frame.min(axis=2), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -6))


def _dashed_footprint_border(outline: np.ndarray) -> bool:
    edge = _POSITIONAL_FOOTPRINT_EDGE_WIDTH
    profiles = (outline[:edge].any(axis=0), outline[-edge:].any(axis=0),
                outline[:, :edge].any(axis=1), outline[:, -edge:].any(axis=1))
    dashed_sides = 0
    for profile in profiles:
        joined = cv2.morphologyEx(np.asarray(profile, dtype=np.uint8)[None, :], cv2.MORPH_CLOSE,
                                 np.ones((1, 2), dtype=np.uint8))
        if np.count_nonzero(np.diff(joined.ravel().astype(np.int8), prepend=0) > 0) >= 2:
            dashed_sides += 1
    return dashed_sides >= _POSITIONAL_FOOTPRINT_MIN_DASHED_SIDES


def _up_arrow_glyph(white: np.ndarray) -> bool:
    height, width = white.shape
    _count, _labels, components, _centers = cv2.connectedComponentsWithStats(white)
    for left, top, glyph_width, glyph_height, area in components[1:]:
        if not (_POSITIONAL_GLYPH_WIDTH[0] <= glyph_width <= _POSITIONAL_GLYPH_WIDTH[1]
                and _POSITIONAL_GLYPH_HEIGHT[0] <= glyph_height <= _POSITIONAL_GLYPH_HEIGHT[1]
                and area >= _POSITIONAL_GLYPH_MIN_AREA):
            continue
        if max(abs(left + glyph_width / 2 - width / 2),
               abs(top + glyph_height / 2 - height / 2)) > _POSITIONAL_GLYPH_CENTER_TOLERANCE:
            continue
        glyph = white[top:top + glyph_height, left:left + glyph_width] != 0
        lower = glyph[glyph_height // 2:]
        if float(lower[:, glyph_width // 2].mean()) < _POSITIONAL_GLYPH_MIN_STEM_FRACTION:
            continue
        edge_width = max(1, glyph_width // 4)
        wings = np.concatenate((lower[:, :edge_width], lower[:, -edge_width:]), axis=1)
        if float(wings.mean()) <= _POSITIONAL_GLYPH_MAX_WING_FRACTION:
            return True
    return False


def _localized_arrow_contours(red: np.ndarray, hsv: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result: list[np.ndarray] = list(contours)
    for contour in contours:
        left, top, width, height = cv2.boundingRect(contour)
        perimeter = cv2.arcLength(contour, True)
        if (max(width, height) <= _POSITIONAL_ARROW_SIZE[1] and perimeter > 0
                and 4 * math.pi * cv2.contourArea(contour) / perimeter ** 2
                >= _POSITIONAL_ARROW_MIN_ROUNDNESS):
            continue
        if width < _POSITIONAL_GLYPH_WIDTH[0] or height < _POSITIONAL_GLYPH_HEIGHT[0]:
            continue
        white = cv2.inRange(hsv[top:top + height, left:left + width],
                           np.array((0, 0, 230)), np.array((179, 45, 255)))
        _count, _labels, components, _centers = cv2.connectedComponentsWithStats(white)
        for glyph_left, glyph_top, glyph_width, glyph_height, area in components[1:]:
            if not (_POSITIONAL_GLYPH_WIDTH[0] <= glyph_width <= _POSITIONAL_GLYPH_WIDTH[1]
                    and _POSITIONAL_GLYPH_HEIGHT[0] <= glyph_height <= _POSITIONAL_GLYPH_HEIGHT[1]
                    and area >= _POSITIONAL_GLYPH_MIN_AREA):
                continue
            center_x = left + glyph_left + glyph_width // 2
            center_y = top + glyph_top + glyph_height // 2
            radius = min(_POSITIONAL_ARROW_SIZE[1] // 2, max(glyph_width, glyph_height) + 1)
            start_x, start_y = max(0, center_x - radius), max(0, center_y - radius)
            patch = red[start_y:center_y + radius, start_x:center_x + radius]
            local, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            result.extend(part + np.array((start_x, start_y), dtype=np.int32) for part in local)
    return result


def _partial_arrow_targets(
    red: np.ndarray, row_tops: tuple[int, ...], offset: Point,
) -> list[MatchCandidate]:
    """Recover an obscured arrow only beside an observed dashed station row."""
    targets: list[MatchCandidate] = []
    minimum_radius, maximum_radius = (size // 2 for size in _POSITIONAL_ARROW_SIZE)
    for row_top in row_tops:
        expected = row_top - offset[1]
        top = max(_WORLD_SEARCH_TOP, expected - maximum_radius - 1)
        bottom = min(red.shape[0], expected + maximum_radius + 1)
        region = cv2.GaussianBlur(red[top:bottom], (3, 3), 0)
        circles = cv2.HoughCircles(region, cv2.HOUGH_GRADIENT, 1, _POSITIONAL_ARROW_SIZE[0],
                                   param1=50, param2=10, minRadius=minimum_radius,
                                   maxRadius=maximum_radius)
        if circles is None:
            continue
        for circle_x, circle_y, radius in circles[0]:
            circle_y += top
            if abs(float(circle_y) - expected) > _POSITIONAL_PARTIAL_ROW_TOLERANCE:
                continue
            left, upper = max(0, int(circle_x - radius)), max(0, int(circle_y - radius))
            right = min(red.shape[1], int(circle_x + radius) + 1)
            lower = min(red.shape[0], int(circle_y + radius) + 1)
            vertical, horizontal = np.ogrid[upper:lower, left:right]
            interior = ((horizontal - circle_x) ** 2 + (vertical - circle_y) ** 2
                        <= (radius * _POSITIONAL_PARTIAL_INTERIOR_SCALE) ** 2)
            if not np.any(interior):
                continue
            fraction = float((red[upper:lower, left:right][interior] != 0).mean())
            if not _POSITIONAL_PARTIAL_RED_FRACTION[0] <= fraction <= (
                _POSITIONAL_PARTIAL_RED_FRACTION[1]
            ):
                continue
            target = _shifted(MatchCandidate("stationPosition", 0.0,
                (round(float(circle_x)), round(float(circle_y))),
                BoundingBox(left, upper, right, lower)), *offset)
            if 0 <= target.center[0] < red.shape[1]:
                targets.append(target)
    return targets


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
        self._panel_template_sizes: dict[str, tuple[int, int]] = {}
        self._panel_cache: tuple[
            np.ndarray, tuple[tuple[int, ...], BoundingBox | None, StrategyConfig],
            BoundingBox | None,
        ] | None = None

    # --- template lifecycle -------------------------------------------------------------

    def load_templates(self) -> list[str]:
        """Try configured assets and report load failures. Startup checks required templates."""
        self._panel_cache = None
        self._panel_template_sizes.clear()
        baseline = self._required_template_names()
        strategy = self._config.strategy
        panel_templates = (strategy.panel_template, strategy.unlock_panel_template)
        strategy_templates = (
            *strategy.max_templates, strategy.locked_template, *panel_templates,
            *strategy.marker_templates,
            strategy.restaurant_template, strategy.city_template, strategy.city_continue_template,
            *strategy.boost_templates,
        )
        failed: list[str] = []
        for name in dict.fromkeys((*baseline, *strategy_templates)):
            path = self._config.paths.assets_dir / f"{name}.png"
            if name not in baseline and not path.exists():
                continue
            try:
                size = self._matcher.load_template(path)
                self._loaded.add(name)
                if size is not None and name in panel_templates:
                    self._panel_template_sizes[name] = size
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
        strategy = self._config.strategy
        required = [NEW_LEVEL_TEMPLATE, UNLOCK_TEMPLATE, UPGRADE_STATION_TEMPLATE,
                    strategy.restaurant_template, strategy.city_template,
                    strategy.city_continue_template]
        active_red_icons = self._active_red_icon_template_names()
        missing = [name for name in required if name not in self._loaded]
        if not any(name in self._loaded for name in active_red_icons):
            missing.append("RedIcon*")
        if self._config.strategy.enabled:
            missing.extend(name for name in (*strategy.max_templates, strategy.locked_template,
                strategy.panel_template, strategy.unlock_panel_template, *strategy.marker_templates,
                strategy.restaurant_template)
                if name not in self._loaded)
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

    def mirror_windows(self) -> list[tuple[int, bool]]:
        return self._capture.mirror_windows()

    def bind_mirror(self, process_id: int) -> None:
        self._capture.bind_mirror(process_id)
        self._last_relocate_at = 0.0

    def ensure_target_ready(self, *, force: bool = False) -> None:
        """Confirms the target window is ready to capture from. The expensive relocate/resize
        query (a full window enumeration) only actually needs to run when the window might have
        moved or changed size, not on every single call — ensure_window(resize=False) still runs
        a cheap liveness check every time and only skips the expensive path."""
        now = time.monotonic()
        interval = self._config.flow_timing.window_relocate_interval
        should_relocate = force or now - self._last_relocate_at >= interval
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
        """Readiness failures are recoverable; programming errors reach the runtime error latch."""
        try:
            return self._capture.environment_ready()
        except CaptureError:
            return False

    def raise_target(self) -> bool:
        """Bring the game window forward, allowing unexpected failures to stop recovery."""
        try:
            return self._capture.raise_window()
        except CaptureError:
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

    def find_unlock_button(
        self, frame: np.ndarray, search_region: BoundingBox | None = None,
    ) -> MatchResult:
        return self._find_one(frame, UNLOCK_TEMPLATE, self._config.thresholds.unlock,
                              search_region=search_region)

    def find_upgrade_station(
        self, frame: np.ndarray, threshold: float | None = None,
        search_region: BoundingBox | None = None,
    ) -> MatchResult:
        if self._config.strategy.enabled:
            panel = self.station_panel_bounds(frame, near=search_region)
            if panel is None:
                return MatchResult(found=False, best=None)
            if search_region is None:
                search_region = panel
            else:
                left = max(panel.x_min, search_region.x_min)
                top = max(panel.y_min, search_region.y_min)
                right = min(panel.x_max, search_region.x_max)
                bottom = min(panel.y_max, search_region.y_max)
                if left >= right or top >= bottom:
                    return MatchResult(found=False, best=None)
                search_region = BoundingBox(left, top, right, bottom)
        return self._find_one(
            frame,
            UPGRADE_STATION_TEMPLATE,
            self._config.thresholds.upgrade_station if threshold is None else threshold,
            hsv_gated=True,
            search_region=search_region,
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
        region = self.station_panel_bounds(frame) if self._config.strategy.enabled else None
        if self._config.strategy.enabled and region is None:
            return []
        return self._matcher.suppress_overlaps(
            self._matcher.find_template_candidates(
                frame, UPGRADE_STATION_TEMPLATE, threshold,
                station.candidate_min_distance, station.hsv, search_region=region,
            ), station.candidate_nms_iou_threshold,
        )

    def _find_one(
        self, frame: np.ndarray, template_name: str, threshold: float, hsv_gated: bool = False,
        search_region: BoundingBox | None = None,
    ) -> MatchResult:
        if template_name not in self._loaded:
            return MatchResult(found=False, best=None)
        gate = self._config.upgrade_station.hsv if hsv_gated else None
        if search_region is not None:
            return self._matcher.find_template(
                frame, template_name, threshold, hsv_gate=gate, search_region=search_region
            )
        return self._matcher.find_template(frame, template_name, threshold, hsv_gate=gate)

    # --- multi-template reads ------------------------------------------------------------

    def _panel_pixels(
        self, frame: np.ndarray, centers: BoundingBox | None, names: tuple[str, str],
    ) -> tuple[np.ndarray, int, int]:
        first, second = (self._panel_template_sizes.get(name) for name in names)
        if centers is None or first is None or second is None:
            return frame, 0, 0
        # Candidate centers need their whole template and a 1px halo for the 3x3 blur.
        x_margin = (max(first[0], second[0]) + 1) // 2 + 1
        y_margin = (max(first[1], second[1]) + 1) // 2 + 1
        left = max(0, centers.x_min - x_margin)
        right = min(frame.shape[1], centers.x_max + x_margin)
        top = max(0, centers.y_min - y_margin)
        bottom = min(frame.shape[0], centers.y_max + y_margin)
        if right <= left or bottom <= top:
            return frame, 0, 0
        return frame[top:bottom, left:right], left, top

    def station_panel_bounds(
        self, frame: np.ndarray, *, near: BoundingBox | None = None,
    ) -> BoundingBox | None:
        config = self._config.strategy
        names = (config.panel_template, config.unlock_panel_template)
        # A panel containing the held point can have its center offset from that point.
        centers = None if near is None else BoundingBox(
            near.x_min - 75, near.y_min - 85, near.x_max + 75, near.y_max + 85,
        )
        pixels, crop_x, crop_y = self._panel_pixels(frame, centers, names)
        cache_key = (frame.shape, near, config)
        cached = self._panel_cache
        if (cached is not None and cached[1] == cache_key
                and cached[0].dtype == pixels.dtype and np.array_equal(cached[0], pixels)):
            return cached[2]
        # Keep only one exact observation; a copy also detects reused capture buffers changing.
        observed = pixels.copy()
        height, width = frame.shape[:2]
        blurred = cv2.GaussianBlur(pixels, (3, 3), 0)
        # A visible masked header can fit while its transparent template border overhangs.
        for padding in (0, width):
            image = (blurred if not padding else cv2.copyMakeBorder(
                blurred, padding, padding, padding, padding, cv2.BORDER_CONSTANT
            ))
            origin_x, origin_y = crop_x - padding, crop_y - padding
            region = (None if not padding else BoundingBox(
                padding, padding, padding + width, padding + height
            ))
            if centers is not None:
                region = BoundingBox(
                    centers.x_min - origin_x, centers.y_min - origin_y,
                    centers.x_max - origin_x, centers.y_max - origin_y,
                )
            for name in names:
                panel = self._find_one(image, name, config.panel_threshold, search_region=region)
                if panel.best is not None:
                    box = panel.best.bounding_box
                    left, top = max(0, box.x_min + origin_x), max(0, box.y_min + origin_y)
                    right = min(width, box.x_max + origin_x)
                    bottom = min(height, box.y_max + origin_y)
                    if left < right and top < bottom:
                        bounds = BoundingBox(left, top, right, bottom)
                        self._panel_cache = (observed, cache_key, bounds)
                        return bounds
        self._panel_cache = (observed, cache_key, None)
        return None

    def station_panel(
        self, frame: np.ndarray, *, held_position: Point | None = None,
    ) -> StationPanel:
        config = self._config.strategy
        tolerance = self._config.upgrade_station.disappear_position_tolerance_px
        near = (None if held_position is None else BoundingBox(
            held_position[0] - tolerance, held_position[1] - tolerance,
            held_position[0] + tolerance + 1, held_position[1] + tolerance + 1,
        ))
        region = (self.station_panel_bounds(frame) if near is None
                  else self.station_panel_bounds(frame, near=near))
        if region is None:
            return StationPanel.UNKNOWN
        max_frame = cv2.threshold(frame, config.max_text_min_value, 255, cv2.THRESH_BINARY)[1]
        max_frame = cv2.GaussianBlur(max_frame, (3, 3), 0)
        if any(self._find_one(max_frame, name, config.max_threshold,
                             search_region=region).found for name in config.max_templates):
            return StationPanel.MAX
        if (self._find_one(frame, UPGRADE_STATION_TEMPLATE,
                          self._config.thresholds.upgrade_station, hsv_gated=True,
                          search_region=region).found
                or self.find_unlock_button(frame, search_region=region).found):
            return StationPanel.AFFORDABLE
        locked = self._find_one(frame, config.locked_template, config.locked_threshold,
                                search_region=region)
        if locked.best is not None:
            box = locked.best.bounding_box
            if held_position is not None and (
                max(abs(locked.best.center[0] - held_position[0]),
                    abs(locked.best.center[1] - held_position[1])) > tolerance
                or not (box.x_min <= held_position[0] < box.x_max
                        and box.y_min <= held_position[1] < box.y_max)
            ):
                return StationPanel.UNKNOWN
            hsv = cv2.cvtColor(frame[box.y_min:box.y_max, box.x_min:box.x_max], cv2.COLOR_BGR2HSV)
            red = hsv_ranges_mask(hsv, config.locked_price_hsv.ranges)
            if cv2.countNonZero(red) / red.size >= config.locked_price_hsv.min_match_ratio:
                return StationPanel.LOCKED
        return StationPanel.UNKNOWN

    def viewport_still(self, before: np.ndarray, after: np.ndarray) -> bool | None:
        """Camera translation in world pixels; uncertain images cannot prove an edge."""
        if before.shape != after.shape:
            return None
        margin = before.shape[1] // 6
        bottom = min(before.shape[0], self._config.capture_regions.max_search_y)
        regions = [frame[_WORLD_SEARCH_TOP:bottom, margin:-margin] for frame in (before, after)]
        if min(regions[0].shape[:2]) < 32:
            return None
        gray = [cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float32) for region in regions]
        if any(float(image.std()) < 1.0 for image in gray):
            return None
        height, width = gray[0].shape
        window = cv2.createHanningWindow((width, height), cv2.CV_32F)
        # Crowd effects obscure fine detail; retry weak matches with calibrated low-pass filtering.
        for sigma in (0, 5):
            images = [cv2.GaussianBlur(image, (0, 0), sigma) if sigma else image.copy()
                      for image in gray]
            shift, response = cv2.phaseCorrelate(images[0], images[1], window)
            if (math.isfinite(response) and response >= _VIEWPORT_MIN_RESPONSE
                    and all(math.isfinite(value) for value in shift)):
                return math.hypot(*shift) <= _VIEWPORT_MAX_DRIFT
        # Animated crowds can overwhelm correlation. Require distributed feature consensus
        # before trusting the fallback; a few matching sprites cannot prove a stationary view.
        images = [image.astype(np.uint8) for image in gray]
        for background_only in (False, True):
            stable: np.ndarray | None = None
            if background_only:
                images, stable = self._viewport_background(before, after)
            height, width = images[0].shape
            detector = cv2.ORB.create(nfeatures=500, fastThreshold=5 if background_only else 20,
                                     edgeThreshold=15 if background_only else 31)
            features = [detector.detectAndCompute(image, stable) for image in images]
            if any(descriptors is None for _points, descriptors in features):
                continue
            matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(features[0][1], features[1][1], k=2)
            good = [pair[0] for pair in matches
                    if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
            if len(good) < 10:
                continue
            source = np.asarray([features[0][0][match.queryIdx].pt for match in good],
                                dtype=np.float32)
            destination = np.asarray([features[1][0][match.trainIdx].pt for match in good],
                                     dtype=np.float32)
            transform, inliers = cv2.estimateAffinePartial2D(
                source, destination, method=cv2.RANSAC, ransacReprojThreshold=1.5,
            )
            if transform is None or inliers is None or not np.isfinite(transform).all():
                continue
            supported = inliers.ravel().astype(bool)
            if supported.sum() < 10 or supported.mean() < 0.5:
                continue
            if np.any(np.ptp(source[supported], axis=0) < (width / 2, height / 2)):
                continue
            corners = np.asarray([(0, 0), (width - 1, 0), (0, height - 1),
                                  (width - 1, height - 1)], dtype=np.float32)
            drift = corners @ transform[:, :2].T + transform[:, 2] - corners
            return bool(np.linalg.norm(drift, axis=1).max() <= _VIEWPORT_MAX_DRIFT)
        return None

    def _viewport_background(
        self, before: np.ndarray, after: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Retain stable world patches across the width, excluding configured HUD regions."""
        top = _WORLD_SEARCH_TOP
        bottom = min(before.shape[0], self._config.capture_regions.max_search_y)
        images = [cv2.cvtColor(frame[top:bottom], cv2.COLOR_BGR2GRAY)
                  for frame in (before, after)]
        stable = (cv2.absdiff(*images) <= 3).astype(np.uint8) * 255
        height, width = stable.shape
        zones = self._config.forbidden_zones
        for zone in (*zones.numbered_zones, zones.click_zone, *zones.event_zone_options.values()):
            left, right = max(0, zone.x_min), min(width, zone.x_max + 1)
            start = max(0, zone.y_min - top)
            end = min(height, (zone.y_max + 1 if zone.y_max is not None else bottom) - top)
            stable[start:max(start, end), left:max(left, right)] = 0
        return images, cv2.erode(stable, np.ones((31, 31), dtype=np.uint8))

    def positional_station_targets(self, frame: np.ndarray) -> list[MatchCandidate]:
        """Template-independent footprints and arrow shapes, ordered by world position."""
        bottom = self._world_bottom(frame)
        if bottom <= _WORLD_SEARCH_TOP:
            return []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, np.array((0, 0, 235)), np.array((179, 25, 255)))
        white[:_WORLD_SEARCH_TOP] = 0
        white[bottom:] = 0
        candidates: list[MatchCandidate] = []
        # Raw borders retain clipped building edges; contrast rejects bright floor-tile fill.
        for mask in (white, white & _station_outline(frame)):
            connected = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))
            connected = cv2.dilate(connected, np.ones((3, 3), dtype=np.uint8))
            contours, _ = cv2.findContours(connected, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                left, top, width, height = cv2.boundingRect(contour)
                for expansion in (0, _POSITIONAL_FOOTPRINT_BORDER_EXPANSION):
                    box = BoundingBox(left - expansion, top - expansion,
                                      left + width + expansion, top + height + expansion)
                    target = self._positional_footprint(frame, mask, box, bottom)
                    if target is not None:
                        candidates.append(target)
        row_tops = tuple(sorted({target.bounding_box.y_min for target in candidates}))
        candidates.extend(self._positional_arrows(hsv, bottom, row_tops))
        targets: list[MatchCandidate] = []
        tolerance = self._config.red_icon.fast_min_distance
        for target in candidates:
            if not any(math.dist(target.center, known.center) < tolerance
                       for known in targets):
                targets.append(target)
        return targets

    def _world_bottom(self, frame: np.ndarray) -> int:
        return int(min(frame.shape[0], self._config.red_icon_zones.new_level_zone.y_min,
                       self._config.red_icon_zones.upgrade_zone.y_min))

    def _positional_arrows(
        self, hsv: np.ndarray, bottom: int, row_tops: tuple[int, ...] = (),
    ) -> list[MatchCandidate]:
        red = hsv_ranges_mask(hsv, self._config.red_icon.hsv.ranges)
        red[:_WORLD_SEARCH_TOP] = 0
        red[bottom:] = 0
        contours = _localized_arrow_contours(red, hsv)
        targets: list[MatchCandidate] = []
        for contour in contours:
            left, top, width, height = cv2.boundingRect(contour)
            if not (_POSITIONAL_ARROW_SIZE[0] <= width <= _POSITIONAL_ARROW_SIZE[1]
                    and _POSITIONAL_ARROW_SIZE[0] <= height <= _POSITIONAL_ARROW_SIZE[1]
                    and 0.75 < width / height < 1.25
                    and top > _WORLD_SEARCH_TOP and top + height < bottom):
                continue
            perimeter = cv2.arcLength(contour, True)
            if (perimeter == 0 or 4 * math.pi * cv2.contourArea(contour) / perimeter ** 2
                    < _POSITIONAL_ARROW_MIN_ROUNDNESS):
                continue
            white = cv2.inRange(hsv[top:top + height, left:left + width],
                               np.array((0, 0, 230)), np.array((179, 45, 255)))
            if not _up_arrow_glyph(white):
                continue
            arrow = _shifted(MatchCandidate(
                "stationPosition", 0.0, (left + width // 2, top + height // 2),
                BoundingBox(left, top, left + width, top + height),
            ), *self._config.red_icon.offset)
            if (_WORLD_SEARCH_TOP <= arrow.center[1] < bottom
                    and 0 <= arrow.center[0] < hsv.shape[1]):
                targets.append(arrow)
        targets.extend(_partial_arrow_targets(red, row_tops, self._config.red_icon.offset))
        return targets

    def _positional_footprint(
        self, frame: np.ndarray, white: np.ndarray, box: BoundingBox, bottom: int,
    ) -> MatchCandidate | None:
        left, top = box.x_min, box.y_min
        width, height = box.x_max - left, box.y_max - top
        if not (_POSITIONAL_FOOTPRINT_MIN_SIZE[0] <= width <= _POSITIONAL_FOOTPRINT_MAX_SIZE[0]
                and _POSITIONAL_FOOTPRINT_MIN_SIZE[1] <= height
                <= _POSITIONAL_FOOTPRINT_MAX_SIZE[1]
                and 0.7 <= width / height <= 2.25
                and top > _WORLD_SEARCH_TOP and top + height < bottom
                and left > 0 and left + width < frame.shape[1]):
            return None
        edge = _POSITIONAL_FOOTPRINT_EDGE_WIDTH
        inset = _POSITIONAL_FOOTPRINT_INSET
        outline = white[top:top + height, left:left + width] != 0
        edges = (outline[:edge], outline[-edge:], outline[:, :edge], outline[:, -edge:])
        if (min(float(side.mean()) for side in edges) < _POSITIONAL_FOOTPRINT_MIN_EDGE_FRACTION
                or float(outline[inset:-inset, inset:-inset].mean())
                > _POSITIONAL_FOOTPRINT_MAX_INTERIOR_FRACTION
                or not _dashed_footprint_border(outline)):
            return None
        interior = frame[top + inset:top + height - inset, left + inset:left + width - inset]
        if float(interior.std(axis=(0, 1)).max()) > _POSITIONAL_FOOTPRINT_MAX_INTERIOR_DEVIATION:
            return None
        return MatchCandidate("stationPosition", 0.0, (left + width // 2, top + height // 2), box)

    def station_targets(self, frame: np.ndarray) -> list[MatchCandidate]:
        """Station click points from footprints or the existing arrow-to-station offset."""
        config = self._config.strategy
        # The legacy scan cap excludes valid edge stations; the footer bounds the world.
        bottom = self._world_bottom(frame)
        if bottom <= _WORLD_SEARCH_TOP:
            return []
        markers = [name for name in config.marker_templates if name in self._loaded]
        targets: list[MatchCandidate] = []
        if markers:
            # Translucent dashed borders keep their local contrast across floor colours.
            border = _station_outline(frame)
            border = cv2.cvtColor(cv2.GaussianBlur(border, (3, 3), 0), cv2.COLOR_GRAY2BGR)
            for built, names in groupby(markers, key=lambda name: name.startswith("stationBuilt")):
                found = self._matcher.find_candidates_across_templates(
                    frame if built else border, tuple(names),
                    _BUILT_STATION_MIN_CONFIDENCE if built else config.marker_threshold, 15, None,
                    search_region=BoundingBox(0, _WORLD_SEARCH_TOP, frame.shape[1], bottom),
                )
                targets.extend(candidate for candidate in found
                               if candidate.bounding_box.y_max <= bottom)
            targets = self._matcher.suppress_overlaps(targets, 0.3)
        offset_x, offset_y = self._config.red_icon.offset
        margin = self._config.red_icon.fast_min_distance
        region = BoundingBox(-offset_x - margin, _WORLD_SEARCH_TOP - offset_y - margin,
                             frame.shape[1] - offset_x + margin, bottom - offset_y + margin)
        icons = self.find_red_icons(
            frame, force_normal_mode=not self.fast_red_icon_mode, search_region=region,
        )
        zones = self._config.forbidden_zones
        protected = (*zones.numbered_zones, zones.click_zone, *zones.event_zone_options.values())
        if self.fast_red_icon_mode and not any(
            _WORLD_SEARCH_TOP <= icon.center[1] + offset_y < bottom
            and 0 <= icon.center[0] + offset_x < frame.shape[1]
            and not any(zone.contains(icon.center[0] + offset_x, icon.center[1] + offset_y)
                        for zone in protected) for icon in icons
        ):
            icons = self.find_red_icons(frame, force_normal_mode=True, search_region=region)
        for icon in icons:
            target = _shifted(icon, *self._config.red_icon.offset)
            if (not _WORLD_SEARCH_TOP <= target.center[1] < bottom
                    or not 0 <= target.center[0] < frame.shape[1]):
                continue
            if not any(math.dist(target.center, other.center) < (
                    self._config.red_icon.fast_min_distance) for other in targets):
                targets.append(target)
        return targets

    def empty_world_point(
        self, frame: np.ndarray, excluded: BoundingBox, occupied: tuple[Point, ...] = (),
    ) -> Point | None:
        if not np.any(frame != frame[0, 0]):
            return None
        height, width = frame.shape[:2]
        allowed = np.zeros((height, width), dtype=np.uint8)
        region = _IDLE_WORLD_REGION
        allowed[region.y_min:min(region.y_max, height),
                region.x_min:min(region.x_max, width)] = 1
        # The configured idle point uses the same panel/station exclusions as inferred ground.
        preferred = self._config.click_targets.idle_click_pos
        inside = 0 <= preferred[0] < width and 0 <= preferred[1] < height
        if inside:
            allowed[preferred[1], preferred[0]] = 1
        allowed[max(0, excluded.y_min):min(height, excluded.y_max + 1),
                max(0, excluded.x_min):min(width, excluded.x_max + 1)] = 0
        # These observations only exclude possible touch areas; they never select a station.
        for point in (*occupied, *(target.center for target in self.station_targets(frame))):
            cv2.circle(allowed, point, _IDLE_STATION_CLEARANCE_PX, 0, -1)
        if not allowed.any():
            return None
        pixels = cv2.medianBlur(frame, _IDLE_MEDIAN_KERNEL).astype(np.int16)
        pixels //= _IDLE_COLOR_BIN_SIZE
        colors = (pixels[:, :, 0] * _IDLE_COLOR_BIN_SIZE ** 2
                  + pixels[:, :, 1] * _IDLE_COLOR_BIN_SIZE + pixels[:, :, 2])
        counts = np.bincount(colors[allowed.astype(bool)])
        dominant = int(counts.argmax())
        if dominant == 0:
            return None
        if inside and allowed[preferred[1], preferred[0]]:
            return preferred
        primary_count = int(counts[dominant])
        counts[dominant] = 0
        secondary = int(counts.argmax())
        palette = np.array([dominant // _IDLE_COLOR_BIN_SIZE ** 2,
                            dominant // _IDLE_COLOR_BIN_SIZE % _IDLE_COLOR_BIN_SIZE,
                            dominant % _IDLE_COLOR_BIN_SIZE], dtype=np.int16)
        secondary_palette = np.array([secondary // _IDLE_COLOR_BIN_SIZE ** 2,
                                      secondary // _IDLE_COLOR_BIN_SIZE % _IDLE_COLOR_BIN_SIZE,
                                      secondary % _IDLE_COLOR_BIN_SIZE], dtype=np.int16)
        palettes = [palette]
        if (int(counts[secondary]) / primary_count >= _IDLE_SECONDARY_MIN_FRACTION
                and np.max(np.abs(palette - secondary_palette)) <= _IDLE_PALETTE_MERGE_DISTANCE):
            palettes.append(secondary_palette)
        floor = np.zeros((height, width), dtype=np.uint8)
        for palette in palettes:
            floor |= (np.max(np.abs(pixels - palette), axis=2)
                      <= _IDLE_COLOR_BIN_TOLERANCE).astype(np.uint8)
        floor &= allowed
        distance = cv2.distanceTransform(floor, cv2.DIST_L2, 5)
        _minimum, radius, _minimum_point, clear_point = cv2.minMaxLoc(distance)
        if radius < _IDLE_MIN_CLEAR_RADIUS_PX:
            return None
        return int(clear_point[0]), int(clear_point[1])

    def boost_level(self, frame: np.ndarray) -> BoostLevel:
        config = self._config.strategy
        # Active multipliers are in the header; offers elsewhere cannot label running time.
        region = BoundingBox(75, 25, 175, 48)
        found = [level for name, level in zip(config.boost_templates,
                 (BoostLevel.X48, BoostLevel.X480), strict=True)
                 if self._find_one(frame, name, config.threshold, search_region=region).found]
        return found[0] if len(found) == 1 else BoostLevel.UNKNOWN

    def restaurant_start(self, frame: np.ndarray) -> bool:
        config = self._config.strategy
        binary = cv2.threshold(frame, config.max_text_min_value, 255, cv2.THRESH_BINARY)[1]
        return self._find_one(binary, config.restaurant_template, config.threshold).found

    def city_arrival(self, frame: np.ndarray) -> bool:
        config = self._config.strategy
        return self._find_one(frame, config.city_template, config.threshold).found

    def find_city_continue(self, frame: np.ndarray) -> MatchResult:
        config = self._config.strategy
        return self._find_one(frame, config.city_continue_template, config.threshold)

    def stats_panel_ready(self, frame: np.ndarray) -> bool:
        return any(candidate.center == self._config.click_targets.stats_upgrade_close_pos
                   for candidate in self.find_close_buttons(frame))

    def find_red_icons(
        self, frame: np.ndarray, force_normal_mode: bool = False,
        *, search_region: BoundingBox | None = None,
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
            search_region=search_region,
        )

    def new_level_icon_seen(self, frame: np.ndarray) -> bool:
        zone = self._config.red_icon_zones.new_level_zone
        if frame.shape[0] <= zone.y_min:
            return False
        region = BoundingBox(zone.x_min, zone.y_min, zone.x_max + 1,
                             frame.shape[0] if zone.y_max is None else zone.y_max + 1)
        return self.split_red_icons(self.find_red_icons(frame, search_region=region))[1]

    def stats_icon_seen(self, frame: np.ndarray) -> bool:
        """Read only the stats signal, keeping the matcher's full-frame coordinates."""
        zone = self._config.red_icon_zones.upgrade_zone
        if frame.shape[0] <= zone.y_min:
            return False
        region = BoundingBox(zone.x_min, zone.y_min, zone.x_max + 1,
                             frame.shape[0] if zone.y_max is None else zone.y_max + 1)
        return self.split_red_icons(self.find_red_icons(frame, search_region=region))[2]

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
        red = hsv_ranges_mask(hsv, config.block_ranges)
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

    def new_level_near_miss(self, frame: np.ndarray) -> list[TemplateExplanation]:
        """Best raw `newLevel` ("Open" button) match with the threshold NOT applied. On a dimmed
        screen it reads far below the gate (a uniform dim of k scores about 1 - (1-k)^2/k) while its
        centre may still name the button, so the early probe logs where it sits and how far under
        the gate it is. One item, or none when the template did not load. Diagnostics only."""
        if NEW_LEVEL_TEMPLATE not in self._loaded:
            return []
        seen = self._matcher.explain_template(frame, NEW_LEVEL_TEMPLATE)
        return [] if seen is None else [seen]

    def close_button_near_misses(self, frame: np.ndarray) -> list[TemplateExplanation]:
        """Best raw match per loaded red-X template with the threshold, the red-block prefilter and
        the HSV gate NOT applied (the gate's ratio is reported): what the X rule saw when it found
        nothing on a screen where an X is plainly visible. Diagnostics only (the early probe)."""
        config = self._config.close_button
        explained = (
            self._matcher.explain_template(frame, name, config.hsv)
            for name in config.template_names
            if name in self._loaded
        )
        return [item for item in explained if item is not None]

    def _gather(
        self,
        frame: np.ndarray,
        template_names: tuple[str, ...],
        threshold: float,
        min_distance: int,
        hsv_gate: HsvGate | None,
        min_matches: int,
        iou_threshold: float,
        *, search_region: BoundingBox | None = None,
    ) -> list[MatchCandidate]:
        raw = self._matcher.find_candidates_across_templates(
            frame, template_names, threshold, min_distance, hsv_gate, search_region=search_region,
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
