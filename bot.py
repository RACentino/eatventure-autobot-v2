import logging
import math
import threading
import time
from collections import deque
from collections.abc import Iterable
from enum import Enum, auto
from pathlib import Path
from typing import Any

import config
from image_matcher import ImageMatcher
from mouse_controller import MouseController
from telegram_notifier import TelegramNotifier
from window_capture import (
    WindowCapture,
    WindowCaptureError,
    WindowNotAvailableError,
)

logger = logging.getLogger(__name__)

TemplatePair = tuple[Any, Any]
BoxCandidate = tuple[float, int, int, int, int, str]
RedIcon = tuple[float, int, int]
ForbiddenZoneBounds = tuple[int, int, int, int]
RUNTIME_TEMPLATE_NAMES = (
    *config.RED_ICON_NORMAL_TEMPLATE_NAMES,
    *(f"box{index}" for index in range(1, 5)),
    "newLevel",
    "unlock",
    "upgradeStation",
)


class State(Enum):
    FIND_RED_ICONS = auto()
    CLICK_RED_ICON = auto()
    CHECK_UNLOCK = auto()
    SEARCH_UPGRADE_STATION = auto()
    HOLD_UPGRADE_STATION = auto()
    OPEN_BOXES = auto()
    UPGRADE_STATS = auto()
    SCROLL = auto()
    CHECK_NEW_LEVEL = auto()
    TRANSITION_LEVEL = auto()
    WAIT_FOR_UNLOCK = auto()


class EatventureBot:
    def __init__(self) -> None:
        logger.info("Initializing Eatventure Bot")
        self._running = threading.Event()
        self._stop_requested = threading.Event()
        self._state_operation_lock = threading.RLock()

        self.image_matcher = ImageMatcher(config.MATCH_THRESHOLD)
        self.templates = self.load_templates()
        self._red_icon_template_names_cache: list[str] | None = None
        self._red_icon_template_names_cache = self._red_icon_template_names()
        self._red_icon_max_width, self._red_icon_max_height = (
            self._red_icon_template_span()
        )

        self.window_capture = WindowCapture(
            config.WINDOW_TITLE,
            config.WINDOW_WIDTH,
            config.WINDOW_HEIGHT,
            stop_event=self._stop_requested,
        )
        try:
            self.mouse_controller = MouseController(
                self.window_capture,
                click_delay=config.CLICK_DELAY,
                move_delay=config.MOUSE_MOVE_DELAY,
                stop_event=self._stop_requested,
            )
        except Exception:
            self.window_capture.close()
            raise
        self.state = State.FIND_RED_ICONS
        self._handlers = {
            State.FIND_RED_ICONS: self.handle_find_red_icons,
            State.CLICK_RED_ICON: self.handle_click_red_icon,
            State.CHECK_UNLOCK: self.handle_check_unlock,
            State.SEARCH_UPGRADE_STATION: self.handle_search_upgrade_station,
            State.HOLD_UPGRADE_STATION: self.handle_hold_upgrade_station,
            State.OPEN_BOXES: self.handle_open_boxes,
            State.UPGRADE_STATS: self.handle_upgrade_stats,
            State.SCROLL: self.handle_scroll,
            State.CHECK_NEW_LEVEL: self.handle_check_new_level,
            State.TRANSITION_LEVEL: self.handle_transition_level,
            State.WAIT_FOR_UNLOCK: self.handle_wait_for_unlock,
        }
        self.telegram = TelegramNotifier(
            config.TELEGRAM_BOT_TOKEN,
            config.TELEGRAM_CHAT_ID,
            config.TELEGRAM_ENABLED,
        )

        self._initialize_runtime_state()

        logger.info("Bot initialized successfully")

    @property
    def running(self) -> bool:
        return self._running.is_set()

    @running.setter
    def running(self, is_running: bool) -> None:
        if is_running:
            self._running.set()
            return
        self._running.clear()

    def _initialize_runtime_state(self) -> None:
        self.running = False
        self.red_icon_fast_mode = True
        self.red_icons: list[RedIcon] = []
        self.current_red_icon_index = 0
        self.wait_for_unlock_attempts = 0
        self.max_wait_for_unlock_attempts = max(1, int(config.UNLOCK_SEARCH_ATTEMPTS))
        self.work_done = False
        self.cycle_counter = 0
        self.upgrade_station_counter = 0
        self.successful_red_icon_positions: deque[int] = deque(maxlen=24)
        self.upgrade_found_in_cycle = False
        self.consecutive_failed_cycles = 0
        self.total_levels_completed = 0
        self.current_level_start_time: float | None = None
        self.upgrade_station_pos: tuple[int, int] | None = None
        self._oscillation_cycle_index = 1
        self._oscillation_leg_direction = 1
        self._oscillation_leg_progress = 0
        self._new_level_red_icon_verified = False
        self._state_entered_at = time.monotonic()

    def set_event_forbidden_zone(self, bounds: ForbiddenZoneBounds) -> None:
        self.mouse_controller.set_event_forbidden_zone(bounds)

    def toggle_red_icon_mode(self) -> str:
        if self.running:
            return "Fast" if self.red_icon_fast_mode else "Normal"
        self.red_icon_fast_mode = not self.red_icon_fast_mode
        return "Fast" if self.red_icon_fast_mode else "Normal"

    def load_templates(self) -> dict[str, TemplatePair]:
        templates: dict[str, TemplatePair] = {}
        templates_path = Path(config.ASSETS_DIR)
        if not templates_path.is_dir():
            logger.warning("Assets directory not found: %s", templates_path)
            return templates

        for template_name in RUNTIME_TEMPLATE_NAMES:
            template_file = templates_path / f"{template_name}.png"
            if not template_file.is_file():
                logger.warning("Template unavailable: %s", template_file.name)
                continue
            try:
                template_img = self.image_matcher.load_template(template_file)
                height, width = template_img[0].shape[:2]
                if width > config.WINDOW_WIDTH or height > config.WINDOW_HEIGHT:
                    raise ValueError(
                        f"template is {width}x{height}, larger than the target client"
                    )
                templates[template_name] = template_img
                logger.info("Loaded template: %s", template_name)
            except Exception as exc:
                logger.warning("Template unavailable: %s: %s", template_file.name, exc)

        return templates

    def _red_icon_min_matches(self, fast_mode: bool | None = None) -> int:
        use_fast_mode = self.red_icon_fast_mode if fast_mode is None else fast_mode
        if use_fast_mode:
            return 1
        available = sum(1 for name in self.templates if name.startswith("RedIcon"))
        if available <= 0:
            return 1
        configured = max(1, int(config.RED_ICON_MIN_MATCHES))
        return min(configured, available)

    def _sleep(self, duration: Any) -> bool:
        try:
            delay = float(duration)
        except (TypeError, ValueError, OverflowError):
            delay = 0.0
        if not math.isfinite(delay):
            logger.error("Rejected non-finite wait duration: %r", duration)
            delay = 0.0
        delay = max(0.0, delay)
        return not self._stop_requested.wait(delay)

    def _click_idle(self) -> bool:
        return self.mouse_controller.click(
            config.IDLE_CLICK_POS[0], config.IDLE_CLICK_POS[1], relative=True
        )

    def _scrcpy_miss_recovery_sleep(self, duration: Any) -> bool:
        try:
            delay = max(0.0, float(duration))
        except (TypeError, ValueError):
            delay = 0.0
        if delay <= 0:
            return True
        return self._sleep(delay)

    def _scan_red_icon_frame(
        self,
        screenshot: Any,
        limited_screenshot: Any,
        scan_threshold: float,
        min_matches: int,
        fast_mode: bool | None = None,
    ) -> tuple[list[RedIcon], RedIcon | None]:
        all_detections = self._collect_red_icon_detections(
            limited_screenshot,
            scan_threshold,
            min_distance=80,
            fast_mode=fast_mode,
        )
        red_icons = self._icons_from_detections(all_detections, min_matches)
        best_new_level_icon = self._find_new_level_red_icon(
            screenshot, scan_threshold, min_matches, fast_mode
        )
        return red_icons, best_new_level_icon

    def _find_new_level_button(self, screenshot: Any) -> tuple[bool, float, int, int]:
        template_pair = self.templates.get("newLevel")
        if template_pair is None:
            return False, 0.0, 0, 0
        template, mask = template_pair
        return self.image_matcher.find_template(
            screenshot,
            template,
            mask=mask,
            threshold=config.NEW_LEVEL_THRESHOLD,
            template_name="newLevel",
        )

    def _red_icon_template_names(self) -> list[str]:
        cached = self._red_icon_template_names_cache
        if cached is not None:
            return cached
        return [
            name
            for name in config.RED_ICON_NORMAL_TEMPLATE_NAMES
            if name in self.templates
        ]

    def _red_icon_template_span(self) -> tuple[int, int]:
        max_width = 0
        max_height = 0
        for template_name in self._red_icon_template_names():
            if template_name not in self.templates:
                continue
            template, _ = self.templates[template_name]
            max_height = max(max_height, int(template.shape[0]))
            max_width = max(max_width, int(template.shape[1]))
        return max_width, max_height

    @staticmethod
    def _extract_region(
        screenshot: Any,
        x_min: Any,
        x_max: Any,
        y_min: Any,
        y_max: Any,
        pad_x: Any = 0,
        pad_y: Any = 0,
    ) -> tuple[Any, int, int]:
        height, width = screenshot.shape[:2]
        left = max(0, int(x_min) - int(pad_x))
        right = min(width, int(x_max) + int(pad_x))
        top = max(0, int(y_min) - int(pad_y))
        bottom = min(height, int(y_max) + int(pad_y))
        if left >= right or top >= bottom:
            return screenshot[0:0, 0:0], 0, 0
        return screenshot[top:bottom, left:right], left, top

    @staticmethod
    def _merge_icon_detection(
        detections: dict[tuple[int, int], list[tuple[str, float]]],
        x: int,
        y: int,
        template_name: str,
        confidence: float,
    ) -> None:
        for existing_x, existing_y in list(detections.keys()):
            if abs(x - existing_x) < 10 and abs(y - existing_y) < 10:
                detections[(existing_x, existing_y)].append((template_name, confidence))
                return
        detections[(x, y)] = [(template_name, confidence)]

    def _collect_red_icon_detections(
        self,
        screenshot: Any,
        threshold: float,
        min_distance: int = 80,
        offset_x: int = 0,
        offset_y: int = 0,
        fast_mode: bool | None = None,
    ) -> dict[tuple[int, int], list[tuple[str, float]]]:
        detections: dict[tuple[int, int], list[tuple[str, float]]] = {}
        if screenshot.size == 0:
            return detections
        use_fast_mode = self.red_icon_fast_mode if fast_mode is None else fast_mode
        template_names = self._red_icon_template_names()
        if use_fast_mode:
            fast_names = [
                name
                for name in config.RED_ICON_FAST_TEMPLATE_NAMES
                if name in self.templates
            ]
            if not fast_names:
                return detections
            template_names = fast_names
            min_distance = int(config.RED_ICON_FAST_MIN_DISTANCE)
        else:
            min_distance = min(min_distance, int(config.RED_ICON_FAST_MIN_DISTANCE))
            discovery_names = [
                name
                for name in ("RedIcon4", "RedIcon8", "RedIcon6")
                if name in template_names
            ]
            if discovery_names:
                template_names = discovery_names
        hsv_mask = self.image_matcher.build_hsv_mask(
            screenshot, config.RED_ICON_HSV_RANGES
        )
        threshold = max(float(threshold), float(config.RED_ICON_THRESHOLD))
        for template_name in template_names:
            if template_name not in self.templates:
                continue
            template, mask = self.templates[template_name]
            icons = self.image_matcher.find_all_templates(
                screenshot,
                template,
                mask=mask,
                threshold=threshold,
                min_distance=min_distance,
                template_name=template_name,
                hsv_ranges=config.RED_ICON_HSV_RANGES,
                hsv_match_threshold=config.RED_ICON_HSV_MIN_MATCH_RATIO,
                hsv_mask=hsv_mask,
            )
            for confidence, x, y in icons:
                self._merge_icon_detection(
                    detections,
                    x + offset_x,
                    y + offset_y,
                    template_name,
                    confidence,
                )

        if use_fast_mode or self._red_icon_min_matches(False) < 2:
            return detections

        available_names = self._red_icon_template_names()
        for (x, y), matches in list(detections.items()):
            seen = {name for name, _ in matches}
            if len(seen) >= 2:
                continue
            local_x, local_y = x - offset_x, y - offset_y
            region, region_x, region_y = self._extract_region(
                screenshot,
                local_x,
                local_x + 1,
                local_y,
                local_y + 1,
                pad_x=self._red_icon_max_width,
                pad_y=self._red_icon_max_height,
            )
            confirmation_order = (
                ("RedIcon5", "RedIcon6", "RedIcon14", "RedIcon4", "RedIcon8")
                if seen & {"RedIcon4", "RedIcon8"}
                else ("RedIcon14", "RedIcon5", "RedIcon4", "RedIcon8", "RedIcon6")
            )
            for template_name in confirmation_order:
                if template_name in seen or template_name not in available_names:
                    continue
                template, mask = self.templates[template_name]
                confirmations = self.image_matcher.find_all_templates(
                    region,
                    template,
                    mask=mask,
                    threshold=threshold,
                    min_distance=10,
                    template_name=template_name,
                    hsv_ranges=config.RED_ICON_HSV_RANGES,
                    hsv_match_threshold=config.RED_ICON_HSV_MIN_MATCH_RATIO,
                )
                for confidence, candidate_x, candidate_y in confirmations:
                    candidate_x += region_x + offset_x
                    candidate_y += region_y + offset_y
                    if abs(candidate_x - x) >= 10 or abs(candidate_y - y) >= 10:
                        continue
                    self._merge_icon_detection(
                        detections,
                        candidate_x,
                        candidate_y,
                        template_name,
                        confidence,
                    )
                    break
                break
        return detections

    @staticmethod
    def _best_confidence_by_template(
        matches: list[tuple[str, float]],
    ) -> dict[str, float]:
        by_template: dict[str, float] = {}
        for template_name, confidence in matches:
            existing = by_template.get(template_name)
            if existing is None or confidence > existing:
                by_template[template_name] = confidence
        return by_template

    @classmethod
    def _icons_from_detections(
        cls: type["EatventureBot"],
        detections: dict[tuple[int, int], list[tuple[str, float]]],
        min_matches: int,
    ) -> list[RedIcon]:
        icons: list[RedIcon] = []
        for (x, y), matches in detections.items():
            by_template = cls._best_confidence_by_template(matches)
            if len(by_template) < min_matches:
                continue
            max_confidence = max(by_template.values())
            icons.append((max_confidence, x, y))
        return icons

    def _find_best_zone_red_icon(
        self,
        screenshot: Any,
        threshold: float,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        min_distance: int = 80,
        fast_mode: bool | None = None,
    ) -> RedIcon | None:
        region, offset_x, offset_y = self._extract_region(
            screenshot,
            x_min,
            x_max,
            y_min,
            y_max,
            pad_x=self._red_icon_max_width,
            pad_y=self._red_icon_max_height,
        )
        if region.size == 0:
            return None

        detections = self._collect_red_icon_detections(
            region,
            threshold,
            min_distance=min_distance,
            offset_x=offset_x,
            offset_y=offset_y,
            fast_mode=fast_mode,
        )
        min_matches = self._red_icon_min_matches(fast_mode)
        icons = self._icons_from_detections(detections, min_matches)
        best_match = None
        for confidence, x, y in icons:
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                continue
            if best_match is None or confidence > best_match[0]:
                best_match = (confidence, x, y)
        return best_match

    def _find_new_level_red_icon(
        self,
        screenshot: Any = None,
        scan_threshold: float | None = None,
        min_matches: int | None = None,
        fast_mode: bool | None = None,
    ) -> RedIcon | None:
        if screenshot is None:
            screenshot = self.window_capture.capture(max_y=config.EXTENDED_SEARCH_Y)
        if scan_threshold is None:
            scan_threshold = config.NEW_LEVEL_RED_ICON_THRESHOLD
        else:
            scan_threshold = min(
                float(scan_threshold), float(config.NEW_LEVEL_RED_ICON_THRESHOLD)
            )
        if min_matches is None:
            min_matches = self._red_icon_min_matches(fast_mode)

        footer_region, offset_x, offset_y = self._extract_region(
            screenshot,
            config.NEW_LEVEL_RED_ICON_X_MIN,
            config.NEW_LEVEL_RED_ICON_X_MAX,
            config.NEW_LEVEL_RED_ICON_Y_MIN,
            config.NEW_LEVEL_RED_ICON_Y_MAX,
            pad_x=self._red_icon_max_width,
            pad_y=self._red_icon_max_height,
        )
        footer_detections = self._collect_red_icon_detections(
            footer_region,
            scan_threshold,
            min_distance=80,
            offset_x=offset_x,
            offset_y=offset_y,
            fast_mode=fast_mode,
        )
        all_red_icons_extended = self._icons_from_detections(
            footer_detections, min_matches
        )

        best_new_level_icon = None
        for confidence, x, y in all_red_icons_extended:
            if not (
                config.NEW_LEVEL_RED_ICON_X_MIN <= x <= config.NEW_LEVEL_RED_ICON_X_MAX
                and config.NEW_LEVEL_RED_ICON_Y_MIN
                <= y
                <= config.NEW_LEVEL_RED_ICON_Y_MAX
            ):
                continue
            if confidence < config.NEW_LEVEL_RED_ICON_THRESHOLD:
                continue
            if best_new_level_icon is None or confidence > best_new_level_icon[0]:
                best_new_level_icon = (confidence, x, y)
        return best_new_level_icon

    def _remember_successful_red_icon_position(self, y_value: Any) -> None:
        y_value = int(y_value)
        for existing_y in self.successful_red_icon_positions:
            if abs(existing_y - y_value) < 12:
                return
        self.successful_red_icon_positions.append(y_value)

    def _record_level_completion(self) -> float:
        self.total_levels_completed += 1
        elapsed = 0.0
        completion_time = time.monotonic()
        if self.current_level_start_time is not None:
            elapsed = max(0.0, completion_time - self.current_level_start_time)
        self.current_level_start_time = completion_time
        self._reset_search_cycle()
        self.telegram.notify_new_level(self.total_levels_completed, elapsed)
        return elapsed

    def _reset_search_cycle(self) -> None:
        self.cycle_counter = 0
        self.wait_for_unlock_attempts = 0
        self._oscillation_cycle_index = 1
        self._oscillation_leg_direction = 1
        self._oscillation_leg_progress = 0
        self._new_level_red_icon_verified = False

    def _advance_oscillation_progress(self) -> None:
        target_steps = max(
            1, int(self._oscillation_cycle_index) * int(config.SCROLL_INCREMENT_STEP)
        )
        self._oscillation_leg_progress += 1
        if self._oscillation_leg_progress < target_steps:
            return
        self._oscillation_leg_progress = 0
        if self._oscillation_leg_direction > 0:
            self._oscillation_leg_direction = -1
            return
        self._oscillation_leg_direction = 1
        self._oscillation_cycle_index += 1
        if self._oscillation_cycle_index > int(config.MAX_SCROLL_CYCLES):
            self._oscillation_cycle_index = 1

    def _perform_oscillating_scroll_step(self) -> bool:
        distance = round(
            float(config.SCROLL_PIXEL_STEP) * float(config.SCROLL_DISTANCE_RATIO)
        )
        start_x, start_y = config.SCROLL_START_POS
        direction = 1 if self._oscillation_leg_direction > 0 else -1
        target_y = start_y - distance if direction > 0 else start_y + distance
        logger.info(
            "Oscillating scroll step: cycle=%s direction=%s progress=%s",
            self._oscillation_cycle_index,
            "down" if direction > 0 else "up",
            self._oscillation_leg_progress + 1,
        )
        moved = self.mouse_controller.drag(
            start_x,
            start_y,
            start_x,
            target_y,
            duration=config.SCROLL_DURATION,
            relative=True,
        )
        if not moved:
            return False
        self._advance_oscillation_progress()
        if not self._sleep(config.POST_SCROLL_SETTLE):
            return False
        return self._sleep(config.SCROLL_INTERVAL_PAUSE)

    def _find_upgrade_station_match(self, threshold: float) -> RedIcon | None:
        if "upgradeStation" not in self.templates:
            return None

        limited_screenshot = self.window_capture.capture(
            max_y=config.UPGRADE_STATION_SEARCH_Y
        )
        template, mask = self.templates["upgradeStation"]
        candidates = self.image_matcher.find_all_templates(
            limited_screenshot,
            template,
            mask=mask,
            threshold=threshold,
            min_distance=15,
            template_name="upgradeStation",
            hsv_ranges=config.UPGRADE_STATION_HSV_RANGES,
            hsv_match_threshold=config.UPGRADE_STATION_HSV_MIN_MATCH_RATIO,
        )
        if not candidates:
            return None

        for confidence, x, y in candidates:
            x = int(x)
            y = int(y)
            if not self.mouse_controller.is_in_forbidden_zone(x, y, relative=True):
                return float(confidence), x, y

        return None

    def _clickable_red_icons(self, red_icons: Iterable[RedIcon]) -> list[RedIcon]:
        return [
            (confidence, x, y)
            for confidence, x, y in red_icons
            if not self.mouse_controller.is_in_forbidden_zone(
                x + config.RED_ICON_OFFSET_X,
                y + config.RED_ICON_OFFSET_Y,
                relative=True,
            )
        ]

    def _red_icon_priority_key(self, icon: RedIcon) -> tuple[int, int, float]:
        confidence, _, y = icon
        for success_y in self.successful_red_icon_positions:
            if abs(y - success_y) < 50:
                return (0, y, -confidence)
        return (1, y, -confidence)

    def _find_verified_upgrade_station_match(
        self,
        base_threshold: float,
        relaxed_threshold: float,
    ) -> tuple[RedIcon | None, bool]:
        verify_attempts = max(1, int(config.UPGRADE_STATION_VERIFY_SEARCH_ATTEMPTS))
        for attempt in range(verify_attempts):
            current_threshold = base_threshold if attempt == 0 else relaxed_threshold
            verified_match = self._find_upgrade_station_match(current_threshold)
            if verified_match is not None:
                return verified_match, True
            if attempt < verify_attempts - 1 and not self._sleep(
                config.UPGRADE_STATION_VERIFY_SEARCH_INTERVAL
            ):
                return None, False
        return None, True

    def _hold_upgrade_station_until_complete(
        self,
        x: int,
        y: int,
        relaxed_threshold: float,
        hold_check_interval: float,
        hold_max_duration: float,
    ) -> tuple[bool, bool, float]:
        hold_started_at = time.monotonic()
        if not math.isfinite(hold_max_duration) or hold_max_duration <= 0.0:
            logger.error(
                "Upgrade station hold rejected because its maximum duration is not positive"
            )
            return False, True, 0.0
        station_disappeared = False

        def should_release() -> bool:
            nonlocal station_disappeared
            station_disappeared = (
                self._find_upgrade_station_match(relaxed_threshold) is None
            )
            return station_disappeared

        completed = self.mouse_controller.hold_at(
            x,
            y,
            hold_max_duration,
            hold_check_interval,
            should_release,
            relative=True,
        )
        elapsed = time.monotonic() - hold_started_at
        return completed, station_disappeared, elapsed

    def _box_template_names(self) -> list[str]:
        return [
            box_name
            for box_name in ("box1", "box2", "box3", "box4")
            if box_name in self.templates
        ]

    def _collect_box_candidates(
        self, limited_screenshot: Any, box_threshold: float
    ) -> list[BoxCandidate]:
        box_candidates: list[BoxCandidate] = []
        box_template_names = self._box_template_names()
        hsv_mask = self.image_matcher.build_hsv_mask(
            limited_screenshot, config.BOX_HSV_RANGES
        )
        for box_name in box_template_names:
            template, mask = self.templates[box_name]
            candidates = self.image_matcher.find_template_candidates(
                limited_screenshot,
                template,
                mask=mask,
                threshold=box_threshold,
                min_distance=12,
                template_name=box_name,
                hsv_ranges=config.BOX_HSV_RANGES,
                hsv_match_threshold=config.BOX_HSV_MIN_MATCH_RATIO,
                hsv_mask=hsv_mask,
            )
            for confidence, x, y, candidate_width, candidate_height in candidates:
                candidate_width = int(candidate_width)
                candidate_height = int(candidate_height)
                box_candidates.append(
                    (
                        confidence,
                        int(x),
                        int(y),
                        candidate_width,
                        candidate_height,
                        box_name,
                    )
                )
        minimum_matches = min(
            max(1, int(config.BOXES_MIN_MATCHES)), len(box_template_names)
        )
        groups: list[list[BoxCandidate]] = []
        for candidate in box_candidates:
            for group in groups:
                if (
                    abs(candidate[1] - group[0][1]) < 10
                    and abs(candidate[2] - group[0][2]) < 10
                ):
                    group.append(candidate)
                    break
            else:
                groups.append([candidate])
        return [
            max(group, key=lambda candidate: candidate[0])
            for group in groups
            if len({candidate[5] for candidate in group}) >= minimum_matches
        ]

    def _click_box_candidates(self, scheduled_boxes: list[BoxCandidate]) -> int:
        boxes_found = 0
        for _, x, y, _, _, _ in scheduled_boxes:
            if self.mouse_controller.is_in_forbidden_zone(x, y, relative=True):
                logger.debug("Box candidate is in a forbidden zone")
                continue
            if self.mouse_controller.click(x, y, relative=True):
                boxes_found += 1
        return boxes_found

    def _next_state_after_box_cycle(self) -> State:
        if (
            self.consecutive_failed_cycles
            >= config.FAILED_UPGRADE_SEARCHES_BEFORE_SCROLL
        ):
            self.consecutive_failed_cycles = 0
            self.cycle_counter = 0
            logger.info("Repeated search failures reached threshold, forcing scroll")
            return State.SCROLL

        if self.upgrade_found_in_cycle:
            self.upgrade_found_in_cycle = False
            self.cycle_counter = 0
            logger.info("Upgrade found in cycle, staying in current area")
            return State.FIND_RED_ICONS

        if self.work_done:
            self.cycle_counter = 0
            logger.info("Work completed in current area, rescanning before scrolling")
            return State.FIND_RED_ICONS

        max_idle_pass_attempts = max(1, int(config.MAX_IDLE_PASS_ATTEMPTS))
        self.cycle_counter += 1
        logger.info(
            "No work detected in current area (idle pass %s/%s)",
            self.cycle_counter,
            max_idle_pass_attempts,
        )
        if self.cycle_counter >= max_idle_pass_attempts:
            self.cycle_counter = 0
            return State.SCROLL

        return State.FIND_RED_ICONS

    def start(self) -> bool:
        if not self._state_operation_lock.acquire(blocking=False):
            logger.warning("Cannot start bot while a state operation is active")
            return False
        try:
            if self.running:
                return True
            self._stop_requested.clear()
            try:
                self.window_capture.ensure_window(resize=True)
            except WindowCaptureError as exc:
                logger.error("Cannot start bot: %s", exc)
                self.running = False
                self._stop_requested.set()
                return False
            if not self.mouse_controller.is_target_foreground():
                logger.error(
                    "Cannot start bot because '%s' is not the foreground window",
                    config.WINDOW_TITLE,
                )
                self.running = False
                self._stop_requested.set()
                return False
            self.running = True
            self._state_entered_at = time.monotonic()
            if self.current_level_start_time is None:
                self.current_level_start_time = time.monotonic()
            return True
        finally:
            self._state_operation_lock.release()

    def request_stop(self) -> None:
        self._stop_requested.set()

    def stop(self) -> None:
        self.request_stop()
        with self._state_operation_lock:
            self.running = False
            self.mouse_controller.release_left_button()
            self.state = State.FIND_RED_ICONS
            self._reset_search_cycle()
            self.red_icons.clear()
            self.current_red_icon_index = 0
            self.upgrade_station_pos = None
            self.upgrade_found_in_cycle = False
            self.work_done = False
            self.consecutive_failed_cycles = 0

    def close(self) -> None:
        self.stop()
        self.telegram.close()
        self.window_capture.close()

    def step(self) -> bool:
        if not self._state_operation_lock.acquire(blocking=False):
            logger.warning("Ignoring reentrant bot step")
            return False
        try:
            if not self.running:
                return False
            if self._stop_requested.is_set():
                self.stop()
                return False
            self.window_capture.ensure_window(resize=True)
            if not self.mouse_controller.is_target_foreground():
                logger.error(
                    "Window '%s' lost foreground ownership; stopping bot",
                    config.WINDOW_TITLE,
                )
                self.stop()
                return False
            previous_state = self.state
            current_state = self._handlers[previous_state]()
            if not self.running:
                return False
            if not self.mouse_controller.is_target_foreground():
                logger.error(
                    "Window '%s' lost foreground ownership during %s; stopping bot",
                    config.WINDOW_TITLE,
                    previous_state.name,
                )
                self.stop()
                return False
            if not isinstance(current_state, State):
                raise TypeError(
                    f"Handler for {previous_state.name} returned {current_state!r}"
                )
            self.state = current_state
            now = time.monotonic()
            if current_state != previous_state:
                self._state_entered_at = now
            elif now - self._state_entered_at >= config.STATE_STALL_TIMEOUT_SECONDS:
                logger.warning(
                    "State %s stalled; resetting search flow", current_state.name
                )
                self._reset_search_cycle()
                self.state = State.FIND_RED_ICONS
                self._state_entered_at = now
            return True
        except (WindowNotAvailableError, WindowCaptureError) as exc:
            logger.error("Stopping bot: %s", exc)
            self.stop()
            return False
        except Exception:
            logger.exception("Stopping bot due to unexpected state-handler failure")
            self.stop()
            return False
        finally:
            self._state_operation_lock.release()

    def _state_from_red_icon_scan(self, best_new_level_icon: RedIcon | None) -> State:
        if best_new_level_icon is not None:
            logger.info(
                "New level red icon detected at (%s, %s) [%.3f]",
                best_new_level_icon[1],
                best_new_level_icon[2],
                best_new_level_icon[0],
            )
            self._new_level_red_icon_verified = False
            return State.CHECK_NEW_LEVEL

        if not self.red_icons:
            return State.OPEN_BOXES

        filtered_icons = self._clickable_red_icons(self.red_icons)
        if not filtered_icons:
            logger.info("No valid red icons after forbidden-zone filtering")
            return State.OPEN_BOXES

        self.red_icons = sorted(filtered_icons, key=self._red_icon_priority_key)
        self.current_red_icon_index = 0
        self.cycle_counter = 0
        logger.info("%s red icons ready to process", len(self.red_icons))
        return State.CLICK_RED_ICON

    def handle_find_red_icons(self) -> State:
        if not self._click_idle():
            return State.FIND_RED_ICONS

        self.work_done = False

        screenshot = self.window_capture.capture(max_y=config.EXTENDED_SEARCH_Y)
        limited_screenshot = screenshot[: config.MAX_SEARCH_Y, :]

        found, _, x, y = self._find_new_level_button(limited_screenshot)
        if found:
            self.cycle_counter = 0
            logger.info("newLevel.png found at (%s, %s)", x, y)
            return State.TRANSITION_LEVEL

        scan_threshold = config.RED_ICON_THRESHOLD

        min_matches = self._red_icon_min_matches()
        self.red_icons, best_new_level_icon = self._scan_red_icon_frame(
            screenshot,
            limited_screenshot,
            scan_threshold,
            min_matches,
        )

        if (
            not self.red_icons
            and best_new_level_icon is None
            and self._scrcpy_miss_recovery_sleep(
                config.SCRCPY_RED_ICON_MISS_RECOVERY_DELAY
            )
        ):
            screenshot = self.window_capture.capture(max_y=config.EXTENDED_SEARCH_Y)
            limited_screenshot = screenshot[: config.MAX_SEARCH_Y, :]
            found, _, x, y = self._find_new_level_button(limited_screenshot)
            if found:
                self.cycle_counter = 0
                logger.info(
                    "newLevel.png found at (%s, %s) after SCRCPY recovery", x, y
                )
                return State.TRANSITION_LEVEL
            self.red_icons, best_new_level_icon = self._scan_red_icon_frame(
                screenshot,
                limited_screenshot,
                scan_threshold,
                min_matches,
            )
            if (
                not self.red_icons
                and best_new_level_icon is None
                and self.red_icon_fast_mode
            ):
                logger.info(
                    "Fast red-icon scan missed twice; retrying second frame in Normal mode"
                )
                self.red_icons, best_new_level_icon = self._scan_red_icon_frame(
                    screenshot,
                    limited_screenshot,
                    scan_threshold,
                    self._red_icon_min_matches(False),
                    fast_mode=False,
                )
        return self._state_from_red_icon_scan(best_new_level_icon)

    def handle_click_red_icon(self) -> State:
        if self.current_red_icon_index >= len(self.red_icons):
            logger.info("All red icons processed, continuing cycle")
            return State.OPEN_BOXES

        confidence, x, y = self.red_icons[self.current_red_icon_index]
        click_x = x + config.RED_ICON_OFFSET_X
        click_y = y + config.RED_ICON_OFFSET_Y

        clicked = self.mouse_controller.click(click_x, click_y, relative=True)
        if not clicked:
            logger.warning("Red icon click failed at (%s, %s)", click_x, click_y)
            self.current_red_icon_index += 1
            if self.current_red_icon_index < len(self.red_icons):
                return State.CLICK_RED_ICON
            return State.OPEN_BOXES

        logger.info(
            "Clicked red icon %s/%s at (%s, %s) [%.3f]",
            self.current_red_icon_index + 1,
            len(self.red_icons),
            click_x,
            click_y,
            confidence,
        )
        self.work_done = True
        return State.CHECK_UNLOCK

    def handle_check_unlock(self) -> State:
        if not self._sleep(config.SCRCPY_ACTION_SETTLE_DELAY):
            return State.CHECK_UNLOCK

        limited_screenshot = self.window_capture.capture(max_y=config.MAX_SEARCH_Y)

        template_pair = self.templates.get("unlock")
        if template_pair is None:
            return State.SEARCH_UPGRADE_STATION

        template, mask = template_pair
        found, confidence, x, y = self.image_matcher.find_template(
            limited_screenshot,
            template,
            mask=mask,
            threshold=config.UNLOCK_THRESHOLD,
            template_name="unlock",
        )
        if not found or self.mouse_controller.is_in_forbidden_zone(x, y, relative=True):
            return State.SEARCH_UPGRADE_STATION

        logger.info("Unlock found at (%s, %s) [%.3f]", x, y, confidence)
        if not self.mouse_controller.click(x, y, relative=True):
            logger.warning("Unlock click failed at (%s, %s)", x, y)
            return State.CHECK_UNLOCK

        if not self._sleep(config.SCRCPY_ACTION_SETTLE_DELAY):
            return State.CHECK_UNLOCK
        return State.SEARCH_UPGRADE_STATION

    def handle_search_upgrade_station(self) -> State:
        base_threshold = config.UPGRADE_STATION_THRESHOLD
        relaxed_threshold = max(0.0, base_threshold - 0.05)
        max_attempts = max(1, int(config.UPGRADE_SEARCH_ATTEMPTS))

        for attempt in range(max_attempts):
            if "upgradeStation" not in self.templates:
                break

            current_threshold = base_threshold if attempt < 2 else relaxed_threshold
            match = self._find_upgrade_station_match(current_threshold)
            if match is not None:
                _, x, y = match
                logger.info(
                    "Upgrade station found at (%s, %s) on attempt %s", x, y, attempt + 1
                )
                self.upgrade_station_pos = (x, y)
                self.upgrade_found_in_cycle = True
                self.consecutive_failed_cycles = 0
                self.cycle_counter = 0
                return State.HOLD_UPGRADE_STATION

            if attempt < max_attempts - 1 and not self._sleep(
                config.UPGRADE_SEARCH_INTERVAL
            ):
                return State.OPEN_BOXES

        self.consecutive_failed_cycles += 1
        logger.info("Upgrade station not found, returning to OPEN_BOXES")
        return State.OPEN_BOXES

    def _verify_upgrade_station_hold_target(
        self,
        x: int,
        y: int,
    ) -> tuple[RedIcon, float, float] | None:
        logger.info(
            "Visually verifying upgrade station at (%s, %s) before holding", x, y
        )
        if not self._sleep(config.UPGRADE_STATION_VERIFY_SETTLE_DELAY):
            return None

        base_threshold = config.UPGRADE_STATION_THRESHOLD
        relaxed_threshold = max(0.0, base_threshold - 0.05)
        verified_match, verification_completed = (
            self._find_verified_upgrade_station_match(
                base_threshold,
                relaxed_threshold,
            )
        )
        if not verification_completed:
            return None

        if verified_match is not None:
            _, verified_x, verified_y = verified_match
            logger.info(
                "Single-clicking visually verified upgrade station at (%s, %s)",
                verified_x,
                verified_y,
            )
            clicked = self.mouse_controller.precise_click(
                verified_x, verified_y, relative=True
            )
            if not clicked:
                logger.warning(
                    "Upgrade station verification click failed at (%s, %s)",
                    verified_x,
                    verified_y,
                )
                return None
            if not self._sleep(config.UPGRADE_STATION_VERIFY_SETTLE_DELAY):
                return None
            verified_match, verification_completed = (
                self._find_verified_upgrade_station_match(
                    base_threshold,
                    relaxed_threshold,
                )
            )
            if not verification_completed:
                return None

        if verified_match is None:
            logger.info(
                "Upgrade station was not visible during visual verification; continuing main flow"
            )
            self.upgrade_station_pos = None
            self.upgrade_found_in_cycle = False
            return None

        confidence, verified_x, verified_y = verified_match
        self.upgrade_station_pos = (verified_x, verified_y)
        logger.info(
            "Upgrade station verified active at (%s, %s) [%.3f]",
            verified_x,
            verified_y,
            confidence,
        )
        return verified_match, base_threshold, relaxed_threshold

    def handle_hold_upgrade_station(self) -> State:
        if not self.upgrade_station_pos:
            return State.OPEN_BOXES

        x, y = self.upgrade_station_pos
        if self.mouse_controller.is_in_forbidden_zone(x, y, relative=True):
            logger.warning(
                "Upgrade station blocked by forbidden zone at (%s, %s)", x, y
            )
            return State.OPEN_BOXES

        hold_check_interval = max(
            config.UPGRADE_HOLD_CHECK_INTERVAL_MIN,
            min(
                config.UPGRADE_HOLD_CHECK_INTERVAL_MAX,
                float(config.UPGRADE_STATION_VERIFY_SEARCH_INTERVAL),
            ),
        )
        verified_target = self._verify_upgrade_station_hold_target(x, y)
        if verified_target is None:
            return State.OPEN_BOXES
        (_, x, y), _, relaxed_threshold = verified_target
        logger.info("Press-and-holding upgrade station at (%s, %s)", x, y)
        hold_completed, station_disappeared, hold_elapsed = (
            self._hold_upgrade_station_until_complete(
                x,
                y,
                relaxed_threshold,
                hold_check_interval,
                float(config.CLICK_HOLD_MAX_DURATION),
            )
        )
        if not hold_completed:
            return State.OPEN_BOXES
        if station_disappeared:
            logger.info(
                "Upgrade station no longer detected after %.2fs hold", hold_elapsed
            )
        else:
            logger.info(
                "Upgrade station hold released at the %.2fs limit", hold_elapsed
            )
        if self.current_red_icon_index < len(self.red_icons):
            _, _, red_y = self.red_icons[self.current_red_icon_index]
            self._remember_successful_red_icon_position(red_y)
        self.upgrade_station_pos = None
        if not self._click_idle():
            return State.OPEN_BOXES
        if not self._sleep(config.SCRCPY_ACTION_SETTLE_DELAY):
            return State.OPEN_BOXES
        self.upgrade_station_counter += 1
        if self.upgrade_station_counter >= config.UPGRADES_BEFORE_STATS:
            self.upgrade_station_counter = 0
            logger.info("Upgrade counter reached stats threshold")
            return State.UPGRADE_STATS

        return State.OPEN_BOXES

    def handle_upgrade_stats(self) -> State:
        if not self._click_idle():
            return State.OPEN_BOXES

        best_stats_match = None
        for attempt in range(2):
            screenshot = self.window_capture.capture(max_y=config.EXTENDED_SEARCH_Y)
            limited_screenshot = screenshot[: config.MAX_SEARCH_Y, :]

            found, _, _, _ = self._find_new_level_button(limited_screenshot)
            if found:
                return State.TRANSITION_LEVEL

            best_stats_match = self._find_best_zone_red_icon(
                screenshot,
                config.STATS_RED_ICON_THRESHOLD,
                config.UPGRADE_RED_ICON_X_MIN,
                config.UPGRADE_RED_ICON_X_MAX,
                config.UPGRADE_RED_ICON_Y_MIN,
                config.UPGRADE_RED_ICON_Y_MAX,
                min_distance=80,
            )
            if best_stats_match is None and attempt and self.red_icon_fast_mode:
                best_stats_match = self._find_best_zone_red_icon(
                    screenshot,
                    config.STATS_RED_ICON_THRESHOLD,
                    config.UPGRADE_RED_ICON_X_MIN,
                    config.UPGRADE_RED_ICON_X_MAX,
                    config.UPGRADE_RED_ICON_Y_MIN,
                    config.UPGRADE_RED_ICON_Y_MAX,
                    min_distance=80,
                    fast_mode=False,
                )
            if best_stats_match is not None:
                break
            if attempt == 0 and self._scrcpy_miss_recovery_sleep(
                config.SCRCPY_RED_ICON_MISS_RECOVERY_DELAY
            ):
                logger.info("No stats icon detected; retrying after capture recovery")
                continue
            logger.info(
                "No stats icon detected%s", " after recovery" if attempt else ""
            )
            return State.SCROLL

        self.cycle_counter = 0
        if best_stats_match is None:
            return State.SCROLL
        confidence, icon_x, icon_y = best_stats_match
        logger.info(
            "Stats icon found at (%s, %s) with confidence %.3f; upgrading",
            icon_x,
            icon_y,
            confidence,
        )
        opened = self.mouse_controller.click(
            config.STATS_UPGRADE_BUTTON_POS[0],
            config.STATS_UPGRADE_BUTTON_POS[1],
            relative=True,
        )
        if not opened:
            return State.OPEN_BOXES

        if not self._sleep(config.SCRCPY_ACTION_SETTLE_DELAY):
            return State.OPEN_BOXES
        clicked = self.mouse_controller.spam_click_at(
            config.STATS_UPGRADE_POS[0],
            config.STATS_UPGRADE_POS[1],
            duration=config.STATS_UPGRADE_CLICK_DURATION,
            click_delay=config.STATS_UPGRADE_CLICK_DELAY,
            mouse_down_duration=config.STATS_UPGRADE_CLICK_DELAY,
            mouse_up_duration=0.0,
            relative=True,
        )
        if not clicked:
            logger.warning(
                "Stats upgrade spam-click failed at %s", config.STATS_UPGRADE_POS
            )
            return State.OPEN_BOXES

        if not self._click_idle():
            return State.OPEN_BOXES
        logger.info("Stats upgrade completed")
        return State.OPEN_BOXES

    def handle_open_boxes(self) -> State:
        if not self._click_idle():
            return State.OPEN_BOXES

        limited_screenshot = self.window_capture.capture(max_y=config.BOX_SEARCH_Y)

        found, _, _, _ = self._find_new_level_button(limited_screenshot)
        if found:
            logger.info("New level found while opening boxes")
            return State.TRANSITION_LEVEL

        box_threshold = config.BOX_THRESHOLD
        box_candidates = self._collect_box_candidates(limited_screenshot, box_threshold)
        if not box_candidates and self._scrcpy_miss_recovery_sleep(
            config.SCRCPY_BOX_MISS_RECOVERY_DELAY
        ):
            limited_screenshot = self.window_capture.capture(max_y=config.BOX_SEARCH_Y)
            found, _, _, _ = self._find_new_level_button(limited_screenshot)
            if found:
                logger.info("New level found while opening boxes after SCRCPY recovery")
                return State.TRANSITION_LEVEL
            box_candidates = self._collect_box_candidates(
                limited_screenshot, box_threshold
            )

        merged_boxes = self.image_matcher.suppress_overlaps(
            box_candidates, config.BOX_NMS_IOU_THRESHOLD
        )
        boxes_found = self._click_box_candidates(merged_boxes)

        if boxes_found > 0:
            self.work_done = True
            self.cycle_counter = 0
            logger.info("Opened %s boxes", boxes_found)

        return self._next_state_after_box_cycle()

    def handle_scroll(self) -> State:
        if not self._click_idle():
            return State.SCROLL
        if not self._perform_oscillating_scroll_step():
            return State.SCROLL
        self.cycle_counter = 0
        return State.FIND_RED_ICONS

    def handle_check_new_level(self) -> State:
        if not self._click_idle():
            logger.warning("Failed to clear focus before confirming the new level")
            return State.CHECK_NEW_LEVEL
        if not self._sleep(config.FOCUS_SETTLE_DELAY):
            return State.CHECK_NEW_LEVEL
        if not self._new_level_red_icon_verified:
            confirmed_icon = None
            for attempt in range(2):
                screenshot = self.window_capture.capture(max_y=config.EXTENDED_SEARCH_Y)
                confirmed_icon = self._find_new_level_red_icon(screenshot)
                if confirmed_icon is not None:
                    break
                if attempt and self.red_icon_fast_mode:
                    confirmed_icon = self._find_new_level_red_icon(
                        screenshot, fast_mode=False
                    )
                    if confirmed_icon is not None:
                        break
                if attempt == 0 and not self._sleep(
                    config.SCRCPY_RED_ICON_MISS_RECOVERY_DELAY
                ):
                    return State.CHECK_NEW_LEVEL
            if confirmed_icon is None:
                logger.warning(
                    "New-level red icon was stale on two fresh frames; rescanning"
                )
                return State.FIND_RED_ICONS

            self._new_level_red_icon_verified = True
            logger.info(
                "New level red icon confirmed at (%s, %s) [%.3f]",
                confirmed_icon[1],
                confirmed_icon[2],
                confirmed_icon[0],
            )

        opened = self.mouse_controller.click(
            config.NEW_LEVEL_BUTTON_POS[0],
            config.NEW_LEVEL_BUTTON_POS[1],
            relative=True,
        )
        if not opened:
            logger.warning(
                "Failed to click the new level button at %s",
                config.NEW_LEVEL_BUTTON_POS,
            )
            return State.CHECK_NEW_LEVEL
        if not self._sleep(config.NEW_LEVEL_CONFIRMATION_DELAY):
            return State.CHECK_NEW_LEVEL
        advanced = self.mouse_controller.click(
            config.LEVEL_TRANSITION_POS[0],
            config.LEVEL_TRANSITION_POS[1],
            relative=True,
        )
        if not advanced:
            logger.warning(
                "Failed to click the level transition button at %s",
                config.LEVEL_TRANSITION_POS,
            )
            return State.CHECK_NEW_LEVEL
        if not self._sleep(config.LEVEL_TRANSITION_SECONDARY_SETTLE_DELAY):
            return State.WAIT_FOR_UNLOCK
        logger.info(
            "Verified red-icon level transition submitted; awaiting unlock confirmation"
        )
        return State.WAIT_FOR_UNLOCK

    def handle_transition_level(self) -> State:
        if not self._click_idle():
            return State.TRANSITION_LEVEL

        max_attempts = max(1, int(config.NEW_LEVEL_SEARCH_ATTEMPTS))
        for attempt in range(max_attempts):
            limited_screenshot = self.window_capture.capture(max_y=config.MAX_SEARCH_Y)

            found, _, x, y = self._find_new_level_button(limited_screenshot)
            if found:
                logger.info(
                    "New level button found at (%s, %s) on attempt %s",
                    x,
                    y,
                    attempt + 1,
                )
                clicked = self.mouse_controller.click(x, y, relative=True)
                if not clicked:
                    logger.warning("New level button click failed at (%s, %s)", x, y)
                    return State.CHECK_NEW_LEVEL
                if self._sleep(config.LEVEL_TRANSITION_SETTLE_DELAY):
                    logger.info(
                        "New-level transition submitted; awaiting unlock confirmation"
                    )
                return State.WAIT_FOR_UNLOCK

            if attempt < max_attempts - 1 and not self._sleep(
                config.NEW_LEVEL_SEARCH_INTERVAL
            ):
                return State.TRANSITION_LEVEL

        logger.warning("New level button not found after %s attempts", max_attempts)
        self._reset_search_cycle()
        return State.FIND_RED_ICONS

    def handle_wait_for_unlock(self) -> State:
        if not self._click_idle():
            logger.warning("Failed to clear focus while waiting for the next unlock")
            return State.WAIT_FOR_UNLOCK
        if not self._sleep(config.FOCUS_SETTLE_DELAY):
            return State.WAIT_FOR_UNLOCK

        self.wait_for_unlock_attempts += 1
        if self.wait_for_unlock_attempts > self.max_wait_for_unlock_attempts:
            logger.warning(
                "Unlock button not found after %s attempts, resetting",
                self.max_wait_for_unlock_attempts,
            )
            self.wait_for_unlock_attempts = 0
            self._reset_search_cycle()
            return State.FIND_RED_ICONS

        screenshot = self.window_capture.capture()
        template_pair = self.templates.get("unlock")
        if template_pair is None:
            if not self._sleep(config.UNLOCK_SEARCH_INTERVAL):
                return State.WAIT_FOR_UNLOCK
            return State.WAIT_FOR_UNLOCK

        template, mask = template_pair
        found, confidence, x, y = self.image_matcher.find_template(
            screenshot,
            template,
            mask=mask,
            threshold=config.UNLOCK_THRESHOLD,
            template_name="unlock",
        )
        if not found:
            if not self._sleep(config.UNLOCK_SEARCH_INTERVAL):
                return State.WAIT_FOR_UNLOCK
            return State.WAIT_FOR_UNLOCK

        logger.info("Unlock button found at (%s, %s) [%.3f]", x, y, confidence)
        if self.mouse_controller.is_in_forbidden_zone(x, y, relative=True):
            logger.warning("Unlock button found in forbidden zone at (%s, %s)", x, y)
            if not self._sleep(config.UNLOCK_SEARCH_INTERVAL):
                return State.WAIT_FOR_UNLOCK
            return State.WAIT_FOR_UNLOCK
        if not self.mouse_controller.click(x, y, relative=True):
            logger.warning("Unlock button click failed at (%s, %s)", x, y)
            return State.WAIT_FOR_UNLOCK

        elapsed = self._record_level_completion()
        logger.info(
            "Level %s confirmed by unlock. Time spent: %.1fs",
            self.total_levels_completed,
            elapsed,
        )
        if not self._sleep(config.UNLOCK_SETTLE_DELAY):
            return State.FIND_RED_ICONS
        return State.FIND_RED_ICONS
