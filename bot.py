import logging
import math
import threading
import time
from collections import deque
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import config
from domain import (
    BOT_STATE_LOOP_SLEEP_SECONDS,
    BOX_TEMPLATE_NAMES,
    CHECK_NEW_LEVEL_PRE_CLICK_DELAY,
    CLICK_BUCKET_SIZE_PIXELS,
    FAILED_CLICK_BUCKET_LIMIT,
    ICON_MERGE_DISTANCE_PIXELS,
    MAX_LEVEL_TRANSITION_ATTEMPTS,
    MAX_UPGRADE_SEARCH_ATTEMPTS,
    RED_ICON_FALLBACK_MIN_DISTANCE,
    RED_ICON_TEMPLATE_NAMES,
    REQUIRED_TEMPLATE_NAMES,
    ROW_PRIORITY_DISTANCE_PIXELS,
    SEARCH_CYCLES_BEFORE_SCROLL,
    SUCCESSFUL_RED_ICON_ROWS_LIMIT,
    SUCCESSFUL_ROW_DEDUP_DISTANCE_PIXELS,
    TRANSITION_LEVEL_BUTTON_WAIT_SECONDS,
    TRANSITION_LEVEL_RETRY_DELAY_SECONDS,
    UPGRADE_STATION_HOLD_MAX_VERIFY_INTERVAL,
    UPGRADE_STATION_HOLD_MIN_VERIFY_INTERVAL,
    UPGRADE_STATION_THRESHOLD_RELAXATION,
    UPGRADE_STATS_CYCLE_INTERVAL,
    WAIT_FOR_UNLOCK_POST_CLICK_DELAY,
    WAIT_FOR_UNLOCK_PRE_SCAN_DELAY,
    WAIT_FOR_UNLOCK_RETRY_DELAY,
    State,
    TemplateName,
)
from image_matcher import ImageMatcher
from mouse_controller import MouseController, wait_event
from telegram_notifier import TelegramNotifier
from window_capture import WindowCapture, WindowCaptureError, WindowNotAvailableError

logger = logging.getLogger(__name__)

TemplatePair = tuple[Any, Any]
RedIcon = tuple[float, int, int]
RedIconRecord = tuple[float, int, int, str]
UpgradeStationCandidate = tuple[float, int, int, int, int]
BoxCandidate = tuple[float, int, int, int, int, str]
StateResult = State | None
UPWARD_OSCILLATING_SCROLL_DIRECTION = -1
RUNTIME_RECOVERY_RETRY_DELAY_SECONDS = 1.0
WINDOW_RECOVERY_MESSAGE = "Target window is unavailable"
SCROLL_RECOVERY_MESSAGE = "Scroll input failed repeatedly"
LEVEL_TRANSITION_RECOVERY_MESSAGE = "Level transition was not verified"


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _same_upgrade_station_target(
    locked_target: UpgradeStationCandidate,
    candidate: UpgradeStationCandidate,
) -> bool:
    _, locked_x, locked_y, locked_width, locked_height = locked_target
    _, candidate_x, candidate_y, candidate_width, candidate_height = candidate
    return abs(candidate_x - locked_x) * 2 <= max(
        candidate_width, locked_width
    ) and abs(candidate_y - locked_y) * 2 <= max(candidate_height, locked_height)


class _UpgradeStationHoldMonitor:
    def __init__(
        self,
        bot: Any,
        target: UpgradeStationCandidate,
        threshold: float,
    ) -> None:
        self.bot = bot
        self.target = target
        self.threshold = max(0.0, float(threshold))
        self.relaxed_threshold = max(
            0.0, self.threshold - UPGRADE_STATION_THRESHOLD_RELAXATION
        )
        interval = _finite_float(
            config.UPGRADE_STATION_VERIFY_SEARCH_INTERVAL,
            BOT_STATE_LOOP_SLEEP_SECONDS,
        )
        self.interval = max(
            UPGRADE_STATION_HOLD_MIN_VERIFY_INTERVAL,
            min(UPGRADE_STATION_HOLD_MAX_VERIFY_INTERVAL, interval),
        )
        self.next_verify_at = time.perf_counter() + self.interval
        self.station_lost = False

    def __call__(self) -> bool:
        if self.bot._stop_requested.is_set():
            return True
        if time.perf_counter() < self.next_verify_at:
            return False
        match = self.bot._verify_upgrade_station(
            self.target,
            self.threshold,
            self.relaxed_threshold,
            wait_between_attempts=False,
        )
        if match is None:
            self.station_lost = True
            return True
        self.target = match
        self.next_verify_at = time.perf_counter() + self.interval
        return False


class EatventureBot:
    def __init__(self) -> None:
        logger.info("Initializing Eatventure Bot")
        self._stop_requested = threading.Event()
        self._lifecycle_lock = threading.RLock()
        self._step_lock = threading.Lock()
        with ExitStack() as pending_resources:
            self.window_capture = WindowCapture(
                config.WINDOW_TITLE, config.WINDOW_WIDTH, config.WINDOW_HEIGHT
            )
            pending_resources.callback(self.window_capture.close)
            self.image_matcher = ImageMatcher(config.MATCH_THRESHOLD)
            self.mouse_controller = MouseController(
                self.window_capture.get_input_window_rect,
                config.CLICK_DELAY,
                config.MOUSE_MOVE_DELAY,
                stop_event=self._stop_requested,
            )
            self.telegram = TelegramNotifier(
                config.TELEGRAM_BOT_TOKEN,
                config.TELEGRAM_CHAT_ID,
                config.TELEGRAM_ENABLED,
            )
            pending_resources.callback(self.telegram.close)
            self.templates = self.load_templates()
            self.state = State.FIND_RED_ICONS
            self._state_handlers = {
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
            self._reset_runtime_state()
            self.ready = self._validate_required_templates()
            pending_resources.pop_all()
        logger.info("Bot initialized successfully")

    def _reset_runtime_state(self) -> None:
        self.running = False
        self.total_levels_completed = 0
        self.current_level_started_at: float | None = None
        self._reset_action_state()

    def _reset_action_state(self) -> None:
        self.red_icon: RedIcon | None = None
        self.active_upgrade_station_target: UpgradeStationCandidate | None = None
        self.empty_cycle_count = 0
        self.upgrade_station_counter = 0
        self.scroll_failures = 0
        self._active_recoveries: set[str] = set()
        self.needs_dismissal = False
        self.last_clicked_pos: tuple[int, int] | None = None
        self.failed_click_tracker: dict[tuple[int, int], int] = {}
        self.successful_red_icon_positions: deque[int] = deque(
            maxlen=SUCCESSFUL_RED_ICON_ROWS_LIMIT
        )
        self.scroll_direction = 1
        self.scroll_cycle_index = 1
        self.scroll_cycle_progress = 0

    def load_templates(self) -> dict[str, TemplatePair]:
        templates: dict[str, TemplatePair] = {}
        assets_path = Path(config.ASSETS_DIR)
        if not assets_path.exists():
            logger.error("Assets directory not found: %s", assets_path)
            return templates
        for name in REQUIRED_TEMPLATE_NAMES:
            template_file = assets_path / f"{name}.png"
            try:
                templates[name] = self.image_matcher.load_template(template_file)
                logger.info("Loaded template: %s", name)
            except Exception as exc:
                logger.error("Failed to load template %s: %s", template_file, exc)
        return templates

    def _validate_required_templates(self) -> bool:
        missing = [
            name for name in REQUIRED_TEMPLATE_NAMES if name not in self.templates
        ]
        if missing:
            logger.error("Missing required templates: %s", ", ".join(missing))
            return False
        return True

    def request_stop(self) -> None:
        self._stop_requested.set()
        self.mouse_controller.release_left_button()

    def start(self) -> bool:
        with self._lifecycle_lock:
            if self.running:
                return True
            if self._step_lock.locked() or not self.ready:
                logger.warning(
                    "Cannot start bot while not ready or while a step is active"
                )
                return False
            try:
                self.window_capture.ensure_window(resize=True)
                self.window_capture.activate_for_input()
                self.window_capture.get_input_window_rect()
            except WindowCaptureError as exc:
                logger.error("Cannot start bot: %s", exc)
                return False
            self._stop_requested.clear()
            self._reset_action_state()
            self.state = State.FIND_RED_ICONS
            self.current_level_started_at = time.monotonic()
            self.running = True
            return True

    def stop(self) -> bool:
        with self._lifecycle_lock:
            self._stop_requested.set()
            self.running = False
            return self.mouse_controller.release_left_button()

    def step(self) -> bool:
        if not self._step_lock.acquire(blocking=False):
            logger.warning("Ignoring reentrant bot step")
            return False
        try:
            if not self.running:
                return False
            if self._stop_requested.is_set():
                self.stop()
                return False
            if not self.window_capture.is_window_active():
                return self._wait_for_window_recovery("Target window is not active")
            if WINDOW_RECOVERY_MESSAGE in self._active_recoveries:
                self.window_capture.ensure_window(resize=True)
                self.window_capture.get_input_window_rect()
            current_state = self.state
            handler = self._state_handlers.get(current_state)
            if handler is None:
                logger.error("No handler registered for state %s", current_state.name)
                raise RuntimeError(f"Missing handler for {current_state.name}")
            started_at = time.perf_counter()
            next_state = handler()
            if next_state is not None:
                if not isinstance(next_state, State):
                    raise TypeError(
                        f"Handler for {current_state.name} returned {next_state!r}"
                    )
                self.state = next_state
            logger.debug(
                "State step %s -> %s in %.1fms",
                current_state.name,
                self.state.name,
                (time.perf_counter() - started_at) * 1000.0,
            )
            self._complete_recovery(WINDOW_RECOVERY_MESSAGE)
            return True
        except (WindowNotAvailableError, WindowCaptureError) as exc:
            return self._wait_for_window_recovery(str(exc))
        except Exception:
            logger.exception("Stopping bot due to unexpected state-handler failure")
        finally:
            self._step_lock.release()
        self.stop()
        return False

    def _sleep(self, duration: Any) -> bool:
        return wait_event(self._stop_requested, duration)

    def _template(self, name: str) -> TemplatePair | None:
        return self.templates.get(name)

    def _click_idle(self) -> bool:
        return self.mouse_controller.click(*config.IDLE_CLICK_POS, relative=True)

    def _dismiss_if_needed(self) -> bool:
        if not self.needs_dismissal:
            return True
        if not self._click_idle():
            return False
        self.needs_dismissal = False
        return True

    def _begin_recovery(self, message: str) -> None:
        if message in self._active_recoveries:
            return
        self._active_recoveries.add(message)
        logger.error("%s; bot will retry", message)
        self.telegram.notify_failure(message)

    def _complete_recovery(self, message: str) -> None:
        if message not in self._active_recoveries:
            return
        self._active_recoveries.remove(message)
        logger.info("Recovered from: %s", message)
        self.telegram.notify_recovered(message)

    def _wait_for_window_recovery(self, detail: str) -> bool:
        self._begin_recovery(WINDOW_RECOVERY_MESSAGE)
        logger.debug("%s: %s", WINDOW_RECOVERY_MESSAGE, detail)
        if not self.mouse_controller.release_left_button():
            logger.error("Could not release mouse input during window recovery")
        return self._sleep(RUNTIME_RECOVERY_RETRY_DELAY_SECONDS)

    def _red_icon_names(self) -> tuple[str, ...]:
        names = (
            (TemplateName.RED_ICON.value,)
            if config.RED_ICON_FAST_MODE_ENABLED
            else RED_ICON_TEMPLATE_NAMES
        )
        return tuple(name for name in names if name in self.templates)

    def _red_icon_min_matches(self) -> int:
        if config.RED_ICON_FAST_MODE_ENABLED:
            return 1
        return min(
            max(1, int(config.RED_ICON_MIN_MATCHES)),
            max(1, len(self._red_icon_names())),
        )

    def _collect_red_icon_map(
        self,
        screenshot: Any,
        threshold: float,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> dict[tuple[int, int], list[tuple[str, float]]]:
        detections: dict[tuple[int, int], list[tuple[str, float]]] = {}
        if getattr(screenshot, "size", 0) == 0:
            return detections
        hsv_mask = self.image_matcher.build_hsv_mask(
            screenshot, config.RED_ICON_HSV_RANGES
        )
        min_distance = (
            config.RED_ICON_FAST_MIN_DISTANCE
            if config.RED_ICON_FAST_MODE_ENABLED
            else RED_ICON_FALLBACK_MIN_DISTANCE
        )
        for name in self._red_icon_names():
            template_pair = self._template(name)
            if template_pair is None:
                continue
            template, mask = template_pair
            matches = self.image_matcher.find_all_color_gated_templates(
                screenshot,
                template,
                mask=mask,
                threshold=threshold,
                min_distance=min_distance,
                template_name=name,
                hsv_ranges=config.RED_ICON_HSV_RANGES,
                use_supervision_nms=True,
                supervision_iou_threshold=config.SUPERVISION_RED_ICON_NMS_IOU_THRESHOLD,
                hsv_mask=hsv_mask,
            )
            height, width = template.shape[:2]
            candidates = self.image_matcher.filter_candidates_by_hsv(
                screenshot,
                [(confidence, x, y, width, height) for confidence, x, y in matches],
                template,
                mask=mask,
                hsv_ranges=config.RED_ICON_HSV_RANGES,
                hsv_match_threshold=config.RED_ICON_HSV_MIN_MATCH_RATIO,
                hsv_mask=hsv_mask,
            )
            for confidence, x, y, _, _ in candidates:
                self._merge_icon(
                    detections,
                    int(x) + offset_x,
                    int(y) + offset_y,
                    name,
                    float(confidence),
                )
        return detections

    @staticmethod
    def _merge_icon(
        detections: dict[tuple[int, int], list[tuple[str, float]]],
        x: int,
        y: int,
        name: str,
        confidence: float,
    ) -> None:
        for (existing_x, existing_y), matches in detections.items():
            if (
                abs(x - existing_x) < ICON_MERGE_DISTANCE_PIXELS
                and abs(y - existing_y) < ICON_MERGE_DISTANCE_PIXELS
            ):
                matches.append((name, confidence))
                return
        detections[(x, y)] = [(name, confidence)]

    def _red_icon_records(
        self, red_icon_map: dict[tuple[int, int], list[tuple[str, float]]]
    ) -> list[RedIconRecord]:
        records = []
        minimum_matches = self._red_icon_min_matches()
        for (x, y), matches in red_icon_map.items():
            best_by_name: dict[str, float] = {}
            for name, confidence in matches:
                best_by_name[name] = max(float(confidence), best_by_name.get(name, 0.0))
            if len(best_by_name) >= minimum_matches:
                name, confidence = max(best_by_name.items(), key=lambda item: item[1])
                records.append((confidence, x, y, name))
        return records

    def _find_new_level_button(self, screenshot: Any) -> tuple[bool, float, int, int]:
        template_pair = self._template(TemplateName.NEW_LEVEL.value)
        if template_pair is None:
            return False, 0.0, 0, 0
        template, mask = template_pair
        return self.image_matcher.find_template(
            screenshot,
            template,
            mask=mask,
            threshold=config.NEW_LEVEL_THRESHOLD,
            template_name=TemplateName.NEW_LEVEL.value,
        )

    def _find_unlock_button(self, screenshot: Any) -> tuple[bool, float, int, int]:
        template_pair = self._template(TemplateName.UNLOCK.value)
        if template_pair is None:
            return False, 0.0, 0, 0
        template, mask = template_pair
        return self.image_matcher.find_template(
            screenshot,
            template,
            mask=mask,
            threshold=config.UNLOCK_THRESHOLD,
            template_name=TemplateName.UNLOCK.value,
        )

    def _find_zone_red_icon(
        self, screenshot: Any, zone: tuple[int, int, int, int], threshold: float
    ) -> RedIcon | None:
        height_pad = width_pad = 0
        for name in self._red_icon_names():
            template_pair = self._template(name)
            if template_pair is None:
                continue
            template, _ = template_pair
            height_pad = max(height_pad, int(template.shape[0]))
            width_pad = max(width_pad, int(template.shape[1]))
        if height_pad == 0 or width_pad == 0:
            return None
        height, width = screenshot.shape[:2]
        left = max(0, zone[0] - width_pad)
        right = min(width, zone[1] + width_pad)
        top = max(0, zone[2] - height_pad)
        bottom = min(height, zone[3] + height_pad)
        region = screenshot[top:bottom, left:right]
        records = self._red_icon_records(
            self._collect_red_icon_map(region, threshold, left, top)
        )
        valid = [
            (confidence, x, y)
            for confidence, x, y, _ in records
            if zone[0] <= x <= zone[1] and zone[2] <= y <= zone[3]
        ]
        return max(valid, default=None, key=lambda icon: icon[0])

    def _scan_red_icons(self, screenshot: Any) -> tuple[list[RedIcon], RedIcon | None]:
        new_level_zone = (
            config.NEW_LEVEL_RED_ICON_X_MIN,
            config.NEW_LEVEL_RED_ICON_X_MAX,
            config.NEW_LEVEL_RED_ICON_Y_MIN,
            config.NEW_LEVEL_RED_ICON_Y_MAX,
        )
        new_level_icon = self._find_zone_red_icon(
            screenshot, new_level_zone, config.NEW_LEVEL_RED_ICON_THRESHOLD
        )
        threshold = min(config.RED_ICON_THRESHOLD, config.NEW_LEVEL_RED_ICON_THRESHOLD)
        records = self._red_icon_records(
            self._collect_red_icon_map(screenshot[: config.MAX_SEARCH_Y, :], threshold)
        )
        return [(confidence, x, y) for confidence, x, y, _ in records], new_level_icon

    def _scrcpy_recovery(self, delay: Any) -> bool:
        if not config.SCRCPY_MISS_RECOVERY_ENABLED:
            return False
        wait_time = max(0.0, _finite_float(delay))
        return True if wait_time == 0 else self._sleep(wait_time)

    def _clickable_icons(self, icons: list[RedIcon]) -> list[RedIcon]:
        return [
            icon
            for icon in icons
            if not self.mouse_controller.is_in_forbidden_zone(
                icon[1] + config.RED_ICON_OFFSET_X,
                icon[2] + config.RED_ICON_OFFSET_Y,
                relative=True,
            )
        ]

    def _stable_icons(self, icons: list[RedIcon]) -> list[RedIcon]:
        return [
            icon
            for icon in icons
            if self.failed_click_tracker.get(self._bucket(icon[1], icon[2]), 0)
            < FAILED_CLICK_BUCKET_LIMIT
        ]

    def _red_icon_priority(self, icon: RedIcon) -> tuple[int, int]:
        _, _, y = icon
        for successful_y in self.successful_red_icon_positions:
            if abs(y - successful_y) < ROW_PRIORITY_DISTANCE_PIXELS:
                return 0, y
        return 1, y

    @staticmethod
    def _bucket(x: int, y: int) -> tuple[int, int]:
        return int(x) // CLICK_BUCKET_SIZE_PIXELS, int(y) // CLICK_BUCKET_SIZE_PIXELS

    def _record_failed_click(self) -> None:
        if self.last_clicked_pos is None:
            return
        bucket = self._bucket(*self.last_clicked_pos)
        self.failed_click_tracker[bucket] = self.failed_click_tracker.get(bucket, 0) + 1

    def _remember_successful_row(self) -> None:
        if self.red_icon is None:
            return
        _, _, red_y = self.red_icon
        if all(
            abs(existing - red_y) >= SUCCESSFUL_ROW_DEDUP_DISTANCE_PIXELS
            for existing in self.successful_red_icon_positions
        ):
            self.successful_red_icon_positions.append(int(red_y))
        if self.last_clicked_pos is not None:
            self.failed_click_tracker.pop(self._bucket(*self.last_clicked_pos), None)

    def _find_upgrade_station(
        self,
        threshold: float,
        locked_target: UpgradeStationCandidate | None = None,
    ) -> UpgradeStationCandidate | None:
        screenshot = self.window_capture.capture(max_y=config.UPGRADE_STATION_SEARCH_Y)
        for candidate in self._upgrade_candidates(screenshot, threshold):
            _, x, y, _, _ = candidate
            if self.mouse_controller.is_in_forbidden_zone(x, y, relative=True):
                continue
            if locked_target is None or _same_upgrade_station_target(
                locked_target, candidate
            ):
                return candidate
        return None

    def _verify_upgrade_station(
        self,
        locked_target: UpgradeStationCandidate,
        base_threshold: float,
        relaxed_threshold: float,
        wait_between_attempts: bool,
    ) -> UpgradeStationCandidate | None:
        attempts = max(1, int(config.UPGRADE_STATION_VERIFY_SEARCH_ATTEMPTS))
        attempts = min(attempts, MAX_UPGRADE_SEARCH_ATTEMPTS)
        for attempt in range(attempts):
            threshold = base_threshold if attempt == 0 else relaxed_threshold
            match = self._find_upgrade_station(threshold, locked_target)
            if match is not None:
                return match
            if (
                wait_between_attempts
                and attempt < attempts - 1
                and not self._sleep(config.UPGRADE_STATION_VERIFY_SEARCH_INTERVAL)
            ):
                return None
        return None

    def _upgrade_candidates(
        self, screenshot: Any, threshold: float
    ) -> list[UpgradeStationCandidate]:
        template_pair = self._template(TemplateName.UPGRADE_STATION.value)
        if template_pair is None:
            return []
        template, mask = template_pair
        hsv_mask = self.image_matcher.build_hsv_mask(
            screenshot, config.UPGRADE_STATION_HSV_RANGES
        )
        candidates = self.image_matcher.find_color_gated_template_candidates(
            screenshot,
            template,
            mask=mask,
            threshold=threshold,
            min_distance=15,
            template_name=TemplateName.UPGRADE_STATION.value,
            hsv_ranges=config.UPGRADE_STATION_HSV_RANGES,
            hsv_mask=hsv_mask,
        )
        candidates = self.image_matcher.filter_candidates_with_supervision_nms(
            candidates,
            iou_threshold=config.SUPERVISION_UPGRADE_STATION_NMS_IOU_THRESHOLD,
        )
        return self.image_matcher.filter_candidates_by_hsv(
            screenshot,
            candidates,
            template,
            mask=mask,
            hsv_ranges=config.UPGRADE_STATION_HSV_RANGES,
            hsv_match_threshold=config.UPGRADE_STATION_HSV_MIN_MATCH_RATIO,
            hsv_mask=hsv_mask,
        )

    def _box_candidates(self, screenshot: Any) -> list[BoxCandidate]:
        candidates: list[BoxCandidate] = []
        hsv_mask = self.image_matcher.build_hsv_mask(screenshot, config.BOX_HSV_RANGES)
        for name in BOX_TEMPLATE_NAMES:
            candidates.extend(self._template_box_candidates(screenshot, name, hsv_mask))
        return self.image_matcher.filter_candidates_with_supervision_nms(
            candidates,
            iou_threshold=config.SUPERVISION_BOX_NMS_IOU_THRESHOLD,
        )

    def _template_box_candidates(
        self, screenshot: Any, name: str, hsv_mask: Any = None
    ) -> list[BoxCandidate]:
        template_pair = self._template(name)
        if template_pair is None:
            return []
        template, mask = template_pair
        matches = self.image_matcher.find_color_gated_template_candidates(
            screenshot,
            template,
            mask=mask,
            threshold=config.BOX_THRESHOLD,
            min_distance=12,
            template_name=name,
            hsv_ranges=config.BOX_HSV_RANGES,
            hsv_match_threshold=config.BOX_HSV_MIN_MATCH_RATIO,
            hsv_mask=hsv_mask,
        )
        matches = self.image_matcher.filter_candidates_by_hsv(
            screenshot,
            matches,
            template,
            mask=mask,
            hsv_ranges=config.BOX_HSV_RANGES,
            hsv_match_threshold=config.BOX_HSV_MIN_MATCH_RATIO,
            hsv_mask=hsv_mask,
        )
        return [
            (float(confidence), int(x), int(y), int(width), int(height), name)
            for confidence, x, y, width, height in matches
        ]

    def _reset_search_cycle(self) -> None:
        self.empty_cycle_count = 0
        self.scroll_failures = 0
        self.scroll_direction = 1
        self.scroll_cycle_index = 1
        self.scroll_cycle_progress = 0

    def _scroll(self) -> bool:
        distance = round(
            float(config.SCROLL_PIXEL_STEP) * float(config.SCROLL_DISTANCE_RATIO)
        )
        start_x, start_y = config.SCROLL_START_POS
        moved = self.mouse_controller.drag(
            start_x,
            start_y,
            start_x,
            start_y - (distance * self.scroll_direction),
            duration=config.SCROLL_DURATION,
            relative=True,
        )
        if not moved or not self._sleep(config.POST_SCROLL_SETTLE):
            return False
        if not self._click_idle():
            return False
        if not self._sleep(config.SCROLL_INTERVAL_PAUSE):
            return False
        self.scroll_cycle_progress += 1
        cycle_limit = max(
            1, self.scroll_cycle_index * int(config.SCROLL_INCREMENT_STEP)
        )
        if self.scroll_cycle_progress >= cycle_limit:
            self.scroll_cycle_progress = 0
            self.scroll_direction *= -1
            if self.scroll_direction > 0:
                maximum_scroll_cycles = int(config.MAX_SCROLL_CYCLES)
                if self.scroll_cycle_index >= maximum_scroll_cycles:
                    self.scroll_cycle_index = 1
                else:
                    self.scroll_cycle_index += 1
        return True

    def _find_new_level_red_icon(self) -> RedIcon | None:
        screenshot = self.window_capture.capture(max_y=config.EXTENDED_SEARCH_Y)
        new_level_red_icon_zone = (
            config.NEW_LEVEL_RED_ICON_X_MIN,
            config.NEW_LEVEL_RED_ICON_X_MAX,
            config.NEW_LEVEL_RED_ICON_Y_MIN,
            config.NEW_LEVEL_RED_ICON_Y_MAX,
        )
        return self._find_zone_red_icon(
            screenshot,
            new_level_red_icon_zone,
            config.NEW_LEVEL_RED_ICON_THRESHOLD,
        )

    def _record_level_completion(self, source: str) -> float:
        self.total_levels_completed += 1
        elapsed = 0.0
        now = time.monotonic()
        if self.current_level_started_at is not None:
            elapsed = max(0.0, now - self.current_level_started_at)
        self.current_level_started_at = now
        self._reset_search_cycle()
        self.telegram.notify_new_level(self.total_levels_completed, elapsed)
        logger.debug("Recorded level completion from %s", source)
        return elapsed

    def handle_find_red_icons(self) -> StateResult:
        if not self._dismiss_if_needed():
            return State.FIND_RED_ICONS
        screenshot = self.window_capture.capture(max_y=config.EXTENDED_SEARCH_Y)
        found, _, x, y = self._find_new_level_button(
            screenshot[: config.MAX_SEARCH_Y, :]
        )
        if found:
            logger.info("newLevel.png found at (%s, %s)", x, y)
            return State.TRANSITION_LEVEL
        icons, new_level_icon = self._scan_red_icons(screenshot)
        if (
            not icons
            and new_level_icon is None
            and self._scrcpy_recovery(config.SCRCPY_RED_ICON_MISS_RECOVERY_DELAY)
        ):
            icons, new_level_icon = self._scan_red_icons(
                self.window_capture.capture(max_y=config.EXTENDED_SEARCH_Y)
            )
        if new_level_icon is not None:
            logger.info(
                "New level red icon detected at (%s, %s)",
                new_level_icon[1],
                new_level_icon[2],
            )
            return State.CHECK_NEW_LEVEL
        stable_icons = self._stable_icons(self._clickable_icons(icons))
        if not stable_icons:
            if icons:
                self.failed_click_tracker.clear()
                return State.SCROLL
            return State.OPEN_BOXES
        self.red_icon = min(stable_icons, key=self._red_icon_priority)
        logger.info("Selected one of %s red icons", len(stable_icons))
        return State.CLICK_RED_ICON

    def handle_click_red_icon(self) -> StateResult:
        if self.red_icon is None:
            return State.OPEN_BOXES
        confidence, x, y = self.red_icon
        self.last_clicked_pos = (x, y)
        click_x = x + config.RED_ICON_OFFSET_X
        click_y = y + config.RED_ICON_OFFSET_Y
        clicked = self.mouse_controller.click(click_x, click_y, relative=True)
        if not clicked:
            self._record_failed_click()
            self.red_icon = None
            return State.FIND_RED_ICONS
        self.empty_cycle_count = 0
        logger.info(
            "Clicked red icon at (%s, %s) [%.3f]",
            click_x,
            click_y,
            confidence,
        )
        return State.CHECK_UNLOCK

    def handle_check_unlock(self) -> StateResult:
        found, confidence, x, y = self._find_unlock_button(
            self.window_capture.capture(max_y=config.MAX_SEARCH_Y)
        )
        if found and not self.mouse_controller.is_in_forbidden_zone(
            x, y, relative=True
        ):
            logger.info("Unlock found at (%s, %s) [%.3f]", x, y, confidence)
            if not self.mouse_controller.click(x, y, relative=True):
                return State.CHECK_UNLOCK
        return State.SEARCH_UPGRADE_STATION

    def handle_search_upgrade_station(self) -> StateResult:
        self.active_upgrade_station_target = None
        base_threshold = float(config.UPGRADE_STATION_THRESHOLD)
        relaxed_threshold = max(
            0.0, base_threshold - UPGRADE_STATION_THRESHOLD_RELAXATION
        )
        for attempt in range(MAX_UPGRADE_SEARCH_ATTEMPTS):
            threshold = (
                base_threshold
                if attempt < UPGRADE_STATS_CYCLE_INTERVAL
                else relaxed_threshold
            )
            match = self._find_upgrade_station(threshold)
            if match is not None:
                _, x, y, _, _ = match
                self.active_upgrade_station_target = match
                self.empty_cycle_count = 0
                logger.info("Upgrade station found at (%s, %s)", x, y)
                return State.HOLD_UPGRADE_STATION
            if attempt < MAX_UPGRADE_SEARCH_ATTEMPTS - 1 and not self._sleep(
                config.UPGRADE_SEARCH_INTERVAL
            ):
                return State.OPEN_BOXES
        self._record_failed_click()
        logger.info("Upgrade station not found")
        return State.OPEN_BOXES

    def handle_hold_upgrade_station(self) -> StateResult:
        locked_target = self.active_upgrade_station_target
        if locked_target is None:
            return State.OPEN_BOXES
        threshold = float(config.UPGRADE_STATION_THRESHOLD)
        try:
            match = self._verified_upgrade_station_or_clear(locked_target, threshold)
            if match is None:
                return State.OPEN_BOXES
            self.active_upgrade_station_target = match
            _, x, y, _, _ = match
            if self.mouse_controller.is_in_forbidden_zone(x, y, relative=True):
                logger.warning(
                    "Upgrade station blocked by forbidden zone at (%s, %s)", x, y
                )
                return State.OPEN_BOXES
            clicked = self.mouse_controller.precise_click(x, y, relative=True)
            if not clicked:
                logger.warning(
                    "Upgrade station verification click failed at (%s, %s)", x, y
                )
                return State.OPEN_BOXES
            if not self._sleep(config.UPGRADE_STATION_VERIFY_SETTLE_DELAY):
                return State.OPEN_BOXES
            match = self._verified_upgrade_station_or_clear(match, threshold)
            if match is None:
                return State.OPEN_BOXES
            self.active_upgrade_station_target = match
            _, x, y, _, _ = match
            monitor = _UpgradeStationHoldMonitor(self, match, threshold)
            held = self.mouse_controller.hold_at(
                x,
                y,
                duration=config.CLICK_HOLD_MAX_DURATION,
                relative=True,
                interrupt_check=monitor,
            )
            if not held and not monitor.station_lost:
                return State.OPEN_BOXES
            self._remember_successful_row()
            self.empty_cycle_count = 0
            self.needs_dismissal = True
            self.upgrade_station_counter += 1
            if self.upgrade_station_counter >= UPGRADE_STATS_CYCLE_INTERVAL:
                self.upgrade_station_counter = 0
                return State.UPGRADE_STATS
            return State.OPEN_BOXES
        finally:
            self.active_upgrade_station_target = None

    def _verified_upgrade_station_or_clear(
        self,
        locked_target: UpgradeStationCandidate,
        threshold: float,
    ) -> UpgradeStationCandidate | None:
        match = self._verify_upgrade_station(
            locked_target,
            threshold,
            max(0.0, threshold - UPGRADE_STATION_THRESHOLD_RELAXATION),
            wait_between_attempts=True,
        )
        if match is not None:
            return match
        logger.info("Locked upgrade station disappeared before hold")
        return None

    def handle_upgrade_stats(self) -> StateResult:
        if not self._dismiss_if_needed():
            return State.OPEN_BOXES
        screenshot = self.window_capture.capture(max_y=config.EXTENDED_SEARCH_Y)
        found, _, _, _ = self._find_new_level_button(
            screenshot[: config.MAX_SEARCH_Y, :]
        )
        if found:
            return State.TRANSITION_LEVEL
        zone = (
            config.UPGRADE_RED_ICON_X_MIN,
            config.UPGRADE_RED_ICON_X_MAX,
            config.UPGRADE_RED_ICON_Y_MIN,
            config.UPGRADE_RED_ICON_Y_MAX,
        )
        if (
            self._find_zone_red_icon(screenshot, zone, config.STATS_RED_ICON_THRESHOLD)
            is None
        ):
            logger.info("No stats icon detected")
            return State.SCROLL
        if not self.mouse_controller.click(
            *config.STATS_UPGRADE_BUTTON_POS, relative=True
        ):
            return State.OPEN_BOXES
        clicked = self.mouse_controller.click_stats_upgrade_at(
            *config.STATS_UPGRADE_POS,
            duration=config.STATS_UPGRADE_CLICK_DURATION,
            click_delay=config.STATS_UPGRADE_CLICK_DELAY,
            relative=True,
            interrupt_check=self._stop_requested.is_set,
        )
        if not clicked:
            return State.OPEN_BOXES
        self.empty_cycle_count = 0
        self.needs_dismissal = True
        logger.info("Stats upgrade completed")
        return State.OPEN_BOXES

    def handle_open_boxes(self) -> StateResult:
        if not self._dismiss_if_needed():
            return State.FIND_RED_ICONS
        screenshot = self.window_capture.capture(max_y=config.BOX_SEARCH_Y)
        found, _, _, _ = self._find_new_level_button(
            screenshot[: config.MAX_SEARCH_Y, :]
        )
        if found:
            return State.TRANSITION_LEVEL
        candidates = self._box_candidates(screenshot)
        if not candidates and self._scrcpy_recovery(
            config.SCRCPY_BOX_MISS_RECOVERY_DELAY
        ):
            candidates = self._box_candidates(
                self.window_capture.capture(max_y=config.BOX_SEARCH_Y)
            )
        box_opened = False
        for _, x, y, _, _, _ in candidates:
            if self.mouse_controller.is_in_forbidden_zone(x, y, relative=True):
                continue
            box_opened = self.mouse_controller.click(x, y, relative=True)
            break
        if box_opened:
            self.empty_cycle_count = 0
            self.needs_dismissal = True
            logger.info("Opened one box")
            return State.OPEN_BOXES
        self.empty_cycle_count += 1
        if self.empty_cycle_count >= SEARCH_CYCLES_BEFORE_SCROLL:
            self.empty_cycle_count = 0
            return State.SCROLL
        return State.FIND_RED_ICONS

    def handle_scroll(self) -> StateResult:
        self.failed_click_tracker.clear()
        if not self._dismiss_if_needed():
            return State.SCROLL
        screenshot = self.window_capture.capture(max_y=config.MAX_SEARCH_Y)
        found, _, _, _ = self._find_new_level_button(screenshot)
        if found:
            return State.TRANSITION_LEVEL
        if not self._scroll():
            self.scroll_failures += 1
            if self.scroll_failures >= 2:
                self.scroll_failures = 0
                self._begin_recovery(SCROLL_RECOVERY_MESSAGE)
            if not self._sleep(RUNTIME_RECOVERY_RETRY_DELAY_SECONDS):
                logger.debug("Scroll recovery wait was interrupted")
            return State.SCROLL
        self._complete_recovery(SCROLL_RECOVERY_MESSAGE)
        self.scroll_failures = 0
        self.empty_cycle_count = 0
        return State.FIND_RED_ICONS

    def handle_check_new_level(self) -> StateResult:
        if not self._dismiss_if_needed():
            return State.CHECK_NEW_LEVEL

        pre_click_delay_completed = self._sleep(CHECK_NEW_LEVEL_PRE_CLICK_DELAY)
        if not pre_click_delay_completed:
            return State.CHECK_NEW_LEVEL

        original_scroll_direction = self.scroll_direction
        original_scroll_cycle_index = self.scroll_cycle_index
        original_scroll_cycle_progress = self.scroll_cycle_progress
        try:
            self.scroll_direction = -UPWARD_OSCILLATING_SCROLL_DIRECTION
            verification_scroll_completed = self._scroll()
        finally:
            self.scroll_direction = original_scroll_direction
            self.scroll_cycle_index = original_scroll_cycle_index
            self.scroll_cycle_progress = original_scroll_cycle_progress
        if not verification_scroll_completed:
            if self._stop_requested.is_set():
                return State.CHECK_NEW_LEVEL
            self._begin_recovery(SCROLL_RECOVERY_MESSAGE)
            if not self._sleep(RUNTIME_RECOVERY_RETRY_DELAY_SECONDS):
                logger.debug("Verification scroll recovery wait was interrupted")
            return State.CHECK_NEW_LEVEL
        self._complete_recovery(SCROLL_RECOVERY_MESSAGE)

        verified_new_level_red_icon = self._find_new_level_red_icon()
        if verified_new_level_red_icon is None:
            self._complete_recovery(LEVEL_TRANSITION_RECOVERY_MESSAGE)
            logger.info(
                "New level red icon disappeared after verification scroll; "
                "resuming main flow"
            )
            return State.FIND_RED_ICONS

        new_level_button_clicked = self.mouse_controller.click(
            *config.NEW_LEVEL_BUTTON_POS,
            relative=True,
        )
        if not new_level_button_clicked:
            return State.CHECK_NEW_LEVEL

        level_transition_confirmation_clicked = self.mouse_controller.click(
            *config.LEVEL_TRANSITION_POS,
            relative=True,
        )
        if not level_transition_confirmation_clicked:
            return State.CHECK_NEW_LEVEL

        return State.TRANSITION_LEVEL

    def handle_transition_level(self) -> StateResult:
        if not self._dismiss_if_needed():
            return State.CHECK_NEW_LEVEL
        for attempt in range(MAX_LEVEL_TRANSITION_ATTEMPTS):
            screenshot = self.window_capture.capture(max_y=config.MAX_SEARCH_Y)
            found, _, x, y = self._find_new_level_button(screenshot)
            if found:
                if not self.mouse_controller.click(x, y, relative=True):
                    continue
                if not self._sleep(TRANSITION_LEVEL_BUTTON_WAIT_SECONDS):
                    return State.CHECK_NEW_LEVEL
                screenshot = self.window_capture.capture(max_y=config.MAX_SEARCH_Y)
            button_still_visible, _, _, _ = self._find_new_level_button(screenshot)
            unlock_found, _, _, _ = self._find_unlock_button(screenshot)
            if not button_still_visible and unlock_found:
                elapsed = self._record_level_completion("transition")
                logger.info(
                    "Level %s completed. Time spent: %.1fs",
                    self.total_levels_completed,
                    elapsed,
                )
                self._complete_recovery(LEVEL_TRANSITION_RECOVERY_MESSAGE)
                return State.WAIT_FOR_UNLOCK
            if attempt < MAX_LEVEL_TRANSITION_ATTEMPTS - 1 and not self._sleep(
                TRANSITION_LEVEL_RETRY_DELAY_SECONDS
            ):
                return State.CHECK_NEW_LEVEL
        self._begin_recovery(LEVEL_TRANSITION_RECOVERY_MESSAGE)
        if not self._sleep(RUNTIME_RECOVERY_RETRY_DELAY_SECONDS):
            logger.debug("Level transition recovery wait was interrupted")
        return State.CHECK_NEW_LEVEL

    def handle_wait_for_unlock(self) -> StateResult:
        if not self._dismiss_if_needed() or not self._sleep(
            WAIT_FOR_UNLOCK_PRE_SCAN_DELAY
        ):
            return State.WAIT_FOR_UNLOCK
        found, confidence, x, y = self._find_unlock_button(
            self.window_capture.capture()
        )
        if not found or self.mouse_controller.is_in_forbidden_zone(x, y, relative=True):
            if not self._sleep(WAIT_FOR_UNLOCK_RETRY_DELAY):
                return State.WAIT_FOR_UNLOCK
            return State.WAIT_FOR_UNLOCK
        logger.info("Unlock button found at (%s, %s) [%.3f]", x, y, confidence)
        if not self.mouse_controller.click(x, y, relative=True):
            return State.WAIT_FOR_UNLOCK
        if not self._sleep(WAIT_FOR_UNLOCK_POST_CLICK_DELAY):
            return State.WAIT_FOR_UNLOCK
        self._reset_search_cycle()
        return State.FIND_RED_ICONS
