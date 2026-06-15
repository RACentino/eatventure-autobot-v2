import json
import logging
import math
import os
import tempfile
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import config
from asset_tracker import AssetDetection, AssetTracker
from domain import (
    BEHAVIOR_KEYS,
    BOT_RUN_ITERATION_LIMIT,
    BOT_STATE_LOOP_SLEEP_SECONDS,
    BOX_TEMPLATE_NAMES,
    CHECK_NEW_LEVEL_POST_CLICK_DELAY,
    CHECK_NEW_LEVEL_PRE_CLICK_DELAY,
    CHECK_NEW_LEVEL_UPGRADE_CLICK_DELAY,
    CLICK_BUCKET_SIZE_PIXELS,
    FAILED_CYCLES_BEFORE_SCROLL,
    FAILED_CLICK_BUCKET_LIMIT,
    ICON_MERGE_DISTANCE_PIXELS,
    LEARNING_LOOP_ITERATION_LIMIT,
    MAX_LEVEL_TRANSITION_ATTEMPTS,
    MAX_TEMPLATE_FILES,
    MAX_TEMPLATE_NAMES,
    MAX_UPGRADE_SEARCH_ATTEMPTS,
    MAX_WAIT_FOR_UNLOCK_ATTEMPTS,
    MIN_TEMPLATE_DIMENSION,
    RED_ICON_FALLBACK_MIN_DISTANCE,
    RED_ICON_MISSING_PATTERN,
    RED_ICON_NO_BACKGROUND_TEMPLATE,
    RED_ICON_PRIMARY_TEMPLATE,
    RED_ICON_TEMPLATE_PREFIX,
    REQUIRED_TEMPLATE_NAMES,
    ROW_PRIORITY_DISTANCE_PIXELS,
    SEARCH_CYCLES_BEFORE_SCROLL,
    SUCCESSFUL_RED_ICON_ROWS_LIMIT,
    SUCCESSFUL_ROW_DEDUP_DISTANCE_PIXELS,
    TRANSITION_LEVEL_BUTTON_WAIT_SECONDS,
    TRANSITION_LEVEL_RETRY_DELAY_SECONDS,
    UPGRADE_STATS_CYCLE_INTERVAL,
    UPGRADE_STATION_HOLD_MAX_VERIFY_INTERVAL,
    UPGRADE_STATION_HOLD_MIN_VERIFY_INTERVAL,
    UPGRADE_STATION_THRESHOLD_RELAXATION,
    WAIT_FOR_UNLOCK_POST_CLICK_DELAY,
    WAIT_FOR_UNLOCK_PRE_SCAN_DELAY,
    WAIT_FOR_UNLOCK_RETRY_DELAY,
    AssetType,
    SupervisionFlag,
    TemplateName,
)
from image_matcher import ImageMatcher
from mouse_controller import MouseController, precise_sleep, wait_event
from state_machine import State, StateMachine
from telegram_notifier import TelegramNotifier
from window_capture import WindowCapture, WindowCaptureError, WindowNotAvailableError

logger = logging.getLogger(__name__)

TemplatePair = tuple[Any, Any]
RedIcon = tuple[float, int, int]
RedIconRecord = tuple[float, int, int, str]
BoxCandidate = tuple[float, int, int, int, int, str]
StateResult = State | None


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


class RuntimePersistence:
    def __init__(self, path: str, save_interval: float) -> None:
        self.path = str(path or "")
        self.save_interval = max(0.0, _finite_float(save_interval))
        self._last_save_time = 0.0
        self._lock = threading.RLock()

    def load(self) -> dict[str, Any]:
        if not self.path:
            return {}
        with self._lock:
            try:
                with open(self.path, "r", encoding="utf-8") as handle:
                    state = json.load(handle)
            except FileNotFoundError:
                return {}
            except (OSError, TypeError, ValueError) as exc:
                logger.warning("Failed to load runtime state: %s", exc)
                return {}
        return state if isinstance(state, dict) else {}

    def save(self, state: dict[str, Any], force: bool = False) -> bool:
        if not self.path:
            return False
        now = time.monotonic()
        if not self._save_due(now, force):
            return False
        with self._lock:
            return self._write_state_file(state, now)

    def _save_due(self, now: float, force: bool) -> bool:
        if force or self.save_interval <= 0:
            return True
        return now - self._last_save_time >= self.save_interval

    def _write_state_file(self, state: dict[str, Any], now: float) -> bool:
        temp_path: str | None = None
        try:
            target_path = Path(self.path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=target_path.parent, delete=False
            ) as handle:
                json.dump(state, handle, indent=2, sort_keys=True)
                temp_path = handle.name
            os.replace(temp_path, self.path)
            self._last_save_time = now
            return True
        except (OSError, TypeError, ValueError) as exc:
            logger.error("Failed to persist runtime state: %s", exc)
            self._remove_temp_state_file(temp_path)
            return False

    @staticmethod
    def _remove_temp_state_file(temp_path: str | None) -> None:
        if temp_path is None:
            return
        try:
            os.remove(temp_path)
        except OSError:
            logger.debug("Temporary state file cleanup failed")


class AdaptiveTuner:
    def __init__(self) -> None:
        self.enabled = bool(config.ADAPTIVE_TUNER_ENABLED)
        self.alpha = max(0.0, min(1.0, _finite_float(config.ADAPTIVE_TUNER_ALPHA, 0.3)))
        self.click_success_rate = 1.0
        self.search_success_rate = 1.0
        self.click_delay = _finite_float(config.CLICK_DELAY)
        self.move_delay = _finite_float(config.MOUSE_MOVE_DELAY)
        self.search_interval = _finite_float(config.UPGRADE_SEARCH_INTERVAL)

    @staticmethod
    def _clamp(value: Any, minimum: Any, maximum: Any) -> float:
        low = _finite_float(minimum)
        high = max(low, _finite_float(maximum, low))
        return max(low, min(high, _finite_float(value, low)))

    def record_click_result(self, success: bool) -> None:
        if not self.enabled:
            return
        score = 1.0 if success else 0.0
        self.click_success_rate = (1.0 - self.alpha) * self.click_success_rate
        self.click_success_rate += self.alpha * score
        if self.click_success_rate < config.ADAPTIVE_TUNER_CLICK_LOW_THRESHOLD:
            click_delta = config.ADAPTIVE_TUNER_CLICK_DELAY_STEP
            move_delta = config.ADAPTIVE_TUNER_MOVE_DELAY_STEP
        elif self.click_success_rate > config.ADAPTIVE_TUNER_CLICK_HIGH_THRESHOLD:
            click_delta = -config.ADAPTIVE_TUNER_CLICK_DECREMENT
            move_delta = -config.ADAPTIVE_TUNER_MOVE_DECREMENT
        else:
            return
        self.click_delay = self._clamp(
            self.click_delay + _finite_float(click_delta),
            config.ADAPTIVE_TUNER_MIN_CLICK_DELAY,
            config.ADAPTIVE_TUNER_MAX_CLICK_DELAY,
        )
        self.move_delay = self._clamp(
            self.move_delay + _finite_float(move_delta),
            config.ADAPTIVE_TUNER_MIN_MOVE_DELAY,
            config.ADAPTIVE_TUNER_MAX_MOVE_DELAY,
        )

    def record_search_result(self, success: bool) -> None:
        if not self.enabled:
            return
        score = 1.0 if success else 0.0
        self.search_success_rate = (1.0 - self.alpha) * self.search_success_rate
        self.search_success_rate += self.alpha * score
        if self.search_success_rate < config.ADAPTIVE_TUNER_SEARCH_LOW_THRESHOLD:
            delta = config.ADAPTIVE_TUNER_SEARCH_INTERVAL_STEP
        elif self.search_success_rate > config.ADAPTIVE_TUNER_SEARCH_HIGH_THRESHOLD:
            delta = -config.ADAPTIVE_TUNER_SEARCH_DECREMENT
        else:
            return
        self.search_interval = self._clamp(
            self.search_interval + _finite_float(delta),
            config.ADAPTIVE_TUNER_MIN_SEARCH_INTERVAL,
            config.ADAPTIVE_TUNER_MAX_SEARCH_INTERVAL,
        )

    def apply_profile(self, profile: dict[str, Any]) -> None:
        behavior = HistoricalLearner.sanitize_behavior(profile)
        self.click_delay = behavior.get("click_delay", self.click_delay)
        self.move_delay = behavior.get("move_delay", self.move_delay)
        self.search_interval = behavior.get("search_interval", self.search_interval)

    def snapshot(self) -> dict[str, float]:
        return {
            "click_delay": float(self.click_delay),
            "move_delay": float(self.move_delay),
            "search_interval": float(self.search_interval),
        }

    def reset(self) -> None:
        self.click_success_rate = 1.0
        self.search_success_rate = 1.0
        self.click_delay = _finite_float(config.CLICK_DELAY)
        self.move_delay = _finite_float(config.MOUSE_MOVE_DELAY)
        self.search_interval = _finite_float(config.UPGRADE_SEARCH_INTERVAL)


class HistoricalLearner:
    def __init__(self, bot: Any, persistence: RuntimePersistence | None = None) -> None:
        self.bot = bot
        self.persistence = persistence
        self.enabled = bool(config.AI_LEARNING_ENABLED)
        self.interval = max(
            _finite_float(config.LEARNING_LOOP_MIN_SLEEP),
            _finite_float(config.AI_LEARNING_THREAD_INTERVAL, 5.0),
        )
        self.batch_window = max(2, int(config.AI_LEARNING_BATCH_WINDOW))
        self.top_k = max(1, int(config.AI_LEARNING_PROFILE_BLEND_TOP_K))
        self.ema_alpha = max(0.01, min(0.8, _finite_float(config.AI_LEARNING_EMA_ALPHA)))
        self.min_improvement_ratio = max(
            0.0, _finite_float(config.AI_LEARNING_MIN_IMPROVEMENT_RATIO)
        )
        self.apply_cooldown = max(0.0, _finite_float(config.AI_LEARNING_APPLY_COOLDOWN))
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._records: list[dict[str, Any]] = []
        self._total_completions = 0
        self._last_processed_batch = 0
        self._last_apply_time = 0.0
        self._tuned_behavior: dict[str, float] = {}
        self._load()

    @staticmethod
    def sanitize_behavior(behavior: Any) -> dict[str, float]:
        if not isinstance(behavior, dict):
            return {}
        bounds = {
            "click_delay": (
                config.AI_LEARNING_MIN_CLICK_DELAY,
                config.AI_LEARNING_MAX_CLICK_DELAY,
            ),
            "move_delay": (
                config.AI_LEARNING_MIN_MOVE_DELAY,
                config.AI_LEARNING_MAX_MOVE_DELAY,
            ),
            "search_interval": (
                config.AI_LEARNING_MIN_SEARCH_INTERVAL,
                config.AI_LEARNING_MAX_SEARCH_INTERVAL,
            ),
        }
        return {
            key: AdaptiveTuner._clamp(behavior.get(key), minimum, maximum)
            for key, (minimum, maximum) in bounds.items()
            if key in behavior
        }

    def _load(self) -> None:
        if not self.enabled or self.persistence is None:
            return
        state = self.persistence.load()
        records = state.get("records", [])
        if isinstance(records, list):
            self._records = [item for item in records if isinstance(item, dict)]
            self._records = self._records[-config.AI_LEARNING_RECORDS_LIMIT :]
        self._total_completions = max(
            0, int(state.get("total_completions", len(self._records)) or 0)
        )
        self._last_processed_batch = max(
            0, int(state.get("last_processed_batch", 0) or 0)
        )
        self._tuned_behavior = self.sanitize_behavior(
            state.get("tuned_behavior", {})
        )
        if self._tuned_behavior:
            self.bot.apply_learned_behavior(self._tuned_behavior)

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="historical_learner", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=config.AI_LEARNING_THREAD_JOIN_TIMEOUT)
        self._persist(force=True)

    def record_completion(self, seconds_spent: float, source: str) -> None:
        if not self.enabled or seconds_spent <= 0:
            return
        with self._lock:
            self._records.append(
                {
                    "timestamp": time.time(),
                    "time_spent": float(seconds_spent),
                    "source": str(source),
                    "behavior": self.bot.get_runtime_behavior_snapshot(),
                }
            )
            self._records = self._records[-config.AI_LEARNING_RECORDS_LIMIT :]
            self._total_completions += 1
        self._persist()

    def reset(self) -> None:
        with self._lock:
            self._records = []
            self._total_completions = 0
            self._last_processed_batch = 0
            self._last_apply_time = 0.0
            self._tuned_behavior = {}
        self._persist(force=True)

    def _loop(self) -> None:
        for _ in range(LEARNING_LOOP_ITERATION_LIMIT):
            if self._stop.is_set():
                return
            try:
                self._apply_pending_profile()
            except Exception:
                logger.exception("Historical learner cycle failed")
            if self._stop.wait(self.interval):
                return
        logger.error("Historical learner loop reached iteration limit")

    def _apply_pending_profile(self) -> None:
        with self._lock:
            records = list(self._records[-self.batch_window :])
            batch_marker = self._total_completions // self.batch_window
        if batch_marker <= self._last_processed_batch:
            return
        self._last_processed_batch = batch_marker
        if time.monotonic() - self._last_apply_time < self.apply_cooldown:
            self._persist()
            return
        valid = [
            (
                _finite_float(item.get("time_spent"), -1.0),
                self.sanitize_behavior(item.get("behavior", {})),
            )
            for item in records
        ]
        valid = [
            (duration, behavior)
            for duration, behavior in valid
            if duration > 0 and behavior
        ]
        if valid and self._profile_improves(valid):
            self._tuned_behavior = self._blend_profiles(sorted(valid)[: self.top_k])
            self.bot.apply_learned_behavior(self._tuned_behavior)
            self._last_apply_time = time.monotonic()
        self._persist()

    def _profile_improves(self, valid: list[tuple[float, dict[str, float]]]) -> bool:
        average = sum(duration for duration, _ in valid) / len(valid)
        best_duration = min(duration for duration, _ in valid)
        improvement_ratio = (average - best_duration) / average
        return improvement_ratio >= self.min_improvement_ratio

    def _blend_profiles(
        self, ranked: list[tuple[float, dict[str, float]]]
    ) -> dict[str, float]:
        current = self.bot.get_runtime_behavior_snapshot()
        blended = {key: 0.0 for key in BEHAVIOR_KEYS}
        for _, behavior in ranked:
            for key in BEHAVIOR_KEYS:
                blended[key] += _finite_float(behavior.get(key), current[key])
        divisor = float(max(1, len(ranked)))
        return {
            key: (1.0 - self.ema_alpha) * current[key]
            + self.ema_alpha * (value / divisor)
            for key, value in blended.items()
        }

    def _persist(self, force: bool = False) -> None:
        if self.persistence is None:
            return
        with self._lock:
            state = {
                "records": self._records[-config.AI_LEARNING_RECORDS_LIMIT :],
                "total_completions": self._total_completions,
                "last_processed_batch": self._last_processed_batch,
                "tuned_behavior": self._tuned_behavior,
            }
        if not self.persistence.save(state, force=force):
            logger.debug("Historical learner state was not persisted")


class _UpgradeStationHoldMonitor:
    def __init__(self, bot: Any, threshold: float) -> None:
        self.bot = bot
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
            self.threshold,
            self.relaxed_threshold,
            wait_between_attempts=False,
        )
        if match is None:
            self.station_lost = True
            return True
        self.next_verify_at = time.perf_counter() + self.interval
        return False


class EatventureBot:
    def __init__(self) -> None:
        logger.info("Initializing Eatventure Bot")
        self._stop_requested = threading.Event()
        self._step_active = threading.Event()
        self.window_capture = WindowCapture(
            config.WINDOW_TITLE, config.WINDOW_WIDTH, config.WINDOW_HEIGHT
        )
        self.image_matcher = ImageMatcher(config.MATCH_THRESHOLD)
        self.mouse_controller = MouseController(
            self.window_capture.get_window_rect,
            config.CLICK_DELAY,
            config.MOUSE_MOVE_DELAY,
        )
        self.state_machine = StateMachine(State.FIND_RED_ICONS)
        self.telegram = TelegramNotifier(
            config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID, config.TELEGRAM_ENABLED
        )
        self.tuner = AdaptiveTuner()
        self.learning_persistence = RuntimePersistence(
            config.AI_LEARNING_STATE_FILE, config.AI_LEARNING_SAVE_INTERVAL
        )
        self.historical_learner = HistoricalLearner(self, self.learning_persistence)
        self.templates = self.load_templates()
        self.asset_tracker = AssetTracker(
            self.window_capture, self._detect_trackable_assets
        )
        self._reset_runtime_state()
        self.register_states()
        self.ready = self._validate_required_templates()
        logger.info("Bot initialized successfully")

    def _reset_runtime_state(self) -> None:
        self.running = False
        self.red_icons: list[RedIcon] = []
        self.current_red_icon_index = 0
        self.wait_for_unlock_attempts = 0
        self.max_wait_for_unlock_attempts = MAX_WAIT_FOR_UNLOCK_ATTEMPTS
        self.work_done = False
        self.cycle_counter = 0
        self.upgrade_station_counter = 0
        self.red_icon_processed_count = 0
        self.consecutive_failed_cycles = 0
        self.upgrade_found_in_cycle = False
        self.upgrade_station_pos: tuple[int, int] | None = None
        self.last_clicked_pos: tuple[int, int] | None = None
        self.failed_click_tracker: dict[tuple[int, int], int] = {}
        self.successful_red_icon_positions: deque[int] = deque(
            maxlen=SUCCESSFUL_RED_ICON_ROWS_LIMIT
        )
        self.total_levels_completed = 0
        self.current_level_start_time: datetime | None = None
        self.scroll_direction = 1
        self.scroll_cycle_index = 1
        self.scroll_cycle_progress = 0
        self.forbidden_zones = list(config.NUMBERED_FORBIDDEN_ZONE_BOUNDS)

    def load_templates(self) -> dict[str, TemplatePair]:
        templates: dict[str, TemplatePair] = {}
        assets_path = Path(config.ASSETS_DIR)
        if not assets_path.exists():
            logger.error("Assets directory not found: %s", assets_path)
            return templates
        for template_file in sorted(assets_path.glob("*.png"))[:MAX_TEMPLATE_FILES]:
            try:
                templates[template_file.stem] = self.image_matcher.load_template(template_file)
                logger.info("Loaded template: %s", template_file.stem)
            except Exception as exc:
                logger.error("Failed to load template %s: %s", template_file, exc)
        return templates

    def register_states(self) -> None:
        for state, handler in (
            (State.FIND_RED_ICONS, self.handle_find_red_icons),
            (State.CLICK_RED_ICON, self.handle_click_red_icon),
            (State.CHECK_UNLOCK, self.handle_check_unlock),
            (State.SEARCH_UPGRADE_STATION, self.handle_search_upgrade_station),
            (State.HOLD_UPGRADE_STATION, self.handle_hold_upgrade_station),
            (State.OPEN_BOXES, self.handle_open_boxes),
            (State.UPGRADE_STATS, self.handle_upgrade_stats),
            (State.SCROLL, self.handle_scroll),
            (State.CHECK_NEW_LEVEL, self.handle_check_new_level),
            (State.TRANSITION_LEVEL, self.handle_transition_level),
            (State.WAIT_FOR_UNLOCK, self.handle_wait_for_unlock),
        ):
            self.state_machine.register_handler(state, handler)

    def _validate_required_templates(self) -> bool:
        missing = [name for name in REQUIRED_TEMPLATE_NAMES if name not in self.templates]
        if not any(name.startswith(RED_ICON_TEMPLATE_PREFIX) for name in self.templates):
            missing.append(RED_ICON_MISSING_PATTERN)
        if missing:
            logger.error("Missing required templates: %s", ", ".join(missing))
            return False
        return True

    def request_stop(self) -> None:
        self._stop_requested.set()

    def start(self) -> bool:
        if self.running:
            return True
        if self._step_active.is_set() or not self.ready:
            logger.warning("Cannot start bot while not ready or while a step is active")
            return False
        try:
            self.window_capture.ensure_window(resize=True)
        except WindowCaptureError as exc:
            logger.error("Cannot start bot: %s", exc)
            return False
        self._stop_requested.clear()
        self.running = True
        self.current_level_start_time = self.current_level_start_time or datetime.now()
        self.asset_tracker.start()
        self.historical_learner.start()
        return True

    def stop(self) -> None:
        self._stop_requested.set()
        self.running = False
        self.asset_tracker.stop()
        self.historical_learner.stop()

    def step(self) -> bool:
        if self._step_active.is_set():
            logger.warning("Ignoring reentrant bot step")
            return False
        self._step_active.set()
        try:
            if self._stop_requested.is_set() or not self.window_capture.is_window_active():
                self.stop()
                return False
            self._apply_tuning()
            if self.state_machine.update():
                return True
            logger.error("State update failed in %s", self.state_machine.get_state_name())
        except (WindowNotAvailableError, WindowCaptureError) as exc:
            logger.error("Stopping bot: %s", exc)
        except Exception:
            logger.exception("Stopping bot due to unexpected state-handler failure")
        finally:
            self._step_active.clear()
        self.stop()
        return False

    def run(self) -> None:
        if not self.start():
            return
        try:
            for _ in range(BOT_RUN_ITERATION_LIMIT):
                if not self.running:
                    return
                self.step()
                precise_sleep(BOT_STATE_LOOP_SLEEP_SECONDS)
            logger.error("Bot run loop reached iteration limit")
        finally:
            self.stop()

    def _sleep(self, duration: Any) -> bool:
        return wait_event(self._stop_requested, duration)

    def _apply_tuning(self) -> None:
        self.mouse_controller.click_delay = float(self.tuner.click_delay)
        self.mouse_controller.move_delay = float(self.tuner.move_delay)

    def get_runtime_behavior_snapshot(self) -> dict[str, float]:
        return self.tuner.snapshot()

    def apply_learned_behavior(self, learned: dict[str, Any]) -> None:
        self.tuner.apply_profile(learned)
        self._apply_tuning()

    def wipe_memory(self) -> None:
        self.tuner.reset()
        self.asset_tracker.reset()
        self.historical_learner.reset()
        self.successful_red_icon_positions.clear()
        self.failed_click_tracker.clear()
        self.current_level_start_time = datetime.now() if self.running else None
        self._apply_tuning()

    def _template(self, name: str) -> TemplatePair | None:
        return self.templates.get(name)

    def _click_idle(self) -> bool:
        return self.mouse_controller.click(*config.IDLE_CLICK_POS, relative=True)

    @staticmethod
    def _supervision_enabled(flag_name: SupervisionFlag | str) -> bool:
        flags = {
            SupervisionFlag.BOX_NMS: config.SUPERVISION_BOX_NMS_ENABLED,
            SupervisionFlag.RED_ICON_NMS: config.SUPERVISION_RED_ICON_NMS_ENABLED,
            SupervisionFlag.UPGRADE_STATION_NMS: config.SUPERVISION_UPGRADE_STATION_NMS_ENABLED,
        }
        try:
            normalized_flag = SupervisionFlag(str(flag_name))
        except ValueError:
            return False
        return bool(config.SUPERVISION_ENABLED and flags.get(normalized_flag, False))

    def _red_icon_names(self) -> list[str]:
        names = [
            name for name in self.templates if name.startswith(RED_ICON_TEMPLATE_PREFIX)
        ]
        if config.RED_ICON_FAST_MODE_ENABLED:
            fast_names = [
                name
                for name in config.RED_ICON_FAST_TEMPLATE_NAMES
                if name in self.templates
            ]
            if fast_names:
                return fast_names[:MAX_TEMPLATE_NAMES]
        return sorted(
            names,
            key=lambda name: (
                name != RED_ICON_PRIMARY_TEMPLATE,
                name == RED_ICON_NO_BACKGROUND_TEMPLATE,
                name,
            ),
        )[:MAX_TEMPLATE_NAMES]

    def _red_icon_min_matches(self) -> int:
        if config.RED_ICON_FAST_MODE_ENABLED:
            return 1
        requested_matches = max(1, int(config.RED_ICON_MIN_MATCHES))
        available_matches = max(1, len(self._red_icon_names()))
        return min(requested_matches, available_matches)

    def _passes_hsv_gate(
        self,
        screenshot: Any,
        template: Any,
        mask: Any,
        center_x: int,
        center_y: int,
        width: int,
        height: int,
        hsv_ranges: Any,
        ratio: float,
    ) -> bool:
        top_left = (
            int(center_x) - int(width) // 2,
            int(center_y) - int(height) // 2,
        )
        return self.image_matcher._check_hsv_gate(
            screenshot, template, top_left, mask, hsv_ranges, ratio
        )

    def _collect_red_icon_map(
        self,
        screenshot: Any,
        threshold: float,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> dict[tuple[int, int], list[tuple[str, float]]]:
        detections: dict[tuple[int, int], list[tuple[str, float]]] = {}
        min_distance = (
            config.RED_ICON_FAST_MIN_DISTANCE
            if config.RED_ICON_FAST_MODE_ENABLED
            else RED_ICON_FALLBACK_MIN_DISTANCE
        )
        for name in self._red_icon_names():
            template_pair = self._template(name)
            if template_pair is None or getattr(screenshot, "size", 0) == 0:
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
                use_supervision_nms=self._supervision_enabled(
                    SupervisionFlag.RED_ICON_NMS
                ),
                supervision_iou_threshold=config.SUPERVISION_RED_ICON_NMS_IOU_THRESHOLD,
                supervision_class_agnostic=config.SUPERVISION_CLASS_AGNOSTIC_NMS,
            )
            height, width = template.shape[:2]
            for confidence, x, y in matches:
                if self._passes_hsv_gate(
                    screenshot,
                    template,
                    mask,
                    x,
                    y,
                    width,
                    height,
                    config.RED_ICON_HSV_RANGES,
                    config.RED_ICON_HSV_MIN_MATCH_RATIO,
                ):
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
        for existing_x, existing_y in list(detections):
            if (
                abs(x - existing_x) < ICON_MERGE_DISTANCE_PIXELS
                and abs(y - existing_y) < ICON_MERGE_DISTANCE_PIXELS
            ):
                detections[(existing_x, existing_y)].append((name, confidence))
                return
        detections[(x, y)] = [(name, confidence)]

    def _red_icon_records(
        self, red_icon_map: dict[tuple[int, int], list[tuple[str, float]]]
    ) -> list[RedIconRecord]:
        records: list[RedIconRecord] = []
        min_matches = self._red_icon_min_matches()
        for (x, y), matches in red_icon_map.items():
            best_by_name: dict[str, float] = {}
            for name, confidence in matches:
                best_by_name[name] = max(
                    float(confidence), best_by_name.get(name, 0.0)
                )
            if len(best_by_name) >= min_matches:
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

    def _find_zone_red_icon(
        self, screenshot: Any, zone: tuple[int, int, int, int], threshold: float
    ) -> RedIcon | None:
        width_pad = height_pad = 0
        for name in self._red_icon_names():
            template_pair = self._template(name)
            if template_pair is None:
                continue
            template, _ = template_pair
            height_pad = max(height_pad, int(template.shape[0]))
            width_pad = max(width_pad, int(template.shape[1]))
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
        tracked_icons = (
            [
                (asset.confidence, asset.center_x, asset.center_y)
                for asset in self.asset_tracker.assets(AssetType.RED_ICON)
                if asset.center_y < config.MAX_SEARCH_Y
            ]
            if config.ASSET_TRACKING_ENABLED
            else []
        )
        if tracked_icons:
            return tracked_icons, new_level_icon
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
        if self.current_red_icon_index >= len(self.red_icons):
            return
        _, _, red_y = self.red_icons[self.current_red_icon_index]
        if all(
            abs(existing - red_y) >= SUCCESSFUL_ROW_DEDUP_DISTANCE_PIXELS
            for existing in self.successful_red_icon_positions
        ):
            self.successful_red_icon_positions.append(int(red_y))
        if self.last_clicked_pos is not None:
            self.failed_click_tracker.pop(self._bucket(*self.last_clicked_pos), None)

    def _find_upgrade_station(
        self, threshold: float, use_tracked: bool = True
    ) -> RedIcon | None:
        if use_tracked and config.ASSET_TRACKING_ENABLED:
            tracked = sorted(
                self.asset_tracker.assets(AssetType.UPGRADE_STATION),
                key=lambda item: item.confidence,
                reverse=True,
            )
            for asset in tracked:
                if not self.mouse_controller.is_in_forbidden_zone(
                    asset.center_x, asset.center_y, relative=True
                ):
                    return asset.confidence, asset.center_x, asset.center_y
        screenshot = self.window_capture.capture(max_y=config.UPGRADE_STATION_SEARCH_Y)
        for confidence, x, y, _, _ in self._upgrade_candidates(screenshot, threshold):
            if not self.mouse_controller.is_in_forbidden_zone(x, y, relative=True):
                return float(confidence), int(x), int(y)
        return None

    def _verify_upgrade_station(
        self,
        base_threshold: float,
        relaxed_threshold: float,
        wait_between_attempts: bool,
    ) -> RedIcon | None:
        attempts = max(1, int(config.UPGRADE_STATION_VERIFY_SEARCH_ATTEMPTS))
        attempts = min(attempts, MAX_UPGRADE_SEARCH_ATTEMPTS)
        for attempt in range(attempts):
            threshold = base_threshold if attempt == 0 else relaxed_threshold
            match = self._find_upgrade_station(threshold, use_tracked=False)
            if match is not None:
                return match
            if wait_between_attempts and attempt < attempts - 1:
                if not self._sleep(config.UPGRADE_STATION_VERIFY_SEARCH_INTERVAL):
                    return None
        return None

    def _upgrade_candidates(
        self, screenshot: Any, threshold: float
    ) -> list[tuple[float, int, int, int, int]]:
        template_pair = self._template(TemplateName.UPGRADE_STATION.value)
        if template_pair is None:
            return []
        template, mask = template_pair
        candidates = self.image_matcher.find_template_candidates(
            screenshot,
            template,
            mask=mask,
            threshold=threshold,
            min_distance=15,
            template_name=TemplateName.UPGRADE_STATION.value,
        )
        if self._supervision_enabled(SupervisionFlag.UPGRADE_STATION_NMS):
            filtered = self.image_matcher.filter_candidates_with_supervision_nms(
                candidates,
                iou_threshold=config.SUPERVISION_UPGRADE_STATION_NMS_IOU_THRESHOLD,
                class_agnostic=config.SUPERVISION_CLASS_AGNOSTIC_NMS,
            )
            candidates = list(filtered) if filtered is not None else candidates
        return [
            item
            for item in candidates
            if self._passes_hsv_gate(
                screenshot,
                template,
                mask,
                item[1],
                item[2],
                item[3],
                item[4],
                config.UPGRADE_STATION_HSV_RANGES,
                config.UPGRADE_STATION_HSV_MIN_MATCH_RATIO,
            )
        ]

    def _box_candidates(
        self, screenshot: Any, include_tracked: bool = True
    ) -> list[BoxCandidate]:
        candidates = self._tracked_box_candidates(include_tracked)
        for name in BOX_TEMPLATE_NAMES:
            candidates.extend(self._template_box_candidates(screenshot, name))
        if self._supervision_enabled(SupervisionFlag.BOX_NMS):
            filtered = self.image_matcher.filter_candidates_with_supervision_nms(
                candidates,
                iou_threshold=config.SUPERVISION_BOX_NMS_IOU_THRESHOLD,
                class_agnostic=config.SUPERVISION_CLASS_AGNOSTIC_NMS,
            )
            return list(filtered) if filtered is not None else candidates
        return candidates

    def _tracked_box_candidates(self, include_tracked: bool) -> list[BoxCandidate]:
        if not include_tracked or not config.ASSET_TRACKING_ENABLED:
            return []
        return [
            (
                asset.confidence,
                asset.center_x,
                asset.center_y,
                asset.width,
                asset.height,
                asset.template_name,
            )
            for asset in self.asset_tracker.assets(AssetType.BOX)
        ]

    def _template_box_candidates(self, screenshot: Any, name: str) -> list[BoxCandidate]:
        template_pair = self._template(name)
        if template_pair is None:
            return []
        template, mask = template_pair
        matches = self.image_matcher.find_template_candidates(
            screenshot,
            template,
            mask=mask,
            threshold=config.BOX_THRESHOLD,
            min_distance=12,
            template_name=name,
        )
        return [
            (float(confidence), int(x), int(y), int(width), int(height), name)
            for confidence, x, y, width, height in matches
            if self._passes_hsv_gate(
                screenshot,
                template,
                mask,
                x,
                y,
                width,
                height,
                config.BOX_HSV_RANGES,
                config.BOX_HSV_MIN_MATCH_RATIO,
            )
        ]

    def _detect_trackable_assets(self, screenshot: Any) -> list[AssetDetection]:
        assets: list[AssetDetection] = []
        assets.extend(self._detect_trackable_red_icons(screenshot))
        assets.extend(self._detect_trackable_upgrade_stations(screenshot))
        assets.extend(self._detect_trackable_boxes(screenshot))
        return assets

    def _detect_trackable_red_icons(self, screenshot: Any) -> list[AssetDetection]:
        if not config.ASSET_TRACKING_RED_ICON_ENABLED:
            return []
        threshold = min(config.RED_ICON_THRESHOLD, config.NEW_LEVEL_RED_ICON_THRESHOLD)
        records = self._red_icon_records(
            self._collect_red_icon_map(screenshot, threshold)
        )
        return [
            self._red_icon_detection(confidence, x, y, name)
            for confidence, x, y, name in records
        ]

    def _red_icon_detection(
        self, confidence: float, x: int, y: int, name: str
    ) -> AssetDetection:
        template_pair = self._template(name)
        height, width = (
            template_pair[0].shape[:2]
            if template_pair
            else (MIN_TEMPLATE_DIMENSION, MIN_TEMPLATE_DIMENSION)
        )
        return AssetDetection(
            AssetType.RED_ICON,
            name,
            confidence,
            x,
            y,
            width,
            height,
            x + config.RED_ICON_OFFSET_X,
            y + config.RED_ICON_OFFSET_Y,
        )

    def _detect_trackable_upgrade_stations(
        self, screenshot: Any
    ) -> list[AssetDetection]:
        if not config.ASSET_TRACKING_UPGRADE_STATION_ENABLED:
            return []
        upgrade_screenshot = screenshot[: config.UPGRADE_STATION_SEARCH_Y, :]
        return [
            AssetDetection(
                AssetType.UPGRADE_STATION,
                TemplateName.UPGRADE_STATION.value,
                confidence,
                x,
                y,
                width,
                height,
                x,
                y,
            )
            for confidence, x, y, width, height in self._upgrade_candidates(
                upgrade_screenshot, config.UPGRADE_STATION_THRESHOLD
            )
        ]

    def _detect_trackable_boxes(self, screenshot: Any) -> list[AssetDetection]:
        if not config.ASSET_TRACKING_BOX_ENABLED:
            return []
        box_screenshot = screenshot[: config.BOX_SEARCH_Y, :]
        return [
            AssetDetection(AssetType.BOX, name, confidence, x, y, width, height, x, y)
            for confidence, x, y, width, height, name in self._box_candidates(
                box_screenshot, include_tracked=False
            )
        ]

    def _reset_search_cycle(self) -> None:
        self.cycle_counter = 0
        self.wait_for_unlock_attempts = 0
        self.scroll_direction = 1
        self.scroll_cycle_index = 1
        self.scroll_cycle_progress = 0

    def _scroll(self) -> bool:
        distance = int(round(float(config.SCROLL_PIXEL_STEP) * float(config.SCROLL_DISTANCE_RATIO)))
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
        self._sleep(config.SCROLL_INTERVAL_PAUSE)
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

    def _record_level_completion(self, source: str) -> float:
        self.total_levels_completed += 1
        elapsed = 0.0
        if self.current_level_start_time is not None:
            elapsed = max(0.0, (datetime.now() - self.current_level_start_time).total_seconds())
        self.current_level_start_time = datetime.now()
        self._reset_search_cycle()
        self.telegram.notify_new_level(self.total_levels_completed, elapsed)
        self.historical_learner.record_completion(elapsed, source)
        return elapsed

    def handle_find_red_icons(self, current_state: State) -> StateResult:
        self._click_idle()
        self.cycle_counter += 1
        if self.cycle_counter >= SEARCH_CYCLES_BEFORE_SCROLL:
            self.cycle_counter = 0
            return State.SCROLL
        self.work_done = False
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
        self.red_icons = sorted(stable_icons, key=self._red_icon_priority)
        self.current_red_icon_index = 0
        self.work_done = True
        logger.info("%s red icons ready to process", len(self.red_icons))
        return State.CLICK_RED_ICON

    def handle_click_red_icon(self, current_state: State) -> StateResult:
        if self.current_red_icon_index >= len(self.red_icons):
            return State.OPEN_BOXES
        confidence, x, y = self.red_icons[self.current_red_icon_index]
        self.last_clicked_pos = (x, y)
        click_x = x + config.RED_ICON_OFFSET_X
        click_y = y + config.RED_ICON_OFFSET_Y
        clicked = self.mouse_controller.click(click_x, click_y, relative=True)
        self.tuner.record_click_result(clicked)
        self._apply_tuning()
        if not clicked:
            self._record_failed_click()
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
        return State.CHECK_UNLOCK

    def handle_check_unlock(self, current_state: State) -> StateResult:
        template_pair = self._template(TemplateName.UNLOCK.value)
        if template_pair is None:
            return State.SEARCH_UPGRADE_STATION
        template, mask = template_pair
        found, confidence, x, y = self.image_matcher.find_template(
            self.window_capture.capture(max_y=config.MAX_SEARCH_Y),
            template,
            mask=mask,
            threshold=config.UNLOCK_THRESHOLD,
            template_name=TemplateName.UNLOCK.value,
        )
        if found and not self.mouse_controller.is_in_forbidden_zone(x, y, relative=True):
            logger.info("Unlock found at (%s, %s) [%.3f]", x, y, confidence)
            self.mouse_controller.click(x, y, relative=True)
        return State.SEARCH_UPGRADE_STATION

    def handle_search_upgrade_station(self, current_state: State) -> StateResult:
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
                _, x, y = match
                self.upgrade_station_pos = (x, y)
                self._remember_successful_row()
                self.upgrade_found_in_cycle = True
                self.consecutive_failed_cycles = 0
                self.cycle_counter = 0
                self.tuner.record_search_result(True)
                self._apply_tuning()
                logger.info("Upgrade station found at (%s, %s)", x, y)
                return State.HOLD_UPGRADE_STATION
            if (
                attempt < MAX_UPGRADE_SEARCH_ATTEMPTS - 1
                and not self._sleep(self.tuner.search_interval)
            ):
                return State.OPEN_BOXES
        self._record_failed_click()
        self.red_icon_processed_count += 1
        self.consecutive_failed_cycles += 1
        self.tuner.record_search_result(False)
        self._apply_tuning()
        logger.info("Upgrade station not found")
        return State.OPEN_BOXES

    def handle_hold_upgrade_station(self, current_state: State) -> StateResult:
        if self.upgrade_station_pos is None:
            return State.OPEN_BOXES
        x, y = self.upgrade_station_pos
        if self.mouse_controller.is_in_forbidden_zone(x, y, relative=True):
            logger.warning(
                "Upgrade station blocked by forbidden zone at (%s, %s)", x, y
            )
            return State.OPEN_BOXES
        clicked = self.mouse_controller.precise_click(x, y, relative=True)
        self.tuner.record_click_result(clicked)
        self._apply_tuning()
        if not clicked:
            logger.warning(
                "Upgrade station verification click failed at (%s, %s)", x, y
            )
            return State.OPEN_BOXES
        if not self._sleep(config.UPGRADE_STATION_VERIFY_SETTLE_DELAY):
            return State.OPEN_BOXES
        threshold = float(config.UPGRADE_STATION_THRESHOLD)
        match = self._verify_upgrade_station(
            threshold,
            max(0.0, threshold - UPGRADE_STATION_THRESHOLD_RELAXATION),
            wait_between_attempts=True,
        )
        if match is None:
            logger.info("Upgrade station disappeared before hold")
            self.upgrade_station_pos = None
            self.upgrade_found_in_cycle = False
            self.tuner.record_search_result(False)
            self._apply_tuning()
            return State.OPEN_BOXES
        _, x, y = match
        self.upgrade_station_pos = (x, y)
        self.tuner.record_search_result(True)
        self._apply_tuning()
        monitor = _UpgradeStationHoldMonitor(self, threshold)
        held = self.mouse_controller.hold_at(
            x,
            y,
            duration=config.CLICK_HOLD_MAX_DURATION,
            relative=True,
            interrupt_check=monitor,
        )
        self.tuner.record_click_result(held or monitor.station_lost)
        self._apply_tuning()
        if not held and not monitor.station_lost:
            return State.OPEN_BOXES
        self._remember_successful_row()
        self.red_icon_processed_count += 1
        self.upgrade_station_pos = None
        self._click_idle()
        self._sleep(config.STATE_DELAY)
        self.upgrade_station_counter += 1
        if self.upgrade_station_counter >= UPGRADE_STATS_CYCLE_INTERVAL:
            self.upgrade_station_counter = 0
            return State.UPGRADE_STATS
        return State.OPEN_BOXES

    def handle_upgrade_stats(self, current_state: State) -> StateResult:
        self._click_idle()
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
        if not self.mouse_controller.click(*config.STATS_UPGRADE_BUTTON_POS, relative=True):
            return State.OPEN_BOXES
        self._sleep(config.STATE_DELAY)
        clicked = self.mouse_controller.click_stats_upgrade_at(
            *config.STATS_UPGRADE_POS,
            duration=config.STATS_UPGRADE_CLICK_DURATION,
            click_delay=config.STATS_UPGRADE_CLICK_DELAY,
            relative=True,
            interrupt_check=self._stop_requested.is_set,
        )
        if not clicked:
            return State.OPEN_BOXES
        self._click_idle()
        logger.info("Stats upgrade completed")
        return State.OPEN_BOXES

    def handle_open_boxes(self, current_state: State) -> StateResult:
        self._click_idle()
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
        boxes_found = 0
        for _, x, y, _, _, _ in candidates:
            if not self.mouse_controller.is_in_forbidden_zone(x, y, relative=True):
                boxes_found += (
                    1 if self.mouse_controller.click(x, y, relative=True) else 0
                )
        if boxes_found:
            self.work_done = True
            logger.info("Opened %s boxes", boxes_found)
        if self.upgrade_found_in_cycle:
            self.upgrade_found_in_cycle = False
            self.cycle_counter = 0
            return State.FIND_RED_ICONS
        self.cycle_counter += 1
        if (
            self.consecutive_failed_cycles >= FAILED_CYCLES_BEFORE_SCROLL
            or self.cycle_counter >= SEARCH_CYCLES_BEFORE_SCROLL
        ):
            if self.consecutive_failed_cycles >= FAILED_CYCLES_BEFORE_SCROLL:
                self.consecutive_failed_cycles = 0
            self.cycle_counter = 0
            return State.SCROLL
        return State.FIND_RED_ICONS

    def handle_scroll(self, current_state: State) -> StateResult:
        self.failed_click_tracker.clear()
        self._click_idle()
        screenshot = self.window_capture.capture(max_y=config.MAX_SEARCH_Y)
        found, _, _, _ = self._find_new_level_button(screenshot)
        if found:
            return State.TRANSITION_LEVEL
        if not self._scroll():
            logger.debug("Scroll action did not complete")
        self._click_idle()
        self.cycle_counter = 0
        return State.FIND_RED_ICONS

    def handle_check_new_level(self, current_state: State) -> StateResult:
        self._click_idle()
        self._sleep(CHECK_NEW_LEVEL_PRE_CLICK_DELAY)
        if not self.mouse_controller.click(*config.NEW_LEVEL_BUTTON_POS, relative=True):
            return State.CHECK_NEW_LEVEL
        self._sleep(CHECK_NEW_LEVEL_POST_CLICK_DELAY)
        match = self._find_upgrade_station(config.UPGRADE_STATION_THRESHOLD)
        if match is None:
            logger.warning("Upgrade station not found for level transition")
            self._reset_search_cycle()
            return State.FIND_RED_ICONS
        _, x, y = match
        if not self.mouse_controller.click(x, y, relative=True):
            return State.CHECK_NEW_LEVEL
        self._sleep(CHECK_NEW_LEVEL_UPGRADE_CLICK_DELAY)
        return State.TRANSITION_LEVEL

    def handle_transition_level(self, current_state: State) -> StateResult:
        self._click_idle()
        for attempt in range(MAX_LEVEL_TRANSITION_ATTEMPTS):
            found, _, x, y = self._find_new_level_button(
                self.window_capture.capture(max_y=config.MAX_SEARCH_Y)
            )
            if found:
                if not self.mouse_controller.click(x, y, relative=True):
                    return State.CHECK_NEW_LEVEL
                self._sleep(TRANSITION_LEVEL_BUTTON_WAIT_SECONDS)
                elapsed = self._record_level_completion("transition")
                logger.info(
                    "Level %s completed. Time spent: %.1fs",
                    self.total_levels_completed,
                    elapsed,
                )
                return State.WAIT_FOR_UNLOCK
            if attempt < MAX_LEVEL_TRANSITION_ATTEMPTS - 1:
                self._sleep(TRANSITION_LEVEL_RETRY_DELAY_SECONDS)
        logger.warning("New level button not found after transition attempts")
        self._reset_search_cycle()
        return State.FIND_RED_ICONS

    def handle_wait_for_unlock(self, current_state: State) -> StateResult:
        self._click_idle()
        self._sleep(WAIT_FOR_UNLOCK_PRE_SCAN_DELAY)
        self.wait_for_unlock_attempts += 1
        if self.wait_for_unlock_attempts > self.max_wait_for_unlock_attempts:
            self._reset_search_cycle()
            return State.FIND_RED_ICONS
        template_pair = self._template(TemplateName.UNLOCK.value)
        if template_pair is None:
            self._sleep(WAIT_FOR_UNLOCK_RETRY_DELAY)
            return State.WAIT_FOR_UNLOCK
        template, mask = template_pair
        found, confidence, x, y = self.image_matcher.find_template(
            self.window_capture.capture(),
            template,
            mask=mask,
            threshold=config.UNLOCK_THRESHOLD,
            template_name=TemplateName.UNLOCK.value,
        )
        if not found or self.mouse_controller.is_in_forbidden_zone(x, y, relative=True):
            self._sleep(WAIT_FOR_UNLOCK_RETRY_DELAY)
            return State.WAIT_FOR_UNLOCK
        logger.info("Unlock button found at (%s, %s) [%.3f]", x, y, confidence)
        if not self.mouse_controller.click(x, y, relative=True):
            return State.WAIT_FOR_UNLOCK
        self._sleep(WAIT_FOR_UNLOCK_POST_CLICK_DELAY)
        self._reset_search_cycle()
        return State.FIND_RED_ICONS
