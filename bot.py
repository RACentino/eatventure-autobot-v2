import json
import logging
import math
import os
import tempfile
import threading
import time
from collections.abc import Iterable
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import config
from image_matcher import ImageMatcher
from mouse_controller import MouseController, precise_sleep, wait_event
from state_machine import State, StateMachine
from telegram_notifier import TelegramNotifier
from window_capture import (
    ForbiddenAreaOverlay,
    WindowCapture,
    WindowCaptureError,
    WindowNotAvailableError,
)

logger = logging.getLogger(__name__)

TemplatePair = tuple[Any, Any]
MatchCandidate = tuple[float, int, int, int, int]
BoxCandidate = tuple[float, int, int, int, int, str]
RedIcon = tuple[float, int, int]
StateResult = State | None


class AdaptiveTuner:
    def __init__(self) -> None:
        self.enabled = bool(config.ADAPTIVE_TUNER_ENABLED)
        self.alpha = float(config.ADAPTIVE_TUNER_ALPHA)
        self.click_success_rate = 1.0
        self.search_success_rate = 1.0
        self.click_delay = float(config.CLICK_DELAY)
        self.move_delay = float(config.MOUSE_MOVE_DELAY)
        self.search_interval = float(config.UPGRADE_SEARCH_INTERVAL)

    def _ema(self, current: float, new_value: float) -> float:
        return (1.0 - self.alpha) * current + self.alpha * new_value

    def record_click_result(self, success: bool) -> None:
        if not self.enabled:
            return
        self.click_success_rate = self._ema(self.click_success_rate, 1.0 if success else 0.0)
        if self.click_success_rate < config.ADAPTIVE_TUNER_CLICK_LOW_THRESHOLD:
            self.click_delay = min(
                self.click_delay + config.ADAPTIVE_TUNER_CLICK_DELAY_STEP,
                config.ADAPTIVE_TUNER_MAX_CLICK_DELAY,
            )
            self.move_delay = min(
                self.move_delay + config.ADAPTIVE_TUNER_MOVE_DELAY_STEP,
                config.ADAPTIVE_TUNER_MAX_MOVE_DELAY,
            )
        elif self.click_success_rate > config.ADAPTIVE_TUNER_CLICK_HIGH_THRESHOLD:
            self.click_delay = max(
                self.click_delay - config.ADAPTIVE_TUNER_CLICK_DECREMENT,
                config.ADAPTIVE_TUNER_MIN_CLICK_DELAY,
            )
            self.move_delay = max(
                self.move_delay - config.ADAPTIVE_TUNER_MOVE_DECREMENT,
                config.ADAPTIVE_TUNER_MIN_MOVE_DELAY,
            )

    def record_search_result(self, success: bool) -> None:
        if not self.enabled:
            return
        self.search_success_rate = self._ema(self.search_success_rate, 1.0 if success else 0.0)
        if self.search_success_rate < config.ADAPTIVE_TUNER_SEARCH_LOW_THRESHOLD:
            self.search_interval = min(
                self.search_interval + config.ADAPTIVE_TUNER_SEARCH_INTERVAL_STEP,
                config.ADAPTIVE_TUNER_MAX_SEARCH_INTERVAL,
            )
        elif self.search_success_rate > config.ADAPTIVE_TUNER_SEARCH_HIGH_THRESHOLD:
            self.search_interval = max(
                self.search_interval - config.ADAPTIVE_TUNER_SEARCH_DECREMENT,
                config.ADAPTIVE_TUNER_MIN_SEARCH_INTERVAL,
            )

    def reset(self) -> None:
        self.click_success_rate = 1.0
        self.search_success_rate = 1.0
        self.click_delay = float(config.CLICK_DELAY)
        self.move_delay = float(config.MOUSE_MOVE_DELAY)
        self.search_interval = float(config.UPGRADE_SEARCH_INTERVAL)


class VisionPersistence:
    def __init__(self, path: str, save_interval: float) -> None:
        self.path = path
        self.save_interval = float(save_interval)
        self._last_save_time = 0.0
        self._lock = threading.RLock()

    def load(self) -> dict[str, Any]:
        if not self.path:
            return {}
        with self._lock:
            if not os.path.exists(self.path):
                return {}
            try:
                with open(self.path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, TypeError, ValueError) as exc:
                logger.warning("Failed to load persisted state from %s: %s", self.path, exc)
                return {}
        return data if isinstance(data, dict) else {}

    def _state_directory(self) -> tuple[str, str]:
        directory = os.path.dirname(self.path)
        return directory, directory or "."

    @staticmethod
    def _remove_temp_file(temp_path: str | None) -> None:
        if not temp_path:
            return
        try:
            os.remove(temp_path)
        except OSError:
            return

    @staticmethod
    def _write_temp_state_file(state: dict[str, Any], target_dir: str) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target_dir,
            prefix=".state-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
            return handle.name

    def save(self, state: dict[str, Any], force: bool = False) -> bool:
        if not self.path:
            return False

        now = time.monotonic()
        with self._lock:
            if not force and self.save_interval > 0 and (now - self._last_save_time) < self.save_interval:
                return False

            directory, target_dir = self._state_directory()
            temp_path = None
            try:
                if directory:
                    os.makedirs(directory, exist_ok=True)

                temp_path = self._write_temp_state_file(state, target_dir)
                os.replace(temp_path, self.path)
                self._last_save_time = now
                return True
            except (OSError, TypeError, ValueError) as exc:
                logger.error("Failed to persist state to %s: %s", self.path, exc)
                self._remove_temp_file(temp_path)
                return False


class VisionOptimizer:
    def __init__(self, persistence: VisionPersistence | None = None) -> None:
        self.enabled = bool(config.AI_VISION_ENABLED)
        self.alpha = float(config.AI_VISION_ALPHA)
        self.alpha_max = float(config.AI_VISION_ALPHA_MAX)
        self.confidence_boost = float(config.AI_VISION_CONFIDENCE_BOOST)
        self.red_icon_threshold = float(config.RED_ICON_THRESHOLD)
        self.new_level_threshold = float(config.NEW_LEVEL_THRESHOLD)
        self.new_level_red_icon_threshold = float(config.NEW_LEVEL_RED_ICON_THRESHOLD)
        self.upgrade_station_threshold = float(config.UPGRADE_STATION_THRESHOLD)
        self.stats_upgrade_threshold = float(config.STATS_RED_ICON_THRESHOLD)
        self.box_threshold = float(config.BOX_THRESHOLD)
        self.persistence = persistence
        self._miss_counts = {
            "red_icon": 0,
            "new_level": 0,
            "new_level_red_icon": 0,
            "upgrade_station": 0,
            "stats_upgrade": 0,
            "box": 0,
        }

    def _ema(self, current: float, new_value: float, alpha: float | None = None) -> float:
        blend = self.alpha if alpha is None else alpha
        return (1.0 - blend) * current + blend * new_value

    @staticmethod
    def _finite_float(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        return number

    def _adaptive_alpha(self, confidence: Any) -> float:
        confidence = self._finite_float(confidence)
        if confidence is None or confidence <= 0:
            return self.alpha
        boost = (
            max(0.0, min(1.0, confidence - config.AI_VISION_CONFIDENCE_THRESHOLD))
            * self.confidence_boost
        )
        return min(self.alpha + boost, self.alpha_max)

    def _update_threshold(self, name: str, confidence: Any, minimum: float, maximum: float) -> None:
        confidence = self._finite_float(confidence)
        if not self.enabled or confidence is None or confidence <= 0:
            return
        self._miss_counts[name] = 0
        current = getattr(self, f"{name}_threshold")
        target = max(minimum, min(maximum, confidence))
        setattr(self, f"{name}_threshold", self._ema(current, target, self._adaptive_alpha(confidence)))
        self._persist()

    def _update_miss(self, name: str, minimum: float, step: float, window: int) -> None:
        if not self.enabled:
            return
        self._miss_counts[name] += 1
        if self._miss_counts[name] < window:
            return
        self._miss_counts[name] = 0
        current = getattr(self, f"{name}_threshold")
        target = max(minimum, current - step)
        setattr(self, f"{name}_threshold", self._ema(current, target, self.alpha_max))
        self._persist()

    def update_red_icon_scan(self, confidences: Iterable[Any]) -> None:
        if not self.enabled:
            return
        if confidences:
            self._miss_counts["red_icon"] = 0
            finite_confidences = [
                value
                for confidence in confidences
                if (value := self._finite_float(confidence)) is not None
            ]
            if not finite_confidences:
                self._update_miss(
                    "red_icon",
                    config.AI_RED_ICON_THRESHOLD_MIN,
                    config.AI_RED_ICON_MISS_STEP,
                    config.AI_RED_ICON_MISS_WINDOW,
                )
                return
            average = sum(finite_confidences) / len(finite_confidences)
            target = max(
                config.AI_RED_ICON_THRESHOLD_MIN,
                min(average - config.AI_RED_ICON_MARGIN, config.AI_RED_ICON_THRESHOLD_MAX),
            )
            self.red_icon_threshold = self._ema(
                self.red_icon_threshold,
                target,
                self._adaptive_alpha(average),
            )
            self._persist()
            return
        self._update_miss(
            "red_icon",
            config.AI_RED_ICON_THRESHOLD_MIN,
            config.AI_RED_ICON_MISS_STEP,
            config.AI_RED_ICON_MISS_WINDOW,
        )

    def update_new_level_confidence(self, confidence: Any) -> None:
        self._update_threshold(
            "new_level",
            confidence,
            config.AI_NEW_LEVEL_THRESHOLD_MIN,
            config.AI_NEW_LEVEL_THRESHOLD_MAX,
        )

    def update_new_level_miss(self) -> None:
        self._update_miss(
            "new_level",
            config.AI_NEW_LEVEL_THRESHOLD_MIN,
            config.AI_NEW_LEVEL_MISS_STEP,
            config.AI_NEW_LEVEL_MISS_WINDOW,
        )

    def update_new_level_red_icon_confidence(self, confidence: Any) -> None:
        self._update_threshold(
            "new_level_red_icon",
            confidence,
            config.AI_NEW_LEVEL_RED_ICON_THRESHOLD_MIN,
            config.AI_NEW_LEVEL_RED_ICON_THRESHOLD_MAX,
        )

    def update_new_level_red_icon_miss(self) -> None:
        self._update_miss(
            "new_level_red_icon",
            config.AI_NEW_LEVEL_RED_ICON_THRESHOLD_MIN,
            config.AI_NEW_LEVEL_RED_ICON_MISS_STEP,
            config.AI_NEW_LEVEL_RED_ICON_MISS_WINDOW,
        )

    def update_upgrade_station_confidence(self, confidence: Any) -> None:
        self._update_threshold(
            "upgrade_station",
            confidence,
            config.AI_UPGRADE_STATION_THRESHOLD_MIN,
            config.AI_UPGRADE_STATION_THRESHOLD_MAX,
        )

    def update_upgrade_station_miss(self) -> None:
        self._update_miss(
            "upgrade_station",
            config.AI_UPGRADE_STATION_THRESHOLD_MIN,
            config.AI_UPGRADE_STATION_MISS_STEP,
            config.AI_UPGRADE_STATION_MISS_WINDOW,
        )

    def update_stats_upgrade_confidence(self, confidence: Any) -> None:
        self._update_threshold(
            "stats_upgrade",
            confidence,
            config.AI_STATS_UPGRADE_THRESHOLD_MIN,
            config.AI_STATS_UPGRADE_THRESHOLD_MAX,
        )

    def update_stats_upgrade_miss(self) -> None:
        self._update_miss(
            "stats_upgrade",
            config.AI_STATS_UPGRADE_THRESHOLD_MIN,
            config.AI_STATS_UPGRADE_MISS_STEP,
            config.AI_STATS_UPGRADE_MISS_WINDOW,
        )

    def update_box_confidence(self, confidence: Any) -> None:
        self._update_threshold(
            "box",
            confidence,
            config.AI_BOX_THRESHOLD_MIN,
            config.AI_BOX_THRESHOLD_MAX,
        )

    def update_box_miss(self) -> None:
        self._update_miss(
            "box",
            config.AI_BOX_THRESHOLD_MIN,
            config.AI_BOX_MISS_STEP,
            config.AI_BOX_MISS_WINDOW,
        )

    def apply_persisted_state(self, state: dict[str, Any]) -> None:
        if not state:
            return
        clamps = {
            "red_icon_threshold": (config.AI_RED_ICON_THRESHOLD_MIN, config.AI_RED_ICON_THRESHOLD_MAX),
            "new_level_threshold": (config.AI_NEW_LEVEL_THRESHOLD_MIN, config.AI_NEW_LEVEL_THRESHOLD_MAX),
            "new_level_red_icon_threshold": (
                config.AI_NEW_LEVEL_RED_ICON_THRESHOLD_MIN,
                config.AI_NEW_LEVEL_RED_ICON_THRESHOLD_MAX,
            ),
            "upgrade_station_threshold": (
                config.AI_UPGRADE_STATION_THRESHOLD_MIN,
                config.AI_UPGRADE_STATION_THRESHOLD_MAX,
            ),
            "stats_upgrade_threshold": (
                config.AI_STATS_UPGRADE_THRESHOLD_MIN,
                config.AI_STATS_UPGRADE_THRESHOLD_MAX,
            ),
            "box_threshold": (config.AI_BOX_THRESHOLD_MIN, config.AI_BOX_THRESHOLD_MAX),
        }
        for key, bounds in clamps.items():
            if key not in state:
                continue
            minimum, maximum = bounds
            try:
                value = float(state[key])
            except (TypeError, ValueError):
                logger.warning("Ignoring invalid persisted vision value for %s", key)
                continue
            if not math.isfinite(value):
                logger.warning("Ignoring non-finite persisted vision value for %s", key)
                continue
            setattr(self, key, max(minimum, min(maximum, value)))

    def reset(self) -> None:
        self.red_icon_threshold = float(config.RED_ICON_THRESHOLD)
        self.new_level_threshold = float(config.NEW_LEVEL_THRESHOLD)
        self.new_level_red_icon_threshold = float(config.NEW_LEVEL_RED_ICON_THRESHOLD)
        self.upgrade_station_threshold = float(config.UPGRADE_STATION_THRESHOLD)
        self.stats_upgrade_threshold = float(config.STATS_RED_ICON_THRESHOLD)
        self.box_threshold = float(config.BOX_THRESHOLD)
        for key in self._miss_counts:
            self._miss_counts[key] = 0
        self._persist(force=True)

    def _persist(self, force: bool = False) -> None:
        if self.persistence is None:
            return
        state = {
            "red_icon_threshold": float(self.red_icon_threshold),
            "new_level_threshold": float(self.new_level_threshold),
            "new_level_red_icon_threshold": float(self.new_level_red_icon_threshold),
            "upgrade_station_threshold": float(self.upgrade_station_threshold),
            "stats_upgrade_threshold": float(self.stats_upgrade_threshold),
            "box_threshold": float(self.box_threshold),
        }
        self.persistence.save(state, force=force)


class HistoricalLearner:
    def __init__(self, bot: Any, persistence: VisionPersistence | None = None) -> None:
        self.bot = bot
        self.persistence = persistence
        self.enabled = bool(config.AI_LEARNING_ENABLED)
        self.interval = max(config.LEARNING_LOOP_MIN_SLEEP, float(config.AI_LEARNING_THREAD_INTERVAL))
        self.pair_window = max(2, int(config.AI_LEARNING_PAIR_WINDOW))
        self.batch_window = max(2, int(config.AI_LEARNING_BATCH_WINDOW))
        self.ema_alpha = max(0.01, min(0.8, float(config.AI_LEARNING_EMA_ALPHA)))
        self.top_k = max(1, int(config.AI_LEARNING_PROFILE_BLEND_TOP_K))
        self.min_improvement_ratio = max(0.0, float(config.AI_LEARNING_MIN_IMPROVEMENT_RATIO))
        self.apply_cooldown = max(0.0, float(config.AI_LEARNING_APPLY_COOLDOWN))
        self._last_apply_time = 0.0
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._records = []
        self._total_completions = 0
        self._last_pair_processed = 0
        self._last_batch_processed = 0
        self._tuned_behavior = {}

        persisted = self.persistence.load() if self.enabled and self.persistence else {}
        if isinstance(persisted, dict) and persisted:
            records = persisted.get("records", [])
            if isinstance(records, list):
                self._records = [record for record in records if isinstance(record, dict)][
                    -config.AI_LEARNING_RECORDS_LIMIT :
                ]
            self._total_completions = max(
                0,
                self._safe_int(persisted.get("total_completions"), len(self._records)),
            )
            self._last_pair_processed = max(0, self._safe_int(persisted.get("last_pair_processed"), 0))
            self._last_batch_processed = max(0, self._safe_int(persisted.get("last_batch_processed"), 0))
            self._tuned_behavior = self._sanitize_behavior(persisted.get("tuned_behavior", {}))
            if self._tuned_behavior:
                self.bot.apply_learned_behavior(self._tuned_behavior)

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return int(default)

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        return number

    def _sanitize_behavior(self, behavior: Any) -> dict[str, float]:
        if not isinstance(behavior, dict):
            return {}
        bounds = {
            "click_delay": (config.AI_LEARNING_MIN_CLICK_DELAY, config.AI_LEARNING_MAX_CLICK_DELAY),
            "move_delay": (config.AI_LEARNING_MIN_MOVE_DELAY, config.AI_LEARNING_MAX_MOVE_DELAY),
            "search_interval": (
                config.AI_LEARNING_MIN_SEARCH_INTERVAL,
                config.AI_LEARNING_MAX_SEARCH_INTERVAL,
            ),
        }
        sanitized = {}
        for key, (minimum, maximum) in bounds.items():
            value = self._safe_float(behavior.get(key))
            if value is not None:
                sanitized[key] = self._clamp(value, minimum, maximum)
        return sanitized

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="historical_learner", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=config.AI_LEARNING_THREAD_JOIN_TIMEOUT)
        self._persist(force=True)

    def record_completion(self, seconds_spent: float, source: str) -> None:
        if not self.enabled or seconds_spent <= 0:
            return
        record = {
            "timestamp": time.time(),
            "time_spent": float(seconds_spent),
            "source": source,
            "behavior": self.bot.get_runtime_behavior_snapshot(),
        }
        with self._lock:
            self._records.append(record)
            self._records = self._records[-config.AI_LEARNING_RECORDS_LIMIT :]
            self._total_completions += 1
        self._persist()

    def reset(self) -> None:
        with self._lock:
            self._records = []
            self._total_completions = 0
            self._last_pair_processed = 0
            self._last_batch_processed = 0
            self._tuned_behavior = {}
            self._last_apply_time = 0.0
        self._persist(force=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._run_cycle()
            except Exception:
                logger.exception("Historical learner cycle failed")
            self._stop.wait(self.interval)

    def _run_cycle(self) -> None:
        with self._lock:
            records = list(self._records)
            total = int(self._total_completions)

        if (time.monotonic() - self._last_apply_time) < self.apply_cooldown:
            return

        changed = False
        if total >= self.pair_window and (total // self.pair_window) > self._last_pair_processed:
            changed = self._apply_profile(records[-self.pair_window :]) or changed
            self._last_pair_processed = total // self.pair_window

        if total >= self.batch_window and (total // self.batch_window) > self._last_batch_processed:
            changed = self._apply_profile(records[-self.batch_window :]) or changed
            self._last_batch_processed = total // self.batch_window

        if changed:
            self._last_apply_time = time.monotonic()
        self._persist()

    def _normalized_record(self, record: Any) -> dict[str, Any] | None:
        if not isinstance(record, dict):
            return None
        duration = self._safe_float(record.get("time_spent"))
        behavior = self._sanitize_behavior(record.get("behavior", {}))
        if duration is None or duration <= 0 or not behavior:
            return None
        return {"time_spent": duration, "behavior": behavior}

    def _apply_profile(self, records: list[dict[str, Any]]) -> bool:
        valid = [
            normalized_record
            for record in records
            if (normalized_record := self._normalized_record(record)) is not None
        ]
        if not valid:
            return False

        durations = [record["time_spent"] for record in valid]
        average = sum(durations) / len(durations)
        best = min(valid, key=lambda item: item["time_spent"])
        if average <= 0:
            return False

        improvement_ratio = (average - best["time_spent"]) / average
        if improvement_ratio < self.min_improvement_ratio:
            return False

        ranked = sorted(valid, key=lambda item: item["time_spent"])
        top = ranked[: self.top_k]
        profile = {"click_delay": 0.0, "move_delay": 0.0, "search_interval": 0.0}
        for record in top:
            behavior = record["behavior"]
            for key in profile:
                profile[key] += float(behavior.get(key, 0.0))

        count = float(len(top))
        for key in profile:
            profile[key] /= count

        current = self.bot.get_runtime_behavior_snapshot()
        tuned = {
            "click_delay": self._clamp(
                self._ema(current["click_delay"], profile["click_delay"]),
                config.AI_LEARNING_MIN_CLICK_DELAY,
                config.AI_LEARNING_MAX_CLICK_DELAY,
            ),
            "move_delay": self._clamp(
                self._ema(current["move_delay"], profile["move_delay"]),
                config.AI_LEARNING_MIN_MOVE_DELAY,
                config.AI_LEARNING_MAX_MOVE_DELAY,
            ),
            "search_interval": self._clamp(
                self._ema(current["search_interval"], profile["search_interval"]),
                config.AI_LEARNING_MIN_SEARCH_INTERVAL,
                config.AI_LEARNING_MAX_SEARCH_INTERVAL,
            ),
        }
        self._tuned_behavior = tuned
        self.bot.apply_learned_behavior(tuned)
        return True

    def _ema(self, current: float, target: float) -> float:
        return (1.0 - self.ema_alpha) * float(current) + self.ema_alpha * float(target)

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    def _persist(self, force: bool = False) -> None:
        if self.persistence is None:
            return
        with self._lock:
            state = {
                "records": self._records[-config.AI_LEARNING_RECORDS_LIMIT :],
                "total_completions": self._total_completions,
                "last_pair_processed": self._last_pair_processed,
                "last_batch_processed": self._last_batch_processed,
                "tuned_behavior": self._tuned_behavior,
            }
        self.persistence.save(state, force=force)


class EatventureBot:
    def __init__(self) -> None:
        logger.info("Initializing Eatventure Bot")
        self._stop_requested = threading.Event()
        self._step_active = threading.Event()

        self.window_capture = WindowCapture(config.WINDOW_TITLE, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        self.image_matcher = ImageMatcher(config.MATCH_THRESHOLD)
        self.mouse_controller = MouseController(
            self.window_capture.get_hwnd,
            config.CLICK_DELAY,
            config.MOUSE_MOVE_DELAY,
        )
        self.state_machine = StateMachine(State.FIND_RED_ICONS)
        self.telegram = TelegramNotifier(
            config.TELEGRAM_BOT_TOKEN,
            config.TELEGRAM_CHAT_ID,
            config.TELEGRAM_ENABLED,
        )

        self.tuner = AdaptiveTuner()
        self.vision_persistence = VisionPersistence(config.AI_VISION_STATE_FILE, config.AI_VISION_SAVE_INTERVAL)
        self.vision_optimizer = VisionOptimizer(self.vision_persistence)
        self.vision_optimizer.apply_persisted_state(self.vision_persistence.load())
        self.learning_persistence = VisionPersistence(
            config.AI_LEARNING_STATE_FILE,
            config.AI_LEARNING_SAVE_INTERVAL,
        )
        self.historical_learner = HistoricalLearner(self, self.learning_persistence)

        self.register_states()
        self.templates = self.load_templates()
        self._red_icon_template_names_cache = self._red_icon_template_names()
        self._red_icon_max_width, self._red_icon_max_height = self._red_icon_template_span()
        self.ready = self._validate_required_templates()
        self._successful_red_icon_history_limit = 24
        self.running = False
        self.red_icons = []
        self.current_red_icon_index = 0
        self.wait_for_unlock_attempts = 0
        self.max_wait_for_unlock_attempts = 4
        self.work_done = False
        self.cycle_counter = 0
        self.upgrade_station_counter = 0
        self.successful_red_icon_positions = deque(maxlen=self._successful_red_icon_history_limit)
        self.upgrade_found_in_cycle = False
        self.consecutive_failed_cycles = 0
        self.total_levels_completed = 0
        self.current_level_start_time = None
        self.upgrade_station_pos = None
        self.overlay = None
        self._oscillation_cycle_index = 1
        self._oscillation_leg_direction = 1
        self._oscillation_leg_progress = 0
        self._new_level_red_icon_verified = False
        self.forbidden_zones = [
            (
                config.FORBIDDEN_ZONE_1_X_MIN,
                config.FORBIDDEN_ZONE_1_X_MAX,
                config.FORBIDDEN_ZONE_1_Y_MIN,
                config.FORBIDDEN_ZONE_1_Y_MAX,
            ),
            (
                config.FORBIDDEN_ZONE_2_X_MIN,
                config.FORBIDDEN_ZONE_2_X_MAX,
                config.FORBIDDEN_ZONE_2_Y_MIN,
                config.FORBIDDEN_ZONE_2_Y_MAX,
            ),
            (
                config.FORBIDDEN_ZONE_3_X_MIN,
                config.FORBIDDEN_ZONE_3_X_MAX,
                config.FORBIDDEN_ZONE_3_Y_MIN,
                config.FORBIDDEN_ZONE_3_Y_MAX,
            ),
            (
                config.FORBIDDEN_ZONE_4_X_MIN,
                config.FORBIDDEN_ZONE_4_X_MAX,
                config.FORBIDDEN_ZONE_4_Y_MIN,
                config.FORBIDDEN_ZONE_4_Y_MAX,
            ),
            (
                config.FORBIDDEN_ZONE_5_X_MIN,
                config.FORBIDDEN_ZONE_5_X_MAX,
                config.FORBIDDEN_ZONE_5_Y_MIN,
                config.FORBIDDEN_ZONE_5_Y_MAX,
            ),
        ]

        logger.info("Bot initialized successfully")

    def load_templates(self) -> dict[str, TemplatePair]:
        templates: dict[str, TemplatePair] = {}
        templates_path = Path(config.ASSETS_DIR)
        if not templates_path.exists():
            logger.error("Assets directory not found: %s", templates_path)
            return templates

        for template_file in sorted(templates_path.glob("*.png")):
            try:
                template_name = template_file.stem
                template_img = self.image_matcher.load_template(template_file)
                templates[template_name] = template_img
                logger.info("Loaded template: %s", template_name)
            except Exception as exc:
                logger.error("Failed to load template %s: %s", template_file, exc)

        return templates

    def _available_red_icon_template_count(self) -> int:
        return sum(1 for name in self.templates if name.startswith("RedIcon"))

    def _red_icon_min_matches(self) -> int:
        if bool(getattr(config, "RED_ICON_FAST_MODE_ENABLED", False)):
            return 1
        available = self._available_red_icon_template_count()
        if available <= 0:
            return 1
        configured = max(1, int(config.RED_ICON_MIN_MATCHES))
        return min(configured, available)

    def _validate_required_templates(self) -> bool:
        missing = [name for name in ("newLevel", "unlock", "upgradeStation") if name not in self.templates]
        red_icon_count = self._available_red_icon_template_count()
        if red_icon_count <= 0:
            missing.append("RedIcon*")
        if missing:
            logger.error("Missing required templates: %s", ", ".join(missing))
            return False
        if red_icon_count < int(config.RED_ICON_MIN_MATCHES):
            logger.warning(
                "Only %s red-icon templates are available; consensus requirement reduced from %s",
                red_icon_count,
                config.RED_ICON_MIN_MATCHES,
            )
        return True

    def register_states(self) -> None:
        self.state_machine.register_handler(State.FIND_RED_ICONS, self.handle_find_red_icons)
        self.state_machine.register_handler(State.CLICK_RED_ICON, self.handle_click_red_icon)
        self.state_machine.register_handler(State.CHECK_UNLOCK, self.handle_check_unlock)
        self.state_machine.register_handler(State.SEARCH_UPGRADE_STATION, self.handle_search_upgrade_station)
        self.state_machine.register_handler(State.HOLD_UPGRADE_STATION, self.handle_hold_upgrade_station)
        self.state_machine.register_handler(State.OPEN_BOXES, self.handle_open_boxes)
        self.state_machine.register_handler(State.UPGRADE_STATS, self.handle_upgrade_stats)
        self.state_machine.register_handler(State.SCROLL, self.handle_scroll)
        self.state_machine.register_handler(State.CHECK_NEW_LEVEL, self.handle_check_new_level)
        self.state_machine.register_handler(State.TRANSITION_LEVEL, self.handle_transition_level)
        self.state_machine.register_handler(State.WAIT_FOR_UNLOCK, self.handle_wait_for_unlock)

    def _sleep(self, duration: Any) -> bool:
        return wait_event(self._stop_requested, duration)

    def _apply_tuning(self) -> None:
        self.mouse_controller.click_delay = float(self.tuner.click_delay)
        self.mouse_controller.move_delay = float(self.tuner.move_delay)

    def _template(self, template_name: str) -> TemplatePair | None:
        return self.templates.get(template_name)

    def _click_idle(self) -> bool:
        return self.mouse_controller.click(config.IDLE_CLICK_POS[0], config.IDLE_CLICK_POS[1], relative=True)

    def _new_level_threshold(self) -> float:
        if self.vision_optimizer.enabled:
            return self.vision_optimizer.new_level_threshold
        return config.NEW_LEVEL_THRESHOLD

    def _red_icon_scan_threshold(self) -> float:
        if not self.vision_optimizer.enabled:
            return config.RED_ICON_THRESHOLD
        return min(
            self.vision_optimizer.red_icon_threshold,
            self.vision_optimizer.new_level_red_icon_threshold,
        )

    def _box_threshold(self) -> float:
        if self.vision_optimizer.enabled:
            return self.vision_optimizer.box_threshold
        return config.BOX_THRESHOLD

    def _stats_upgrade_threshold(self) -> float:
        if self.vision_optimizer.enabled:
            return self.vision_optimizer.stats_upgrade_threshold
        return config.STATS_RED_ICON_THRESHOLD

    @staticmethod
    def _supervision_nms_enabled(flag_name: str) -> bool:
        return bool(getattr(config, "SUPERVISION_ENABLED", False) and getattr(config, flag_name, False))

    def _scrcpy_miss_recovery_sleep(self, duration: Any) -> bool:
        if not bool(getattr(config, "SCRCPY_MISS_RECOVERY_ENABLED", False)):
            return False
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
    ) -> tuple[list[RedIcon], list[float], RedIcon | None]:
        all_detections = self._collect_red_icon_detections(
            limited_screenshot,
            scan_threshold,
            min_distance=80,
        )
        red_icons, valid_red_icon_confidences = self._icons_from_detections(
            all_detections,
            min_matches,
        )
        best_new_level_icon = self._find_new_level_red_icon(screenshot, scan_threshold, min_matches)
        return red_icons, valid_red_icon_confidences, best_new_level_icon

    def _find_new_level_button(self, screenshot: Any) -> tuple[bool, float, int, int]:
        template_pair = self._template("newLevel")
        if template_pair is None:
            return False, 0.0, 0, 0
        template, mask = template_pair
        return self.image_matcher.find_template(
            screenshot,
            template,
            mask=mask,
            threshold=self._new_level_threshold(),
            template_name="newLevel",
        )

    def get_runtime_behavior_snapshot(self) -> dict[str, float]:
        return {
            "click_delay": float(self.tuner.click_delay),
            "move_delay": float(self.tuner.move_delay),
            "search_interval": float(self.tuner.search_interval),
        }

    def apply_learned_behavior(self, learned: dict[str, Any]) -> None:
        learned = self.historical_learner._sanitize_behavior(learned) if hasattr(self, "historical_learner") else learned
        if not learned:
            return
        self.tuner.click_delay = float(learned.get("click_delay", self.tuner.click_delay))
        self.tuner.move_delay = float(learned.get("move_delay", self.tuner.move_delay))
        self.tuner.search_interval = float(learned.get("search_interval", self.tuner.search_interval))
        self._apply_tuning()

    def wipe_memory(self) -> None:
        self.tuner.reset()
        self.vision_optimizer.reset()
        self.historical_learner.reset()
        self.successful_red_icon_positions = deque(maxlen=self._successful_red_icon_history_limit)
        self.current_level_start_time = datetime.now() if self.running else None
        self._apply_tuning()

    def _red_icon_template_names(self) -> list[str]:
        cached = getattr(self, "_red_icon_template_names_cache", None)
        if cached is not None:
            return cached
        template_names = [name for name in self.templates if name.startswith("RedIcon")]
        if not template_names:
            return ["RedIcon"]

        def sort_key(name: str) -> tuple[int, Any]:
            if name == "RedIcon":
                return (0, 0)
            if name == "RedIconNoBG":
                return (2, 0)
            suffix = name.replace("RedIcon", "", 1)
            if suffix.isdigit():
                return (1, int(suffix))
            return (3, suffix)

        return sorted(template_names, key=sort_key)

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
    def _box_iou(first: MatchCandidate, second: MatchCandidate) -> float:
        _, x1, y1, w1, h1 = first[:5]
        _, x2, y2, w2, h2 = second[:5]
        left = max(x1 - w1 / 2, x2 - w2 / 2)
        top = max(y1 - h1 / 2, y2 - h2 / 2)
        right = min(x1 + w1 / 2, x2 + w2 / 2)
        bottom = min(y1 + h1 / 2, y2 + h2 / 2)
        intersection = max(0, right - left) * max(0, bottom - top)
        union = (w1 * h1) + (w2 * h2) - intersection
        return intersection / union if union > 0 else 0

    @classmethod
    def _dedupe_box_candidates(
        cls: type["EatventureBot"],
        candidates: list[BoxCandidate],
        iou_threshold: float,
    ) -> list[BoxCandidate]:
        merged = []
        for candidate in sorted(candidates, key=lambda item: item[0], reverse=True):
            if all(cls._box_iou(candidate, existing) <= iou_threshold for existing in merged):
                merged.append(candidate)
        return merged

    @classmethod
    def _merge_box_candidates(cls: type["EatventureBot"], candidates: list[BoxCandidate]) -> list[BoxCandidate]:
        strict = cls._dedupe_box_candidates(candidates, 0.20)
        relaxed = cls._dedupe_box_candidates(candidates, 0.25)
        if len(relaxed) - len(strict) == 1:
            return relaxed
        return strict

    def _merge_box_candidates_with_supervision(self, candidates: list[BoxCandidate]) -> list[BoxCandidate]:
        legacy_candidates = self._merge_box_candidates(candidates)
        if not candidates or not self._supervision_nms_enabled("SUPERVISION_BOX_NMS_ENABLED"):
            return legacy_candidates
        supervision_candidates = self.image_matcher.filter_candidates_with_supervision_nms(
            candidates,
            iou_threshold=getattr(config, "SUPERVISION_BOX_NMS_IOU_THRESHOLD", 0.25),
            class_agnostic=getattr(config, "SUPERVISION_CLASS_AGNOSTIC_NMS", True),
        )
        if supervision_candidates is None or (not supervision_candidates and legacy_candidates):
            return legacy_candidates
        return [candidate for candidate in supervision_candidates]

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
    ) -> dict[tuple[int, int], list[tuple[str, float]]]:
        detections: dict[tuple[int, int], list[tuple[str, float]]] = {}
        if screenshot.size == 0:
            return detections
        template_names = self._red_icon_template_names()
        if bool(getattr(config, "RED_ICON_FAST_MODE_ENABLED", False)):
            configured_names = getattr(config, "RED_ICON_FAST_TEMPLATE_NAMES", ())
            fast_names = [name for name in configured_names if name in self.templates]
            if fast_names:
                template_names = fast_names
                min_distance = int(getattr(config, "RED_ICON_FAST_MIN_DISTANCE", min_distance))
        for template_name in template_names:
            if template_name not in self.templates:
                continue
            template, mask = self.templates[template_name]
            use_supervision_nms = self._supervision_nms_enabled("SUPERVISION_RED_ICON_NMS_ENABLED")
            supervision_iou_threshold = getattr(config, "SUPERVISION_RED_ICON_NMS_IOU_THRESHOLD", 0.20)
            supervision_class_agnostic = getattr(config, "SUPERVISION_CLASS_AGNOSTIC_NMS", True)
            icons = self.image_matcher.find_all_templates(
                screenshot,
                template,
                mask=mask,
                threshold=threshold,
                min_distance=min_distance,
                template_name=template_name,
                use_supervision_nms=use_supervision_nms,
                supervision_iou_threshold=supervision_iou_threshold,
                supervision_class_agnostic=supervision_class_agnostic,
            )
            template_height, template_width = template.shape[:2]
            for confidence, x, y in icons:
                location = (int(x) - template_width // 2, int(y) - template_height // 2)
                if not self._candidate_passes_template_gates(
                    screenshot,
                    template,
                    mask,
                    location,
                    config.RED_ICON_HSV_COLOR_GATE_ENABLED,
                    config.RED_ICON_HSV_RANGES,
                    config.RED_ICON_HSV_MIN_MATCH_RATIO,
                ):
                    continue
                self._merge_icon_detection(
                    detections,
                    x + offset_x,
                    y + offset_y,
                    template_name,
                    confidence,
                )
        return detections

    @staticmethod
    def _best_confidence_by_template(matches: list[tuple[str, float]]) -> dict[str, float]:
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
    ) -> tuple[list[RedIcon], list[float]]:
        icons: list[RedIcon] = []
        confidences: list[float] = []
        for (x, y), matches in detections.items():
            by_template = cls._best_confidence_by_template(matches)
            if len(by_template) < min_matches:
                continue
            max_confidence = max(by_template.values())
            icons.append((max_confidence, x, y))
            confidences.append(max_confidence)
        return icons, confidences

    def _find_best_zone_red_icon(
        self,
        screenshot: Any,
        threshold: float,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        min_distance: int = 80,
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
        )
        min_matches = self._red_icon_min_matches()
        icons, _ = self._icons_from_detections(detections, min_matches)
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
    ) -> RedIcon | None:
        if screenshot is None:
            screenshot = self.window_capture.capture(max_y=config.EXTENDED_SEARCH_Y)
        if scan_threshold is None:
            scan_threshold = self._red_icon_scan_threshold()
        if min_matches is None:
            min_matches = self._red_icon_min_matches()

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
        )
        all_red_icons_extended, _ = self._icons_from_detections(footer_detections, min_matches)

        new_level_icon_threshold = (
            self.vision_optimizer.new_level_red_icon_threshold
            if self.vision_optimizer.enabled
            else config.NEW_LEVEL_RED_ICON_THRESHOLD
        )
        best_new_level_icon = None
        for confidence, x, y in all_red_icons_extended:
            if not (
                config.NEW_LEVEL_RED_ICON_X_MIN <= x <= config.NEW_LEVEL_RED_ICON_X_MAX
                and config.NEW_LEVEL_RED_ICON_Y_MIN <= y <= config.NEW_LEVEL_RED_ICON_Y_MAX
            ):
                continue
            if confidence < new_level_icon_threshold:
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
        if self.current_level_start_time is not None:
            elapsed = (datetime.now() - self.current_level_start_time).total_seconds()
        self.current_level_start_time = datetime.now()
        self._reset_search_cycle()
        self.telegram.notify_new_level(self.total_levels_completed, elapsed)
        self.historical_learner.record_completion(elapsed, "transition")
        return elapsed

    def _reset_search_cycle(self) -> None:
        self.cycle_counter = 0
        self.wait_for_unlock_attempts = 0
        self._oscillation_cycle_index = 1
        self._oscillation_leg_direction = 1
        self._oscillation_leg_progress = 0
        self._new_level_red_icon_verified = False

    def _advance_oscillation_progress(self) -> None:
        target_steps = max(1, int(self._oscillation_cycle_index) * int(config.SCROLL_INCREMENT_STEP))
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
        distance = int(round(float(config.SCROLL_PIXEL_STEP) * float(config.SCROLL_DISTANCE_RATIO)))
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
        if moved:
            self._sleep(config.POST_SCROLL_SETTLE)
            self._sleep(config.SCROLL_INTERVAL_PAUSE)
            self._advance_oscillation_progress()
            self.mouse_controller.click(config.IDLE_CLICK_POS[0], config.IDLE_CLICK_POS[1], relative=True)
        return bool(moved)

    def _perform_single_down_scroll(self) -> bool:
        distance = int(round(float(config.SCROLL_PIXEL_STEP) * float(config.SCROLL_DISTANCE_RATIO)))
        start_x, start_y = config.SCROLL_START_POS
        target_y = start_y - distance
        logger.info("Verification scroll down before confirming new level red icon")
        moved = self.mouse_controller.drag(
            start_x,
            start_y,
            start_x,
            target_y,
            duration=config.SCROLL_DURATION,
            relative=True,
        )
        if moved:
            self._sleep(config.POST_SCROLL_SETTLE)
            self._sleep(config.SCROLL_INTERVAL_PAUSE)
        return bool(moved)

    def _upgrade_station_threshold(self) -> float:
        if self.vision_optimizer.enabled:
            return self.vision_optimizer.upgrade_station_threshold
        return config.UPGRADE_STATION_THRESHOLD

    def _candidate_passes_template_gates(
        self,
        screenshot: Any,
        template: Any,
        mask: Any,
        location: tuple[int, int],
        hsv_gate_enabled: bool,
        hsv_ranges: Any,
        hsv_min_match_ratio: float,
    ) -> bool:
        if not hsv_gate_enabled:
            return True
        return self.image_matcher._check_hsv_gate(
            screenshot,
            template,
            location,
            mask,
            hsv_ranges,
            hsv_min_match_ratio,
        )

    def _find_upgrade_station_match(self, threshold: float) -> RedIcon | None:
        if "upgradeStation" not in self.templates:
            return None

        limited_screenshot = self.window_capture.capture(max_y=config.UPGRADE_STATION_SEARCH_Y)
        template, mask = self.templates["upgradeStation"]
        candidates = self.image_matcher.find_all_templates(
            limited_screenshot,
            template,
            mask=mask,
            threshold=threshold,
            min_distance=15,
            template_name="upgradeStation",
            use_supervision_nms=self._supervision_nms_enabled("SUPERVISION_UPGRADE_STATION_NMS_ENABLED"),
            supervision_iou_threshold=getattr(config, "SUPERVISION_UPGRADE_STATION_NMS_IOU_THRESHOLD", 0.20),
            supervision_class_agnostic=getattr(config, "SUPERVISION_CLASS_AGNOSTIC_NMS", True),
        )
        if not candidates:
            return None

        template_height, template_width = template.shape[:2]
        for confidence, x, y in candidates:
            x = int(x)
            y = int(y)
            location = (x - template_width // 2, y - template_height // 2)

            if not self._candidate_passes_template_gates(
                limited_screenshot,
                template,
                mask,
                location,
                config.UPGRADE_STATION_HSV_COLOR_GATE_ENABLED,
                config.UPGRADE_STATION_HSV_RANGES,
                config.UPGRADE_STATION_HSV_MIN_MATCH_RATIO,
            ):
                continue

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
            if attempt < verify_attempts - 1 and not self._sleep(config.UPGRADE_STATION_VERIFY_SEARCH_INTERVAL):
                return None, False
        return None, True

    @staticmethod
    def _hold_max_duration() -> float:
        hold_max_duration = float(config.CLICK_HOLD_MAX_DURATION)
        if not math.isfinite(hold_max_duration):
            return 0.0
        return max(0.0, hold_max_duration)

    def _position_cursor_for_upgrade_hold(self, x: int, y: int) -> tuple[int, int] | None:
        screen_pos = self.mouse_controller._resolve_screen_position(x, y, relative=True)
        if screen_pos is None:
            logger.warning("Upgrade station hold position could not be resolved at (%s, %s)", x, y)
            return None

        screen_x, screen_y = screen_pos
        if self.mouse_controller._set_cursor_pos(screen_x, screen_y):
            if self.mouse_controller.move_delay > 0:
                precise_sleep(self.mouse_controller.move_delay)
            return screen_x, screen_y

        self.tuner.record_click_result(False)
        self._apply_tuning()
        logger.warning("Failed to position cursor for Upgrade Station hold at (%s, %s)", x, y)
        return None

    def _find_current_upgrade_hold_match(
        self,
        base_threshold: float,
        relaxed_threshold: float,
    ) -> RedIcon | None:
        current_match = self._find_upgrade_station_match(base_threshold)
        if current_match is not None:
            return current_match
        current_match = self._find_upgrade_station_match(relaxed_threshold)
        if current_match is not None:
            return current_match
        if self._scrcpy_miss_recovery_sleep(getattr(config, "SCRCPY_UPGRADE_MISS_RECOVERY_DELAY", 0.0)):
            return self._find_upgrade_station_match(relaxed_threshold)
        return None

    def _reposition_held_upgrade_station(
        self,
        screen_position: tuple[int, int],
        x: int,
        y: int,
    ) -> tuple[int, int] | None:
        next_screen_pos = self.mouse_controller._resolve_screen_position(x, y, relative=True)
        if next_screen_pos is None:
            logger.warning("Upgrade station hold target became invalid at (%s, %s)", x, y)
            return None
        if next_screen_pos == screen_position:
            return screen_position
        if self.mouse_controller._set_cursor_pos(next_screen_pos[0], next_screen_pos[1]):
            return next_screen_pos
        logger.warning("Failed to reposition held cursor to (%s, %s)", x, y)
        return None

    @staticmethod
    def _hold_duration_reached(hold_max_duration: float, hold_elapsed: float) -> bool:
        return hold_max_duration > 0.0 and hold_elapsed >= hold_max_duration

    def _monitor_upgrade_station_hold(
        self,
        screen_position_holder: list[tuple[int, int]],
        base_threshold: float,
        relaxed_threshold: float,
        hold_check_interval: float,
        hold_max_duration: float,
        hold_started_at: float,
    ) -> tuple[bool, bool, float]:
        while True:
            hold_elapsed = time.monotonic() - hold_started_at
            if self._stop_requested.is_set():
                logger.warning("Upgrade station hold interrupted after %.2fs", hold_elapsed)
                return False, False, hold_elapsed
            if self._hold_duration_reached(hold_max_duration, hold_elapsed):
                logger.warning(
                    "Upgrade station hold max duration %.2fs reached after %.2fs; releasing hold",
                    hold_max_duration,
                    hold_elapsed,
                )
                return True, True, hold_elapsed
            if not self._sleep(hold_check_interval):
                return False, False, hold_elapsed

            current_match = self._find_current_upgrade_hold_match(base_threshold, relaxed_threshold)
            if current_match is None:
                return True, False, time.monotonic() - hold_started_at

            confidence, x, y = current_match
            self.upgrade_station_pos = (x, y)
            self.vision_optimizer.update_upgrade_station_confidence(confidence)

            next_position = self._reposition_held_upgrade_station(screen_position_holder[0], x, y)
            if next_position is None:
                return False, False, time.monotonic() - hold_started_at
            screen_position_holder[0] = next_position

    def _hold_upgrade_station_until_complete(
        self,
        screen_position: tuple[int, int],
        base_threshold: float,
        relaxed_threshold: float,
        hold_check_interval: float,
        hold_max_duration: float,
    ) -> tuple[bool, bool, float]:
        screen_x, screen_y = screen_position
        hold_started_at = time.monotonic()
        screen_position_holder = [screen_position]
        holding = False

        try:
            if not self.mouse_controller._left_down_at_screen(
                screen_x,
                screen_y,
                interrupt_check=self._stop_requested.is_set,
            ):
                self.tuner.record_click_result(False)
                self._apply_tuning()
                logger.warning("Upgrade station hold press failed at (%s, %s)", screen_x, screen_y)
                return False, False, time.monotonic() - hold_started_at

            holding = True
            self.tuner.record_click_result(True)
            self._apply_tuning()
            return self._monitor_upgrade_station_hold(
                screen_position_holder,
                base_threshold,
                relaxed_threshold,
                hold_check_interval,
                hold_max_duration,
                hold_started_at,
            )
        finally:
            if holding:
                release_x, release_y = screen_position_holder[0]
                self.mouse_controller._left_up_at_screen(release_x, release_y)

    def _box_template_names(self) -> list[str]:
        return [box_name for box_name in ("box1", "box2", "box3", "box4") if box_name in self.templates]

    def _collect_box_candidates(self, limited_screenshot: Any, box_threshold: float) -> list[BoxCandidate]:
        box_candidates: list[BoxCandidate] = []
        for box_name in self._box_template_names():
            template, mask = self.templates[box_name]
            candidates = self.image_matcher.find_template_candidates(
                limited_screenshot,
                template,
                mask=mask,
                threshold=box_threshold,
                min_distance=12,
                template_name=box_name,
            )
            for confidence, x, y, candidate_width, candidate_height in candidates:
                candidate_width = int(candidate_width)
                candidate_height = int(candidate_height)
                location = (int(x) - candidate_width // 2, int(y) - candidate_height // 2)
                if not self._candidate_passes_template_gates(
                    limited_screenshot,
                    template,
                    mask,
                    location,
                    config.BOX_HSV_COLOR_GATE_ENABLED,
                    config.BOX_HSV_RANGES,
                    config.BOX_HSV_MIN_MATCH_RATIO,
                ):
                    continue
                box_candidates.append((confidence, int(x), int(y), candidate_width, candidate_height, box_name))
        return box_candidates

    def _click_box_candidates(self, merged_boxes: list[BoxCandidate]) -> tuple[int, float]:
        boxes_found = 0
        best_box_confidence = 0.0
        for confidence, x, y, _, _, _ in merged_boxes:
            if self.mouse_controller.is_in_forbidden_zone(x, y, relative=True):
                logger.debug("Box candidate is in a forbidden zone")
                continue
            if self.mouse_controller.click(x, y, relative=True):
                boxes_found += 1
                best_box_confidence = max(best_box_confidence, confidence)
        return boxes_found, best_box_confidence

    def _next_state_after_box_cycle(self) -> State:
        if self.consecutive_failed_cycles >= 3:
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

        self.cycle_counter += 1
        logger.info("No work detected in current area (idle pass %s/2)", self.cycle_counter)
        if self.cycle_counter >= 2:
            self.cycle_counter = 0
            return State.SCROLL

        return State.FIND_RED_ICONS

    def start(self) -> bool:
        if self.running:
            return True
        if self._step_active.is_set():
            logger.warning("Cannot start bot while a previous state step is still stopping")
            return False
        if not self.ready:
            logger.error("Cannot start bot because required templates are missing")
            return False
        try:
            self.window_capture.ensure_window(resize=True)
        except WindowCaptureError as exc:
            logger.error("Cannot start bot: %s", exc)
            self.running = False
            return False
        self._stop_requested.clear()
        self.running = True
        if self.current_level_start_time is None:
            self.current_level_start_time = datetime.now()
        self.historical_learner.start()
        if config.SHOW_FORBIDDEN_AREA and self.overlay is None:
            self.overlay = ForbiddenAreaOverlay(self.window_capture, self.forbidden_zones)
            self.overlay.start()
        return True

    def stop(self) -> None:
        self._stop_requested.set()
        if not self.running and self.overlay is None:
            self.historical_learner.stop()
            return
        self.running = False
        self.historical_learner.stop()
        if self.overlay is not None:
            self.overlay.stop()
            self.overlay = None

    def step(self) -> bool:
        if self._step_active.is_set():
            logger.warning("Ignoring reentrant bot step")
            return False
        self._step_active.set()
        try:
            if self._stop_requested.is_set():
                self.stop()
                return False
            if not self.window_capture.is_window_active():
                logger.error("Window '%s' is not available", config.WINDOW_TITLE)
                self.stop()
                return False
            self._apply_tuning()
            updated = bool(self.state_machine.update())
            if not updated:
                logger.error("State machine update failed in state %s; stopping bot", self.state_machine.get_state_name())
                self.stop()
                return False
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
            self._step_active.clear()

    def run(self) -> None:
        if not self.start():
            return
        try:
            while self.running:
                if not self.window_capture.is_window_active():
                    logger.error("Window '%s' is no longer active", config.WINDOW_TITLE)
                    break
                self.step()
                precise_sleep(0.1)
        finally:
            self.stop()

    def handle_find_red_icons(self, current_state: State) -> StateResult:
        self._click_idle()

        self.work_done = False

        screenshot = self.window_capture.capture(max_y=config.EXTENDED_SEARCH_Y)
        limited_screenshot = screenshot[: config.MAX_SEARCH_Y, :]

        has_new_level_template = self._template("newLevel") is not None
        found, confidence, x, y = self._find_new_level_button(limited_screenshot)
        if found:
            self.cycle_counter = 0
            self.vision_optimizer.update_new_level_confidence(confidence)
            logger.info("newLevel.png found at (%s, %s)", x, y)
            return State.TRANSITION_LEVEL
        if has_new_level_template:
            self.vision_optimizer.update_new_level_miss()

        scan_threshold = self._red_icon_scan_threshold()

        min_matches = self._red_icon_min_matches()
        self.red_icons, valid_red_icon_confidences, best_new_level_icon = self._scan_red_icon_frame(
            screenshot,
            limited_screenshot,
            scan_threshold,
            min_matches,
        )

        self.vision_optimizer.update_red_icon_scan(valid_red_icon_confidences)

        if (
            not self.red_icons
            and best_new_level_icon is None
            and self._scrcpy_miss_recovery_sleep(getattr(config, "SCRCPY_RED_ICON_MISS_RECOVERY_DELAY", 0.0))
        ):
            screenshot = self.window_capture.capture(max_y=config.EXTENDED_SEARCH_Y)
            limited_screenshot = screenshot[: config.MAX_SEARCH_Y, :]
            found, confidence, x, y = self._find_new_level_button(limited_screenshot)
            if found:
                self.cycle_counter = 0
                self.vision_optimizer.update_new_level_confidence(confidence)
                logger.info("newLevel.png found at (%s, %s) after SCRCPY recovery", x, y)
                return State.TRANSITION_LEVEL
            self.red_icons, valid_red_icon_confidences, best_new_level_icon = self._scan_red_icon_frame(
                screenshot,
                limited_screenshot,
                scan_threshold,
                min_matches,
            )
            self.vision_optimizer.update_red_icon_scan(valid_red_icon_confidences)

        if best_new_level_icon is not None:
            self.vision_optimizer.update_new_level_red_icon_confidence(best_new_level_icon[0])
            logger.info(
                "New level red icon detected at (%s, %s) [%.3f]",
                best_new_level_icon[1],
                best_new_level_icon[2],
                best_new_level_icon[0],
            )
            self._new_level_red_icon_verified = False
            return State.CHECK_NEW_LEVEL
        self.vision_optimizer.update_new_level_red_icon_miss()

        if self.red_icons:
            filtered_icons = self._clickable_red_icons(self.red_icons)

            if not filtered_icons:
                logger.info("No valid red icons after forbidden-zone filtering")
                return State.OPEN_BOXES

            self.red_icons = sorted(filtered_icons, key=self._red_icon_priority_key)
            self.current_red_icon_index = 0
            self.cycle_counter = 0
            self.work_done = True
            logger.info("%s red icons ready to process", len(self.red_icons))
            return State.CLICK_RED_ICON

        return State.OPEN_BOXES

    def handle_click_red_icon(self, current_state: State) -> StateResult:
        if self.current_red_icon_index >= len(self.red_icons):
            logger.info("All red icons processed, continuing cycle")
            return State.OPEN_BOXES

        confidence, x, y = self.red_icons[self.current_red_icon_index]
        click_x = x + config.RED_ICON_OFFSET_X
        click_y = y + config.RED_ICON_OFFSET_Y

        clicked = self.mouse_controller.click(click_x, click_y, relative=True)
        self.tuner.record_click_result(clicked)
        self._apply_tuning()
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
        return State.CHECK_UNLOCK

    def handle_check_unlock(self, current_state: State) -> StateResult:
        limited_screenshot = self.window_capture.capture(max_y=config.MAX_SEARCH_Y)

        template_pair = self._template("unlock")
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

        return State.SEARCH_UPGRADE_STATION

    def handle_search_upgrade_station(self, current_state: State) -> StateResult:
        base_threshold = self._upgrade_station_threshold()
        relaxed_threshold = max(0.0, base_threshold - 0.05)
        max_attempts = 5

        for attempt in range(max_attempts):
            if "upgradeStation" not in self.templates:
                break

            current_threshold = base_threshold if attempt < 2 else relaxed_threshold
            match = self._find_upgrade_station_match(current_threshold)
            if match is not None:
                confidence, x, y = match
                logger.info("Upgrade station found at (%s, %s) on attempt %s", x, y, attempt + 1)
                self.upgrade_station_pos = (x, y)
                self.upgrade_found_in_cycle = True
                self.consecutive_failed_cycles = 0
                self.cycle_counter = 0
                self.vision_optimizer.update_upgrade_station_confidence(confidence)
                self.tuner.record_search_result(True)
                self._apply_tuning()
                return State.HOLD_UPGRADE_STATION

            if attempt < max_attempts - 1:
                if not self._sleep(self.tuner.search_interval):
                    return State.OPEN_BOXES

        self.vision_optimizer.update_upgrade_station_miss()
        self.tuner.record_search_result(False)
        self._apply_tuning()
        self.consecutive_failed_cycles += 1
        logger.info("Upgrade station not found, returning to OPEN_BOXES")
        return State.OPEN_BOXES

    def handle_hold_upgrade_station(self, current_state: State) -> StateResult:
        if not self.upgrade_station_pos:
            return State.OPEN_BOXES

        x, y = self.upgrade_station_pos
        if self.mouse_controller.is_in_forbidden_zone(x, y, relative=True):
            logger.warning("Upgrade station blocked by forbidden zone at (%s, %s)", x, y)
            return State.OPEN_BOXES

        logger.info("Single-clicking upgrade station before verification at (%s, %s)", x, y)
        clicked = self.mouse_controller.precise_click(x, y, relative=True)
        self.tuner.record_click_result(clicked)
        self._apply_tuning()
        if not clicked:
            logger.warning("Upgrade station verification click failed at (%s, %s)", x, y)
            return State.OPEN_BOXES

        if not self._sleep(config.UPGRADE_STATION_VERIFY_SETTLE_DELAY):
            return State.OPEN_BOXES

        base_threshold = self._upgrade_station_threshold()
        relaxed_threshold = max(0.0, base_threshold - 0.05)
        verified_match, verification_completed = self._find_verified_upgrade_station_match(
            base_threshold,
            relaxed_threshold,
        )
        if not verification_completed:
            return State.OPEN_BOXES

        if verified_match is None:
            logger.info("Upgrade station disappeared after verification click; continuing main flow")
            self.upgrade_station_pos = None
            self.upgrade_found_in_cycle = False
            self.vision_optimizer.update_upgrade_station_miss()
            self.tuner.record_search_result(False)
            self._apply_tuning()
            return State.OPEN_BOXES

        confidence, x, y = verified_match
        self.upgrade_station_pos = (x, y)
        self.vision_optimizer.update_upgrade_station_confidence(confidence)
        self.tuner.record_search_result(True)
        self._apply_tuning()
        logger.info("Upgrade station verified active at (%s, %s) [%.3f]", x, y, confidence)
        if self.current_red_icon_index < len(self.red_icons):
            _, _, red_y = self.red_icons[self.current_red_icon_index]
            self._remember_successful_red_icon_position(red_y)

        hold_check_interval = max(0.05, min(0.20, float(config.UPGRADE_STATION_VERIFY_SEARCH_INTERVAL)))
        hold_max_duration = self._hold_max_duration()
        screen_position = self._position_cursor_for_upgrade_hold(x, y)
        if screen_position is None:
            return State.OPEN_BOXES

        logger.info("Press-and-holding upgrade station at (%s, %s)", x, y)
        hold_completed, hold_stopped_by_max_duration, hold_elapsed = self._hold_upgrade_station_until_complete(
            screen_position,
            base_threshold,
            relaxed_threshold,
            hold_check_interval,
            hold_max_duration,
        )
        if not hold_completed:
            return State.OPEN_BOXES

        if hold_stopped_by_max_duration:
            logger.info("Upgrade station hold released by max duration fallback after %.2fs", hold_elapsed)
        else:
            logger.info("Upgrade station no longer detected after %.2fs hold", hold_elapsed)
        self.upgrade_station_pos = None

        self.mouse_controller.click(config.IDLE_CLICK_POS[0], config.IDLE_CLICK_POS[1], relative=True)
        self._sleep(config.STATE_DELAY)
        self.upgrade_station_counter += 1
        if self.upgrade_station_counter >= 2:
            self.upgrade_station_counter = 0
            logger.info("Upgrade counter reached stats threshold")
            return State.UPGRADE_STATS

        return State.OPEN_BOXES

    def handle_upgrade_stats(self, current_state: State) -> StateResult:
        self._click_idle()

        screenshot = self.window_capture.capture(max_y=config.EXTENDED_SEARCH_Y)
        limited_screenshot = screenshot[: config.MAX_SEARCH_Y, :]

        found, confidence, _, _ = self._find_new_level_button(limited_screenshot)
        if found:
            self.vision_optimizer.update_new_level_confidence(confidence)
            return State.TRANSITION_LEVEL

        best_stats_match = self._find_best_zone_red_icon(
            screenshot,
            self._stats_upgrade_threshold(),
            config.UPGRADE_RED_ICON_X_MIN,
            config.UPGRADE_RED_ICON_X_MAX,
            config.UPGRADE_RED_ICON_Y_MIN,
            config.UPGRADE_RED_ICON_Y_MAX,
            min_distance=80,
        )

        if best_stats_match is None:
            self.vision_optimizer.update_stats_upgrade_miss()
            logger.info("No stats icon detected")
            return State.SCROLL

        best_stats_confidence, _, _ = best_stats_match
        self.vision_optimizer.update_stats_upgrade_confidence(best_stats_confidence)
        self.cycle_counter = 0
        logger.info("Stats icon found, upgrading")
        opened = self.mouse_controller.click(
            config.STATS_UPGRADE_BUTTON_POS[0],
            config.STATS_UPGRADE_BUTTON_POS[1],
            relative=True,
        )
        if not opened:
            return State.OPEN_BOXES

        self._sleep(config.STATE_DELAY)
        clicked = self.mouse_controller.spam_click_at(
            config.STATS_UPGRADE_POS[0],
            config.STATS_UPGRADE_POS[1],
            duration=config.STATS_UPGRADE_CLICK_DURATION,
            click_delay=config.STATS_UPGRADE_CLICK_DELAY,
            mouse_down_duration=config.STATS_UPGRADE_CLICK_DELAY,
            mouse_up_duration=0.0,
            relative=True,
            interrupt_check=self._stop_requested.is_set,
        )
        if not clicked:
            logger.warning("Stats upgrade spam-click failed at %s", config.STATS_UPGRADE_POS)
            return State.OPEN_BOXES

        self.mouse_controller.click(config.IDLE_CLICK_POS[0], config.IDLE_CLICK_POS[1], relative=True)
        logger.info("Stats upgrade completed")
        return State.OPEN_BOXES

    def handle_open_boxes(self, current_state: State) -> StateResult:
        self._click_idle()

        limited_screenshot = self.window_capture.capture(
            max_y=getattr(config, "BOX_SEARCH_Y", config.MAX_SEARCH_Y)
        )

        found, confidence, _, _ = self._find_new_level_button(limited_screenshot)
        if found:
            self.vision_optimizer.update_new_level_confidence(confidence)
            logger.info("New level found while opening boxes")
            return State.TRANSITION_LEVEL

        box_threshold = self._box_threshold()
        box_candidates = self._collect_box_candidates(limited_screenshot, box_threshold)
        if not box_candidates and self._scrcpy_miss_recovery_sleep(getattr(config, "SCRCPY_BOX_MISS_RECOVERY_DELAY", 0.0)):
            limited_screenshot = self.window_capture.capture(
                max_y=getattr(config, "BOX_SEARCH_Y", config.MAX_SEARCH_Y)
            )
            found, confidence, _, _ = self._find_new_level_button(limited_screenshot)
            if found:
                self.vision_optimizer.update_new_level_confidence(confidence)
                logger.info("New level found while opening boxes after SCRCPY recovery")
                return State.TRANSITION_LEVEL
            box_candidates = self._collect_box_candidates(limited_screenshot, box_threshold)

        merged_boxes = self._merge_box_candidates_with_supervision(box_candidates)
        boxes_found, best_box_confidence = self._click_box_candidates(merged_boxes)

        if boxes_found > 0:
            self.work_done = True
            self.cycle_counter = 0
            self.vision_optimizer.update_box_confidence(best_box_confidence)
            logger.info("Opened %s boxes", boxes_found)
        else:
            self.vision_optimizer.update_box_miss()

        return self._next_state_after_box_cycle()

    def handle_scroll(self, current_state: State) -> StateResult:
        self._click_idle()
        self._perform_oscillating_scroll_step()
        self.cycle_counter = 0
        return State.FIND_RED_ICONS

    def handle_check_new_level(self, current_state: State) -> StateResult:
        if not self._click_idle():
            logger.warning("Failed to clear focus before confirming the new level")
            return State.CHECK_NEW_LEVEL
        self._sleep(0.05)
        if not self._new_level_red_icon_verified:
            if not self._perform_single_down_scroll():
                logger.warning("Failed to perform verification scroll for new level red icon")
                return State.CHECK_NEW_LEVEL

            confirmed_icon = self._find_new_level_red_icon()
            if confirmed_icon is None:
                logger.info("New level red icon disappeared after verification scroll; resuming main flow")
                self._new_level_red_icon_verified = False
                self.vision_optimizer.update_new_level_red_icon_miss()
                self._reset_search_cycle()
                return State.FIND_RED_ICONS

            self._new_level_red_icon_verified = True
            self.vision_optimizer.update_new_level_red_icon_confidence(confirmed_icon[0])
            logger.info(
                "New level red icon confirmed after verification scroll at (%s, %s) [%.3f]",
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
            logger.warning("Failed to click the new level button at %s", config.NEW_LEVEL_BUTTON_POS)
            return State.CHECK_NEW_LEVEL
        self._sleep(0.30)
        advanced = self.mouse_controller.click(
            config.LEVEL_TRANSITION_POS[0],
            config.LEVEL_TRANSITION_POS[1],
            relative=True,
        )
        if not advanced:
            logger.warning("Failed to click the level transition button at %s", config.LEVEL_TRANSITION_POS)
            return State.CHECK_NEW_LEVEL
        if not self._sleep(0.20):
            return State.OPEN_BOXES
        elapsed = self._record_level_completion()
        logger.info(
            "Level %s completed via verified red-icon path. Time spent: %.1fs",
            self.total_levels_completed,
            elapsed,
        )
        return State.WAIT_FOR_UNLOCK

    def handle_transition_level(self, current_state: State) -> StateResult:
        self._click_idle()

        max_attempts = 5
        for attempt in range(max_attempts):
            limited_screenshot = self.window_capture.capture(max_y=config.MAX_SEARCH_Y)

            found, confidence, x, y = self._find_new_level_button(limited_screenshot)
            if found:
                self.vision_optimizer.update_new_level_confidence(confidence)
                logger.info("New level button found at (%s, %s) on attempt %s", x, y, attempt + 1)
                clicked = self.mouse_controller.click(x, y, relative=True)
                if not clicked:
                    logger.warning("New level button click failed at (%s, %s)", x, y)
                    return State.CHECK_NEW_LEVEL
                self._sleep(1.0)
                elapsed = self._record_level_completion()
                logger.info(
                    "Level %s completed. Time spent: %.1fs",
                    self.total_levels_completed,
                    elapsed,
                )
                return State.WAIT_FOR_UNLOCK

            if attempt < max_attempts - 1:
                self._sleep(0.20)

        self.vision_optimizer.update_new_level_miss()
        logger.warning("New level button not found after %s attempts", max_attempts)
        self._reset_search_cycle()
        return State.FIND_RED_ICONS

    def handle_wait_for_unlock(self, current_state: State) -> StateResult:
        if not self._click_idle():
            logger.warning("Failed to clear focus while waiting for the next unlock")
            return State.WAIT_FOR_UNLOCK
        self._sleep(0.05)

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
        template_pair = self._template("unlock")
        if template_pair is None:
            self._sleep(0.30)
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
            self._sleep(0.30)
            return State.WAIT_FOR_UNLOCK

        logger.info("Unlock button found at (%s, %s) [%.3f]", x, y, confidence)
        if self.mouse_controller.is_in_forbidden_zone(x, y, relative=True):
            logger.warning("Unlock button found in forbidden zone at (%s, %s)", x, y)
            self._sleep(0.30)
            return State.WAIT_FOR_UNLOCK
        if not self.mouse_controller.click(x, y, relative=True):
            logger.warning("Unlock button click failed at (%s, %s)", x, y)
            return State.WAIT_FOR_UNLOCK

        self._sleep(0.50)
        self.wait_for_unlock_attempts = 0
        self._reset_search_cycle()
        return State.FIND_RED_ICONS
