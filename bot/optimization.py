import json
import logging
import math
import tempfile
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from core import config

logger = logging.getLogger(__name__)
LEARNING_LOOP_ITERATION_LIMIT = 2_147_483_647
MIN_LEARNING_LOOP_INTERVAL = 0.1
MIN_LEARNING_JOIN_TIMEOUT = 0.1


@dataclass(frozen=True)
class AdaptiveTunerState:
    click_success_rate: float
    search_success_rate: float
    click_delay: float
    move_delay: float
    search_interval: float


class AdaptiveTuner:
    def __init__(self) -> None:
        self.enabled = bool(config.ADAPTIVE_TUNER_ENABLED)
        self.alpha = config.bounded_float(config.ADAPTIVE_TUNER_ALPHA, 0.18, minimum=0.0, maximum=1.0)
        self._state = self._default_state()

    @staticmethod
    def _default_state() -> AdaptiveTunerState:
        return AdaptiveTunerState(
            click_success_rate=1.0,
            search_success_rate=1.0,
            click_delay=config.bounded_float(config.CLICK_DELAY, 0.08, minimum=0.0),
            move_delay=config.bounded_float(config.MOUSE_MOVE_DELAY, 0.025, minimum=0.0),
            search_interval=config.bounded_float(config.UPGRADE_SEARCH_INTERVAL, 0.08, minimum=0.0),
        )

    @property
    def click_success_rate(self) -> float:
        return self._state.click_success_rate

    @property
    def search_success_rate(self) -> float:
        return self._state.search_success_rate

    @property
    def click_delay(self) -> float:
        return self._state.click_delay

    @property
    def move_delay(self) -> float:
        return self._state.move_delay

    @property
    def search_interval(self) -> float:
        return self._state.search_interval

    def _ema(self, current: float, new_value: float) -> float:
        return (1.0 - self.alpha) * current + self.alpha * new_value

    def record_click_result(self, success: bool) -> None:
        if not self.enabled:
            return
        state = self._state
        click_success_rate = self._ema(state.click_success_rate, 1.0 if success else 0.0)
        click_delay = state.click_delay
        move_delay = state.move_delay
        if click_success_rate < config.ADAPTIVE_TUNER_CLICK_LOW_THRESHOLD:
            click_delay = min(
                state.click_delay + config.ADAPTIVE_TUNER_CLICK_DELAY_STEP,
                config.ADAPTIVE_TUNER_MAX_CLICK_DELAY,
            )
            move_delay = min(
                state.move_delay + config.ADAPTIVE_TUNER_MOVE_DELAY_STEP,
                config.ADAPTIVE_TUNER_MAX_MOVE_DELAY,
            )
        elif click_success_rate > config.ADAPTIVE_TUNER_CLICK_HIGH_THRESHOLD:
            click_delay = max(
                state.click_delay - config.ADAPTIVE_TUNER_CLICK_DECREMENT,
                config.ADAPTIVE_TUNER_MIN_CLICK_DELAY,
            )
            move_delay = max(
                state.move_delay - config.ADAPTIVE_TUNER_MOVE_DECREMENT,
                config.ADAPTIVE_TUNER_MIN_MOVE_DELAY,
            )
        self._state = replace(
            state,
            click_success_rate=click_success_rate,
            click_delay=click_delay,
            move_delay=move_delay,
        )

    def record_search_result(self, success: bool) -> None:
        if not self.enabled:
            return
        state = self._state
        search_success_rate = self._ema(state.search_success_rate, 1.0 if success else 0.0)
        search_interval = state.search_interval
        if search_success_rate < config.ADAPTIVE_TUNER_SEARCH_LOW_THRESHOLD:
            search_interval = min(
                state.search_interval + config.ADAPTIVE_TUNER_SEARCH_INTERVAL_STEP,
                config.ADAPTIVE_TUNER_MAX_SEARCH_INTERVAL,
            )
        elif search_success_rate > config.ADAPTIVE_TUNER_SEARCH_HIGH_THRESHOLD:
            search_interval = max(
                state.search_interval - config.ADAPTIVE_TUNER_SEARCH_DECREMENT,
                config.ADAPTIVE_TUNER_MIN_SEARCH_INTERVAL,
            )
        self._state = replace(
            state,
            search_success_rate=search_success_rate,
            search_interval=search_interval,
        )

    def reset(self) -> None:
        self._state = self._default_state()

    def apply_runtime_behavior(self, behavior: dict[str, Any]) -> None:
        state = self._state
        self._state = replace(
            state,
            click_delay=config.bounded_float(
                behavior.get("click_delay"),
                state.click_delay,
                minimum=config.ADAPTIVE_TUNER_MIN_CLICK_DELAY,
                maximum=config.ADAPTIVE_TUNER_MAX_CLICK_DELAY,
            ),
            move_delay=config.bounded_float(
                behavior.get("move_delay"),
                state.move_delay,
                minimum=config.ADAPTIVE_TUNER_MIN_MOVE_DELAY,
                maximum=config.ADAPTIVE_TUNER_MAX_MOVE_DELAY,
            ),
            search_interval=config.bounded_float(
                behavior.get("search_interval"),
                state.search_interval,
                minimum=config.ADAPTIVE_TUNER_MIN_SEARCH_INTERVAL,
                maximum=config.ADAPTIVE_TUNER_MAX_SEARCH_INTERVAL,
            ),
        )


class VisionPersistence:
    def __init__(self, path: str | Path, save_interval: float) -> None:
        self.path = Path(path) if path else None
        try:
            normalized_interval = float(save_interval)
        except (TypeError, ValueError):
            normalized_interval = 0.0
        if not math.isfinite(normalized_interval):
            normalized_interval = 0.0
        self.save_interval = max(0.0, normalized_interval)
        self._last_save_time = 0.0
        self._lock = threading.RLock()

    def load(self) -> dict[str, Any]:
        if self.path is None:
            return {}
        with self._lock:
            if not self.path.exists():
                return {}
            try:
                with self.path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, TypeError, ValueError) as exc:
                logger.warning("Failed to load persisted state from %s: %s", self.path, exc)
                return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _remove_temp_file(temp_path: Path | None) -> None:
        if temp_path is None:
            return
        try:
            temp_path.unlink()
        except OSError:
            return

    @staticmethod
    def _write_temp_state_file(state: dict[str, Any], target_dir: Path) -> Path:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target_dir,
            prefix=".state-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
            return Path(handle.name)

    def save(self, state: dict[str, Any], force: bool = False) -> bool:
        if self.path is None:
            return False

        now = time.monotonic()
        with self._lock:
            if not force and self.save_interval > 0 and (now - self._last_save_time) < self.save_interval:
                return False

            temp_path = None
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = self._write_temp_state_file(state, self.path.parent)
                temp_path.replace(self.path)
                self._last_save_time = now
                return True
            except (OSError, TypeError, ValueError) as exc:
                logger.error("Failed to persist state to %s: %s", self.path, exc)
                self._remove_temp_file(temp_path)
                return False


class VisionOptimizer:
    def __init__(self, persistence: VisionPersistence | None = None) -> None:
        self.enabled = bool(config.AI_VISION_ENABLED)
        self.alpha = config.bounded_float(config.AI_VISION_ALPHA, 0.18, minimum=0.0, maximum=1.0)
        self.alpha_max = config.bounded_float(config.AI_VISION_ALPHA_MAX, 0.35, minimum=self.alpha, maximum=1.0)
        self.confidence_boost = config.bounded_float(config.AI_VISION_CONFIDENCE_BOOST, 0.10, minimum=0.0)
        self.red_icon_threshold = config.bounded_float(config.RED_ICON_THRESHOLD, 0.92, minimum=0.0, maximum=1.0)
        self.new_level_threshold = config.bounded_float(config.NEW_LEVEL_THRESHOLD, 0.965, minimum=0.0, maximum=1.0)
        self.new_level_red_icon_threshold = config.bounded_float(
            config.NEW_LEVEL_RED_ICON_THRESHOLD,
            0.942,
            minimum=0.0,
            maximum=1.0,
        )
        self.upgrade_station_threshold = config.bounded_float(
            config.UPGRADE_STATION_THRESHOLD,
            0.91,
            minimum=0.0,
            maximum=1.0,
        )
        self.stats_upgrade_threshold = config.bounded_float(
            config.STATS_RED_ICON_THRESHOLD,
            0.943,
            minimum=0.0,
            maximum=1.0,
        )
        self.box_threshold = config.bounded_float(config.BOX_THRESHOLD, 0.93, minimum=0.0, maximum=1.0)
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

    @staticmethod
    def _threshold_clamps() -> dict[str, tuple[float, float]]:
        return {
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

    def _apply_persisted_threshold(self, state: dict[str, Any], key: str, minimum: float, maximum: float) -> None:
        if key not in state:
            return
        try:
            value = float(state[key])
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid persisted vision value for %s", key)
            return
        if not math.isfinite(value):
            logger.warning("Ignoring non-finite persisted vision value for %s", key)
            return
        setattr(self, key, max(minimum, min(maximum, value)))

    def apply_persisted_state(self, state: dict[str, Any]) -> None:
        if not state:
            return
        for key, (minimum, maximum) in self._threshold_clamps().items():
            self._apply_persisted_threshold(state, key, minimum, maximum)

    def reset(self) -> None:
        self.red_icon_threshold = config.bounded_float(config.RED_ICON_THRESHOLD, 0.92, minimum=0.0, maximum=1.0)
        self.new_level_threshold = config.bounded_float(config.NEW_LEVEL_THRESHOLD, 0.965, minimum=0.0, maximum=1.0)
        self.new_level_red_icon_threshold = config.bounded_float(
            config.NEW_LEVEL_RED_ICON_THRESHOLD,
            0.942,
            minimum=0.0,
            maximum=1.0,
        )
        self.upgrade_station_threshold = config.bounded_float(
            config.UPGRADE_STATION_THRESHOLD,
            0.91,
            minimum=0.0,
            maximum=1.0,
        )
        self.stats_upgrade_threshold = config.bounded_float(
            config.STATS_RED_ICON_THRESHOLD,
            0.943,
            minimum=0.0,
            maximum=1.0,
        )
        self.box_threshold = config.bounded_float(config.BOX_THRESHOLD, 0.93, minimum=0.0, maximum=1.0)
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
        configured_interval = self._safe_float(config.AI_LEARNING_THREAD_INTERVAL)
        minimum_interval = self._safe_float(config.LEARNING_LOOP_MIN_SLEEP)
        self.interval = max(
            MIN_LEARNING_LOOP_INTERVAL,
            configured_interval if configured_interval is not None else MIN_LEARNING_LOOP_INTERVAL,
            minimum_interval if minimum_interval is not None else MIN_LEARNING_LOOP_INTERVAL,
        )
        configured_join_timeout = self._safe_float(config.AI_LEARNING_THREAD_JOIN_TIMEOUT)
        self.join_timeout = max(
            MIN_LEARNING_JOIN_TIMEOUT,
            configured_join_timeout if configured_join_timeout is not None else MIN_LEARNING_JOIN_TIMEOUT,
        )
        self.records_limit = config.bounded_int(config.AI_LEARNING_RECORDS_LIMIT, 256, minimum=1, maximum=10_000)
        self.pair_window = config.bounded_int(config.AI_LEARNING_PAIR_WINDOW, 5, minimum=2, maximum=self.records_limit)
        self.batch_window = config.bounded_int(config.AI_LEARNING_BATCH_WINDOW, 12, minimum=2, maximum=self.records_limit)
        self.ema_alpha = config.bounded_float(config.AI_LEARNING_EMA_ALPHA, 0.14, minimum=0.01, maximum=0.8)
        self.top_k = config.bounded_int(config.AI_LEARNING_PROFILE_BLEND_TOP_K, 3, minimum=1, maximum=self.records_limit)
        self.min_improvement_ratio = config.bounded_float(
            config.AI_LEARNING_MIN_IMPROVEMENT_RATIO,
            0.05,
            minimum=0.0,
            maximum=1.0,
        )
        self.apply_cooldown = config.bounded_float(config.AI_LEARNING_APPLY_COOLDOWN, 60.0, minimum=0.0)
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
                self._records = [record for record in records if isinstance(record, dict)][-self.records_limit :]
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
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(number):
            return None
        return number

    @staticmethod
    def _behavior_bounds() -> dict[str, tuple[float, float]]:
        return {
            "click_delay": (config.AI_LEARNING_MIN_CLICK_DELAY, config.AI_LEARNING_MAX_CLICK_DELAY),
            "move_delay": (config.AI_LEARNING_MIN_MOVE_DELAY, config.AI_LEARNING_MAX_MOVE_DELAY),
            "search_interval": (
                config.AI_LEARNING_MIN_SEARCH_INTERVAL,
                config.AI_LEARNING_MAX_SEARCH_INTERVAL,
            ),
        }

    def _sanitize_behavior(self, behavior: Any) -> dict[str, float]:
        if not isinstance(behavior, dict):
            return {}
        return {
            key: self._clamp(value, minimum, maximum)
            for key, (minimum, maximum) in self._behavior_bounds().items()
            if (value := self._safe_float(behavior.get(key))) is not None
        }

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
            self._thread.join(timeout=self.join_timeout)
        if self.enabled:
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
            self._records = self._records[-self.records_limit :]
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
        for _ in range(LEARNING_LOOP_ITERATION_LIMIT):
            if self._stop.is_set():
                return
            try:
                self._run_cycle()
            except Exception:
                logger.exception("Historical learner cycle failed")
            self._stop.wait(self.interval)
        logger.warning("Historical learner loop reached iteration limit")

    def _run_cycle(self) -> None:
        with self._lock:
            records = list(self._records)
            total = int(self._total_completions)
            last_pair_processed = int(self._last_pair_processed)
            last_batch_processed = int(self._last_batch_processed)
            last_apply_time = float(self._last_apply_time)

        if (time.monotonic() - last_apply_time) < self.apply_cooldown:
            return

        changed = False
        next_pair_processed = last_pair_processed
        next_batch_processed = last_batch_processed
        if total >= self.pair_window and (total // self.pair_window) > last_pair_processed:
            changed = self._apply_profile(records[-self.pair_window :]) or changed
            next_pair_processed = total // self.pair_window

        if total >= self.batch_window and (total // self.batch_window) > last_batch_processed:
            changed = self._apply_profile(records[-self.batch_window :]) or changed
            next_batch_processed = total // self.batch_window

        applied_at = time.monotonic() if changed else last_apply_time
        with self._lock:
            self._last_pair_processed = max(self._last_pair_processed, next_pair_processed)
            self._last_batch_processed = max(self._last_batch_processed, next_batch_processed)
            if changed:
                self._last_apply_time = applied_at
        if changed:
            logger.debug("Historical learner queued tuned behavior")
        self._persist()

    def _normalized_record(self, record: Any) -> dict[str, Any] | None:
        if not isinstance(record, dict):
            return None
        duration = self._safe_float(record.get("time_spent"))
        behavior = self._sanitize_behavior(record.get("behavior", {}))
        if duration is None or duration <= 0 or not behavior:
            return None
        return {"time_spent": duration, "behavior": behavior}

    @staticmethod
    def _average_duration(records: list[dict[str, Any]]) -> float:
        durations = [record["time_spent"] for record in records]
        return sum(durations) / len(durations)

    def _average_behavior_profile(self, records: list[dict[str, Any]]) -> dict[str, float]:
        record_count = float(len(records))
        return {
            key: sum(float(record["behavior"].get(key, 0.0)) for record in records) / record_count
            for key in self._behavior_bounds()
        }

    def _build_tuned_behavior(self, profile: dict[str, float]) -> dict[str, float]:
        current_behavior = self.bot.get_runtime_behavior_snapshot()
        return {
            key: self._clamp(self._ema(current_behavior[key], profile[key]), minimum, maximum)
            for key, (minimum, maximum) in self._behavior_bounds().items()
        }

    def _apply_profile(self, records: list[dict[str, Any]]) -> bool:
        valid = [
            normalized_record
            for record in records
            if (normalized_record := self._normalized_record(record)) is not None
        ]
        if not valid:
            return False

        average = self._average_duration(valid)
        best = min(valid, key=lambda item: item["time_spent"])
        if average <= 0:
            return False

        improvement_ratio = (average - best["time_spent"]) / average
        if improvement_ratio < self.min_improvement_ratio:
            return False

        ranked = sorted(valid, key=lambda item: item["time_spent"])
        profile = self._average_behavior_profile(ranked[: self.top_k])
        tuned = self._build_tuned_behavior(profile)
        with self._lock:
            self._tuned_behavior = dict(tuned)
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
                "records": self._records[-self.records_limit :],
                "total_completions": self._total_completions,
                "last_pair_processed": self._last_pair_processed,
                "last_batch_processed": self._last_batch_processed,
                "tuned_behavior": self._tuned_behavior,
            }
        self.persistence.save(state, force=force)


__all__ = [
    "AdaptiveTuner",
    "HistoricalLearner",
    "VisionOptimizer",
    "VisionPersistence",
]
