import json
import logging
import math
import os
import tempfile
import threading
import time
from collections import deque
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import config
from image_matcher import ImageMatcher
from mouse_controller import MouseController, precise_sleep, wait_event
from state_machine import State, StateMachine
from telegram_notifier import TelegramNotifier
from window_capture import ForbiddenAreaOverlay, WindowCapture, WindowCaptureError, WindowNotAvailableError

logger = logging.getLogger(__name__)

TemplatePair = tuple[Any, Any]
RedIcon = tuple[float, int, int]
BoxCandidate = tuple[float, int, int, int, int, str]
StateResult = State | None


class VisionPersistence:
    def __init__(self, path: str, save_interval: float) -> None:
        self.path = path
        self.save_interval = max(0.0, float(save_interval))
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

    def save(self, state: dict[str, Any], force: bool = False) -> bool:
        if not self.path:
            return False
        with self._lock:
            now = time.monotonic()
            if not force and self.save_interval > 0 and now - self._last_save_time < self.save_interval:
                return False
            directory = os.path.dirname(self.path)
            target_dir = directory or "."
            temp_path = None
            try:
                if directory:
                    os.makedirs(directory, exist_ok=True)
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target_dir, delete=False) as handle:
                    json.dump(state, handle, indent=2, sort_keys=True)
                    temp_path = handle.name
                os.replace(temp_path, self.path)
                self._last_save_time = now
                return True
            except (OSError, TypeError, ValueError) as exc:
                logger.error("Failed to persist state to %s: %s", self.path, exc)
                if temp_path:
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                return False


class AdaptiveTuner:
    def __init__(self) -> None:
        self.enabled = bool(config.ADAPTIVE_TUNER_ENABLED)
        self.alpha = float(config.ADAPTIVE_TUNER_ALPHA)
        self.click_success_rate = 1.0
        self.search_success_rate = 1.0
        self.click_delay = float(config.CLICK_DELAY)
        self.move_delay = float(config.MOUSE_MOVE_DELAY)
        self.search_interval = float(config.UPGRADE_SEARCH_INTERVAL)

    def _ema(self, current: float, value: float) -> float:
        return (1.0 - self.alpha) * current + self.alpha * value

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(float(minimum), min(float(maximum), float(value)))

    def record_click_result(self, success: bool) -> None:
        if not self.enabled:
            return
        self.click_success_rate = self._ema(self.click_success_rate, 1.0 if success else 0.0)
        if self.click_success_rate < config.ADAPTIVE_TUNER_CLICK_LOW_THRESHOLD:
            self.click_delay = self._clamp(
                self.click_delay + config.ADAPTIVE_TUNER_CLICK_DELAY_STEP,
                config.ADAPTIVE_TUNER_MIN_CLICK_DELAY,
                config.ADAPTIVE_TUNER_MAX_CLICK_DELAY,
            )
            self.move_delay = self._clamp(
                self.move_delay + config.ADAPTIVE_TUNER_MOVE_DELAY_STEP,
                config.ADAPTIVE_TUNER_MIN_MOVE_DELAY,
                config.ADAPTIVE_TUNER_MAX_MOVE_DELAY,
            )
        elif self.click_success_rate > config.ADAPTIVE_TUNER_CLICK_HIGH_THRESHOLD:
            self.click_delay = self._clamp(
                self.click_delay - config.ADAPTIVE_TUNER_CLICK_DECREMENT,
                config.ADAPTIVE_TUNER_MIN_CLICK_DELAY,
                config.ADAPTIVE_TUNER_MAX_CLICK_DELAY,
            )
            self.move_delay = self._clamp(
                self.move_delay - config.ADAPTIVE_TUNER_MOVE_DECREMENT,
                config.ADAPTIVE_TUNER_MIN_MOVE_DELAY,
                config.ADAPTIVE_TUNER_MAX_MOVE_DELAY,
            )

    def record_search_result(self, success: bool) -> None:
        if not self.enabled:
            return
        self.search_success_rate = self._ema(self.search_success_rate, 1.0 if success else 0.0)
        if self.search_success_rate < config.ADAPTIVE_TUNER_SEARCH_LOW_THRESHOLD:
            self.search_interval = self._clamp(
                self.search_interval + config.ADAPTIVE_TUNER_SEARCH_INTERVAL_STEP,
                config.ADAPTIVE_TUNER_MIN_SEARCH_INTERVAL,
                config.ADAPTIVE_TUNER_MAX_SEARCH_INTERVAL,
            )
        elif self.search_success_rate > config.ADAPTIVE_TUNER_SEARCH_HIGH_THRESHOLD:
            self.search_interval = self._clamp(
                self.search_interval - config.ADAPTIVE_TUNER_SEARCH_DECREMENT,
                config.ADAPTIVE_TUNER_MIN_SEARCH_INTERVAL,
                config.ADAPTIVE_TUNER_MAX_SEARCH_INTERVAL,
            )

    def reset(self) -> None:
        self.click_success_rate = 1.0
        self.search_success_rate = 1.0
        self.click_delay = float(config.CLICK_DELAY)
        self.move_delay = float(config.MOUSE_MOVE_DELAY)
        self.search_interval = float(config.UPGRADE_SEARCH_INTERVAL)


class VisionOptimizer:
    def __init__(self, persistence: VisionPersistence | None = None) -> None:
        self.enabled = bool(config.AI_VISION_ENABLED)
        self.persistence = persistence
        self.alpha = float(config.AI_VISION_ALPHA)
        self.alpha_max = float(config.AI_VISION_ALPHA_MAX)
        self.confidence_boost = float(config.AI_VISION_CONFIDENCE_BOOST)
        self.settings = self._settings()
        self.thresholds = {name: values["default"] for name, values in self.settings.items()}
        self.misses = {name: 0 for name in self.settings}

    @staticmethod
    def _settings() -> dict[str, dict[str, float | int]]:
        return {
            "red_icon": {
                "default": config.RED_ICON_THRESHOLD,
                "minimum": config.AI_RED_ICON_THRESHOLD_MIN,
                "maximum": config.AI_RED_ICON_THRESHOLD_MAX,
                "miss_window": config.AI_RED_ICON_MISS_WINDOW,
                "miss_step": config.AI_RED_ICON_MISS_STEP,
                "margin": config.AI_RED_ICON_MARGIN,
            },
            "new_level": {
                "default": config.NEW_LEVEL_THRESHOLD,
                "minimum": config.AI_NEW_LEVEL_THRESHOLD_MIN,
                "maximum": config.AI_NEW_LEVEL_THRESHOLD_MAX,
                "miss_window": config.AI_NEW_LEVEL_MISS_WINDOW,
                "miss_step": config.AI_NEW_LEVEL_MISS_STEP,
                "margin": 0.0,
            },
            "new_level_red_icon": {
                "default": config.NEW_LEVEL_RED_ICON_THRESHOLD,
                "minimum": config.AI_NEW_LEVEL_RED_ICON_THRESHOLD_MIN,
                "maximum": config.AI_NEW_LEVEL_RED_ICON_THRESHOLD_MAX,
                "miss_window": config.AI_NEW_LEVEL_RED_ICON_MISS_WINDOW,
                "miss_step": config.AI_NEW_LEVEL_RED_ICON_MISS_STEP,
                "margin": 0.0,
            },
            "upgrade_station": {
                "default": config.UPGRADE_STATION_THRESHOLD,
                "minimum": config.AI_UPGRADE_STATION_THRESHOLD_MIN,
                "maximum": config.AI_UPGRADE_STATION_THRESHOLD_MAX,
                "miss_window": config.AI_UPGRADE_STATION_MISS_WINDOW,
                "miss_step": config.AI_UPGRADE_STATION_MISS_STEP,
                "margin": 0.0,
            },
            "stats_upgrade": {
                "default": config.STATS_RED_ICON_THRESHOLD,
                "minimum": config.AI_STATS_UPGRADE_THRESHOLD_MIN,
                "maximum": config.AI_STATS_UPGRADE_THRESHOLD_MAX,
                "miss_window": config.AI_STATS_UPGRADE_MISS_WINDOW,
                "miss_step": config.AI_STATS_UPGRADE_MISS_STEP,
                "margin": 0.0,
            },
            "box": {
                "default": config.BOX_THRESHOLD,
                "minimum": config.AI_BOX_THRESHOLD_MIN,
                "maximum": config.AI_BOX_THRESHOLD_MAX,
                "miss_window": config.AI_BOX_MISS_WINDOW,
                "miss_step": config.AI_BOX_MISS_STEP,
                "margin": 0.0,
            },
        }

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def _ema(self, current: float, value: float, alpha: float) -> float:
        return (1.0 - alpha) * current + alpha * value

    def _adaptive_alpha(self, confidence: float) -> float:
        boost = max(0.0, min(1.0, confidence - config.AI_VISION_CONFIDENCE_THRESHOLD))
        return min(self.alpha + boost * self.confidence_boost, self.alpha_max)

    def threshold(self, name: str) -> float:
        values = self.settings[name]
        return float(self.thresholds[name] if self.enabled else values["default"])

    def record_confidence(self, name: str, confidence: Any) -> None:
        confidence = self._number(confidence)
        if not self.enabled or confidence is None or confidence <= 0 or name not in self.settings:
            return
        values = self.settings[name]
        self.misses[name] = 0
        target = confidence - float(values["margin"])
        target = AdaptiveTuner._clamp(target, float(values["minimum"]), float(values["maximum"]))
        self.thresholds[name] = self._ema(self.thresholds[name], target, self._adaptive_alpha(confidence))
        self._persist()

    def record_miss(self, name: str) -> None:
        if not self.enabled or name not in self.settings:
            return
        values = self.settings[name]
        self.misses[name] += 1
        if self.misses[name] < int(values["miss_window"]):
            return
        self.misses[name] = 0
        target = max(float(values["minimum"]), self.thresholds[name] - float(values["miss_step"]))
        self.thresholds[name] = self._ema(self.thresholds[name], target, self.alpha_max)
        self._persist()

    def apply_persisted_state(self, state: dict[str, Any]) -> None:
        for name, values in self.settings.items():
            value = self._number(state.get(f"{name}_threshold"))
            if value is None:
                continue
            self.thresholds[name] = AdaptiveTuner._clamp(value, float(values["minimum"]), float(values["maximum"]))

    def reset(self) -> None:
        self.thresholds = {name: values["default"] for name, values in self.settings.items()}
        self.misses = {name: 0 for name in self.settings}
        self._persist(force=True)

    def _persist(self, force: bool = False) -> None:
        if self.persistence is not None:
            state = {f"{name}_threshold": float(value) for name, value in self.thresholds.items()}
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
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._records: list[dict[str, Any]] = []
        self._total_completions = 0
        self._last_pair_processed = 0
        self._last_batch_processed = 0
        self._last_apply_time = 0.0
        self._tuned_behavior: dict[str, float] = {}
        self._load()

    @staticmethod
    def _float(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @classmethod
    def _sanitize_behavior(cls: type["HistoricalLearner"], behavior: Any) -> dict[str, float]:
        if not isinstance(behavior, dict):
            return {}
        bounds = {
            "click_delay": (config.AI_LEARNING_MIN_CLICK_DELAY, config.AI_LEARNING_MAX_CLICK_DELAY),
            "move_delay": (config.AI_LEARNING_MIN_MOVE_DELAY, config.AI_LEARNING_MAX_MOVE_DELAY),
            "search_interval": (config.AI_LEARNING_MIN_SEARCH_INTERVAL, config.AI_LEARNING_MAX_SEARCH_INTERVAL),
        }
        sanitized = {}
        for key, (minimum, maximum) in bounds.items():
            value = cls._float(behavior.get(key))
            if value is not None:
                sanitized[key] = AdaptiveTuner._clamp(value, minimum, maximum)
        return sanitized

    def _load(self) -> None:
        if not self.enabled or self.persistence is None:
            return
        state = self.persistence.load()
        records = state.get("records", []) if isinstance(state, dict) else []
        if isinstance(records, list):
            self._records = [record for record in records if isinstance(record, dict)][-config.AI_LEARNING_RECORDS_LIMIT :]
        self._total_completions = max(0, int(state.get("total_completions", len(self._records)) or 0))
        self._last_pair_processed = max(0, int(state.get("last_pair_processed", 0) or 0))
        self._last_batch_processed = max(0, int(state.get("last_batch_processed", 0) or 0))
        self._tuned_behavior = self._sanitize_behavior(state.get("tuned_behavior", {}))
        if self._tuned_behavior:
            self.bot.apply_learned_behavior(self._tuned_behavior)

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
            self._last_apply_time = 0.0
            self._tuned_behavior = {}
        self._persist(force=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._apply_pending_profiles()
            except Exception:
                logger.exception("Historical learner cycle failed")
            self._stop.wait(self.interval)

    def _apply_pending_profiles(self) -> None:
        with self._lock:
            records = list(self._records)
            total = self._total_completions
        if time.monotonic() - self._last_apply_time < self.apply_cooldown:
            return
        changed = False
        for window, attr in ((self.pair_window, "_last_pair_processed"), (self.batch_window, "_last_batch_processed")):
            marker = total // window
            if total >= window and marker > getattr(self, attr):
                changed = self._apply_profile(records[-window:]) or changed
                setattr(self, attr, marker)
        if changed:
            self._last_apply_time = time.monotonic()
        self._persist()

    def _apply_profile(self, records: list[dict[str, Any]]) -> bool:
        valid = []
        for record in records:
            duration = self._float(record.get("time_spent"))
            behavior = self._sanitize_behavior(record.get("behavior", {}))
            if duration and duration > 0 and behavior:
                valid.append({"time_spent": duration, "behavior": behavior})
        if not valid:
            return False
        average = sum(record["time_spent"] for record in valid) / len(valid)
        ranked = sorted(valid, key=lambda item: item["time_spent"])
        if average <= 0 or (average - ranked[0]["time_spent"]) / average < self.min_improvement_ratio:
            return False
        profile = {key: 0.0 for key in ("click_delay", "move_delay", "search_interval")}
        for record in ranked[: self.top_k]:
            for key in profile:
                profile[key] += float(record["behavior"].get(key, 0.0))
        for key in profile:
            profile[key] /= float(min(self.top_k, len(ranked)))
        current = self.bot.get_runtime_behavior_snapshot()
        tuned = {key: (1.0 - self.ema_alpha) * current[key] + self.ema_alpha * profile[key] for key in profile}
        self._tuned_behavior = self._sanitize_behavior(tuned)
        self.bot.apply_learned_behavior(self._tuned_behavior)
        return True

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
        self.mouse_controller = MouseController(self.window_capture.get_window_rect, config.CLICK_DELAY, config.MOUSE_MOVE_DELAY)
        self.state_machine = StateMachine(State.FIND_RED_ICONS)
        self.telegram = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID, config.TELEGRAM_ENABLED)
        self.tuner = AdaptiveTuner()
        self.vision_persistence = VisionPersistence(config.AI_VISION_STATE_FILE, config.AI_VISION_SAVE_INTERVAL)
        self.vision_optimizer = VisionOptimizer(self.vision_persistence)
        self.vision_optimizer.apply_persisted_state(self.vision_persistence.load())
        self.learning_persistence = VisionPersistence(config.AI_LEARNING_STATE_FILE, config.AI_LEARNING_SAVE_INTERVAL)
        self.historical_learner = HistoricalLearner(self, self.learning_persistence)
        self.templates = self.load_templates()
        self.register_states()
        self.ready = self._validate_required_templates()
        self.running = False
        self.red_icons: list[RedIcon] = []
        self.current_red_icon_index = 0
        self.wait_for_unlock_attempts = 0
        self.max_wait_for_unlock_attempts = 4
        self.work_done = False
        self.cycle_counter = 0
        self.upgrade_station_counter = 0
        self.consecutive_failed_cycles = 0
        self.upgrade_found_in_cycle = False
        self.upgrade_station_pos: tuple[int, int] | None = None
        self.total_levels_completed = 0
        self.current_level_start_time: datetime | None = None
        self.successful_red_icon_positions = deque(maxlen=24)
        self._oscillation_cycle_index = 1
        self._oscillation_leg_direction = 1
        self._oscillation_leg_progress = 0
        self._new_level_red_icon_verified = False
        self.forbidden_zones = [
            (config.FORBIDDEN_ZONE_1_X_MIN, config.FORBIDDEN_ZONE_1_X_MAX, config.FORBIDDEN_ZONE_1_Y_MIN, config.FORBIDDEN_ZONE_1_Y_MAX),
            (config.FORBIDDEN_ZONE_2_X_MIN, config.FORBIDDEN_ZONE_2_X_MAX, config.FORBIDDEN_ZONE_2_Y_MIN, config.FORBIDDEN_ZONE_2_Y_MAX),
            (config.FORBIDDEN_ZONE_3_X_MIN, config.FORBIDDEN_ZONE_3_X_MAX, config.FORBIDDEN_ZONE_3_Y_MIN, config.FORBIDDEN_ZONE_3_Y_MAX),
            (config.FORBIDDEN_ZONE_4_X_MIN, config.FORBIDDEN_ZONE_4_X_MAX, config.FORBIDDEN_ZONE_4_Y_MIN, config.FORBIDDEN_ZONE_4_Y_MAX),
            (config.FORBIDDEN_ZONE_5_X_MIN, config.FORBIDDEN_ZONE_5_X_MAX, config.FORBIDDEN_ZONE_5_Y_MIN, config.FORBIDDEN_ZONE_5_Y_MAX),
        ]
        self.overlay: ForbiddenAreaOverlay | None = None
        logger.info("Bot initialized successfully")

    def load_templates(self) -> dict[str, TemplatePair]:
        templates = {}
        assets_path = Path(config.ASSETS_DIR)
        if not assets_path.exists():
            logger.error("Assets directory not found: %s", assets_path)
            return templates
        for template_file in sorted(assets_path.glob("*.png")):
            try:
                templates[template_file.stem] = self.image_matcher.load_template(template_file)
                logger.info("Loaded template: %s", template_file.stem)
            except Exception as exc:
                logger.error("Failed to load template %s: %s", template_file, exc)
        return templates

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

    def _validate_required_templates(self) -> bool:
        missing = [name for name in ("newLevel", "unlock", "upgradeStation") if name not in self.templates]
        if not any(name.startswith("RedIcon") for name in self.templates):
            missing.append("RedIcon*")
        if missing:
            logger.error("Missing required templates: %s", ", ".join(missing))
            return False
        return True

    def _sleep(self, duration: Any) -> bool:
        return wait_event(self._stop_requested, duration)

    def _apply_tuning(self) -> None:
        self.mouse_controller.click_delay = float(self.tuner.click_delay)
        self.mouse_controller.move_delay = float(self.tuner.move_delay)

    def _template(self, name: str) -> TemplatePair | None:
        return self.templates.get(name)

    def _click_idle(self) -> bool:
        return self.mouse_controller.click(*config.IDLE_CLICK_POS, relative=True)

    @staticmethod
    def _supervision_enabled(flag_name: str) -> bool:
        flags = {
            "SUPERVISION_BOX_NMS_ENABLED": config.SUPERVISION_BOX_NMS_ENABLED,
            "SUPERVISION_RED_ICON_NMS_ENABLED": config.SUPERVISION_RED_ICON_NMS_ENABLED,
            "SUPERVISION_UPGRADE_STATION_NMS_ENABLED": config.SUPERVISION_UPGRADE_STATION_NMS_ENABLED,
        }
        return bool(config.SUPERVISION_ENABLED and flags.get(flag_name, False))

    def _red_icon_template_names(self) -> list[str]:
        names = [name for name in self.templates if name.startswith("RedIcon")]
        if config.RED_ICON_FAST_MODE_ENABLED:
            fast_names = [name for name in config.RED_ICON_FAST_TEMPLATE_NAMES if name in self.templates]
            if fast_names:
                return list(fast_names)
        return sorted(names, key=lambda name: (name != "RedIcon", name == "RedIconNoBG", name))

    def _red_icon_min_matches(self) -> int:
        if config.RED_ICON_FAST_MODE_ENABLED:
            return 1
        available = max(1, len([name for name in self.templates if name.startswith("RedIcon")]))
        return min(max(1, int(config.RED_ICON_MIN_MATCHES)), available)

    def _template_span(self, names: Iterable[str]) -> tuple[int, int]:
        width = 0
        height = 0
        for name in names:
            template_pair = self._template(name)
            if template_pair is None:
                continue
            template, _ = template_pair
            height = max(height, int(template.shape[0]))
            width = max(width, int(template.shape[1]))
        return width, height

    @staticmethod
    def _region(screenshot: Any, x_min: int, x_max: int, y_min: int, y_max: int, pad: tuple[int, int]) -> tuple[Any, int, int]:
        height, width = screenshot.shape[:2]
        left = max(0, int(x_min) - pad[0])
        right = min(width, int(x_max) + pad[0])
        top = max(0, int(y_min) - pad[1])
        bottom = min(height, int(y_max) + pad[1])
        if left >= right or top >= bottom:
            return screenshot[0:0, 0:0], 0, 0
        return screenshot[top:bottom, left:right], left, top

    def _collect_red_icons(self, screenshot: Any, threshold: float, offset_x: int = 0, offset_y: int = 0) -> dict[tuple[int, int], list[tuple[str, float]]]:
        detections: dict[tuple[int, int], list[tuple[str, float]]] = {}
        min_distance = config.RED_ICON_FAST_MIN_DISTANCE if config.RED_ICON_FAST_MODE_ENABLED else 80
        for name in self._red_icon_template_names():
            template_pair = self._template(name)
            if template_pair is None or screenshot.size == 0:
                continue
            template, mask = template_pair
            matches = self.image_matcher.find_all_templates(
                screenshot,
                template,
                mask=mask,
                threshold=threshold,
                min_distance=min_distance,
                template_name=name,
                use_supervision_nms=self._supervision_enabled("SUPERVISION_RED_ICON_NMS_ENABLED"),
                supervision_iou_threshold=config.SUPERVISION_RED_ICON_NMS_IOU_THRESHOLD,
                supervision_class_agnostic=config.SUPERVISION_CLASS_AGNOSTIC_NMS,
            )
            template_h, template_w = template.shape[:2]
            for confidence, x, y in matches:
                top_left = (int(x) - template_w // 2, int(y) - template_h // 2)
                if config.RED_ICON_HSV_COLOR_GATE_ENABLED and not self.image_matcher._check_hsv_gate(
                    screenshot,
                    template,
                    top_left,
                    mask,
                    config.RED_ICON_HSV_RANGES,
                    config.RED_ICON_HSV_MIN_MATCH_RATIO,
                ):
                    continue
                self._merge_icon(detections, int(x) + offset_x, int(y) + offset_y, name, float(confidence))
        return detections

    @staticmethod
    def _merge_icon(detections: dict[tuple[int, int], list[tuple[str, float]]], x: int, y: int, name: str, confidence: float) -> None:
        for existing_x, existing_y in list(detections):
            if abs(x - existing_x) < 10 and abs(y - existing_y) < 10:
                detections[(existing_x, existing_y)].append((name, confidence))
                return
        detections[(x, y)] = [(name, confidence)]

    def _icons_from_detections(self, detections: dict[tuple[int, int], list[tuple[str, float]]]) -> tuple[list[RedIcon], list[float]]:
        icons = []
        confidences = []
        min_matches = self._red_icon_min_matches()
        for (x, y), matches in detections.items():
            best_by_template: dict[str, float] = {}
            for name, confidence in matches:
                best_by_template[name] = max(confidence, best_by_template.get(name, 0.0))
            if len(best_by_template) >= min_matches:
                confidence = max(best_by_template.values())
                icons.append((confidence, x, y))
                confidences.append(confidence)
        return icons, confidences

    def _find_new_level_button(self, screenshot: Any) -> tuple[bool, float, int, int]:
        template_pair = self._template("newLevel")
        if template_pair is None:
            return False, 0.0, 0, 0
        template, mask = template_pair
        return self.image_matcher.find_template(
            screenshot,
            template,
            mask=mask,
            threshold=self.vision_optimizer.threshold("new_level"),
            template_name="newLevel",
        )

    def _find_zone_red_icon(self, screenshot: Any, zone: tuple[int, int, int, int], threshold: float, kind: str) -> RedIcon | None:
        names = self._red_icon_template_names()
        region, x_offset, y_offset = self._region(screenshot, zone[0], zone[1], zone[2], zone[3], self._template_span(names))
        detections = self._collect_red_icons(region, threshold, x_offset, y_offset)
        icons, _ = self._icons_from_detections(detections)
        best = None
        for confidence, x, y in icons:
            if zone[0] <= x <= zone[1] and zone[2] <= y <= zone[3] and (best is None or confidence > best[0]):
                best = (confidence, x, y)
        if best is None:
            self.vision_optimizer.record_miss(kind)
        else:
            self.vision_optimizer.record_confidence(kind, best[0])
        return best

    def _find_new_level_red_icon(self, screenshot: Any | None = None) -> RedIcon | None:
        if screenshot is None:
            screenshot = self.window_capture.capture(max_y=config.EXTENDED_SEARCH_Y)
        zone = (
            config.NEW_LEVEL_RED_ICON_X_MIN,
            config.NEW_LEVEL_RED_ICON_X_MAX,
            config.NEW_LEVEL_RED_ICON_Y_MIN,
            config.NEW_LEVEL_RED_ICON_Y_MAX,
        )
        threshold = self.vision_optimizer.threshold("new_level_red_icon")
        return self._find_zone_red_icon(screenshot, zone, threshold, "new_level_red_icon")

    def _scan_red_icons(self, screenshot: Any) -> tuple[list[RedIcon], list[float], RedIcon | None]:
        limited = screenshot[: config.MAX_SEARCH_Y, :]
        threshold = min(self.vision_optimizer.threshold("red_icon"), self.vision_optimizer.threshold("new_level_red_icon"))
        icons, confidences = self._icons_from_detections(self._collect_red_icons(limited, threshold))
        new_level_icon = self._find_new_level_red_icon(screenshot)
        if confidences:
            self.vision_optimizer.record_confidence("red_icon", sum(confidences) / len(confidences))
        else:
            self.vision_optimizer.record_miss("red_icon")
        return icons, confidences, new_level_icon

    def _scrcpy_recovery(self, delay: Any) -> bool:
        if not config.SCRCPY_MISS_RECOVERY_ENABLED:
            return False
        try:
            delay = max(0.0, float(delay))
        except (TypeError, ValueError):
            delay = 0.0
        return True if delay == 0 else self._sleep(delay)

    def _clickable_icons(self, icons: Iterable[RedIcon]) -> list[RedIcon]:
        return [
            icon
            for icon in icons
            if not self.mouse_controller.is_in_forbidden_zone(
                icon[1] + config.RED_ICON_OFFSET_X,
                icon[2] + config.RED_ICON_OFFSET_Y,
                relative=True,
            )
        ]

    def _red_icon_priority(self, icon: RedIcon) -> tuple[int, int, float]:
        confidence, _, y = icon
        for success_y in self.successful_red_icon_positions:
            if abs(y - success_y) < 50:
                return 0, y, -confidence
        return 1, y, -confidence

    def _remember_successful_y(self, y: int) -> None:
        if all(abs(existing_y - y) >= 12 for existing_y in self.successful_red_icon_positions):
            self.successful_red_icon_positions.append(int(y))

    def _upgrade_station_match(self, threshold: float) -> RedIcon | None:
        template_pair = self._template("upgradeStation")
        if template_pair is None:
            return None
        screenshot = self.window_capture.capture(max_y=config.UPGRADE_STATION_SEARCH_Y)
        template, mask = template_pair
        candidates = self.image_matcher.find_template_candidates(
            screenshot,
            template,
            mask=mask,
            threshold=threshold,
            min_distance=15,
            template_name="upgradeStation",
        )
        if self._supervision_enabled("SUPERVISION_UPGRADE_STATION_NMS_ENABLED"):
            filtered = self.image_matcher.filter_candidates_with_supervision_nms(
                candidates,
                iou_threshold=config.SUPERVISION_UPGRADE_STATION_NMS_IOU_THRESHOLD,
                class_agnostic=config.SUPERVISION_CLASS_AGNOSTIC_NMS,
            )
            candidates = filtered if filtered is not None else candidates
        for confidence, x, y, width, height in candidates:
            top_left = (int(x) - int(width) // 2, int(y) - int(height) // 2)
            if config.UPGRADE_STATION_HSV_COLOR_GATE_ENABLED:
                if not self.image_matcher._check_hsv_gate(screenshot, template, top_left, mask, config.UPGRADE_STATION_HSV_RANGES, config.UPGRADE_STATION_HSV_MIN_MATCH_RATIO):
                    continue
            if not self.mouse_controller.is_in_forbidden_zone(x, y, relative=True):
                return float(confidence), int(x), int(y)
        return None

    def _reset_search_cycle(self) -> None:
        self.cycle_counter = 0
        self.wait_for_unlock_attempts = 0
        self._oscillation_cycle_index = 1
        self._oscillation_leg_direction = 1
        self._oscillation_leg_progress = 0
        self._new_level_red_icon_verified = False

    def _advance_scroll(self) -> None:
        target_steps = max(1, int(self._oscillation_cycle_index) * int(config.SCROLL_INCREMENT_STEP))
        self._oscillation_leg_progress += 1
        if self._oscillation_leg_progress < target_steps:
            return
        self._oscillation_leg_progress = 0
        if self._oscillation_leg_direction > 0:
            self._oscillation_leg_direction = -1
            return
        self._oscillation_leg_direction = 1
        self._oscillation_cycle_index = 1 if self._oscillation_cycle_index >= int(config.MAX_SCROLL_CYCLES) else self._oscillation_cycle_index + 1

    def _scroll(self, verify_down: bool = False) -> bool:
        distance = int(round(float(config.SCROLL_PIXEL_STEP) * float(config.SCROLL_DISTANCE_RATIO)))
        start_x, start_y = config.SCROLL_START_POS
        direction = 1 if verify_down or self._oscillation_leg_direction > 0 else -1
        moved = self.mouse_controller.drag(start_x, start_y, start_x, start_y - (distance * direction), duration=config.SCROLL_DURATION, relative=True)
        if moved:
            self._sleep(config.POST_SCROLL_SETTLE)
            self._sleep(config.SCROLL_INTERVAL_PAUSE)
            if not verify_down:
                self._advance_scroll()
            self._click_idle()
        return bool(moved)

    def _record_level_completion(self, source: str) -> float:
        self.total_levels_completed += 1
        elapsed = 0.0
        if self.current_level_start_time is not None:
            elapsed = (datetime.now() - self.current_level_start_time).total_seconds()
        self.current_level_start_time = datetime.now()
        self._reset_search_cycle()
        self.telegram.notify_new_level(self.total_levels_completed, elapsed)
        self.historical_learner.record_completion(elapsed, source)
        return elapsed

    def _box_candidates(self, screenshot: Any) -> list[BoxCandidate]:
        candidates = []
        for name in ("box1", "box2", "box3", "box4", "box5"):
            template_pair = self._template(name)
            if template_pair is None:
                continue
            template, mask = template_pair
            for confidence, x, y, width, height in self.image_matcher.find_template_candidates(screenshot, template, mask=mask, threshold=self.vision_optimizer.threshold("box"), min_distance=12, template_name=name):
                top_left = (int(x) - int(width) // 2, int(y) - int(height) // 2)
                if config.BOX_HSV_COLOR_GATE_ENABLED:
                    if not self.image_matcher._check_hsv_gate(screenshot, template, top_left, mask, config.BOX_HSV_RANGES, config.BOX_HSV_MIN_MATCH_RATIO):
                        continue
                candidates.append((float(confidence), int(x), int(y), int(width), int(height), name))
        if self._supervision_enabled("SUPERVISION_BOX_NMS_ENABLED"):
            filtered = self.image_matcher.filter_candidates_with_supervision_nms(candidates, iou_threshold=config.SUPERVISION_BOX_NMS_IOU_THRESHOLD, class_agnostic=config.SUPERVISION_CLASS_AGNOSTIC_NMS)
            return list(filtered) if filtered is not None else candidates
        return candidates

    def request_stop(self) -> None:
        self._stop_requested.set()

    def start(self) -> bool:
        if self.running:
            return True
        if self._step_active.is_set() or not self.ready:
            logger.warning("Cannot start bot while not ready or while a state step is active")
            return False
        try:
            self.window_capture.ensure_window(resize=True)
        except WindowCaptureError as exc:
            logger.error("Cannot start bot: %s", exc)
            return False
        self._stop_requested.clear()
        self.running = True
        self.current_level_start_time = self.current_level_start_time or datetime.now()
        self.historical_learner.start()
        if config.SHOW_FORBIDDEN_AREA and self.overlay is None:
            self.overlay = ForbiddenAreaOverlay(self.window_capture, self.forbidden_zones)
            self.overlay.start()
        return True

    def stop(self) -> None:
        self._stop_requested.set()
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
            if self._stop_requested.is_set() or not self.window_capture.is_window_active():
                self.stop()
                return False
            self._apply_tuning()
            if not self.state_machine.update():
                logger.error("State machine update failed in state %s", self.state_machine.get_state_name())
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
                self.step()
                precise_sleep(0.1)
        finally:
            self.stop()

    def get_runtime_behavior_snapshot(self) -> dict[str, float]:
        return {
            "click_delay": float(self.tuner.click_delay),
            "move_delay": float(self.tuner.move_delay),
            "search_interval": float(self.tuner.search_interval),
        }

    def apply_learned_behavior(self, learned: dict[str, Any]) -> None:
        behavior = HistoricalLearner._sanitize_behavior(learned)
        if not behavior:
            return
        self.tuner.click_delay = float(behavior.get("click_delay", self.tuner.click_delay))
        self.tuner.move_delay = float(behavior.get("move_delay", self.tuner.move_delay))
        self.tuner.search_interval = float(behavior.get("search_interval", self.tuner.search_interval))
        self._apply_tuning()

    def wipe_memory(self) -> None:
        self.tuner.reset()
        self.vision_optimizer.reset()
        self.historical_learner.reset()
        self.successful_red_icon_positions.clear()
        self.current_level_start_time = datetime.now() if self.running else None
        self._apply_tuning()

    def handle_find_red_icons(self, current_state: State) -> StateResult:
        self._click_idle()
        self.work_done = False
        screenshot = self.window_capture.capture(max_y=config.EXTENDED_SEARCH_Y)
        found, confidence, x, y = self._find_new_level_button(screenshot[: config.MAX_SEARCH_Y, :])
        if found:
            self.vision_optimizer.record_confidence("new_level", confidence)
            logger.info("newLevel.png found at (%s, %s)", x, y)
            return State.TRANSITION_LEVEL
        self.vision_optimizer.record_miss("new_level")
        self.red_icons, confidences, new_level_icon = self._scan_red_icons(screenshot)
        if not self.red_icons and new_level_icon is None and self._scrcpy_recovery(config.SCRCPY_RED_ICON_MISS_RECOVERY_DELAY):
            screenshot = self.window_capture.capture(max_y=config.EXTENDED_SEARCH_Y)
            self.red_icons, confidences, new_level_icon = self._scan_red_icons(screenshot)
        if new_level_icon is not None:
            self._new_level_red_icon_verified = False
            logger.info("New level red icon detected at (%s, %s) [%.3f]", new_level_icon[1], new_level_icon[2], new_level_icon[0])
            return State.CHECK_NEW_LEVEL
        filtered = self._clickable_icons(self.red_icons)
        if not filtered:
            return State.OPEN_BOXES
        self.red_icons = sorted(filtered, key=self._red_icon_priority)
        self.current_red_icon_index = 0
        self.cycle_counter = 0
        self.work_done = True
        logger.info("%s red icons ready to process", len(self.red_icons))
        return State.CLICK_RED_ICON

    def handle_click_red_icon(self, current_state: State) -> StateResult:
        if self.current_red_icon_index >= len(self.red_icons):
            return State.OPEN_BOXES
        confidence, x, y = self.red_icons[self.current_red_icon_index]
        click_x = x + config.RED_ICON_OFFSET_X
        click_y = y + config.RED_ICON_OFFSET_Y
        clicked = self.mouse_controller.click(click_x, click_y, relative=True)
        self.tuner.record_click_result(clicked)
        self._apply_tuning()
        if not clicked:
            self.current_red_icon_index += 1
            return State.CLICK_RED_ICON if self.current_red_icon_index < len(self.red_icons) else State.OPEN_BOXES
        logger.info("Clicked red icon %s/%s at (%s, %s) [%.3f]", self.current_red_icon_index + 1, len(self.red_icons), click_x, click_y, confidence)
        return State.CHECK_UNLOCK

    def handle_check_unlock(self, current_state: State) -> StateResult:
        template_pair = self._template("unlock")
        if template_pair is None:
            return State.SEARCH_UPGRADE_STATION
        template, mask = template_pair
        found, confidence, x, y = self.image_matcher.find_template(self.window_capture.capture(max_y=config.MAX_SEARCH_Y), template, mask=mask, threshold=config.UNLOCK_THRESHOLD, template_name="unlock")
        if found and not self.mouse_controller.is_in_forbidden_zone(x, y, relative=True):
            logger.info("Unlock found at (%s, %s) [%.3f]", x, y, confidence)
            if not self.mouse_controller.click(x, y, relative=True):
                return State.CHECK_UNLOCK
        return State.SEARCH_UPGRADE_STATION

    def handle_search_upgrade_station(self, current_state: State) -> StateResult:
        base_threshold = self.vision_optimizer.threshold("upgrade_station")
        relaxed_threshold = max(0.0, base_threshold - 0.05)
        for attempt in range(5):
            match = self._upgrade_station_match(base_threshold if attempt < 2 else relaxed_threshold)
            if match is not None:
                confidence, x, y = match
                self.upgrade_station_pos = (x, y)
                self.upgrade_found_in_cycle = True
                self.consecutive_failed_cycles = 0
                self.cycle_counter = 0
                self.vision_optimizer.record_confidence("upgrade_station", confidence)
                self.tuner.record_search_result(True)
                self._apply_tuning()
                logger.info("Upgrade station found at (%s, %s) on attempt %s", x, y, attempt + 1)
                return State.HOLD_UPGRADE_STATION
            if attempt < 4 and not self._sleep(self.tuner.search_interval):
                return State.OPEN_BOXES
        self.vision_optimizer.record_miss("upgrade_station")
        self.tuner.record_search_result(False)
        self._apply_tuning()
        self.consecutive_failed_cycles += 1
        return State.OPEN_BOXES

    def handle_hold_upgrade_station(self, current_state: State) -> StateResult:
        if self.upgrade_station_pos is None:
            return State.OPEN_BOXES
        x, y = self.upgrade_station_pos
        clicked = self.mouse_controller.precise_click(x, y, relative=True)
        self.tuner.record_click_result(clicked)
        self._apply_tuning()
        if not clicked or not self._sleep(config.UPGRADE_STATION_VERIFY_SETTLE_DELAY):
            return State.OPEN_BOXES
        base_threshold = self.vision_optimizer.threshold("upgrade_station")
        relaxed_threshold = max(0.0, base_threshold - 0.05)
        verified = None
        for attempt in range(max(1, int(config.UPGRADE_STATION_VERIFY_SEARCH_ATTEMPTS))):
            verified = self._upgrade_station_match(base_threshold if attempt == 0 else relaxed_threshold)
            if verified is not None:
                break
            if attempt == 0:
                self._sleep(config.UPGRADE_STATION_VERIFY_SEARCH_INTERVAL)
        if verified is None and self._scrcpy_recovery(config.SCRCPY_UPGRADE_MISS_RECOVERY_DELAY):
            verified = self._upgrade_station_match(relaxed_threshold)
        if verified is None:
            self.vision_optimizer.record_miss("upgrade_station")
            return State.OPEN_BOXES
        confidence, x, y = verified
        self.vision_optimizer.record_confidence("upgrade_station", confidence)
        if self.current_red_icon_index < len(self.red_icons):
            self._remember_successful_y(self.red_icons[self.current_red_icon_index][2])
        if not self.mouse_controller.hold_at(x, y, duration=max(0.0, float(config.CLICK_HOLD_MAX_DURATION)), relative=True, interrupt_check=self._stop_requested.is_set):
            return State.OPEN_BOXES
        self.upgrade_station_pos = None
        self._click_idle()
        self._sleep(config.STATE_DELAY)
        self.upgrade_station_counter += 1
        if self.upgrade_station_counter >= 2:
            self.upgrade_station_counter = 0
            return State.UPGRADE_STATS
        return State.OPEN_BOXES

    def handle_upgrade_stats(self, current_state: State) -> StateResult:
        self._click_idle()
        screenshot = self.window_capture.capture(max_y=config.EXTENDED_SEARCH_Y)
        found, confidence, _, _ = self._find_new_level_button(screenshot[: config.MAX_SEARCH_Y, :])
        if found:
            self.vision_optimizer.record_confidence("new_level", confidence)
            return State.TRANSITION_LEVEL
        zone = (config.UPGRADE_RED_ICON_X_MIN, config.UPGRADE_RED_ICON_X_MAX, config.UPGRADE_RED_ICON_Y_MIN, config.UPGRADE_RED_ICON_Y_MAX)
        stats_icon = self._find_zone_red_icon(screenshot, zone, self.vision_optimizer.threshold("stats_upgrade"), "stats_upgrade")
        if stats_icon is None:
            logger.info("No stats icon detected")
            return State.SCROLL
        if not self.mouse_controller.click(*config.STATS_UPGRADE_BUTTON_POS, relative=True):
            return State.OPEN_BOXES
        self._sleep(config.STATE_DELAY)
        if not self.mouse_controller.click_stats_upgrade_at(*config.STATS_UPGRADE_POS, duration=config.STATS_UPGRADE_CLICK_DURATION, click_delay=config.STATS_UPGRADE_CLICK_DELAY, relative=True, interrupt_check=self._stop_requested.is_set):
            return State.OPEN_BOXES
        self._click_idle()
        logger.info("Stats upgrade completed")
        return State.OPEN_BOXES

    def handle_open_boxes(self, current_state: State) -> StateResult:
        self._click_idle()
        screenshot = self.window_capture.capture(max_y=config.BOX_SEARCH_Y)
        found, confidence, _, _ = self._find_new_level_button(screenshot[: config.MAX_SEARCH_Y, :])
        if found:
            self.vision_optimizer.record_confidence("new_level", confidence)
            return State.TRANSITION_LEVEL
        candidates = self._box_candidates(screenshot)
        if not candidates and self._scrcpy_recovery(config.SCRCPY_BOX_MISS_RECOVERY_DELAY):
            screenshot = self.window_capture.capture(max_y=config.BOX_SEARCH_Y)
            candidates = self._box_candidates(screenshot)
        boxes_found = 0
        best_confidence = 0.0
        for confidence, x, y, _, _, _ in candidates:
            if not self.mouse_controller.is_in_forbidden_zone(x, y, relative=True) and self.mouse_controller.click(x, y, relative=True):
                boxes_found += 1
                best_confidence = max(best_confidence, confidence)
        if boxes_found:
            self.work_done = True
            self.cycle_counter = 0
            self.vision_optimizer.record_confidence("box", best_confidence)
            logger.info("Opened %s boxes", boxes_found)
        else:
            self.vision_optimizer.record_miss("box")
        return self._next_state_after_boxes()

    def _next_state_after_boxes(self) -> State:
        if self.consecutive_failed_cycles >= 3:
            self.consecutive_failed_cycles = 0
            self.cycle_counter = 0
            return State.SCROLL
        if self.upgrade_found_in_cycle or self.work_done:
            self.upgrade_found_in_cycle = False
            self.cycle_counter = 0
            return State.FIND_RED_ICONS
        self.cycle_counter += 1
        if self.cycle_counter >= 2:
            self.cycle_counter = 0
            return State.SCROLL
        return State.FIND_RED_ICONS

    def handle_scroll(self, current_state: State) -> StateResult:
        self._click_idle()
        self._scroll()
        self.cycle_counter = 0
        return State.FIND_RED_ICONS

    def handle_check_new_level(self, current_state: State) -> StateResult:
        if not self._click_idle():
            return State.CHECK_NEW_LEVEL
        self._sleep(0.05)
        if not self._new_level_red_icon_verified:
            if not self._scroll(verify_down=True):
                return State.CHECK_NEW_LEVEL
            confirmed = self._find_new_level_red_icon()
            if confirmed is None:
                self._reset_search_cycle()
                return State.FIND_RED_ICONS
            self._new_level_red_icon_verified = True
        if not self.mouse_controller.click(*config.NEW_LEVEL_BUTTON_POS, relative=True):
            return State.CHECK_NEW_LEVEL
        self._sleep(0.30)
        if not self.mouse_controller.click(*config.LEVEL_TRANSITION_POS, relative=True):
            return State.CHECK_NEW_LEVEL
        self._sleep(0.20)
        elapsed = self._record_level_completion("verified_red_icon")
        logger.info("Level %s completed via verified red-icon path. Time spent: %.1fs", self.total_levels_completed, elapsed)
        return State.WAIT_FOR_UNLOCK

    def handle_transition_level(self, current_state: State) -> StateResult:
        self._click_idle()
        for attempt in range(5):
            found, confidence, x, y = self._find_new_level_button(self.window_capture.capture(max_y=config.MAX_SEARCH_Y))
            if found:
                self.vision_optimizer.record_confidence("new_level", confidence)
                if not self.mouse_controller.click(x, y, relative=True):
                    return State.CHECK_NEW_LEVEL
                self._sleep(1.0)
                elapsed = self._record_level_completion("transition")
                logger.info("Level %s completed. Time spent: %.1fs", self.total_levels_completed, elapsed)
                return State.WAIT_FOR_UNLOCK
            if attempt < 4:
                self._sleep(0.20)
        self.vision_optimizer.record_miss("new_level")
        self._reset_search_cycle()
        return State.FIND_RED_ICONS

    def handle_wait_for_unlock(self, current_state: State) -> StateResult:
        self._click_idle()
        self._sleep(0.05)
        self.wait_for_unlock_attempts += 1
        if self.wait_for_unlock_attempts > self.max_wait_for_unlock_attempts:
            self._reset_search_cycle()
            return State.FIND_RED_ICONS
        template_pair = self._template("unlock")
        if template_pair is None:
            self._sleep(0.30)
            return State.WAIT_FOR_UNLOCK
        template, mask = template_pair
        found, confidence, x, y = self.image_matcher.find_template(self.window_capture.capture(), template, mask=mask, threshold=config.UNLOCK_THRESHOLD, template_name="unlock")
        if not found or self.mouse_controller.is_in_forbidden_zone(x, y, relative=True):
            self._sleep(0.30)
            return State.WAIT_FOR_UNLOCK
        logger.info("Unlock button found at (%s, %s) [%.3f]", x, y, confidence)
        if not self.mouse_controller.click(x, y, relative=True):
            return State.WAIT_FOR_UNLOCK
        self._sleep(0.50)
        self._reset_search_cycle()
        return State.FIND_RED_ICONS
