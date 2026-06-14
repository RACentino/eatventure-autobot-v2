"""JSON-persisted historical learning for EatventureBot.

Extracted from bot.py to keep HistoricalLearner independently testable and
to reduce bot.py's line count.  Communicates back to the bot via a typed
Protocol to avoid circular imports.

Disabled by default: set AI_LEARNING_ENABLED = True in config.py to activate.

No RGB gate code.  No Windows-specific imports.  Fully OS-agnostic.
"""

import logging
import math
import threading
import time
from typing import Any, Protocol

import config
from persistence import RuntimePersistence
from tuner import AdaptiveTuner

logger = logging.getLogger(__name__)

LEARNING_LOOP_ITERATION_LIMIT = 2_147_483_647


class _BotLike(Protocol):
    """Minimal bot interface consumed by HistoricalLearner."""

    def apply_learned_behavior(self, learned: dict[str, Any]) -> None: ...

    def get_runtime_behavior_snapshot(self) -> dict[str, float]: ...


class HistoricalLearner:
    """Background learner that blends past level-completion profiles
    into a tuned timing profile.

    Records each level completion with its elapsed time and the active
    timing behavior, then periodically applies the blend of the fastest
    observed profiles back to the bot via *apply_learned_behavior()*.

    Persistence is handled by a RuntimePersistence object so state
    survives bot restarts.
    """

    def __init__(
        self, bot: _BotLike, persistence: RuntimePersistence | None = None
    ) -> None:
        self.bot = bot
        self.persistence = persistence
        self.enabled = bool(config.AI_LEARNING_ENABLED)
        self.interval = max(
            config.LEARNING_LOOP_MIN_SLEEP, float(config.AI_LEARNING_THREAD_INTERVAL)
        )
        self.pair_window = max(2, int(config.AI_LEARNING_PAIR_WINDOW))
        self.batch_window = max(2, int(config.AI_LEARNING_BATCH_WINDOW))
        self.ema_alpha = max(0.01, min(0.8, float(config.AI_LEARNING_EMA_ALPHA)))
        self.top_k = max(1, int(config.AI_LEARNING_PROFILE_BLEND_TOP_K))
        self.min_improvement_ratio = max(
            0.0, float(config.AI_LEARNING_MIN_IMPROVEMENT_RATIO)
        )
        self.apply_cooldown = max(0.0, float(config.AI_LEARNING_APPLY_COOLDOWN))
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._records: list[dict[str, Any]] = []
        self._total_completions = 0
        self._last_pair_processed = 0
        self._last_batch_processed = 0
        self._last_apply_time = 0.0
        self._tuned_behavior: dict[str, float] = {}
        self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _float(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @classmethod
    def _sanitize_behavior(
        cls: type["HistoricalLearner"], behavior: Any
    ) -> dict[str, float]:
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
            self._records = [
                record for record in records if isinstance(record, dict)
            ][-config.AI_LEARNING_RECORDS_LIMIT:]
        self._total_completions = max(
            0, int(state.get("total_completions", len(self._records)) or 0)
        )
        self._last_pair_processed = max(
            0, int(state.get("last_pair_processed", 0) or 0)
        )
        self._last_batch_processed = max(
            0, int(state.get("last_batch_processed", 0) or 0)
        )
        self._tuned_behavior = self._sanitize_behavior(
            state.get("tuned_behavior", {})
        )
        if self._tuned_behavior:
            self.bot.apply_learned_behavior(self._tuned_behavior)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

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
        record = {
            "timestamp": time.time(),
            "time_spent": float(seconds_spent),
            "source": source,
            "behavior": self.bot.get_runtime_behavior_snapshot(),
        }
        with self._lock:
            self._records.append(record)
            self._records = self._records[-config.AI_LEARNING_RECORDS_LIMIT:]
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

    # ------------------------------------------------------------------
    # Background learning loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        for _ in range(LEARNING_LOOP_ITERATION_LIMIT):
            if self._stop.is_set():
                return
            try:
                self._apply_pending_profiles()
            except Exception:
                logger.exception("Historical learner cycle failed")
            if self._stop.wait(self.interval):
                return
        logger.error("Historical learner loop reached iteration limit")

    def _apply_pending_profiles(self) -> None:
        with self._lock:
            records = list(self._records)
            total = self._total_completions
        if time.monotonic() - self._last_apply_time < self.apply_cooldown:
            return
        changed = False
        for window, attr in (
            (self.pair_window, "_last_pair_processed"),
            (self.batch_window, "_last_batch_processed"),
        ):
            marker = total // window
            if total >= window and marker > getattr(self, attr):
                changed = self._apply_profile(records[-window:]) or changed
                setattr(self, attr, marker)
        if changed:
            self._last_apply_time = time.monotonic()
        self._persist()

    # ------------------------------------------------------------------
    # Profile blending
    # ------------------------------------------------------------------

    def _apply_profile(self, records: list[dict[str, Any]]) -> bool:
        valid = self._valid_profile_records(records)
        if not valid:
            return False
        ranked = sorted(valid, key=lambda item: item[0])
        if not self._profile_improves(valid, ranked):
            return False
        profile = self._blended_profile(ranked)
        self._tuned_behavior = self._sanitize_behavior(
            self._tuned_profile(profile)
        )
        self.bot.apply_learned_behavior(self._tuned_behavior)
        return True

    def _valid_profile_records(
        self, records: list[dict[str, Any]]
    ) -> list[tuple[float, dict[str, float]]]:
        valid: list[tuple[float, dict[str, float]]] = []
        for record in records:
            duration = self._float(record.get("time_spent"))
            behavior = self._sanitize_behavior(record.get("behavior", {}))
            if duration and duration > 0 and behavior:
                valid.append((duration, behavior))
        return valid

    def _profile_improves(
        self,
        valid: list[tuple[float, dict[str, float]]],
        ranked: list[tuple[float, dict[str, float]]],
    ) -> bool:
        average = sum(duration for duration, _ in valid) / len(valid)
        return (
            average > 0
            and (average - ranked[0][0]) / average >= self.min_improvement_ratio
        )

    def _blended_profile(
        self, ranked: list[tuple[float, dict[str, float]]]
    ) -> dict[str, float]:
        profile: dict[str, float] = {
            key: 0.0 for key in ("click_delay", "move_delay", "search_interval")
        }
        for _, behavior in ranked[: self.top_k]:
            for key in profile:
                profile[key] += float(behavior.get(key, 0.0))
        divisor = float(min(self.top_k, len(ranked)))
        return {key: value / divisor for key, value in profile.items()}

    def _tuned_profile(self, profile: dict[str, float]) -> dict[str, float]:
        current = self.bot.get_runtime_behavior_snapshot()
        return {
            key: (1.0 - self.ema_alpha) * current[key]
            + self.ema_alpha * profile[key]
            for key in profile
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self, force: bool = False) -> None:
        if self.persistence is None:
            return
        with self._lock:
            state = {
                "records": self._records[-config.AI_LEARNING_RECORDS_LIMIT:],
                "total_completions": self._total_completions,
                "last_pair_processed": self._last_pair_processed,
                "last_batch_processed": self._last_batch_processed,
                "tuned_behavior": self._tuned_behavior,
            }
        if not self.persistence.save(state, force=force):
            logger.debug("Historical learner state was not persisted")
