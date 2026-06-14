"""EMA-based adaptive timing tuner for EatventureBot.

Extracted from bot.py to keep AdaptiveTuner independently testable and
to reduce bot.py's line count.  Disabled by default via config flag.

No RGB gate code.  No Windows-specific imports.  Fully OS-agnostic.
"""

import logging

import config

logger = logging.getLogger(__name__)


class AdaptiveTuner:
    """Exponential-moving-average tuner for click/search timing delays.

    When enabled, records success/failure events from bot operations and
    nudges click_delay, move_delay, and search_interval toward values
    that correlate with higher success rates.

    Disabled by default: set ADAPTIVE_TUNER_ENABLED = True in config.py
    to activate.
    """

    def __init__(self) -> None:
        self.enabled = bool(config.ADAPTIVE_TUNER_ENABLED)
        self.alpha = float(config.ADAPTIVE_TUNER_ALPHA)
        self.click_success_rate = 1.0
        self.search_success_rate = 1.0
        self.click_delay = float(config.CLICK_DELAY)
        self.move_delay = float(config.MOUSE_MOVE_DELAY)
        self.search_interval = float(config.UPGRADE_SEARCH_INTERVAL)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ema(self, current: float, value: float) -> float:
        return (1.0 - self.alpha) * current + self.alpha * value

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(float(minimum), min(float(maximum), float(value)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_click_result(self, success: bool) -> None:
        """Update click success EMA and adjust delays accordingly."""
        if not self.enabled:
            return
        self.click_success_rate = self._ema(
            self.click_success_rate, 1.0 if success else 0.0
        )
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
        """Update search success EMA and adjust search_interval accordingly."""
        if not self.enabled:
            return
        self.search_success_rate = self._ema(
            self.search_success_rate, 1.0 if success else 0.0
        )
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
        """Reset all EMA state and delays to config defaults."""
        self.click_success_rate = 1.0
        self.search_success_rate = 1.0
        self.click_delay = float(config.CLICK_DELAY)
        self.move_delay = float(config.MOUSE_MOVE_DELAY)
        self.search_interval = float(config.UPGRADE_SEARCH_INTERVAL)
