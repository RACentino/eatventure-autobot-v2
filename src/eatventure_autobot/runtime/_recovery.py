"""Gameplay popup handling and state watchdog."""

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from eatventure_autobot.domain.config import BotConfig
from eatventure_autobot.domain.protocols import InputController
from eatventure_autobot.domain.state import X_LOCKED_STATES, State
from eatventure_autobot.domain.types import MatchCandidate, Point
from eatventure_autobot.resilience.watchdog import StallWatchdog, WatchdogVerdict
from eatventure_autobot.runtime.vision import GameVision
from eatventure_autobot.state.context import FlowContext

logger = logging.getLogger("eatventure_autobot.runtime.bot")

class _RecoveryMixin:
    # What this mixin reaches on self. EatventureBot.__init__ (bot.py) creates the attributes;
    # the methods live in bot.py or another mixin.
    _clock: Callable[[], float]
    _asset_frame_at: float | None
    _close_check_seconds: float
    _close_last_pos: Point | None
    _close_level_lock_until: float
    _close_streak: int
    _close_suppressed_until: float
    _config: BotConfig
    _input: InputController
    _last_handled_state: State | None
    _vision: GameVision
    _watchdog: StallWatchdog
    context: FlowContext
    state: State

    if TYPE_CHECKING:
        def _sleep(self, duration: float) -> bool: ...

    def _dismiss_popup(self) -> tuple[bool, np.ndarray | None]:
        """Red-X priority: runs before the state handler, so it outranks every handler and asset.
        A tap consumes the step. A clean frame can serve the next read during this step.

        Hard lock for the New Level dialog (its X looks exactly like this one), two layers, either
        of which blocks the tap:
        1. state: while CHECK_NEW_LEVEL / TRANSITION_LEVEL / WAIT_FOR_UNLOCK owns the screen, return
           before any capture;
        2. afterwards: for new_level_lock_seconds after any step in those states (attempt, miss or
           completion), leave the screen alone so a dialog still open is left to the flow.
        Deliberately NOT a lock on seeing the `unlock` button: that template also matches the blue
        "Collect x2" button of the Offline Earnings popup (0.935 vs a 0.905 threshold), which would
        have locked out the very popup this rule exists for.
        Stand-down guard: the same spot tapped max_consecutive_clicks times in a row means the tap
        is not closing anything, so the rule pauses for suppress_seconds instead of starving the
        flow."""
        config = self._config.close_button
        if not config.enabled or self.state in X_LOCKED_STATES:
            return False, None
        now = time.monotonic()
        if now < self._close_level_lock_until or now < self._close_suppressed_until:
            return False, None
        frame = self._vision.capture()
        self._asset_frame_at = self._clock() if self.state == State.HOLD_UPGRADE_STATION else None
        found = self._vision.find_close_buttons(frame)
        if self._config.measurement.detailed:
            self._close_check_seconds += time.monotonic() - now
        if not found:
            self._close_last_pos, self._close_streak = None, 0
            return False, frame
        target = found[0]
        x, y = target.center
        last = self._close_last_pos
        same_spot = last is not None and abs(last[0] - x) <= 3 and abs(last[1] - y) <= 3
        if same_spot and self._close_streak >= config.max_consecutive_clicks:
            self.context.close_suppressed += 1
            self._close_last_pos, self._close_streak = None, 0
            self._close_suppressed_until = now + config.suppress_seconds
            logger.warning(
                "Red X at (%d, %d) still there after %d taps; ignoring X's for %.0f s",
                x,
                y,
                config.max_consecutive_clicks,
                config.suppress_seconds,
            )
            return False, None
        self._close_streak = self._close_streak + 1 if same_spot else 1
        self._close_last_pos = (x, y)
        self._tap_close_button(target)
        return True, None

    def _tap_close_button(self, target: MatchCandidate) -> bool:
        x, y = target.center
        context = self.context
        # check_forbidden=False: the X is the intended target, wherever the popup puts it (same
        # precedent as the stats button).
        clicked = self._input.click(x, y, check_forbidden=False)
        if clicked:
            context.close_clicks += 1
        context.close_positions[(x, y)] = context.close_positions.get((x, y), 0) + 1
        after = self._last_handled_state
        logger.info(
            "Closed popup at (%d, %d) conf=%.3f state=%s%s%s",
            x,
            y,
            target.confidence,
            self.state.name,
            "" if clicked else " (tap failed)",
            "" if after is None else f" after={after.name}",
        )
        settle = self._config.close_button.settle_delay
        self._sleep(settle)
        return clicked

    def _apply_watchdog(self, current_state: State) -> None:
        if current_state == State.CONFIRM_RESTAURANT:
            self._watchdog.reset()  # these observations have their own longer deadlines
            return
        verdict = self._watchdog.tick(current_state)
        if verdict is WatchdogVerdict.OK:
            return
        logger.warning("Resetting search flow (%s)", self._watchdog.last_reset_reason)
        self.context.reset_search_cycle()
        self.context.invalidate_viewport()
        self.state = State.FIND_RED_ICONS
        self.context.state_attempt = 1
