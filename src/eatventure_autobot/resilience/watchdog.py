"""The same-state stall watchdog, matching v1 exactly.

Verified behavior in both source repos, kept byte-for-byte in spirit — force a transition back to
a fallback state (FIND_RED_ICONS) after state_stall_timeout_seconds in the same state, forever,
with no escalation, no counting, no cap. v1's own mechanism (`bot.py`:
`elif now - state_entered_at >= STATE_STALL_TIMEOUT_SECONDS: reset_search_cycle()`) is exactly
this: one detector, one action.

This module has carried, and shed, two greenfield-only additions that were never verified source
behavior: a consecutive-reset cap escalating to a fail-closed `stop()` (removed — required a human
to restart the bot for something both source repos just loop through), and a second "no productive
progress across changing states" detector (removed — live evidence showed it firing before the
oscillating scroll search could complete a widening sweep, repeatedly resetting the search back to
its narrowest range and actively preventing exactly the productive progress it was meant to
detect). Both were reasonable ideas that real operation disproved; this module now does exactly
what v1 does, nothing more."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto

from eatventure_autobot.domain.config import FlowTimingConfig
from eatventure_autobot.domain.state import State


class WatchdogVerdict(Enum):
    OK = auto()
    RESET = auto()


@dataclass
class StallWatchdog:
    config: FlowTimingConfig
    fallback_state: State = State.FIND_RED_ICONS
    clock: Callable[[], float] = field(default=time.monotonic)

    _last_state: State | None = field(default=None, init=False, repr=False)
    _entered_at: float = field(default=0.0, init=False, repr=False)
    last_reset_reason: str = field(default="", init=False, repr=False)

    def tick(self, current_state: State) -> WatchdogVerdict:
        now = self.clock()
        if current_state != self._last_state:
            self._last_state = current_state
            self._entered_at = now
            return WatchdogVerdict.OK

        if now - self._entered_at < self.config.state_stall_timeout_seconds:
            return WatchdogVerdict.OK

        self.last_reset_reason = f"{current_state.name} stalled"
        self._entered_at = now
        self._last_state = self.fallback_state
        return WatchdogVerdict.RESET

    def reset(self) -> None:
        self._last_state = None
