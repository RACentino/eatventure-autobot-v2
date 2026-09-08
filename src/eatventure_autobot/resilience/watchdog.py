"""Two independent stall detectors sharing one escalation path.

Same-state watchdog: verified behavior in both source repos — force a transition back to a
fallback state (FIND_RED_ICONS) after state_stall_timeout_seconds in the same state, forever, with
no escalation. This module keeps that self-healing behavior but adds a bounded-retry-then-
fail-closed cap (GREENFIELD_PLAN.md decision 3): if the fallback state itself keeps stalling for
max_consecutive_watchdog_resets resets in a row with no real state change in between, escalate
instead of resetting again — neither source repo can currently get unstuck from that scenario.

No-progress watchdog (greenfield addition, not a source-repo behavior): the same-state check alone
can't catch a non-productive loop that keeps cycling between several different states (e.g.
FIND_RED_ICONS -> OPEN_BOXES -> SCROLL -> FIND_RED_ICONS forever) without ever completing anything
real, since every tick sees a "new" state and resets the same-state timer. This tracks
FlowContext.progress_marker (bumped only when something concretely productive happens — a click
lands, a box opens, a hold completes, a level completes) and escalates if it hasn't moved in
max_no_progress_seconds, independent of which states were visited in between. Long idle-farming
stretches with nothing to click/open are normal for this bot, so this threshold is deliberately
much longer than state_stall_timeout_seconds — tune it against real logs/bot.log timestamps for
your own play session, not against this default blindly."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto

from eatventure_autobot.domain.config import FlowTimingConfig
from eatventure_autobot.domain.state import State


class WatchdogVerdict(Enum):
    OK = auto()
    RESET = auto()
    ESCALATE = auto()


@dataclass
class StallWatchdog:
    config: FlowTimingConfig
    fallback_state: State = State.FIND_RED_ICONS
    clock: Callable[[], float] = field(default=time.monotonic)

    _last_state: State | None = field(default=None, init=False, repr=False)
    _entered_at: float = field(default=0.0, init=False, repr=False)
    _consecutive_resets: int = field(default=0, init=False, repr=False)
    _last_progress_marker: int | None = field(default=None, init=False, repr=False)
    _progress_seen_at: float = field(default=0.0, init=False, repr=False)
    last_escalation_reason: str = field(default="", init=False, repr=False)

    def tick(self, current_state: State, progress_marker: int = 0) -> WatchdogVerdict:
        now = self.clock()
        if self._last_progress_marker is None or progress_marker != self._last_progress_marker:
            self._last_progress_marker = progress_marker
            self._progress_seen_at = now

        verdict = self._tick_same_state(current_state, now)
        if verdict is not WatchdogVerdict.OK:
            return verdict

        if now - self._progress_seen_at >= self.config.max_no_progress_seconds:
            self.last_escalation_reason = (
                f"no productive progress in {self.config.max_no_progress_seconds:.0f}s "
                f"(last state: {current_state.name})"
            )
            return WatchdogVerdict.ESCALATE
        return WatchdogVerdict.OK

    def _tick_same_state(self, current_state: State, now: float) -> WatchdogVerdict:
        if current_state != self._last_state:
            self._last_state = current_state
            self._entered_at = now
            self._consecutive_resets = 0
            return WatchdogVerdict.OK

        if now - self._entered_at < self.config.state_stall_timeout_seconds:
            return WatchdogVerdict.OK

        self._consecutive_resets += 1
        self._entered_at = now
        self._last_state = self.fallback_state
        if self._consecutive_resets >= self.config.max_consecutive_watchdog_resets:
            self.last_escalation_reason = (
                f"{current_state.name} stalled repeatedly after "
                f"{self.config.max_consecutive_watchdog_resets} recovery attempts"
            )
            return WatchdogVerdict.ESCALATE
        return WatchdogVerdict.RESET

    def reset(self) -> None:
        self._last_state = None
        self._consecutive_resets = 0
        self._last_progress_marker = None
        self._progress_seen_at = self.clock()
