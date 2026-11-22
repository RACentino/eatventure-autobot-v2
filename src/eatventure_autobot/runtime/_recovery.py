"""The red-X popup rule, the stall watchdog and the dead-game relaunch.

Moved out of bot.py unchanged; EatventureBot inherits it. What it reaches on self is declared
at the top of the class."""

import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from eatventure_autobot.domain.config import BotConfig
from eatventure_autobot.domain.errors import CaptureError
from eatventure_autobot.domain.protocols import InputController, Notifier
from eatventure_autobot.domain.state import X_LOCKED_STATES, GameplayPhase, PauseReason, State
from eatventure_autobot.domain.types import MatchCandidate, Point
from eatventure_autobot.resilience.watchdog import StallWatchdog, WatchdogVerdict
from eatventure_autobot.runtime._telemetry import _frame_thumbnail
from eatventure_autobot.runtime.game_recovery import GameLauncher
from eatventure_autobot.runtime.progress import ProgressLedger
from eatventure_autobot.runtime.vision import GameVision
from eatventure_autobot.state.context import FlowContext

logger = logging.getLogger("eatventure_autobot.runtime.bot")

# The recovery window judges "is the screen alive" from its last few seconds.
_RECOVERY_TAIL_SECONDS = 5.0


class _RecoveryMixin:
    # What this mixin reaches on self. EatventureBot.__init__ (bot.py) creates the attributes;
    # the methods live in bot.py or another mixin.
    _clock: Callable[[], float]
    _close_check_seconds: float
    _close_last_pos: Point | None
    _close_level_lock_until: float
    _close_streak: int
    _close_suppressed_until: float
    _config: BotConfig
    _idle_started_at: float
    _input: InputController
    _last_handled_state: State | None
    _last_probe_thumb: np.ndarray | None
    _launcher: GameLauncher | None
    _notifier: Notifier
    _recent_thumb: np.ndarray | None
    _relaunch_progress: tuple[int, int] | None
    _stop_requested: threading.Event
    _streak_thumb: np.ndarray | None
    _strategy_alive_at: float
    _vision: GameVision
    _watchdog: StallWatchdog
    context: FlowContext
    state: State
    pause_reason: PauseReason | None
    _progress: ProgressLedger | None
    _game_recovery_pending: bool
    _phone_game_foreground: bool | None

    if TYPE_CHECKING:  # annotation only: nothing here exists at runtime

        def _sleep(self, duration: float) -> bool:
            ...

        def _device_available(self, *, force: bool = False) -> bool: ...
        def stop(self, *, manual: bool = False) -> None: ...
        def _arm_resume(self, reason: str) -> None: ...

        def _restart_mirror(self, cancelled: threading.Event) -> bool: ...

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
        if self._config.strategy.enabled and self.context.strategy_waiting_liveness:
            return False, None
        if not config.enabled or self.state in X_LOCKED_STATES:
            return False, None
        now = time.monotonic()
        if now < self._close_level_lock_until or now < self._close_suppressed_until:
            return False, None
        frame = self._vision.capture()
        found = self._vision.find_close_buttons(frame)
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

    def _tap_close_button(
        self, target: MatchCandidate, *, remaining: float | None = None, recovery: bool = False
    ) -> bool:
        """Taps one detected X and waits out the modal settle (clamped to `remaining` seconds when
        the recovery window is closing). Shared by the main-flow rule and the recovery window."""
        x, y = target.center
        context = self.context
        # check_forbidden=False: the X is the intended target, wherever the popup puts it (same
        # precedent as the stats button).
        clicked = self._input.click(x, y, check_forbidden=False)
        if clicked:
            context.close_clicks += 1
            context.recovery_taps += recovery
        context.close_positions[(x, y)] = context.close_positions.get((x, y), 0) + 1
        after = self._last_handled_state
        logger.info(
            "Closed popup at (%d, %d) conf=%.3f state=%s%s%s%s",
            x,
            y,
            target.confidence,
            self.state.name,
            " (recovery)" if recovery else "",
            "" if clicked else " (tap failed)",
            "" if after is None else f" after={after.name}",
        )
        settle = self._config.close_button.settle_delay
        self._sleep(settle if remaining is None else max(0.0, min(settle, remaining)))
        return clicked

    def _recover_popups(self) -> bool:
        """The recovery window after an adb relaunch: for the whole `relaunch_settle_seconds`
        ONLY the red-X rule runs. No handler, no other asset (red icons, boxes, unlock, scroll, idle
        click), no New Level lock and no stand-down guard (nothing is in flight in a freshly
        restarted game, and the window is time-bounded). It always runs the full window, even after
        the X is gone, because popups can arrive late in a cold start. A capture error is retried
        until the deadline; a stop request ends the window early."""
        config = self._config.game_recovery
        started = self._clock()
        deadline = started + config.relaunch_settle_seconds
        tail_from = deadline - _RECOVERY_TAIL_SECONDS
        tail_min: np.ndarray | None = None
        tail_max: np.ndarray | None = None
        taps = capture_errors = 0
        prepared = False
        while self._clock() < deadline and not self._stop_requested.is_set():
            if self._progress is not None:
                self._progress.checkpoint()
            if not self._device_available():
                return False
            remaining = deadline - self._clock()
            if self._phone_game_foreground is False:
                self._sleep(min(config.recovery_poll_seconds, remaining))
                continue
            try:
                self._vision.ensure_target_ready(force=not prepared)
                prepared = True
                frame = self._vision.capture()
                found = self._vision.find_close_buttons(frame)
            except CaptureError as exc:
                prepared = False
                capture_errors += 1
                if capture_errors == 1:
                    logger.warning("Recovery: capture failed (%s); retrying until it ends", exc)
                self._sleep(min(config.recovery_poll_seconds, remaining))
                continue
            last = _frame_thumbnail(frame)
            if self._clock() >= tail_from:
                # Periodic animation can return to its first pose; retain the whole tail's motion.
                tail_min = last if tail_min is None else np.minimum(tail_min, last)
                tail_max = last if tail_max is None else np.maximum(tail_max, last)
            if found:
                taps += 1
                self._tap_close_button(found[0], remaining=remaining, recovery=True)
            else:
                self._sleep(min(config.recovery_poll_seconds, remaining))
        elapsed = self._clock() - started
        if tail_min is None or tail_max is None:  # stopped early or never captured: no verdict
            logger.info("Recovery window ended after %.0f s: %d X tapped", elapsed, taps)
            return False
        delta = float((tail_max - tail_min).mean())
        changing = delta >= self._config.game_recovery.static_frame_delta
        (logger.info if changing else logger.warning)(
            "Recovery window ended after %.0f s: %d X tapped, screen %s "
            "(frame span %.2f over the last %.0f s)%s",
            elapsed,
            taps,
            "changing" if changing else "still static",
            delta,
            _RECOVERY_TAIL_SECONDS,
            "" if changing else "; remaining in recovery with capped backoff",
        )
        return changing

    def _apply_watchdog(self, current_state: State) -> None:
        if current_state == State.CONFIRM_RESTAURANT or (
            current_state == State.FIND_RED_ICONS and self._config.strategy.enabled
            and self.context.strategy_waiting_liveness
        ):
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

    def _game_looks_dead(self, thumb: np.ndarray) -> bool:
        """Runs only on a stall probe (>= stall_scrolls_before_alert idle scrolls, ~10 min at the
        no-work loop's ~4.2 s per scroll; it was ~7.5 min at 3.0 s). It runs on the scan right
        after a drag, when a live game has just moved and a frozen one has not: the picture has not
        changed since the streak began. Healthy idle streaks peak near 66 scrolls (measured at 3.0 s
        per scroll, p99 of 671 metrics samples), so the scroll count alone is no evidence; the
        frame is. Static means unchanged since the streak began OR since a recent scan (the early
        probe's, then the previous alert's): a screen that changed once and then froze (a dimmed
        New Level panel, 2026-10-16) differs from the streak start for good yet is static now."""
        before = self._streak_thumb
        if before is None or self._launcher is None:  # no launcher = recovery switched off
            return False
        config = self._config.game_recovery
        limit = config.static_frame_delta
        delta = float(np.abs(thumb - before).mean())
        static = delta < limit
        recent_note = ""
        if self._recent_thumb is not None:
            recent_delta = float(np.abs(thumb - self._recent_thumb).mean())
            static = static or recent_delta < config.static_recent_delta
            recent_note = (
                f", vs recent={recent_delta:.2f} (static below {config.static_recent_delta:.2f})"
            )
        # The decision's own inputs: the stall probe's "Frame delta vs last probe" compares a
        # different pair of frames, so on its own it cannot tell whether a relaunch was justified.
        logger.info(
            "Dead-game check: frame delta vs streak start=%.2f (static below %.2f)%s, "
            "idle_scrolls=%d -> %s",
            delta,
            limit,
            recent_note,
            self.context.idle_scrolls,
            "static" if static else "alive",
        )
        if static and self._launcher.package is None:
            logger.warning("Game appears dead (frame unchanged) but adb relaunch is not available")
            return False
        return static

    def _queue_game_recovery(self) -> State:
        if self._config.strategy.enabled:
            self._idle_started_at = self._strategy_alive_at
        self._game_recovery_pending = True
        self.pause_reason = PauseReason.RECOVERY
        self.stop()
        self._arm_resume("game appears frozen")
        return State.FIND_RED_ICONS

    def _relaunch_game(self) -> State:
        """adb force-stop + launch, wait out the cold start, and restart the search from scratch.
        No cap on repeats (by choice): the next chance is another full stall alert, ~8 min away."""
        launcher, context = self._launcher, self.context
        assert launcher is not None  # _game_looks_dead only says yes when a launcher is armed
        if not self._device_available(force=True) or self._stop_requested.is_set():
            return State.FIND_RED_ICONS
        self._input.release_left_button()
        context.invalidate_viewport()
        if self._progress is not None:
            self._progress.checkpoint("recovery")
        progress = (context.holds_completed, context.boxes_opened_total)
        again = ""
        viewport_frozen = context.strategy_waiting_liveness
        if viewport_frozen or progress == self._relaunch_progress:
            again = (" (viewport frozen; restarting owned scrcpy)" if viewport_frozen else
                     " (still static after the previous relaunch; the scrcpy window may be frozen)")
            restarted = self._restart_mirror(self._stop_requested)
            if viewport_frozen and not restarted or self._stop_requested.is_set():
                return State.FIND_RED_ICONS
        logger.warning(
            "Game appears dead: frame unchanged for %.0f min%s; relaunching %s",
            (time.monotonic() - self._idle_started_at) / 60,
            again,
            launcher.package,
        )
        self._relaunch_progress = progress
        minutes = (time.monotonic() - self._idle_started_at) / 60
        self._game_recovery_pending = True
        if launcher.relaunch(self._stop_requested):
            context.gameplay_phase = GameplayPhase.FIRST
            context.relaunches += 1
            self._notifier.notify_game_relaunch(True, minutes)
            self._game_recovery_pending = (
                not self._recover_popups() and not self._stop_requested.is_set()
            )
        else:
            # Nothing restarted, so there is nothing to wait for: log (done by the launcher) and
            # carry on with the main flow as before.
            context.relaunch_failures += 1
            self._notifier.notify_game_relaunch(False, minutes)
        context.idle_scrolls = 0
        context.reset_search_cycle()
        self._streak_thumb = None
        self._recent_thumb = None
        self._last_probe_thumb = None
        # The restarted game starts the X rule from scratch: no stale stand-down or lock window.
        self._close_last_pos, self._close_streak = None, 0
        self._close_suppressed_until = self._close_level_lock_until = 0.0
        return State.FIND_RED_ICONS
