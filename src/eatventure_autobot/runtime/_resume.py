"""Auto-resume after an environment self-stop.

Moved out of bot.py unchanged; EatventureBot inherits it. What it reaches on self is declared
at the top of the class."""

import dataclasses
import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from eatventure_autobot.domain.config import BotConfig
from eatventure_autobot.domain.errors import CaptureError
from eatventure_autobot.domain.protocols import InputController, Notifier
from eatventure_autobot.domain.state import PauseReason, State
from eatventure_autobot.domain.types import Point
from eatventure_autobot.runtime.game_recovery import GameLauncher
from eatventure_autobot.runtime.progress import ProgressLedger
from eatventure_autobot.runtime.vision import GameVision

logger = logging.getLogger("eatventure_autobot.runtime.bot")

# Auto-resume (see EatventureBot.poll_resume): how often the cursor is sampled for movement, how
# often an unchanged check is logged anyway, and the longest wait between failed live attempts.
_RESUME_SAMPLE_SECONDS = 1.0
_RESUME_HEARTBEAT_SECONDS = 600.0
_RESUME_MAX_BACKOFF_SECONDS = 900.0


@dataclasses.dataclass
class _ResumeWatch:
    """One auto-resume episode, created when the bot stops itself on a window or focus error."""

    reason: str
    since: float  # when the bot stopped itself
    activity_at: float  # the last key press or cursor movement seen
    sample_at: float  # next cursor sample
    check_at: float  # next readiness check
    cursor: Point | None = None
    attempts: int = 0
    logged: tuple[bool, bool] | None = None  # (ready, quiet) of the last logged check
    logged_at: float = 0.0
    restart_mirror: bool = False


class _ResumeMixin:
    # What this mixin reaches on self. EatventureBot.__init__ (bot.py) creates the attributes;
    # the methods live in bot.py or another mixin.
    _clock: Callable[[], float]
    _config: BotConfig
    _desktop_notifier: Callable[[str, str], None] | None
    _input: InputController
    _notifier: Notifier
    _resume: _ResumeWatch | None
    _started_at: float
    _vision: GameVision
    _launcher: GameLauncher | None
    _progress: ProgressLedger | None
    _stop_requested: threading.Event
    _manual_stop_requested: threading.Event
    pause_reason: PauseReason | None
    _game_recovery_pending: bool

    if TYPE_CHECKING:  # annotation only: nothing here exists at runtime

        @property
        def running(self) -> bool:
            ...

        def start(self, *, report: bool=True) -> bool:
            ...

        def _device_available(self, *, force: bool = False) -> bool: ...
        def _relaunch_game(self) -> State: ...

        def _restart_mirror(self, cancelled: threading.Event) -> bool: ...

        def _bind_existing_mirror(self) -> bool: ...

    def _arm_resume(self, reason: str) -> None:
        """Retry known environment or screen-observation failures. Manual/disconnected stops
        remain latched; unexpected handler failures never call this path."""
        if self._manual_stop_requested.is_set() or self.pause_reason == PauseReason.DISCONNECTED:
            return
        if not self._device_available(force=True):
            return
        self.pause_reason = PauseReason.RECOVERY
        if self._progress is not None:
            self._progress.checkpoint("recovery")
        now = self._clock()
        self._resume = _ResumeWatch(
            reason, now, now, now, now + self._config.resume.check_seconds
        )

    def poll_resume(self, last_key_at: float) -> None:
        """Called on every pass of the main loop while the bot is stopped; a no-op unless armed.
        last_key_at is the latest key press on this PC, on the same clock as self._clock.

        Live mode (resume.enabled, the default since 2026-10-27): once nobody has pressed a key or
        moved the mouse for resume.idle_seconds it calls start(). A window start() would touch
        (GameVision.environment_ready False: another app has the focus) is first brought up with
        raise_window, the one focus change the bot makes on its own; with that off, only a window
        that is already in front is started. Shadow mode (enabled False): each check only logs
        whether it WOULD start the bot. Connected failures retry indefinitely with capped backoff.
        Unexpected exceptions latch ERROR and cancel retries."""
        watch = self._resume
        if self._progress is not None:
            self._progress.checkpoint()
        if watch is None or self.running:
            return
        now = self._clock()
        if now < watch.sample_at:
            return
        watch.sample_at = now + _RESUME_SAMPLE_SECONDS
        try:
            if not self._device_available(force=True):
                return
            cursor = self._input.get_cursor_position()
            # The first sample is only a baseline; a later change means a person moved the mouse.
            if watch.cursor is not None and cursor != watch.cursor:
                watch.activity_at = now
            watch.cursor = cursor
            if now >= watch.check_at:
                self._check_resume(watch, now, last_key_at)
        except CaptureError:
            watch.check_at = now + self._config.resume.check_seconds
        except Exception:
            logger.exception("Resume check failed")
            self._stop_requested.set()
            self._input.release_left_button()
            self._resume = None
            self.pause_reason = PauseReason.ERROR
            if self._progress is not None:
                self._progress.checkpoint("error")

    def _check_resume(self, watch: _ResumeWatch, now: float, last_key_at: float) -> None:
        config = self._config.resume
        idle = now - max(watch.activity_at, last_key_at)
        ready = self._vision.environment_ready()
        quiet = idle >= config.idle_seconds
        watch.check_at = now + config.check_seconds
        # One line per change plus a heartbeat, not one per minute of a 20 h stop.
        if (ready, quiet) != watch.logged or now - watch.logged_at >= _RESUME_HEARTBEAT_SECONDS:
            watch.logged, watch.logged_at = (ready, quiet), now
            go = "resuming" if config.enabled else "would resume"
            logger.info(
                "Resume check (%s): ready=%s idle_s=%.0f/%.0f down_min=%.0f -> %s | %s",
                "live" if config.enabled else "shadow",
                ready,
                idle,
                config.idle_seconds,
                (now - watch.since) / 60,
                go if quiet and (ready or config.raise_window) else "waiting",
                self._vision.describe_environment(),
            )
        if not (config.enabled and quiet):
            return
        if self._manual_stop_requested.is_set() or self._resume is not watch:
            return
        if not ready and self._launcher is not None and not self._launcher.mirror_bound():
            self._bind_existing_mirror()
            ready = self._vision.environment_ready()
        watch.attempts += 1
        backoff = min(_RESUME_MAX_BACKOFF_SECONDS,
                      config.check_seconds * 2 ** min(watch.attempts, 20))
        watch.check_at = now + backoff
        if watch.restart_mirror:
            if self._restart_mirror(self._manual_stop_requested):
                watch.restart_mirror = False
                watch.check_at = self._clock() + config.check_seconds
            return
        if not ready:
            # Not ready means start() would touch the window (usually: another app has the focus).
            # Nobody has used the PC for idle_seconds, so this is the one place the bot brings the
            # window up itself; a failed raise (locked session, window gone) just waits a check.
            if not (config.raise_window and self._vision.raise_target()):
                try:
                    self._vision.ensure_target_ready()
                except CaptureError:
                    self._restart_mirror(self._manual_stop_requested)
                return
            logger.info("Resume: raised the game window after %.0f s of nobody at the PC", idle)
        if self._game_recovery_pending:
            if self._manual_stop_requested.is_set() or not self._device_available(force=True):
                return
            self._stop_requested.clear()
            self._relaunch_game()
            self._stop_requested.set()
            watch.activity_at = self._clock()
            if not self._game_recovery_pending:
                watch.attempts = 0
                backoff = config.check_seconds
            watch.check_at = self._clock() + backoff
            return
        if self.start(report=False):
            logger.info(
                "Auto-resumed after %.0f min (attempt %d): %s",
                (now - watch.since) / 60,
                watch.attempts,
                watch.reason,
            )
            return

    def _report_self_stop(self, reason: str) -> None:
        """Whenever the bot stops itself (not a Z press) it stays stopped until a person comes back
        (44.5 h of 205 over Sep 28-Oct 6), so log the facts that may explain why and alert once.
        Diagnostics only: never raises, never changes what the caller does next. ran_min is "-"
        for a start that failed, since there was no run."""
        try:
            ran = f"{(time.monotonic() - self._started_at) / 60:.0f}" if self.running else "-"
            logger.warning(
                "Self-stop: %s | ran_min=%s | %s", reason, ran, self._vision.describe_environment()
            )
            self._notifier.notify_bot_self_stopped(reason, 0.0 if ran == "-" else float(ran))
            if self._desktop_notifier is not None:
                self._desktop_notifier("Eatventure bot stopped itself", reason)
        except Exception:
            logger.exception("Self-stop report failed")
