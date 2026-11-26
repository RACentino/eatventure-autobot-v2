"""Watch the selected device and resume when the game and desktop are ready."""

import dataclasses
import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from eatventure_autobot.domain.config import BotConfig
from eatventure_autobot.domain.errors import CaptureError, WindowNotAvailableError
from eatventure_autobot.domain.protocols import InputController, Notifier
from eatventure_autobot.domain.state import GameplayPhase, PauseReason, State
from eatventure_autobot.domain.types import Point
from eatventure_autobot.runtime.game_recovery import GameLauncher
from eatventure_autobot.runtime.progress import ProgressLedger
from eatventure_autobot.runtime.vision import GameVision
from eatventure_autobot.state.context import FlowContext

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
    key_activity: Callable[[], float] | None = None
    attempts: int = 0
    logged: tuple[bool, bool, bool | None] | None = None
    logged_at: float = 0.0
    repair_at: float = 0.0
    launch_at: float = 0.0  # A mirror repair must not postpone the first crash launch.


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
    context: FlowContext
    state: State
    pause_reason: PauseReason | None
    _phone_game_foreground: bool | None

    if TYPE_CHECKING:  # annotation only: nothing here exists at runtime

        @property
        def running(self) -> bool:
            ...

        def start(self, *, report: bool = True) -> bool:
            ...

        def _device_available(self, *, force: bool = False) -> bool: ...
        def _restart_mirror(self, cancelled: threading.Event) -> bool: ...

        def _bind_existing_mirror(self) -> bool: ...

    def _arm_resume(self, reason: str) -> None:
        """Arm only after losing the environment or the selected Android game."""
        if self._manual_stop_requested.is_set() or self.pause_reason == PauseReason.DISCONNECTED:
            return
        self.pause_reason = PauseReason.RECOVERY
        if self._progress is not None:
            self._progress.checkpoint("recovery")
        now = self._clock()
        self._resume = _ResumeWatch(reason, now, now, now, now)

    @property
    def resume_poll_delay(self) -> float:
        """Seconds until the next cursor sample, or the next stopped-time checkpoint."""
        if self._resume is None:
            return _RESUME_SAMPLE_SECONDS
        return max(0.0, self._resume.sample_at - self._clock())

    def poll_resume(self, last_key_at: float | Callable[[], float]) -> None:
        """Sample the desktop and phone while an automatic resume is armed."""
        watch = self._resume
        if self._config.measurement.detailed and self._progress is not None:
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
            watch.key_activity = last_key_at if callable(last_key_at) else None
            key_at = last_key_at() if callable(last_key_at) else last_key_at
            watch.activity_at = max(watch.activity_at, key_at)
            if now >= watch.check_at:
                self._check_resume(watch, now, key_at)
        except CaptureError:
            watch.check_at = self._clock() + self._config.resume.check_seconds
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
        status = (ready, quiet, self._phone_game_foreground)
        if status != watch.logged or now - watch.logged_at >= _RESUME_HEARTBEAT_SECONDS:
            watch.logged, watch.logged_at = status, now
            logger.info("Resume observation: mirror_ready=%s phone_foreground=%s "
                        "desktop_idle_s=%.0f elapsed_s=%.0f", ready, self._phone_game_foreground,
                        idle, now - watch.since)
        if not config.enabled or self._manual_stop_requested.is_set() or self._resume is not watch:
            return
        launcher = self._launcher
        if not ready or (launcher is not None and not launcher.mirror_bound()):
            if not quiet or now < watch.repair_at or not self._desktop_idle(watch):
                return
            if (launcher is not None
                    and not (launcher.mirror_bound() or self._bind_existing_mirror())):
                self._back_off_repair(watch)
                self._restart_mirror(self._manual_stop_requested)
                return
            if not self._vision.environment_ready():
                if not self._desktop_idle(watch):
                    return
                if not config.raise_window or not self._vision.raise_target():
                    try:
                        self._vision.ensure_target_ready()
                    except WindowNotAvailableError:
                        self._back_off_repair(watch)
                        self._restart_mirror(self._manual_stop_requested)
                    return
                if not self._desktop_idle(watch):
                    return
                self._vision.ensure_target_ready(force=True)
        resume_state = State.FIND_RED_ICONS
        if launcher is not None and self._config.game_recovery.enabled:
            if launcher.package is None:
                if now < watch.repair_at:
                    return
                if not launcher.detect_package():
                    self._back_off_repair(watch)
                    return
                if not self._device_available(force=True):
                    return
            if self._resume is not watch or self._manual_stop_requested.is_set():
                return
            if self._phone_game_foreground is False:
                if now >= watch.launch_at and launcher.game_exited():
                    self._launch_exited_game(watch, launcher)
                return
            if self._phone_game_foreground is not True:
                return
            # A foreground package can still be on the loading screen.
            frame = self._vision.capture()
            if self._vision.restaurant_start(frame):
                resume_state = State.TRANSITION_LEVEL  # The Open dialog owns its close button.
            elif not (self._vision.city_arrival(frame) or self._vision.find_close_buttons(frame)
                      or self._vision.station_panel_bounds(frame)
                      or self._vision.restaurant_world_visible(frame)):
                return
        if (self._resume is watch and self._vision.environment_ready()
                and self.start(report=False)):
            self.state = resume_state
            logger.info("Gameplay resumed after %.1f s: %s", self._clock() - watch.since,
                        watch.reason)

    def _launch_exited_game(self, watch: _ResumeWatch, launcher: GameLauncher) -> None:
        # Recheck after the first pidof: ADB may have blocked while the user returned or stopped.
        if (not self._device_available(force=True) or self._phone_game_foreground is not False
                or not launcher.mirror_bound() or not self._vision.environment_ready()
                or self._resume is not watch or not launcher.game_exited()
                or not self._input.is_target_foreground() or self._manual_stop_requested.is_set()):
            return
        if self._clock() < watch.launch_at:
            return
        watch.launch_at = float("inf")  # Reserve this attempt while the ADB command is in flight.
        ok = launcher.launch(self._manual_stop_requested)
        self._back_off_repair(watch)
        watch.launch_at = watch.repair_at
        if ok:
            self.context.gameplay_phase = GameplayPhase.FIRST
            self.context.strategy_endpoint = None
            self.context.relaunches += 1
            # Watch for readiness on every poll; give the new process time to appear before
            # another confirmed absence can authorize a retry.
            watch.launch_at = max(watch.launch_at,
                                  self._clock() + self._config.game_recovery.command_timeout)
        else:
            self.context.relaunch_failures += 1
        logger.log(logging.INFO if ok else logging.WARNING,
                   "Launch after confirmed process exit: package=%s accepted=%s",
                   launcher.package, ok)
        self._notifier.notify_game_relaunch(ok, max(0.0, self._clock() - watch.since) / 60)

    def _desktop_idle(self, watch: _ResumeWatch) -> bool:
        if self._manual_stop_requested.is_set() or self._resume is not watch:
            return False
        now = self._clock()
        cursor = self._input.get_cursor_position()
        if watch.cursor is not None and watch.cursor != cursor:
            watch.activity_at = now
        watch.cursor = cursor
        if watch.key_activity is not None:
            watch.activity_at = max(watch.activity_at, watch.key_activity())
        return now - watch.activity_at >= self._config.resume.idle_seconds

    def _back_off_repair(self, watch: _ResumeWatch) -> None:
        watch.attempts = min(watch.attempts + 1, 20)
        watch.repair_at = self._clock() + min(
            _RESUME_MAX_BACKOFF_SECONDS, 15.0 * 2 ** (watch.attempts - 1)
        )

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
