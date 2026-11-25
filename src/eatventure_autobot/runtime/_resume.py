"""Auto-resume after an environment self-stop.

Moved out of bot.py unchanged; EatventureBot inherits it. What it reaches on self is declared
at the top of the class."""

import dataclasses
import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from eatventure_autobot.domain.config import BotConfig
from eatventure_autobot.domain.errors import CaptureError
from eatventure_autobot.domain.protocols import InputController, Notifier
from eatventure_autobot.domain.state import PauseReason, State
from eatventure_autobot.domain.types import Point
from eatventure_autobot.runtime._recovery_probe import (
    ScreenProbe,
    same_game_area,
    scroll_response,
    scroll_surface_current,
)
from eatventure_autobot.runtime.game_recovery import GameLauncher
from eatventure_autobot.runtime.progress import ProgressLedger
from eatventure_autobot.runtime.vision import GameVision

logger = logging.getLogger("eatventure_autobot.runtime.bot")

# Auto-resume (see EatventureBot.poll_resume): how often the cursor is sampled for movement, how
# often an unchanged check is logged anyway, and the longest wait between failed live attempts.
_RESUME_SAMPLE_SECONDS = 1.0
_RESUME_HEARTBEAT_SECONDS = 600.0
_RESUME_MAX_BACKOFF_SECONDS = 900.0
_RECOVERY_PROBE_SECONDS = 15.0
_RECOVERY_SCROLL_SETTLE = .65  # Recorded camera drift lasted up to 602 ms after drag release.


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
    logged: tuple[bool, bool] | None = None  # (ready, quiet) of the last logged check
    logged_at: float = 0.0
    repair_at: float = 0.0
    evidence: str | None = None
    action_screen: str | None = None
    action_at: float = 0.0
    action_signature: np.ndarray | None = None
    checked_world: np.ndarray | None = None


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
    _verify_recovered_unlock: bool
    _recovery_restart_used: bool
    _phone_game_foreground: bool | None
    _idle_started_at: float
    state: State

    if TYPE_CHECKING:  # annotation only: nothing here exists at runtime

        @property
        def running(self) -> bool:
            ...

        def start(self, *, report: bool=True) -> bool:
            ...

        def _device_available(self, *, force: bool = False) -> bool: ...
        def _relaunch_game(self, *, cold: bool = True) -> State: ...
        def _probe_recovery_screen(self, *, after_scroll: bool = False) -> ScreenProbe: ...
        def _sleep(self, duration: float) -> bool: ...
        def _continue_city(self, frame: np.ndarray, *, recovering: bool = False) -> bool: ...
        def _handle_transition_level(self, *, require_open: bool = False) -> State: ...

        def _restart_mirror(self, cancelled: threading.Event) -> bool: ...

        def _bind_existing_mirror(self) -> bool: ...

    def _arm_resume(self, reason: str) -> None:
        """Retry known environment or screen-observation failures. Manual/disconnected stops
        remain latched; unexpected handler failures never call this path."""
        if self._manual_stop_requested.is_set():
            return
        self._device_available(force=True)
        if self._manual_stop_requested.is_set() or self.pause_reason == PauseReason.DISCONNECTED:
            return
        self.pause_reason = PauseReason.RECOVERY
        if self._progress is not None:
            self._progress.checkpoint("recovery")
        now = self._clock()
        self._resume = _ResumeWatch(
            reason, now, now, now, now + self._config.resume.check_seconds
        )

    def poll_resume(self, last_key_at: float | Callable[[], float]) -> None:
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
            if (not self._game_recovery_pending and self._config.game_recovery.enabled
                    and self._launcher is not None and self._launcher.package is not None
                    and self._phone_game_foreground is not True):
                self._game_recovery_pending = True
                watch.check_at = min(watch.check_at, now)
            watch.key_activity = last_key_at if callable(last_key_at) else None
            key_at = last_key_at() if callable(last_key_at) else last_key_at
            watch.activity_at = max(watch.activity_at, key_at)
            if now >= watch.check_at:
                self._check_resume(watch, now, key_at)
        except CaptureError:
            delay = (_RECOVERY_PROBE_SECONDS if self._game_recovery_pending
                     else self._config.resume.check_seconds)
            watch.check_at = self._clock() + delay
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
        if self._game_recovery_pending and config.enabled:
            self._check_game_recovery(watch, now, ready, quiet, last_key_at)
            return
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
        if self.start(report=False):
            logger.info(
                "Auto-resumed after %.0f min (attempt %d): %s",
                (now - watch.since) / 60,
                watch.attempts,
                watch.reason,
            )
            return

    def _check_game_recovery(
        self, watch: _ResumeWatch, now: float, ready: bool, quiet: bool, last_key_at: float,
    ) -> None:
        """Observe on a fixed cadence; only repairs incur exponential backoff."""
        launcher = self._launcher
        if launcher is None or self._resume is not watch or self._manual_stop_requested.is_set():
            return
        watch.check_at = now + _RECOVERY_PROBE_SECONDS
        if not quiet:
            watch.check_at = min(watch.check_at, max(watch.activity_at, last_key_at)
                                 + self._config.resume.idle_seconds)
        can_repair = quiet and now >= watch.repair_at
        if not ready:
            if not can_repair:
                return
            if not launcher.mirror_bound():
                self._bind_existing_mirror()
            if not self._recovery_idle():
                return
            if not self._vision.environment_ready() and not (
                self._config.resume.raise_window and self._vision.raise_target()
            ):
                try:
                    self._vision.ensure_target_ready()
                except CaptureError:
                    self._back_off_repair(watch)
                    self._restart_mirror(self._manual_stop_requested)
                return
        if launcher.package is None and now >= watch.repair_at:
            self._back_off_repair(watch)
            if launcher.detect_package():
                self._device_available(force=True)
        foreground = self._phone_game_foreground
        if foreground is None or launcher.package is None:
            return
        view = self._probe_recovery_screen() if foreground else ScreenProbe("unknown")
        evidence = f"{view.status}/{view.screen}" if foreground else "outside game"
        if evidence != watch.evidence:
            logger.info("Recovery observation trigger=%s evidence=%s elapsed_s=%.0f",
                        watch.reason, evidence, now - watch.since)
            watch.evidence = evidence
        if foreground and (view.status == "unknown" or (view.status == "still"
                and self._recovery_restart_used and view.world_views is None)):
            return
        if (not quiet or self._manual_stop_requested.is_set() or self._resume is not watch
                or not self._device_available(force=True)
                or self._phone_game_foreground is not foreground
                or not self._recovery_input_allowed()):
            return
        # The normal stop event also interrupts handlers and ADB between commands.
        self._stop_requested.clear()
        try:
            if self._manual_stop_requested.is_set() or self._resume is not watch:
                return
            if not foreground:
                if can_repair and launcher.game_exited():
                    self._back_off_repair(watch)
                    self._relaunch_game(cold=False)
                return
            recovery = self._config.game_recovery
            if (view.status == "still" and view.world_views is not None and can_repair
                    and watch.action_screen is None and view.signature is not None
                    and not same_game_area(watch.checked_world, view.signature,
                                           3 * recovery.static_recent_delta)
                    and now - self._idle_started_at >= recovery.stale_after_seconds):
                view = self._verify_recovery_scroll(watch, view)
                if view.status == "unknown":
                    self._back_off_repair(watch)
                    return
            if view.status == "mirror-stale":
                if can_repair:
                    self._back_off_repair(watch)
                    logger.warning("Recovery action=repair-mirror elapsed_s=%.0f",
                                   now - watch.since)
                    if self._restart_mirror(self._stop_requested):
                        watch.checked_world = None
                return
            if view.status == "playable":
                frozen = self._check_playable_screen(watch, view, now)
            elif view.status == "still":
                frozen = not self._recovery_restart_used and launcher.game_unresponsive()
            else:
                return
            if (frozen and can_repair and not self._recovery_restart_used
                    and now - self._idle_started_at >=
                    self._config.game_recovery.stale_after_seconds
                    and not self._manual_stop_requested.is_set()):
                proof = ("Android reports game unresponsive" if view.status == "still"
                         else f"{view.screen} unchanged after accepted tap")
                logger.info("Recovery freeze verified evidence=%s elapsed_s=%.0f",
                            proof, now - watch.since)
                self._back_off_repair(watch)
                self._relaunch_game()
        finally:
            if not self.running:
                self._stop_requested.set()
            # A 38-second cold start or slow ADB command must not cause catch-up polls.
            if self._clock() > now + _RECOVERY_PROBE_SECONDS:
                watch.check_at = self._clock() + _RECOVERY_PROBE_SECONDS

    def _verify_recovery_scroll(self, watch: _ResumeWatch, before: ScreenProbe) -> ScreenProbe:
        scroll = self._config.scroll
        x, y = self._config.click_targets.scroll_start_pos
        distance = max(1, round(scroll.pixel_step * scroll.distance_ratio) // 2)
        window = self._config.window
        bottom = min(window.height, self._config.capture_regions.max_search_y)
        for direction in (-1, 1):
            end_y = y + direction * distance
            low, high = sorted((y, end_y))
            if (not 0 <= x < window.width or not 0 <= low < high < bottom
                    or any(self._input.is_in_forbidden_zone(x, row) for row in range(low, high + 1))
                    or not scroll_surface_current(self._vision, self._config, before)
                    or not self._scroll_input_ready(watch)):
                return ScreenProbe("unknown")
            if not self._input.drag(x, y, x, end_y, duration=scroll.duration):
                return ScreenProbe("unknown")
            self._note_recovery_click((x, end_y))
            if not self._sleep(max(_RECOVERY_SCROLL_SETTLE,
                                   scroll.post_scroll_settle + scroll.interval_pause)):
                return ScreenProbe("unknown")
            after = self._probe_recovery_screen(after_scroll=True)
            if not self._scroll_input_ready(watch):
                return ScreenProbe("unknown")
            result = scroll_response(self._vision, before, after, direction)
            logger.info("Recovery scroll drag_dy=%d evidence=%s", end_y - y, result.status)
            if result.status != "still":
                return result
            before = after  # One direction can be blocked at a healthy camera edge.
        watch.checked_world = before.signature
        return before

    def _scroll_input_ready(self, watch: _ResumeWatch) -> bool:
        return (self._resume is watch and not self._stop_requested.is_set()
                and self._launcher is not None and self._launcher.mirror_bound()
                and self._device_available(force=True) and self._phone_game_foreground is True
                and not self._stop_requested.is_set() and self._recovery_input_allowed())

    def _check_playable_screen(self, watch: _ResumeWatch, view: ScreenProbe, now: float) -> bool:
        """Return freeze evidence only for a stationary control whose accepted tap had no effect."""
        if watch.action_screen is not None and watch.action_screen != view.screen:
            self._recovery_restart_used = False
        if view.screen in (None, "unlock"):
            if self.start(report=False):
                self._game_recovery_pending = False
                self._verify_recovered_unlock = view.screen == "unlock"
                if self._verify_recovered_unlock:
                    self.state = State.WAIT_FOR_UNLOCK
                logger.info("Recovered gameplay after %.0f s", self._clock() - watch.since)
            return False
        if watch.action_screen != view.screen:
            self._try_dialog_once(watch, view)
            return False
        if not view.static or not same_game_area(
            watch.action_signature, view.signature, self._config.game_recovery.static_frame_delta,
        ):
            watch.action_signature = None
            return False
        return now - watch.action_at >= self._config.game_recovery.stale_after_seconds

    def _try_dialog_once(self, watch: _ResumeWatch, view: ScreenProbe) -> None:
        if view.screen == "welcome":
            frame = self._vision.capture()
            if (not self._vision.city_arrival(frame)
                    or not self._continue_city(frame, recovering=True)):
                return
        elif view.screen == "open":
            if self._handle_transition_level(require_open=True) != State.WAIT_FOR_UNLOCK:
                return
        else:
            return
        watch.action_screen = view.screen
        watch.action_signature = view.signature
        watch.action_at = self._clock()
        logger.info("Recovery action=guarded-dialog screen=%s", view.screen)

    def _note_recovery_click(self, point: Point) -> None:
        # The controller moves the desktop cursor to click. Only exempt that exact position.
        watch = self._resume
        if watch is not None:
            position = self._input.get_cursor_position()
            if self._vision.cursor_position_in_window(self._input) == point:
                watch.cursor = position

    def _clear_recovery_dialog(self) -> None:
        if self._resume is not None:
            self._resume.action_screen = None
            self._resume.action_signature = None
            self._resume.checked_world = None  # The launcher confirmed a new app instance.

    def _recovery_input_allowed(self) -> bool:
        return self._input.is_target_foreground() and self._recovery_idle()

    def _recovery_idle(self) -> bool:
        if self._manual_stop_requested.is_set():
            return False
        watch = self._resume
        if watch is None:
            return True
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
            _RESUME_MAX_BACKOFF_SECONDS, _RECOVERY_PROBE_SECONDS * 2 ** (watch.attempts - 1)
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
