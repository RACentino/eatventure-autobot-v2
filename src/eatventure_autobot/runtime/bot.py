"""Orchestration: the assembly line. Every handler here does real I/O (capture, match, click,
sleep), packages what it observed, and hands the decision to a pure function in state/transitions.
No branching on flow state happens in this file that isn't delegated — if a handler starts deciding
where to go next on its own, that logic belongs in state/transitions instead.
"""

import dataclasses
import hashlib
import logging
import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np

from eatventure_autobot.domain.config import BotConfig
from eatventure_autobot.domain.errors import CaptureError, WindowNotAvailableError
from eatventure_autobot.domain.protocols import InputController, Notifier
from eatventure_autobot.domain.state import X_LOCKED_STATES, State
from eatventure_autobot.domain.types import MatchCandidate, Point, Zone
from eatventure_autobot.resilience.watchdog import StallWatchdog, WatchdogVerdict
from eatventure_autobot.runtime.game_recovery import GameLauncher
from eatventure_autobot.runtime.vision import GameVision
from eatventure_autobot.state import transitions as flow
from eatventure_autobot.state.context import FlowContext, LevelMark

logger = logging.getLogger(__name__)


def _config_fingerprint(config: BotConfig) -> str:
    """Short stable id of the tuning in effect, so a metrics line says which config produced it.
    paths and telegram are left out: one is machine-specific, the other holds a secret."""
    tuning = {
        field.name: getattr(config, field.name)
        for field in dataclasses.fields(config)
        if field.name not in ("paths", "telegram")
    }
    return hashlib.sha1(repr(tuning).encode(), usedforsecurity=False).hexdigest()[:8]


# The recovery window judges "is the screen alive" from its last few seconds.
_RECOVERY_TAIL_SECONDS = 5.0

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


def _top_close_position(positions: dict[Point, int]) -> str:
    """The most-tapped X position as 'x,yxCOUNT' text (e.g. 308,250x3698), or 'none': the unattended
    logs' only clue to which popup keeps coming back."""
    if not positions:
        return "none"
    (x, y), count = max(positions.items(), key=lambda item: item[1])
    return f"{x},{y}x{count}"


def _frame_thumbnail(frame: np.ndarray) -> np.ndarray:
    """64x36 grayscale subsample, enough to tell a frozen or dimmed view from a live one.
    ponytail: nearest-pixel sampling, not an area average; it only has to distinguish an identical
    frame (delta 0) from an animated one, so no cv2 call and any frame size works."""
    gray = frame.mean(axis=2) if frame.ndim == 3 else frame.astype(np.float64)
    rows = np.linspace(0, gray.shape[0] - 1, 36).astype(int)
    cols = np.linspace(0, gray.shape[1] - 1, 64).astype(int)
    return np.asarray(gray[rows][:, cols])


class EatventureBot:
    def __init__(
        self,
        config: BotConfig,
        vision: GameVision,
        input_controller: InputController,
        notifier: Notifier,
        stop_event: threading.Event | None = None,
        game_launcher: GameLauncher | None = None,
        desktop_notifier: Callable[[str, str], None] | None = None,
    ) -> None:
        self._config = config
        self._vision = vision
        self._input = input_controller
        self._notifier = notifier
        # (title, body) popup shown when the bot stops itself; None = off (tests, other platforms).
        self._desktop_notifier = desktop_notifier
        self._started_at = 0.0  # monotonic start() time of this run, for the self-stop line
        # Armed (not None) only while the bot is stopped after an environment self-stop and nobody
        # has restarted it since; see poll_resume.
        self._resume: _ResumeWatch | None = None
        self._stop_requested = stop_event or threading.Event()
        self._running = threading.Event()
        self._step_lock = threading.RLock()

        self.state = State.FIND_RED_ICONS
        self.context = FlowContext()
        self._watchdog = StallWatchdog(config=config.flow_timing)
        # ponytail: cumulative since process start, no rolling window; subtract consecutive
        # metrics lines for a per-period rate. Upgrade path: a windowed deque if that gets tedious.
        self._config_fingerprint = _config_fingerprint(config)
        self._state_seconds: dict[State, float] = dict.fromkeys(State, 0.0)
        # The state whose handler ran last step; self.state is already the NEXT one by the time the
        # popup pre-check runs, which is why the "Closed popup" line carries it as after=.
        self._last_handled_state: State | None = None
        self._last_metrics_at = 0.0
        # Where the last few boxes were clicked and how sure the match was, logged when the
        # dead-loop guard trips so a UI-fixed stuck target (same coordinates every pass) and a
        # weak-match false positive (low confidence) are both visible in bot.log.
        self._recent_box_clicks: deque[tuple[int, int, float]] = deque(maxlen=8)
        # Stall alert (see _maybe_alert_stall): when the idle streak began, and whether the next
        # OPEN_BOXES scan should log what box detection actually saw.
        self._idle_started_at = 0.0
        self._stall_probe_pending = False
        self._last_probe_thumb: np.ndarray | None = None
        # Dead-game detection: thumbnail of the first OPEN_BOXES scan of the current idle streak
        # (the "before" picture), the latest scan, and the one at the previous metrics line.
        # None launcher = recovery off (tests, or disabled in config); the thumbnails still feed
        # the stall probe and the frame_delta_5m metric.
        self._launcher = game_launcher
        self._package_probed = False
        self._streak_thumb: np.ndarray | None = None
        self._latest_thumb: np.ndarray | None = None
        self._metrics_thumb: np.ndarray | None = None
        self._relaunch_progress: tuple[int, int] | None = None
        # Red-X rule (see _dismiss_popup): where the last X was tapped and how many times in a row,
        # when the rule may act again after standing down, when the post-level lock ends, and the
        # time the pre-check has cost (capture + match), reported in the metrics line.
        self._close_last_pos: Point | None = None
        self._close_streak = 0
        self._close_suppressed_until = 0.0
        self._close_level_lock_until = 0.0
        self._close_check_seconds = 0.0
        # Clock for the recovery window (a test advances it from a fake sleep); everything else
        # keeps using time.monotonic directly.
        self._clock: Callable[[], float] = time.monotonic
        self._handlers: dict[State, Callable[[], State]] = {
            State.FIND_RED_ICONS: self._handle_find_red_icons,
            State.CLICK_RED_ICON: self._handle_click_red_icon,
            State.CHECK_UNLOCK: self._handle_check_unlock,
            State.SEARCH_UPGRADE_STATION: self._handle_search_upgrade_station,
            State.HOLD_UPGRADE_STATION: self._handle_hold_upgrade_station,
            State.UPGRADE_STATS: self._handle_upgrade_stats,
            State.OPEN_BOXES: self._handle_open_boxes,
            State.SCROLL: self._handle_scroll,
            State.CHECK_NEW_LEVEL: self._handle_check_new_level,
            State.TRANSITION_LEVEL: self._handle_transition_level,
            State.WAIT_FOR_UNLOCK: self._handle_wait_for_unlock,
        }
        missing = set(State) - self._handlers.keys()
        if missing:
            raise ValueError(f"Missing state handlers: {sorted(s.name for s in missing)}")

    # --- lifecycle ------------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running.is_set()

    def start(self, *, report: bool = True) -> bool:
        """report=False is for the auto-resume retries, which must not alert on every failed try."""
        with self._step_lock:
            if self.running:
                return True
            missing = self._vision.validate_required_templates()
            if missing:
                # v1's start gate, which v2 dropped: refuse rather than run half-blind.
                logger.error("Cannot start, missing required templates: %s", ", ".join(missing))
                return False
            self._stop_requested.clear()
            try:
                self._vision.ensure_target_ready()
            except CaptureError as exc:
                logger.error("Cannot start bot: %s", exc)
                if report:
                    self._report_self_stop(f"cannot start: {exc}")
                self._stop_requested.set()
                return False
            if not self._input.is_target_foreground():
                logger.error("Cannot start: '%s' is not foreground", self._config.window.title)
                if report:
                    self._report_self_stop("cannot start: target window is not foreground")
                self._stop_requested.set()
                return False
            if self._launcher is not None and not self._package_probed:
                self._package_probed = True  # once per process: adb is only touched at start
                self._launcher.detect_package()
            self._running.set()
            self._resume = None  # running again, by a person or by poll_resume
            self._watchdog.reset()
            self._started_at = self._last_metrics_at = time.monotonic()
            self.context.state_attempt = 1
            if self.context.current_level_start_time is None:
                self.context.current_level_start_time = time.monotonic()
            self._notifier.notify_bot_started()
            return True

    def request_stop(self) -> None:
        self._stop_requested.set()

    def stop(self, *, manual: bool = False) -> None:
        self.request_stop()
        with self._step_lock:
            was_running = self.running
            self._running.clear()
            self._input.release_left_button()
            self.state = State.FIND_RED_ICONS
            self.context.reset_run()
            self._watchdog.reset()
            if was_running and self._config.flow_timing.metrics_log_interval > 0:
                self._log_metrics()
        # v1 only sends "Bot Stopped" from the manual Z-toggle stop path; an internal auto-stop
        # (stop-requested race, lost foreground, an unhandled handler exception, or close()'s
        # own cleanup call) never notifies there, so it doesn't here either.
        if was_running and manual:
            self._notifier.notify_bot_stopped()

    def close(self) -> None:
        self.stop()
        self._notifier.close()
        self._vision.close()

    def set_event_forbidden_zone(self, zone: Zone) -> None:
        self._input.set_event_forbidden_zone(zone)

    def toggle_red_icon_mode(self) -> str:
        return self._vision.toggle_red_icon_mode()

    def cursor_position_in_window(self) -> Point | None:
        return self._vision.cursor_position_in_window(self._input)

    # --- step loop ------------------------------------------------------------------------

    def step(self) -> bool:
        if not self._step_lock.acquire(blocking=False):
            logger.warning("Ignoring reentrant bot step")
            return False
        try:
            if not self.running:
                return False
            if self._stop_requested.is_set():
                self.stop()
                return False
            self._vision.ensure_target_ready()
            if not self._input.is_target_foreground():
                logger.error("Target lost foreground; stopping bot")
                self._report_self_stop("target lost foreground")
                self.stop()
                self._arm_resume("target lost foreground")
                return False

            if self._dismiss_popup():
                return True  # a popup was closed this step; the handler runs on the next one

            previous_state = self.state
            started = time.monotonic()
            next_state = self._handlers[previous_state]()
            self._state_seconds[previous_state] += time.monotonic() - started
            self._last_handled_state = previous_state
            if previous_state in X_LOCKED_STATES:
                # Hard-lock layer 2: any New Level step (attempt, miss or completion) keeps the X
                # rule off afterwards, so a dialog still open when control returns is left to it.
                self._close_level_lock_until = max(
                    self._close_level_lock_until,
                    time.monotonic() + self._config.close_button.new_level_lock_seconds,
                )
            # v1's StateMachine.update() contract: a handler must return a State, and anything
            # else fails loudly (caught below -> stop) instead of being committed as flow state.
            if not isinstance(next_state, State):
                raise RuntimeError(f"{previous_state.name} handler returned {next_state!r}")
            if not self.running:
                return False
            if next_state != previous_state:
                logger.debug("State %s -> %s", previous_state.name, next_state.name)

            self.context.state_attempt = (
                self.context.state_attempt + 1 if next_state == previous_state else 1
            )
            self.state = next_state
            self._apply_watchdog(next_state)
            self._maybe_log_metrics()
            return True
        except (WindowNotAvailableError, CaptureError) as exc:
            logger.error("Stopping bot: %s", exc)
            self._report_self_stop(str(exc))
            self.stop()
            self._arm_resume(str(exc))
            return False
        except Exception:
            logger.exception("Stopping bot after an unexpected handler failure")
            self._report_self_stop("unexpected handler failure")
            self.stop()
            return False
        finally:
            self._step_lock.release()

    # --- auto-resume after an environment self-stop -----------------------------------------

    def _arm_resume(self, reason: str) -> None:
        """Only the two environment self-stops (window gone, focus lost) call this. A Z press, a
        stop request and a handler bug leave the decision to a person or a code fix, so they never
        arm it: the stopped-but-primed state alone cannot tell those apart."""
        now = self._clock()
        self._resume = _ResumeWatch(
            reason, now, now, now, now + self._config.resume.check_seconds
        )

    def poll_resume(self, last_key_at: float) -> None:
        """Called on every pass of the main loop while the bot is stopped; a no-op unless armed.
        last_key_at is the latest key press on this PC, on the same clock as self._clock.

        Shadow mode (resume.enabled False, the default): each check only logs whether it WOULD
        start the bot, so the share of downtime it could recover is measured first. Live mode: when
        the window is already in a state start() will not touch (GameVision.environment_ready: so
        no resize, no restore(), no focus change) and nobody has pressed a key or moved the mouse
        for resume.idle_seconds, it calls start(). A failed attempt backs off exponentially, and
        max_attempts failures in one stop make it give up and wait for a person.
        Never raises: the main loop has to survive whatever happens in here."""
        watch = self._resume
        if watch is None or self.running:
            return
        now = self._clock()
        if now < watch.sample_at:
            return
        watch.sample_at = now + _RESUME_SAMPLE_SECONDS
        try:
            cursor = self._input.get_cursor_position()
            # The first sample is only a baseline; a later change means a person moved the mouse.
            if watch.cursor is not None and cursor != watch.cursor:
                watch.activity_at = now
            watch.cursor = cursor
            if now >= watch.check_at:
                self._check_resume(watch, now, last_key_at)
        except Exception:
            logger.exception("Resume check failed")
            watch.check_at = now + self._config.resume.check_seconds

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
                go if ready and quiet else "waiting",
                self._vision.describe_environment(),
            )
        if not (config.enabled and ready and quiet):
            return
        watch.attempts += 1
        if self.start(report=False):
            logger.info(
                "Auto-resumed after %.0f min (attempt %d): %s",
                (now - watch.since) / 60,
                watch.attempts,
                watch.reason,
            )
            return
        if watch.attempts >= config.max_attempts:
            logger.warning(
                "Auto-resume gave up after %d attempts: %s", watch.attempts, watch.reason
            )
            self._resume = None
            return
        backoff = min(_RESUME_MAX_BACKOFF_SECONDS, config.check_seconds * 2**watch.attempts)
        watch.check_at = now + backoff

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

    def _dismiss_popup(self) -> bool:
        """Red-X priority: runs before the state handler, so it outranks every handler and asset.
        Returns True when it tapped an X (the step is then spent on that).

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
            return False
        now = time.monotonic()
        if now < self._close_level_lock_until or now < self._close_suppressed_until:
            return False
        frame = self._vision.capture()
        found = self._vision.find_close_buttons(frame)
        self._close_check_seconds += time.monotonic() - now
        if not found:
            self._close_last_pos, self._close_streak = None, 0
            return False
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
            return False
        self._close_streak = self._close_streak + 1 if same_spot else 1
        self._close_last_pos = (x, y)
        self._tap_close_button(target)
        return True

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

    def _recover_popups(self) -> None:
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
        tail: np.ndarray | None = None
        last: np.ndarray | None = None
        taps = capture_errors = 0
        while self._clock() < deadline and not self._stop_requested.is_set():
            remaining = deadline - self._clock()
            try:
                frame = self._vision.capture()
                found = self._vision.find_close_buttons(frame)
            except CaptureError as exc:
                capture_errors += 1
                if capture_errors == 1:
                    logger.warning("Recovery: capture failed (%s); retrying until it ends", exc)
                self._sleep(min(config.recovery_poll_seconds, remaining))
                continue
            last = _frame_thumbnail(frame)
            if tail is None and self._clock() >= tail_from:
                tail = last
            if found:
                taps += 1
                self._tap_close_button(found[0], remaining=remaining, recovery=True)
            else:
                self._sleep(min(config.recovery_poll_seconds, remaining))
        elapsed = self._clock() - started
        if tail is None or last is None:  # stopped early or never captured: no verdict to give
            logger.info("Recovery window ended after %.0f s: %d X tapped", elapsed, taps)
            return
        delta = float(np.abs(last - tail).mean())
        changing = delta >= self._config.game_recovery.static_frame_delta
        (logger.info if changing else logger.warning)(
            "Recovery window ended after %.0f s: %d X tapped, screen %s "
            "(frame delta %.2f over the last %.0f s)%s",
            elapsed,
            taps,
            "changing" if changing else "still static",
            delta,
            _RECOVERY_TAIL_SECONDS,
            "" if changing else "; the next stall alert will relaunch again",
        )

    def _apply_watchdog(self, current_state: State) -> None:
        verdict = self._watchdog.tick(current_state)
        if verdict is WatchdogVerdict.OK:
            return
        logger.warning("Resetting search flow (%s)", self._watchdog.last_reset_reason)
        self.context.reset_search_cycle()
        self.state = State.FIND_RED_ICONS
        self.context.state_attempt = 1

    def _maybe_log_metrics(self) -> None:
        interval = self._config.flow_timing.metrics_log_interval
        now = time.monotonic()
        if interval <= 0 or now - self._last_metrics_at < interval:
            return
        self._last_metrics_at = now
        self._log_metrics()

    def _log_metrics(self) -> None:
        """One parseable line: run_s is time spent inside handlers (not the 5 ms loop wake), and
        each state's share of it — where the run actually went."""
        total = sum(self._state_seconds.values())
        shares = " ".join(
            f"{state.name}={100 * seconds / total:.1f}%"
            for state, seconds in self._state_seconds.items()
            if seconds > 0
        )
        context = self.context
        logger.info(
            "metrics cfg=%s run_s=%.0f levels=%d holds=%d boxes=%d guard_trips=%d "
            "idle_scrolls=%d stall_alerts=%d hold_capped=%d hold_avg_s=%.1f nl_unlock_miss=%d "
            "relaunches=%d relaunch_failures=%d frame_delta_5m=%s close_clicks=%d "
            "close_suppressed=%d close_check_s=%.1f recovery_taps=%d close_top=%s | %s",
            self._config_fingerprint,
            total,
            context.total_levels_completed,
            context.holds_completed,
            context.boxes_opened_total,
            context.box_guard_trips,
            context.idle_scrolls,
            context.stall_alerts,
            context.holds_capped,
            context.hold_seconds_total / max(1, context.holds_completed),
            context.nl_unlock_miss,
            context.relaunches,
            context.relaunch_failures,
            self._frame_delta_since_last_metrics(),
            context.close_clicks,
            context.close_suppressed,
            self._close_check_seconds,
            context.recovery_taps,
            _top_close_position(context.close_positions),
            shares or "-",
        )

    def _frame_delta_since_last_metrics(self) -> str:
        """Mean-abs-diff between the latest scan and the one at the previous metrics line: the
        healthy-farm baseline that static_frame_delta is judged against (a dead game reads ~0)."""
        latest, before = self._latest_thumb, self._metrics_thumb
        self._metrics_thumb = latest
        if latest is None or before is None:
            return "n/a"
        return f"{np.abs(latest - before).mean():.1f}"

    def _maybe_alert_stall(self) -> None:
        """After every landed scroll: the pure counter (context.idle_scrolls) says how long the
        search has produced nothing. Every `stall_scrolls_before_alert` scrolls, log a WARNING and
        arm a one-shot probe of box detection on the next OPEN_BOXES frame, so a silent stall
        names its own cause. No extra capture, gesture or click, so no slowdown.
        ponytail: boxes only. A red-icon near-miss probe and a scroll re-anchor are deferred until
        a stall report shows they are the problem."""
        idle = self.context.idle_scrolls
        if idle == 1:
            self._idle_started_at = time.monotonic()
        every = self._config.scroll.stall_scrolls_before_alert
        if every <= 0 or idle % every != 0:
            return
        context = self.context
        context.stall_alerts += 1
        self._stall_probe_pending = True
        logger.warning(
            "Stall: %d scrolls over %.0f min with no box opened, upgrade held, stats upgrade or "
            "level completed (levels=%d holds=%d boxes=%d); probing box detection on the next scan",
            idle,
            (time.monotonic() - self._idle_started_at) / 60,
            context.total_levels_completed,
            context.holds_completed,
            context.boxes_opened_total,
        )

    def _log_box_near_misses(self, frame: np.ndarray, detected: int) -> None:
        """One text line: how many boxes passed detection on this frame, and for each box template
        its best raw match against the two gates a box must clear (template score, HSV ratio) and
        whether that spot is a forbidden zone. Tells a template/HSV miss from a zone block."""
        parts = []
        for seen in self._vision.box_near_misses(frame):
            ratio = "n/a" if seen.hsv_ratio is None else f"{seen.hsv_ratio:.2f}"
            zone = "" if self._is_clickable(seen.center) else " IN-FORBIDDEN-ZONE"
            parts.append(
                f"{seen.template_name} conf={seen.confidence:.3f} at {seen.center} "
                f"hsv={ratio}{zone}"
            )
        thumb = _frame_thumbnail(frame)
        previous, self._last_probe_thumb = self._last_probe_thumb, thumb
        # delta ~0 between alerts (minutes apart) = frozen stream or static overlay; the quadrant
        # means (TL TR BL BR) show a dimmed screen. Text only: no pixels leave the process.
        delta = "n/a" if previous is None else f"{np.abs(thumb - previous).mean():.2f}"
        quads = " ".join(
            f"{thumb[r:r + 18, c:c + 32].mean():.0f}" for r in (0, 18) for c in (0, 32)
        )
        logger.warning(
            "Stall probe: %d box(es) passed detection on this scan. Frame delta vs last probe="
            "%s, quadrant brightness=[%s]. Best raw match per template "
            "(a box needs conf>=%.3f and hsv>=%.2f, outside forbidden zones): %s",
            detected,
            delta,
            quads,
            self._config.thresholds.box,
            self._config.box.hsv.min_match_ratio,
            "; ".join(parts) or "none",
        )

    # --- shared helpers -------------------------------------------------------------------

    def _sleep(self, duration: float) -> bool:
        if duration <= 0:
            return not self._stop_requested.is_set()
        return not self._stop_requested.wait(duration)

    def _click_idle(self) -> bool:
        return self._input.click(*self._config.click_targets.idle_click_pos)

    def _scrcpy_recovery(self, delay: float) -> bool:
        if not self._config.scrcpy_recovery.enabled:
            return False
        return self._sleep(delay)

    def _is_clickable(self, point: Point) -> bool:
        return not self._input.is_in_forbidden_zone(*point)

    # --- handlers -------------------------------------------------------------------------

    def _handle_find_red_icons(self) -> State:
        if not self._click_idle():
            return State.FIND_RED_ICONS
        regions = self._config.capture_regions
        frame = self._vision.capture(max_y=regions.extended_search_y)
        # v1 checks the newLevel button against a frame limited to max_search_y, distinct from the
        # full extended_search_y frame used for the red-icon scan below.
        if self._vision.find_new_level_button(frame[: regions.max_search_y]).found:
            logger.info("New level button visible during red-icon scan")
            return flow.decide_find_red_icons(
                self.context, flow.RedIconScanObservation(True, False, ())
            )

        candidates = self._vision.find_red_icons(frame)
        actionable, new_level_seen, _ = self._vision.split_red_icons(candidates)
        # Verified v1 gate: retry only when neither an ordinary icon nor the footer new-level icon
        # was seen. A lone stats badge (classified out by split_red_icons) must not suppress it,
        # but a new-level badge does: that goes straight to CHECK_NEW_LEVEL.
        if (
            not actionable
            and not new_level_seen
            and self._scrcpy_recovery(self._config.scrcpy_recovery.red_icon_delay)
        ):
            frame = self._vision.capture(max_y=regions.extended_search_y)
            if self._vision.find_new_level_button(frame[: regions.max_search_y]).found:
                return flow.decide_find_red_icons(
                    self.context, flow.RedIconScanObservation(True, False, ())
                )
            candidates = self._vision.find_red_icons(frame)
            actionable, new_level_seen, _ = self._vision.split_red_icons(candidates)

        offset_x, offset_y = self._config.red_icon.offset
        clickable = tuple(
            candidate
            for candidate in actionable
            if self._is_clickable((candidate.center[0] + offset_x, candidate.center[1] + offset_y))
        )
        return flow.decide_find_red_icons(
            self.context, flow.RedIconScanObservation(False, new_level_seen, clickable)
        )

    def _handle_click_red_icon(self) -> State:
        target = flow.current_red_icon_target(self.context)
        if target is None:
            return State.OPEN_BOXES
        offset_x, offset_y = self._config.red_icon.offset
        x, y = target.center[0] + offset_x, target.center[1] + offset_y
        # Verified v1 behavior: a plain click() (with its post-click delay), not precise_click();
        # CHECK_UNLOCK already does the scrcpy settle when it opens.
        return flow.decide_click_red_icon(self.context, self._input.click(x, y))

    def _handle_check_unlock(self) -> State:
        if not self._sleep(self._config.scrcpy_recovery.action_settle_delay):
            return State.CHECK_UNLOCK
        frame = self._vision.capture(max_y=self._config.capture_regions.max_search_y)
        result = self._vision.find_unlock_button(frame)
        if result.best is None or not self._is_clickable(result.best.center):
            return flow.decide_check_unlock(False, None)
        clicked = self._input.click(*result.best.center)
        if clicked:
            self._sleep(self._config.level_transition.unlock_settle_delay)
        return flow.decide_check_unlock(True, clicked)

    def _handle_search_upgrade_station(self) -> State:
        station = self._config.upgrade_station
        attempt = self.context.state_attempt
        threshold = self._config.thresholds.upgrade_station
        if attempt > 2:
            threshold -= station.threshold_relaxation
        frame = self._vision.capture(max_y=self._config.capture_regions.upgrade_station_search_y)
        candidates = self._vision.find_upgrade_station_candidates(frame, threshold)
        # v1 picks the first candidate that isn't in a forbidden zone, not necessarily the
        # highest-confidence one — so a blocked best match doesn't hide a usable second one.
        position = next((c.center for c in candidates if self._is_clickable(c.center)), None)
        found = position is not None
        # Verified v1: no scrcpy-recovery delay in the search itself (only the hold monitor has
        # one), and no sleep before giving up on the last attempt.
        if not found and attempt < station.search_attempts:
            self._sleep(station.search_interval)
        return flow.decide_search_upgrade_station(
            self.context,
            station,
            flow.UpgradeStationSearchObservation(found, position, attempt),
        )

    def _handle_hold_upgrade_station(self) -> State:
        station = self._config.upgrade_station
        position = self.context.upgrade_station_pos
        if position is None or not self._is_clickable(position):
            return flow.decide_hold_upgrade_station(
                self.context,
                self._config.flow_timing,
                flow.UpgradeHoldObservation(False, False, False),
            )

        verified = self._verify_upgrade_station()
        if verified is None:
            return flow.decide_hold_upgrade_station(
                self.context,
                self._config.flow_timing,
                flow.UpgradeHoldObservation(True, False, False),
            )

        target, relaxed_threshold = verified
        check_interval = max(
            station.hold_check_interval_min,
            min(station.hold_check_interval_max, station.verify_search_interval),
        )
        hold_started = time.monotonic()
        held = self._input.hold_at(
            target[0],
            target[1],
            station.click_hold_max_duration,
            check_interval,
            self._make_disappearance_check(
                relaxed_threshold,
                target,
                station.disappear_confirmation_count,
            ),
        )
        if held:
            self._record_hold(time.monotonic() - hold_started)
        post_ok = (
            held
            and self._click_idle()
            and self._sleep(self._config.flow_timing.state_delay)
        )
        return flow.decide_hold_upgrade_station(
            self.context,
            self._config.flow_timing,
            flow.UpgradeHoldObservation(True, True, held, post_ok, verified_position=target),
        )

    def _record_hold(self, seconds: float) -> None:
        self.context.hold_seconds_total += seconds
        # Within one poll gap of the cap = the clock ran out, not the popup vanishing.
        if seconds >= self._config.upgrade_station.click_hold_max_duration - 0.1:
            self.context.holds_capped += 1

    def _verify_upgrade_station_round(self) -> tuple[Point, float] | None:
        """One full verify_search_attempts retry loop: base threshold on the first attempt,
        relaxed threshold afterward. Used for the single verification round in
        _verify_upgrade_station."""
        station = self._config.upgrade_station
        base = self._config.thresholds.upgrade_station
        relaxed = base - station.threshold_relaxation
        attempts = max(1, station.verify_search_attempts)
        for attempt in range(attempts):
            threshold = base if attempt == 0 else relaxed
            frame = self._vision.capture(
                max_y=self._config.capture_regions.upgrade_station_search_y
            )
            result = self._vision.find_upgrade_station(frame, threshold)
            if result.best is not None and self._is_clickable(result.best.center):
                return result.best.center, relaxed
            # v1 doesn't sleep before giving up on the last attempt.
            if attempt < attempts - 1 and not self._sleep(station.verify_search_interval):
                return None
        return None

    def _verify_upgrade_station(self) -> tuple[Point, float] | None:
        """Two-round pre-hold verification, matching v1's _verify_upgrade_station_hold_target
        exactly: a settle delay, then round 1 verifies the current on-screen state with no click
        at all; only if the station is still there does a priming click fire on the
        freshly-verified position (not the stale stored one), followed by another settle delay
        and a second round that must also confirm it before committing to a hold."""
        station = self._config.upgrade_station
        if not self._sleep(station.verify_settle_delay):
            return None

        first = self._verify_upgrade_station_round()
        if first is None:
            logger.info("Upgrade station was not visible during verification")
            return None

        target, _ = first
        if not self._input.precise_click(*target):
            logger.info("Upgrade station verification click failed at (%s, %s)", *target)
            return None
        if not self._sleep(station.verify_settle_delay):
            return None

        second = self._verify_upgrade_station_round()
        if second is None:
            logger.info("Upgrade station was not visible during verification")
            return None
        return second

    def _station_disappeared(self, threshold: float, position: Point) -> bool:
        """One poll tick's verdict. Verified v1 (_find_current_upgrade_hold_match): a first miss
        is only believed after one delayed scrcpy-recovery re-check, so a single dropped frame
        doesn't end the hold. Recovery disabled or its sleep interrupted means gone, as in v1.
        Debouncing across consecutive ticks is the caller's job (_make_disappearance_check)."""
        if self._station_seen_near(threshold, position):
            return False
        if not self._scrcpy_recovery(self._config.scrcpy_recovery.upgrade_delay):
            return True
        return not self._station_seen_near(threshold, position)

    def _station_seen_near(self, threshold: float, position: Point) -> bool:
        """Single capture+check at the relaxed threshold (the same leniency used for later
        verification attempts).

        A match is only accepted if it's near `position` (the position we're actually holding).
        Live-verified this matters: at the relaxed threshold, once the real station's popup
        changes to its "MAX" state late in a hold, the template can false-positive match an
        unrelated button elsewhere on screen (observed: the footer's "Boost x12" banner, ~389px
        away, at a confidence around/above the base threshold) — high enough that tightening the
        threshold alone would not reliably exclude it. Rejecting matches far from the held
        position catches this regardless of where the false positive happens to appear, without
        assuming the real popup is confined to any particular screen region."""
        station = self._config.upgrade_station
        frame = self._vision.capture(max_y=self._config.capture_regions.upgrade_station_search_y)
        result = self._vision.find_upgrade_station(frame, threshold)
        if not result.found or result.best is None:
            return False
        tolerance = station.disappear_position_tolerance_px
        dx = abs(result.best.center[0] - position[0])
        dy = abs(result.best.center[1] - position[1])
        return dx <= tolerance and dy <= tolerance

    def _make_disappearance_check(
        self, threshold: float, position: Point, required_misses: int
    ) -> Callable[[], bool]:
        """Requires `required_misses` CONSECUTIVE misses before treating the station as gone —
        debounces a single flaky/transient miss so a real hold isn't cut short by one bad frame."""
        misses = 0

        def check() -> bool:
            nonlocal misses
            gone = self._station_disappeared(threshold, position)
            misses = misses + 1 if gone else 0
            return misses >= max(1, required_misses)

        return check

    def _handle_upgrade_stats(self) -> State:
        # Verified v1 behavior: single-shot, no retry loop and no scrcpy-recovery here. This is
        # also the one state where an idle-click failure does NOT retry itself.
        if not self._click_idle():
            return State.OPEN_BOXES
        frame = self._vision.capture(max_y=self._config.capture_regions.extended_search_y)
        if self._vision.find_new_level_button(
            frame[: self._config.capture_regions.max_search_y]
        ).found:
            return flow.decide_upgrade_stats(self.context, flow.StatsIconObservation(True, False))

        # Reuses the shared red-icon scan; split_red_icons applies the stats-icon threshold.
        candidates = self._vision.find_red_icons(frame)
        _, _, stats_seen = self._vision.split_red_icons(candidates)
        if not stats_seen:
            return flow.decide_upgrade_stats(self.context, flow.StatsIconObservation(False, False))

        targets = self._config.click_targets
        button_clicked = self._input.click(*targets.stats_upgrade_button_pos)
        if button_clicked and self._sleep(self._config.flow_timing.state_delay):
            # stats_upgrade_pos (290, 310) sits inside the 2-/3-event forbidden zones
            # (ForbiddenZoneConfig.event_zone_options) meant to protect the event banner from
            # stray clicks elsewhere in the flow. By the time we click here the stats panel is
            # already open and covers that banner, so the protection is a false positive for
            # this one call — bypass it rather than shrinking the zones or moving this
            # confirmed-correct click target.
            if self._input.spam_click_at(
                *targets.stats_upgrade_pos,
                self._config.stats_upgrade.click_duration,
                self._config.stats_upgrade.click_delay,
                down_duration=self._config.stats_upgrade.mouse_down_duration,
                up_duration=self._config.stats_upgrade.mouse_up_duration,
                check_forbidden=False,
            ):
                self._click_idle()
            else:
                logger.warning("Stats upgrade spam-click failed at %s", targets.stats_upgrade_pos)
        return flow.decide_upgrade_stats(self.context, flow.StatsIconObservation(False, True))

    def _handle_open_boxes(self) -> State:
        if not self._click_idle():
            return State.OPEN_BOXES
        box_search_y = self._config.capture_regions.box_search_y
        frame = self._vision.capture(max_y=box_search_y)
        thumb = _frame_thumbnail(frame)
        self._latest_thumb = thumb
        if self.context.idle_scrolls == 1:  # first scan after the first scroll of an idle streak
            self._streak_thumb = thumb
        if self._vision.find_new_level_button(frame).found:
            logger.info("New level found while opening boxes")
            return flow.decide_open_boxes(
                self.context,
                self._config.upgrade_station,
                self._config.scroll.max_idle_pass_attempts,
                flow.BoxCycleObservation(True),
            )

        boxes = self._vision.find_boxes(frame)
        if not boxes and self._scrcpy_recovery(self._config.scrcpy_recovery.box_delay):
            frame = self._vision.capture(max_y=box_search_y)
            if self._vision.find_new_level_button(frame).found:
                return flow.decide_open_boxes(
                    self.context,
                    self._config.upgrade_station,
                    self._config.scroll.max_idle_pass_attempts,
                    flow.BoxCycleObservation(True),
                )
            boxes = self._vision.find_boxes(frame)

        if self._stall_probe_pending:
            self._stall_probe_pending = False
            self._log_box_near_misses(frame, len(boxes))
            if self._game_looks_dead(_frame_thumbnail(frame)):
                return self._relaunch_game()

        opened = 0
        for box in boxes:
            if not self._is_clickable(box.center):
                continue
            if self._input.click(*box.center):
                opened += 1
                self._recent_box_clicks.append((*box.center, box.confidence))
        if opened:
            logger.info("Opened %s boxes", opened)
        trips = self.context.box_guard_trips
        next_state = flow.decide_open_boxes(
            self.context,
            self._config.upgrade_station,
            self._config.scroll.max_idle_pass_attempts,
            flow.BoxCycleObservation(False, opened),
        )
        if self.context.box_guard_trips != trips:
            logger.warning(
                "Box loop guard: %s box passes without a scroll or upgrade; forcing a scroll. "
                "Last clicked positions: %s",
                self._config.upgrade_station.max_box_only_passes,
                [(x, y, round(conf, 3)) for x, y, conf in self._recent_box_clicks],
            )
        return next_state

    def _game_looks_dead(self, thumb: np.ndarray) -> bool:
        """Runs only on a stall probe (>= stall_scrolls_before_alert idle scrolls, ~7.5 min): the
        picture has not changed since the streak began. Healthy idle streaks peak near 66 scrolls
        (p99 of 671 metrics samples), so the scroll count alone is no evidence; the frame is."""
        before = self._streak_thumb
        if before is None or self._launcher is None:  # no launcher = recovery switched off
            return False
        limit = self._config.game_recovery.static_frame_delta
        delta = float(np.abs(thumb - before).mean())
        static = delta < limit
        # The decision's own inputs: the stall probe's "Frame delta vs last probe" compares a
        # different pair of frames, so on its own it cannot tell whether a relaunch was justified.
        logger.info(
            "Dead-game check: frame delta vs streak start=%.2f (static below %.2f), "
            "idle_scrolls=%d -> %s",
            delta,
            limit,
            self.context.idle_scrolls,
            "static" if static else "alive",
        )
        if static and self._launcher.package is None:
            logger.warning("Game appears dead (frame unchanged) but adb relaunch is not available")
            return False
        return static

    def _relaunch_game(self) -> State:
        """adb force-stop + launch, wait out the cold start, and restart the search from scratch.
        No cap on repeats (by choice): the next chance is another full stall alert, ~8 min away."""
        launcher, context = self._launcher, self.context
        assert launcher is not None  # _game_looks_dead only says yes when a launcher is armed
        progress = (context.holds_completed, context.boxes_opened_total)
        again = ""
        if progress == self._relaunch_progress:
            again = " (still static after the previous relaunch; the scrcpy window may be frozen)"
        logger.warning(
            "Game appears dead: frame unchanged for %.0f min%s; relaunching %s",
            (time.monotonic() - self._idle_started_at) / 60,
            again,
            launcher.package,
        )
        self._relaunch_progress = progress
        minutes = (time.monotonic() - self._idle_started_at) / 60
        if launcher.relaunch():
            context.relaunches += 1
            self._notifier.notify_game_relaunch(True, minutes)
            self._recover_popups()  # exclusive X-only window; the main flow resumes after it
        else:
            # Nothing restarted, so there is nothing to wait for: log (done by the launcher) and
            # carry on with the main flow as before.
            context.relaunch_failures += 1
            self._notifier.notify_game_relaunch(False, minutes)
        context.idle_scrolls = 0
        context.reset_search_cycle()
        self._streak_thumb = None
        self._last_probe_thumb = None
        # The restarted game starts the X rule from scratch: no stale stand-down or lock window.
        self._close_last_pos, self._close_streak = None, 0
        self._close_suppressed_until = self._close_level_lock_until = 0.0
        return State.FIND_RED_ICONS

    def _handle_scroll(self) -> State:
        if not self._click_idle():
            return State.SCROLL
        scroll = self._config.scroll
        distance = round(scroll.pixel_step * scroll.distance_ratio)
        start_x, start_y = self._config.click_targets.scroll_start_pos
        direction = 1 if self.context.oscillation_leg_direction > 0 else -1
        target_y = start_y - distance if direction > 0 else start_y + distance
        logger.info(
            "Oscillating scroll: cycle=%s direction=%s",
            self.context.oscillation_cycle_index,
            "down" if direction > 0 else "up",
        )
        moved = self._input.drag(start_x, start_y, start_x, target_y, duration=scroll.duration)
        if moved:
            self.context.advance_oscillation(scroll.increment_step, scroll.max_cycles)
            moved = self._sleep(scroll.post_scroll_settle) and self._sleep(scroll.interval_pause)
        next_state = flow.decide_scroll(self.context, moved)
        if moved:  # decide_scroll only advances the idle streak on a landed scroll
            self._maybe_alert_stall()
        return next_state

    def _perform_new_level_verification_scroll(self) -> bool:
        """Restored from v1 (73f5db0/eccd810): one mouse-drag "scroll" gesture used solely to
        force a fresh render before re-scanning for the new-level red icon. Reuses the shared
        scroll_start_pos origin; distance/duration/settle timing are dedicated to this step on
        LevelTransitionConfig, independent of the oscillating ScrollConfig used by _handle_scroll.
        """
        level = self._config.level_transition
        start_x, start_y = self._config.click_targets.scroll_start_pos
        # Dragging from start_y to start_y - distance scrolls the CONTENT DOWN (finger moves up).
        target_y = start_y - level.verification_scroll_distance
        if not self._input.drag(
            start_x, start_y, start_x, target_y, duration=level.verification_scroll_duration
        ):
            return False
        if not self._sleep(level.verification_scroll_settle_delay):
            return False
        return self._sleep(level.verification_scroll_interval_pause)

    def _handle_check_new_level(self) -> State:
        # Verified v1 behavior: no attempt caps and no scrcpy-recovery here — a verify miss resets
        # immediately (single-shot), and the click-retry tail is an unbounded, watchdog-only loop.
        if self.context.nl_first_mark is None:  # every sighting routes here; first one per level
            self.context.nl_first_mark = self._level_mark()
        if not self._click_idle():
            return State.CHECK_NEW_LEVEL
        if not self._sleep(self._config.flow_timing.focus_settle_delay):
            return State.CHECK_NEW_LEVEL

        if not self.context.new_level_red_icon_verified:
            if not self._perform_new_level_verification_scroll():
                logger.warning("Failed to perform verification scroll for new level red icon")
                return flow.decide_check_new_level(
                    self.context,
                    flow.NewLevelVerificationObservation(False, None, None, scroll_succeeded=False),
                )
            frame = self._vision.capture(max_y=self._config.capture_regions.extended_search_y)
            candidates = self._vision.find_red_icons(frame)
            _, verified, _ = self._vision.split_red_icons(candidates)
            if not verified:
                return flow.decide_check_new_level(
                    self.context, flow.NewLevelVerificationObservation(False, None, None)
                )

        targets = self._config.click_targets
        level = self._config.level_transition
        button_ok = self._input.click(*targets.new_level_button_pos)
        if button_ok and not self._sleep(level.confirmation_delay):
            return State.CHECK_NEW_LEVEL
        transition_ok: bool | None = None
        if button_ok:
            transition_ok = self._input.click(*targets.level_transition_pos)
            if transition_ok:
                self._sleep(level.secondary_settle_delay)
        return flow.decide_check_new_level(
            self.context,
            flow.NewLevelVerificationObservation(True, button_ok, transition_ok),
        )

    def _handle_transition_level(self) -> State:
        level = self._config.level_transition
        attempt = self.context.state_attempt
        # Verified v1: one idle click ahead of the whole search (not one per capture), and a failed
        # idle click retries without consuming a search attempt.
        if attempt == 1 and not self._click_idle():
            self.context.refund_attempt()
            return State.TRANSITION_LEVEL
        # v1 captures at max_search_y for this state's search loop, not extended_search_y.
        frame = self._vision.capture(max_y=self._config.capture_regions.max_search_y)
        result = self._vision.find_new_level_button(frame)
        # No forbidden-zone pre-gate: like v1, a forbidden match is handed to click(), which
        # rejects it, and the failed click routes to CHECK_NEW_LEVEL.
        if result.best is None:
            # v1 doesn't sleep before giving up on the last attempt.
            if attempt < level.search_attempts:
                self._sleep(level.search_interval)
            return flow.decide_transition_level(
                self.context, level, flow.TransitionLevelObservation(False, attempt, None)
            )
        clicked = self._input.click(*result.best.center)
        if clicked:
            self._sleep(level.settle_delay)
        return flow.decide_transition_level(
            self.context, level, flow.TransitionLevelObservation(True, attempt, clicked)
        )

    def _handle_wait_for_unlock(self) -> State:
        if not self._click_idle():
            return State.WAIT_FOR_UNLOCK
        if not self._sleep(self._config.flow_timing.focus_settle_delay):
            return State.WAIT_FOR_UNLOCK
        level = self._config.level_transition
        # v1 captures the full window height here (no max_y limit) — distinct from CHECK_UNLOCK,
        # which uses the same "unlock" template against a max_search_y-limited frame.
        frame = self._vision.capture()
        result = self._vision.find_unlock_button(frame)
        if result.best is None or not self._is_clickable(result.best.center):
            self._sleep(level.unlock_search_interval)
            missed = flow.decide_wait_for_unlock(
                self.context, level, flow.WaitForUnlockObservation(False, None)
            )
            if missed != State.WAIT_FOR_UNLOCK:  # attempts exhausted: back to the economy loop
                self._note_unlock_miss(frame)
            return missed

        clicked = self._input.click(*result.best.center)
        previous_total = self.context.total_levels_completed
        next_state = flow.decide_wait_for_unlock(
            self.context, level, flow.WaitForUnlockObservation(True, clicked)
        )
        if (
            self.context.total_levels_completed == previous_total
            and next_state != State.WAIT_FOR_UNLOCK
        ):
            self._note_unlock_miss(frame)
        if self.context.total_levels_completed > previous_total:
            self._report_level_completion()
            # v1 sleeps here regardless of outcome before returning to FIND_RED_ICONS.
            self._sleep(level.unlock_settle_delay)
        return next_state

    def _note_unlock_miss(self, frame: np.ndarray) -> None:
        self.context.nl_unlock_miss += 1
        # best_conf = the raw best `unlock` match, threshold not applied: low = the button is greyed
        # out (cash is the limit), near the threshold = an appearance problem. Diagnostics only.
        best = self._vision.unlock_near_miss(frame)
        logger.info(
            "New level: unlock button not found (miss %d, levels=%d, best_conf=%s)",
            self.context.nl_unlock_miss,
            self.context.total_levels_completed,
            "-" if best is None else f"{best.confidence:.3f}",
        )

    def _level_mark(self) -> LevelMark:
        context = self.context
        return LevelMark(
            sum(self._state_seconds.values()),
            context.holds_completed,
            context.boxes_opened_total,
            context.nl_unlock_miss,
        )

    def _report_level_completion(self) -> None:
        now = time.monotonic()
        started = self.context.current_level_start_time
        elapsed = now - started if started is not None else 0.0
        self.context.current_level_start_time = now
        logger.info(
            "Restaurant %s completed in %.1fs", self.context.total_levels_completed, elapsed
        )
        # Breakdown in RUNNING seconds (the line above is wall-clock and includes stopped time).
        # "before" = up to the first New Level sighting, "after" = from it to the completion; a
        # level with no recorded sighting counts wholly as "before". The running time of the handler
        # that is completing the level is not in _state_seconds yet (a few seconds at most).
        context = self.context
        start, end = context.level_start_mark, self._level_mark()
        seen = context.nl_first_mark or end
        context.level_start_mark, context.nl_first_mark = end, None
        logger.info(
            "Level %d breakdown: run_s=%.0f holds=%d boxes=%d unlock_miss=%d | "
            "before_nl: run_s=%.0f holds=%d boxes=%d | after_nl: run_s=%.0f holds=%d boxes=%d",
            context.total_levels_completed,
            end.run_s - start.run_s,
            end.holds - start.holds,
            end.boxes - start.boxes,
            end.unlock_miss - start.unlock_miss,
            seen.run_s - start.run_s,
            seen.holds - start.holds,
            seen.boxes - start.boxes,
            end.run_s - seen.run_s,
            end.holds - seen.holds,
            end.boxes - seen.boxes,
        )
        self._notifier.notify_new_level(self.context.total_levels_completed, elapsed)
