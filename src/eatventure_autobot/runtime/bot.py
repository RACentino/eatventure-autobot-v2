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
from eatventure_autobot.domain.types import Point, Zone
from eatventure_autobot.resilience.watchdog import StallWatchdog
from eatventure_autobot.runtime._handlers import _HandlersMixin
from eatventure_autobot.runtime._recovery import _RecoveryMixin
from eatventure_autobot.runtime._resume import _ResumeMixin, _ResumeWatch
from eatventure_autobot.runtime._telemetry import (  # noqa: F401  (re-exported: tests import them)
    _format_near_misses,
    _frame_thumbnail,
    _nl_routes,
    _popup_summary,
    _TelemetryMixin,
    _top_close_position,
)
from eatventure_autobot.runtime.game_recovery import GameLauncher
from eatventure_autobot.runtime.vision import GameVision
from eatventure_autobot.state.context import FlowContext

logger = logging.getLogger(__name__)


# Names the state flow in effect. Bump it when the flow changes without a config change: the
# fingerprint below hashes config only, so metrics lines from before and after would otherwise
# carry the same cfg= id and could not be told apart.
FLOW_REVISION = (
    "no-work: FIND>STATS>OPEN>SCROLL>OPEN>FIND; unlock: greyed early-out; box flood cap"
)


def _config_fingerprint(config: BotConfig) -> str:
    """Short stable id of the tuning in effect, so a metrics line says which config produced it.
    paths and telegram are left out: one is machine-specific, the other holds a secret."""
    tuning = {
        field.name: getattr(config, field.name)
        for field in dataclasses.fields(config)
        if field.name not in ("paths", "telegram")
    }
    tuning["flow_revision"] = FLOW_REVISION
    return hashlib.sha1(repr(tuning).encode(), usedforsecurity=False).hexdigest()[:8]


class EatventureBot(_TelemetryMixin, _RecoveryMixin, _ResumeMixin, _HandlersMixin):
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
        # Handler calls per state (a self-loop counts every pass): a state's seconds divided by it
        # is the cost per entry, and the OPEN_BOXES count gives the crate-scan cadence.
        self._state_calls: dict[State, int] = dict.fromkeys(State, 0)
        # When the last OPEN_BOXES scan was captured (None after a start), for the box_gap_ms tally.
        self._last_box_scan_at: float | None = None
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
        # Early probe (see _maybe_alert_stall): armed once per idle streak and logged on the next
        # OPEN_BOXES scan. _scan_delta is how far that scan's picture is from the previous scan's
        # (taken before the last drag): near the animation-only level means the drag moved nothing.
        self._early_probe_pending = False
        self._scan_delta: float | None = None
        # Dead-game detection: thumbnail of the first OPEN_BOXES scan of the current idle streak
        # (the "before" picture), the latest scan, and the one at the previous metrics line.
        # None launcher = recovery off (tests, or disabled in config); the thumbnails still feed
        # the stall probe and the frame_delta_5m metric.
        self._launcher = game_launcher
        self._package_probed = False
        self._streak_thumb: np.ndarray | None = None
        # Armed by the first landed scroll of an idle streak (_maybe_alert_stall); the next
        # OPEN_BOXES scan takes the "before" picture and disarms it, whatever else happened between.
        self._streak_anchor_pending = False
        # A recent picture of the same streak (the early-probe scan, then each alert's scan): the
        # dead-game check also asks whether the screen stopped moving SINCE then, because the
        # "before" picture above can predate a freeze (see GameRecoveryConfig.static_recent_delta).
        self._recent_thumb: np.ndarray | None = None
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
            self._last_box_scan_at = None  # a gap that spans a stop is not a scan cadence
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
            self._state_calls[previous_state] += 1
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
