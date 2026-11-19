"""Bot lifecycle, handler dispatch, and shared runtime guards."""

import dataclasses
import hashlib
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import ExitStack

import cv2
import numpy as np

from eatventure_autobot.domain.config import BotConfig
from eatventure_autobot.domain.errors import CaptureError, ObservationError, WindowNotAvailableError
from eatventure_autobot.domain.protocols import InputController, Notifier
from eatventure_autobot.domain.state import (
    X_LOCKED_STATES,
    GameplayPhase,
    PauseReason,
    State,
    StationPanel,
)
from eatventure_autobot.domain.types import BoundingBox, Point, Zone
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
from eatventure_autobot.runtime.progress import ProgressLedger
from eatventure_autobot.runtime.vision import GameVision
from eatventure_autobot.state.context import FlowContext

logger = logging.getLogger(__name__)
# The calibrated panel touch area extends beyond its visible/template bounds.
_IDLE_PANEL_PADDING_PX = 30


# Names the state flow in effect. Bump it when the flow changes without a config change: the
# fingerprint below hashes config only, so metrics lines from before and after would otherwise
# carry the same cfg= id and could not be told apart.
FLOW_REVISION = (
    "confirmed-transitions; device-latched-resume; bounded-hold; first-last-reverse-v1; "
    "neutral-idle-v1; two-frame-targets-v1; viewport-mirror-first-v1; phone-progress-proof-v1; "
    "visual-manual-reconstruction-v1; rightmost-column-last-v2; protected-station-reframe-v1; "
    "observation-recovery-v1; footer-only-find-v1; candidate-no-drag-handoff-v1; "
    "scroll-pause-observation-v1; known-lock-stats-v1; narrow-terminal-row-v1; "
    "edge-gated-station-matching-v1; bounded-renovation-verification-v1; "
    "consistent-last-two-views-v1; economy-footer-preemption-v1; "
    "open-upgrade-handoff-v1; repeat-hold-observation-v1; shared-renovation-frame-v1"
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
        progress: ProgressLedger | None = None,
    ) -> None:
        self._config = config
        self._progress = progress
        self.pause_reason: PauseReason | None = PauseReason.MANUAL
        self._manual_stop_requested = threading.Event()
        self._device_check_at = 0.0
        self._game_recovery_pending = False
        self._phone_game_foreground: bool | None = None
        self._strategy_health_thumb: np.ndarray | None = None
        self._strategy_target_frame: np.ndarray | None = None
        self._strategy_alive_at = 0.0
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
        if progress is not None:
            self.context.total_levels_completed = int(progress.record["completions"])
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
            State.CONFIRM_RESTAURANT: self._handle_confirm_restaurant,
        }
        missing = set(State) - self._handlers.keys()
        if missing:
            raise ValueError(f"Missing state handlers: {sorted(s.name for s in missing)}")

    # --- lifecycle ------------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running.is_set()

    @property
    def recovering(self) -> bool:
        return self._resume is not None

    @property
    def manual_stop_pending(self) -> bool:
        return self._manual_stop_requested.is_set()

    def acknowledge_manual_stop(self) -> None:
        self._manual_stop_requested.clear()

    def _restart_mirror(self, cancelled: threading.Event) -> bool:
        if self._launcher is None:
            return False
        window = self._config.window
        if not self._launcher.restart_mirror(window.title, window.width, window.height, cancelled):
            return False
        if self._launcher.mirror_pid is not None:
            self._vision.bind_mirror(self._launcher.mirror_pid)
        return True

    def _bind_existing_mirror(self) -> bool:
        if self._launcher is None:
            return False
        process_id = self._launcher.bind_existing_mirror(self._vision.mirror_windows())
        if process_id is None:
            return False
        self._vision.bind_mirror(process_id)
        return True

    def _prepare_mirror(self, *, report: bool) -> bool:
        launcher = self._launcher
        if launcher is None:
            return True
        missing_window = False
        if report or not launcher.mirror_bound():
            missing_window = not self._bind_existing_mirror() and launcher.mirror_pid is not None
        if launcher.mirror_bound() and not missing_window:
            return True
        recovery = self._config.game_recovery
        if recovery.command_timeout <= 0 or recovery.recovery_poll_seconds <= 0:
            logger.error("scrcpy startup needs a positive timeout and polling interval")
            return False
        if not self._restart_mirror(self._manual_stop_requested):
            return False
        attempts = math.ceil(recovery.command_timeout / recovery.recovery_poll_seconds)
        for _ in range(attempts):
            if self._stop_requested.is_set() or not launcher.connected():
                return False
            try:
                self._vision.ensure_target_ready(force=True)
                return True
            except WindowNotAvailableError:
                if not self._sleep(recovery.recovery_poll_seconds):
                    return False
        logger.error("Timed out waiting for the selected-device scrcpy window")
        return False

    def _prepare_start(self, *, report: bool) -> bool:
        if self._launcher is not None:
            available = self._launcher.select_device() if report else self._launcher.connected()
            if not available:
                self._disconnect()
                return False
        missing = self._vision.validate_required_templates()
        if missing:
            logger.error("Cannot start, missing required templates: %s", ", ".join(missing))
            return False
        self._stop_requested.clear()
        try:
            if not self._prepare_mirror(report=report):
                self._stop_requested.set()
                return False
            self._vision.ensure_target_ready(force=True)
        except CaptureError as exc:
            logger.error("Cannot start bot: %s", exc)
            if report:
                self._report_self_stop(f"cannot start: {exc}")
            self._stop_requested.set()
            return False
        if self._launcher is not None and not self._launcher.mirror_bound():
            logger.error("Selected-device scrcpy binding was lost before startup")
            self._stop_requested.set()
            return False
        if not self._input.is_target_foreground():
            logger.error("Cannot start: '%s' is not foreground", self._config.window.title)
            if report:
                self._report_self_stop("cannot start: target window is not foreground")
            self._stop_requested.set()
            return False
        if (self._launcher is not None and self._config.game_recovery.enabled
                and not self._package_probed):
            self._package_probed = True  # package discovery is once per process
            self._launcher.detect_package()
        return True

    def start(self, *, report: bool = True) -> bool:
        """report=False is for the auto-resume retries, which must not alert on every failed try."""
        with self._step_lock:
            if self.running:
                return True
            if not report and (self._manual_stop_requested.is_set() or self.pause_reason in (
                PauseReason.MANUAL, PauseReason.DISCONNECTED, PauseReason.ERROR
            )):
                return False
            if report:
                self._manual_stop_requested.clear()
                self._game_recovery_pending = False
            if not self._prepare_start(report=report):
                return False
            phase = GameplayPhase.FIRST if report else self.context.gameplay_phase
            if self._manual_stop_requested.is_set() or self._stop_requested.is_set():
                return False
            self._running.set()
            self.pause_reason = None
            self.context.invalidate_viewport()
            self.context.gameplay_phase = phase
            self.context.strategy_seek = True
            self.context.strategy_waiting_liveness = False
            self.context.strategy_scrolls = self.context.strategy_stationary_count = 0
            self._strategy_health_thumb = None
            if self._progress is not None:
                self._progress.checkpoint("active")
            self._resume = None  # running again, by a person or by poll_resume
            self._watchdog.reset()
            self._last_box_scan_at = None  # a gap that spans a stop is not a scan cadence
            self._started_at = self._last_metrics_at = time.monotonic()
            self.context.state_attempt = 1
            if self.context.current_level_start_time is None:
                self.context.current_level_start_time = time.monotonic()
            self._notifier.notify_bot_started()
            return True

    def request_stop(self, *, manual: bool = False) -> None:
        if manual:
            self._manual_stop_requested.set()
            self.pause_reason = PauseReason.MANUAL
            self._resume = None
            self._game_recovery_pending = False
        self._stop_requested.set()

    def stop(self, *, manual: bool = False) -> None:
        self.request_stop(manual=manual or self._manual_stop_requested.is_set())
        with self._step_lock:
            was_running = self.running
            self._running.clear()
            self._input.release_left_button()
            self.state = State.FIND_RED_ICONS
            self.context.reset_run()
            self.context.invalidate_viewport()
            self.context.transition_pending = False
            if self._progress is not None:
                reason = self.pause_reason or PauseReason.ERROR
                self._progress.checkpoint(reason.value)
                self._progress.cancel_transition()
            self._watchdog.reset()
            if was_running and self._config.flow_timing.metrics_log_interval > 0:
                self._log_metrics()
        # v1 only sends "Bot Stopped" from the manual Z-toggle stop path; an internal auto-stop
        # (stop-requested race, lost foreground, an unhandled handler exception, or close()'s
        # own cleanup call) never notifies there, so it doesn't here either.
        if was_running and manual:
            self._notifier.notify_bot_stopped()

    def close(self) -> None:
        with ExitStack() as cleanup:
            if self._launcher is not None:
                cleanup.callback(self._launcher.close)
            cleanup.callback(self._vision.close)
            cleanup.callback(self._notifier.close)
            if self._progress is not None:
                cleanup.callback(self._progress.close)
            self.stop()

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
            if not self._device_available():
                return False
            self._vision.ensure_target_ready()
            if not self._input.is_target_foreground():
                logger.error("Target lost foreground; stopping bot")
                self._report_self_stop("target lost foreground")
                self.pause_reason = PauseReason.RECOVERY
                self.stop()
                self._arm_resume("target lost foreground")
                return False

            if self._dismiss_popup():
                return True  # a popup was closed this step; the handler runs on the next one

            return self._run_state_handler()
        except (ObservationError, CaptureError) as exc:
            if isinstance(exc, ObservationError):
                logger.warning(
                    "Screen observation unavailable; scheduling localization retry: %s", exc
                )
            else:
                logger.error("Stopping bot: %s", exc)
            self._report_self_stop(str(exc))
            if (not self._manual_stop_requested.is_set()
                    and self.pause_reason != PauseReason.DISCONNECTED):
                self.pause_reason = PauseReason.RECOVERY
            self.stop()
            self._arm_resume(str(exc))
            return False
        except Exception:
            logger.exception("Stopping bot after an unexpected handler failure")
            self._report_self_stop("unexpected handler failure")
            self.pause_reason = PauseReason.ERROR
            self.stop()
            return False
        finally:
            self._step_lock.release()

    def _run_state_handler(self) -> bool:
        previous_state = self.state
        started = time.monotonic()
        next_state = self._handlers[previous_state]()
        self._state_seconds[previous_state] += time.monotonic() - started
        self._state_calls[previous_state] += 1
        self._last_handled_state = previous_state
        if previous_state in X_LOCKED_STATES:
            # Avoid dismissing a restaurant dialog that still owns the next UI action.
            self._close_level_lock_until = max(
                self._close_level_lock_until,
                time.monotonic() + self._config.close_button.new_level_lock_seconds,
            )
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
        if self._progress is not None:
            self._progress.checkpoint()
        return True

    # --- shared helpers -------------------------------------------------------------------

    def _disconnect(self) -> None:
        self.pause_reason = PauseReason.DISCONNECTED
        self._game_recovery_pending = False
        self._resume = None
        self.stop()
        logger.warning("Device disconnected; manual resume required")

    def _recover_stalled_mirror(self) -> bool:
        """Only failed UI observations pay for an independent phone-progress check."""
        if self._launcher is None or not self._config.game_recovery.enabled:
            return False
        if not self._device_available(force=True) or self._stop_requested.is_set():
            return True
        before = self._vision.capture().copy()
        phone_before = None
        for _ in range(3):
            encoded = self._launcher.capture_device_png()
            if self._stop_requested.is_set():
                return True
            if encoded is None:
                return not self._device_available(force=True)
            try:
                phone = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
            except cv2.error:
                return False
            if phone is None:
                return False
            if phone_before is not None and not np.array_equal(phone_before, phone):
                break
            phone_before = phone
        else:
            return False
        if not np.array_equal(before, self._vision.capture()):
            return False
        if not self._device_available(force=True) or self._stop_requested.is_set():
            return True
        logger.warning("Phone is changing but scrcpy is frozen; suspending input for mirror repair")
        self.pause_reason = PauseReason.RECOVERY
        self.stop()
        self._game_recovery_pending = False
        self._arm_resume("phone progresses while scrcpy display is frozen")
        if self._resume is not None:
            self._resume.restart_mirror = True
        return True

    def _device_available(self, *, force: bool = False) -> bool:
        if self._launcher is None:
            return True
        now = self._clock()
        if not force and now < self._device_check_at:
            return self.pause_reason != PauseReason.DISCONNECTED
        self._device_check_at = now + 1.0
        if self._launcher.connected():
            self._phone_game_foreground = (
                self._launcher.game_foreground() if self._launcher.package is not None else None
            )
            if self.running and self._phone_game_foreground is False:
                logger.warning("Selected game left Android foreground; releasing input")
                if self._config.game_recovery.enabled:
                    self._queue_game_recovery()
                else:
                    self.pause_reason = PauseReason.ERROR
                    self.stop()
                return False
            return True
        self._disconnect()
        return False

    def _sleep(self, duration: float) -> bool:
        if duration <= 0:
            return not self._stop_requested.is_set()
        return not self._stop_requested.wait(duration)

    def _click_idle(
        self, exclude: BoundingBox | None = None, *, alternate: bool = False,
    ) -> bool:
        if self._stop_requested.is_set():
            return False
        targets = self._config.click_targets
        anchor = self.context.strategy_anchor
        if self._config.strategy.enabled:
            if not self._input.is_target_foreground():
                return False
            if exclude is None:
                return True
            padding = _IDLE_PANEL_PADDING_PX
            blocked = BoundingBox(exclude.x_min - padding, exclude.y_min - padding,
                                  exclude.x_max + padding, exclude.y_max + padding)
            occupied = tuple(self.context.maxed_positions) + (() if anchor is None else (anchor,))
            frame = self._vision.capture()
            point = (None if alternate
                     else self._vision.empty_world_point(frame, blocked, occupied))
            if point is None or not self._is_clickable(point):
                panel = self._vision.station_panel_bounds(frame)
                if panel is None or self._vision.station_panel(frame) == StationPanel.UNKNOWN:
                    return False
                x, y = targets.scroll_start_pos
                positions: tuple[Point, ...] = (
                    (x, y), (panel.x_max + padding + 1, y),
                    (panel.x_min - padding - 1, y),
                )
                for candidate in positions:
                    x, y = candidate
                    if (not (0 <= x < self._config.window.width
                             and 0 <= y < self._config.window.height)
                            or not self._is_clickable(candidate)
                            or (panel.x_min - padding <= x <= panel.x_max + padding
                                and panel.y_min - padding <= y <= panel.y_max + padding)):
                        continue
                    return self._input.click(*candidate)
                return False
            return self._is_clickable(point) and self._input.click(*point)
        positions = (targets.idle_click_pos, targets.scroll_start_pos)
        for position in positions:
            if exclude is not None and (
                exclude.x_min - _IDLE_PANEL_PADDING_PX <= position[0]
                <= exclude.x_max + _IDLE_PANEL_PADDING_PX
                and exclude.y_min - _IDLE_PANEL_PADDING_PX <= position[1]
                <= exclude.y_max + _IDLE_PANEL_PADDING_PX
            ):
                continue
            return self._input.click(*position)
        return False

    def _scrcpy_recovery(self, delay: float) -> bool:
        if not self._config.scrcpy_recovery.enabled:
            return False
        return self._sleep(delay)

    def _is_clickable(self, point: Point) -> bool:
        return not self._input.is_in_forbidden_zone(*point)
