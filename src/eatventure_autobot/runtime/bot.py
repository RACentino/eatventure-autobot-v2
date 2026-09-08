"""Orchestration: the assembly line. Every handler here does real I/O (capture, match, click,
sleep), packages what it observed, and hands the decision to a pure function in state/transitions.
No branching on flow state happens in this file that isn't delegated — if a handler starts deciding
where to go next on its own, that logic belongs in state/transitions instead.
"""

import logging
import threading
import time
from collections.abc import Callable

from eatventure_autobot.domain.config import BotConfig
from eatventure_autobot.domain.errors import CaptureError, WindowNotAvailableError
from eatventure_autobot.domain.protocols import InputController, Notifier
from eatventure_autobot.domain.state import State
from eatventure_autobot.domain.types import Point, Zone
from eatventure_autobot.resilience.watchdog import StallWatchdog, WatchdogVerdict
from eatventure_autobot.runtime.vision import GameVision
from eatventure_autobot.state import transitions as flow
from eatventure_autobot.state.context import FlowContext

logger = logging.getLogger(__name__)


class EatventureBot:
    def __init__(
        self,
        config: BotConfig,
        vision: GameVision,
        input_controller: InputController,
        notifier: Notifier,
        stop_event: threading.Event | None = None,
    ) -> None:
        self._config = config
        self._vision = vision
        self._input = input_controller
        self._notifier = notifier
        self._stop_requested = stop_event or threading.Event()
        self._running = threading.Event()
        self._step_lock = threading.RLock()

        self.state = State.FIND_RED_ICONS
        self.context = FlowContext()
        self._watchdog = StallWatchdog(config=config.flow_timing)
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

    def start(self) -> bool:
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
                self._stop_requested.set()
                return False
            if not self._input.is_target_foreground():
                logger.error("Cannot start: '%s' is not foreground", self._config.window.title)
                self._stop_requested.set()
                return False
            self._running.set()
            self._watchdog.reset()
            self.context.state_attempt = 1
            if self.context.current_level_start_time is None:
                self.context.current_level_start_time = time.monotonic()
            self._notifier.notify_bot_started()
            return True

    def request_stop(self) -> None:
        self._stop_requested.set()

    def stop(self) -> None:
        self.request_stop()
        with self._step_lock:
            was_running = self.running
            self._running.clear()
            self._input.release_left_button()
            self.state = State.FIND_RED_ICONS
            self.context.reset_run()
            self._watchdog.reset()
        if was_running:
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
                self.stop()
                return False

            previous_state = self.state
            next_state = self._handlers[previous_state]()
            if not self.running:
                return False
            # v2's post-handler recheck (v1 only checks before): catches focus lost mid-click.
            if not self._input.is_target_foreground():
                logger.error("Target lost foreground during %s; stopping", previous_state.name)
                self.stop()
                return False

            self.context.state_attempt = (
                self.context.state_attempt + 1 if next_state == previous_state else 1
            )
            self.state = next_state
            self._apply_watchdog(next_state)
            return True
        except (WindowNotAvailableError, CaptureError) as exc:
            logger.error("Stopping bot: %s", exc)
            self.stop()
            return False
        except Exception:
            logger.exception("Stopping bot after an unexpected handler failure")
            self.stop()
            return False
        finally:
            self._step_lock.release()

    def _apply_watchdog(self, current_state: State) -> None:
        verdict = self._watchdog.tick(current_state, self.context.progress_marker)
        if verdict is WatchdogVerdict.OK:
            return
        if verdict is WatchdogVerdict.RESET:
            logger.warning("State %s stalled; resetting search flow", current_state.name)
            self.context.reset_search_cycle()
            self.state = State.FIND_RED_ICONS
            self.context.state_attempt = 1
            return
        logger.error(
            "Watchdog escalating to fail-closed stop: %s", self._watchdog.last_escalation_reason
        )
        self.stop()

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
        if self._vision.find_new_level_button(frame).found:
            logger.info("New level button visible during red-icon scan")
            return flow.decide_find_red_icons(
                self.context, flow.RedIconScanObservation(True, False, ())
            )

        candidates = self._vision.find_red_icons(frame)
        actionable, new_level_seen, _ = self._vision.split_red_icons(candidates)
        # Gated on "no actionable icons," not "no raw candidates": a lone footer badge (new-level
        # or stats icon, classified out by split_red_icons above) would otherwise make candidates
        # non-empty and silently skip both the recovery retry and the fast->normal fallback even
        # with zero actionable icons and real ones potentially elsewhere on screen.
        if not actionable and self._scrcpy_recovery(self._config.scrcpy_recovery.red_icon_delay):
            frame = self._vision.capture(max_y=regions.extended_search_y)
            if self._vision.find_new_level_button(frame).found:
                return flow.decide_find_red_icons(
                    self.context, flow.RedIconScanObservation(True, False, ())
                )
            candidates = self._vision.find_red_icons(frame)
            actionable, new_level_seen, _ = self._vision.split_red_icons(candidates)
            if not actionable and self._vision.fast_red_icon_mode:
                # v2's fast->normal fallback on a second empty frame; v1 has no runtime toggle.
                candidates = self._vision.find_red_icons(frame, force_normal_mode=True)
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
        clicked = self._input.precise_click(x, y)
        if clicked:
            self._sleep(self._config.scrcpy_recovery.action_settle_delay)
        return flow.decide_click_red_icon(self.context, clicked)

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
        result = self._vision.find_upgrade_station(frame, threshold)
        position = result.best.center if result.best is not None else None
        found = position is not None and self._is_clickable(position)
        if not found:
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

        verified = self._verify_upgrade_station(position)
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
        held = self._input.hold_at(
            target[0],
            target[1],
            station.click_hold_max_duration,
            check_interval,
            self._make_disappearance_check(
                self._config.thresholds.upgrade_station,
                relaxed_threshold,
                station.disappear_confirmation_count,
            ),
        )
        if held:
            current = flow.current_red_icon_target(self.context)
            if current is not None:
                self.context.remember_successful_red_icon_row(current.center[1])
        post_ok = (
            held
            and self._click_idle()
            and self._sleep(self._config.scrcpy_recovery.action_settle_delay)
        )
        return flow.decide_hold_upgrade_station(
            self.context,
            self._config.flow_timing,
            flow.UpgradeHoldObservation(True, True, held, post_ok),
        )

    def _verify_upgrade_station_round(self) -> tuple[Point, float] | None:
        """One full verify_search_attempts retry loop: base threshold on the first attempt,
        relaxed threshold afterward. Used for both rounds around the pre-hold wake click."""
        station = self._config.upgrade_station
        base = self._config.thresholds.upgrade_station
        relaxed = base - station.threshold_relaxation
        for attempt in range(max(1, station.verify_search_attempts)):
            threshold = base if attempt == 0 else relaxed
            frame = self._vision.capture(
                max_y=self._config.capture_regions.upgrade_station_search_y
            )
            result = self._vision.find_upgrade_station(frame, threshold)
            if result.best is not None and self._is_clickable(result.best.center):
                return result.best.center, relaxed
            if not self._sleep(station.verify_search_interval):
                return None
        return None

    def _verify_upgrade_station(self, position: Point) -> tuple[Point, float] | None:
        """Re-checks on fresh frames that the station is still there before committing to a hold
        (verified behavior in both repos). A successful first round is followed by a deliberate
        "wake click" on the freshly-found position (the upgrade UI sometimes needs one priming
        tap before the button truly registers/renders) and a second, identical verification
        round — only a station that survives both rounds is held. Both rounds, and the very
        first attempt, are preceded by the same settle delay v1 uses."""
        station = self._config.upgrade_station
        if not self._sleep(station.verify_settle_delay):
            return None

        first = self._verify_upgrade_station_round()
        if first is None:
            logger.info("Upgrade station was not visible during verification")
            return None

        wake_target, _ = first
        if not self._input.precise_click(*wake_target):
            logger.info(
                "Upgrade station verification wake click failed at (%s, %s)", *wake_target
            )
            return None
        if not self._sleep(station.verify_settle_delay):
            return None

        second = self._verify_upgrade_station_round()
        if second is None:
            logger.info("Upgrade station was not visible during second verification round")
            return None

        target, relaxed = second
        self.context.upgrade_station_pos = target
        return target, relaxed

    def _station_disappeared(self, base_threshold: float, relaxed_threshold: float) -> bool:
        """Base-threshold check, then relaxed-threshold check, then (if scrcpy recovery is
        enabled) one recovery sleep and a final relaxed-threshold recheck before concluding the
        station is genuinely gone for this poll tick — mirrors v1's
        _find_current_upgrade_hold_match so a single dropped/stale scrcpy frame isn't mistaken
        for the station having actually disappeared."""
        regions = self._config.capture_regions
        frame = self._vision.capture(max_y=regions.upgrade_station_search_y)
        if self._vision.find_upgrade_station(frame, base_threshold).found:
            return False
        frame = self._vision.capture(max_y=regions.upgrade_station_search_y)
        if self._vision.find_upgrade_station(frame, relaxed_threshold).found:
            return False
        if self._scrcpy_recovery(self._config.scrcpy_recovery.upgrade_delay):
            frame = self._vision.capture(max_y=regions.upgrade_station_search_y)
            if self._vision.find_upgrade_station(frame, relaxed_threshold).found:
                return False
        return True

    def _make_disappearance_check(
        self, base_threshold: float, relaxed_threshold: float, required_misses: int
    ) -> Callable[[], bool]:
        """Requires `required_misses` CONSECUTIVE misses before treating the station as gone —
        debounces a single flaky/transient miss so a real hold isn't cut short by one bad frame."""
        misses = 0

        def check() -> bool:
            nonlocal misses
            gone = self._station_disappeared(base_threshold, relaxed_threshold)
            misses = misses + 1 if gone else 0
            return misses >= max(1, required_misses)

        return check

    def _handle_upgrade_stats(self) -> State:
        max_attempts = self._config.stats_upgrade.search_max_attempts
        if not self._click_idle():
            return State.UPGRADE_STATS
        attempt = self.context.state_attempt
        frame = self._vision.capture(max_y=self._config.capture_regions.extended_search_y)
        if self._vision.find_new_level_button(frame).found:
            return flow.decide_upgrade_stats(
                flow.StatsIconObservation(True, False, attempt), max_attempts
            )

        # Reuses the red-icon scan rather than a second pass at the stats-specific threshold; the
        # two thresholds differ by at most 0.003 in either source config.
        candidates = self._vision.find_red_icons(frame, force_normal_mode=attempt >= max_attempts)
        _, _, stats_seen = self._vision.split_red_icons(candidates)
        if not stats_seen:
            if attempt < max_attempts:
                self._scrcpy_recovery(self._config.scrcpy_recovery.red_icon_delay)
            return flow.decide_upgrade_stats(
                flow.StatsIconObservation(False, False, attempt), max_attempts
            )

        targets = self._config.click_targets
        if self._input.click(*targets.stats_upgrade_button_pos):
            self._sleep(self._config.scrcpy_recovery.action_settle_delay)
            self._input.spam_click_at(
                *targets.stats_upgrade_pos,
                self._config.stats_upgrade.click_duration,
                self._config.stats_upgrade.click_delay,
            )
            self._click_idle()
        return flow.decide_upgrade_stats(
            flow.StatsIconObservation(False, True, attempt), max_attempts
        )

    def _handle_open_boxes(self) -> State:
        if not self._click_idle():
            return State.OPEN_BOXES
        box_search_y = self._config.capture_regions.box_search_y
        frame = self._vision.capture(max_y=box_search_y)
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

        opened = 0
        for box in boxes:
            if not self._is_clickable(box.center):
                continue
            if self._input.click(*box.center):
                opened += 1
        if opened:
            logger.info("Opened %s boxes", opened)
        return flow.decide_open_boxes(
            self.context,
            self._config.upgrade_station,
            self._config.scroll.max_idle_pass_attempts,
            flow.BoxCycleObservation(False, opened),
        )

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
        return flow.decide_scroll(self.context, moved)

    def _handle_check_new_level(self) -> State:
        max_attempts = self._config.level_transition.new_level_verify_max_attempts
        if not self._click_idle():
            return State.CHECK_NEW_LEVEL
        if not self._sleep(self._config.flow_timing.focus_settle_delay):
            return State.CHECK_NEW_LEVEL
        attempt = self.context.state_attempt

        if not self.context.new_level_red_icon_verified:
            frame = self._vision.capture(max_y=self._config.capture_regions.extended_search_y)
            candidates = self._vision.find_red_icons(
                frame, force_normal_mode=attempt >= max_attempts
            )
            _, verified, _ = self._vision.split_red_icons(candidates)
            if not verified:
                if attempt < max_attempts:
                    self._scrcpy_recovery(self._config.scrcpy_recovery.red_icon_delay)
                return flow.decide_check_new_level(
                    self.context,
                    max_attempts,
                    flow.NewLevelVerificationObservation(False, attempt, None, None),
                )
            self.context.new_level_red_icon_verified = True

        targets = self._config.click_targets
        level = self._config.level_transition
        button_ok = self._input.click(*targets.new_level_button_pos)
        transition_ok: bool | None = None
        if button_ok:
            self._sleep(level.confirmation_delay)
            transition_ok = self._input.click(*targets.level_transition_pos)
            if transition_ok:
                self._sleep(level.secondary_settle_delay)
        return flow.decide_check_new_level(
            self.context,
            max_attempts,
            flow.NewLevelVerificationObservation(True, attempt, button_ok, transition_ok),
        )

    def _handle_transition_level(self) -> State:
        if not self._click_idle():
            return State.TRANSITION_LEVEL
        level = self._config.level_transition
        attempt = self.context.state_attempt
        frame = self._vision.capture(max_y=self._config.capture_regions.extended_search_y)
        result = self._vision.find_new_level_button(frame)
        if result.best is None or not self._is_clickable(result.best.center):
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
        frame = self._vision.capture(max_y=self._config.capture_regions.max_search_y)
        result = self._vision.find_unlock_button(frame)
        if result.best is None or not self._is_clickable(result.best.center):
            self._sleep(level.unlock_search_interval)
            return flow.decide_wait_for_unlock(
                self.context, level, flow.WaitForUnlockObservation(False, None)
            )

        clicked = self._input.click(*result.best.center)
        previous_total = self.context.total_levels_completed
        next_state = flow.decide_wait_for_unlock(
            self.context, level, flow.WaitForUnlockObservation(True, clicked)
        )
        if self.context.total_levels_completed > previous_total:
            self._report_level_completion()
        return next_state

    def _report_level_completion(self) -> None:
        now = time.monotonic()
        started = self.context.current_level_start_time
        elapsed = now - started if started is not None else 0.0
        self.context.current_level_start_time = now
        logger.info(
            "Restaurant %s completed in %.1fs", self.context.total_levels_completed, elapsed
        )
        self._notifier.notify_new_level(self.context.total_levels_completed, elapsed)
