"""Screen observations and guarded actions for each bot state."""

import logging
import math
import time
from collections import deque
from collections.abc import Callable
from threading import Event
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from eatventure_autobot.domain.config import BotConfig
from eatventure_autobot.domain.errors import ObservationError
from eatventure_autobot.domain.protocols import InputController
from eatventure_autobot.domain.state import GameplayPhase, State, StationPanel
from eatventure_autobot.domain.types import BoundingBox, MatchCandidate, Point, TemplateExplanation
from eatventure_autobot.runtime._recovery import RECOVERY_PROBE_SECONDS, STATION_SEARCH_ANR
from eatventure_autobot.runtime._recovery_probe import (
    ScreenProbe,
    _restaurant_visible,
    recovery_unlock_visible,
)
from eatventure_autobot.runtime._telemetry import _frame_thumbnail
from eatventure_autobot.runtime.game_recovery import GameLauncher
from eatventure_autobot.runtime.progress import ProgressLedger
from eatventure_autobot.runtime.vision import GameVision, StationPanelRead
from eatventure_autobot.state import transitions as flow
from eatventure_autobot.state.context import ROW_DEDUPE_DISTANCE, FlowContext, LevelMark

logger = logging.getLogger("eatventure_autobot.runtime.bot")

# Two consecutive scans of the station popup this close (px) count as "has stopped moving"
# (ScrcpyRecoveryConfig.popup_early_exit). The matcher reports whole-pixel positions, so a settled
# panel reads 0; 2 only absorbs a single sub-pixel rounding flip.
_POPUP_STILL_PX = 2.0
_VIEWPORT_MAX_SETTLE_SAMPLES = 12
_VIEWPORT_SETTLE_WAIT_BUDGET = 3.6  # Seconds of observation waits available to a moving camera.
_VIEWPORT_REQUIRED_STILL_SAMPLES = 2
_VIEWPORT_MAX_PANEL_DISMISSALS = 2
_VIEWPORT_FRESH_ENTRY_PROBES = 3
_VIEWPORT_SAME_PANEL_IOU = 0.5
_MIN_STATION_TARGET_Y = 300


class _ViewportError(ObservationError):
    """Bounded viewport readiness failure, separate from unrelated detector failures."""


class _StationSearchSpent(_ViewportError):
    """A search pass ended without a verified station."""


class _StationLanding(NamedTuple):
    viewport: int
    frame: np.ndarray
    targets: list[MatchCandidate]


class _SettlingControl(NamedTuple):
    point: Point
    panel: BoundingBox
    name_ink: np.ndarray

    def same_panel(self, other: "_SettlingControl") -> bool:
        return (self.panel.iou(other.panel) >= _VIEWPORT_SAME_PANEL_IOU
                and bool(np.array_equal(self.name_ink, other.name_ink)))


class _HandlersMixin:
    # What this mixin reaches on self. EatventureBot.__init__ (bot.py) creates the attributes;
    # the methods live in bot.py or another mixin.
    _clock: Callable[[], float]
    _config: BotConfig
    _asset_frame: np.ndarray | None
    _early_probe_pending: bool
    _input: InputController
    _latest_thumb: np.ndarray | None
    _recent_box_clicks: deque[tuple[int, int, float]]
    _recent_thumb: np.ndarray | None
    _scan_delta: float | None
    _stall_probe_pending: bool
    _streak_anchor_pending: bool
    _streak_thumb: np.ndarray | None
    _vision: GameVision
    context: FlowContext
    _progress: ProgressLedger | None
    _launcher: GameLauncher | None
    _verify_recovered_unlock: bool
    _recovery_station: tuple[int, Point] | None
    _recovery_restart_used: bool
    _strategy_health_thumb: np.ndarray | None
    _strategy_target_frame: np.ndarray | None
    _strategy_endpoint_frame: np.ndarray | None = None
    _station_landing: _StationLanding | None = None
    _station_recheck_at: float | None = None
    _strategy_alive_at: float
    _phone_game_foreground: bool | None
    _stop_requested: Event

    if TYPE_CHECKING:  # annotation only: nothing here exists at runtime

        def _maybe_alert_stall(self) -> None:
            ...

        def _log_box_near_misses(self, frame: np.ndarray, detected: int) -> None:
            ...

        def _log_early_probe(self, frame: np.ndarray, detected: int) -> None:
            ...

        def _sleep(self, duration: float) -> bool:
            ...

        def _click_idle(
            self, exclude: BoundingBox | None = None, *, alternate: bool = False,
        ) -> bool:
            ...

        def _scrcpy_recovery(self, delay: float) -> bool:
            ...

        def _record_box_scan(self) -> None:
            ...

        def _record_popup_seen(self, milliseconds: int) -> None:
            ...

        def _is_clickable(self, point: Point) -> bool:
            ...

        def _record_hold(self, seconds: float) -> None:
            ...

        def _game_looks_dead(self, thumb: np.ndarray) -> bool:
            ...

        def _recovery_input_allowed(self) -> bool: ...
        def _note_recovery_click(self, point: Point) -> None: ...

        def _queue_game_recovery(self, reason: str = ...) -> State: ...

        def _arm_resume(self, reason: str) -> None: ...

        def stop(self, *, manual: bool = False) -> None: ...

        def _level_mark(self) -> LevelMark:
            ...

        def _report_level_completion(self) -> None:
            ...

        def _device_available(self, *, force: bool = False) -> bool: ...

        def _recover_stalled_mirror(self) -> bool: ...
        def _probe_recovery_screen(self) -> ScreenProbe: ...

    def _settle_and_watch(self, duration: float) -> bool:
        """_sleep(duration) that also notes when the popup the last click opened first shows (tally
        in the metrics line). It clicks nothing and never raises: any trouble just stops the looking
        and sleeps the rest.
        With popup_early_exit it may end BEFORE the deadline, only when the popup is fresh and
        still: some scan since the click saw no popup (so it did not predate the click) and the
        station match then sat at the same spot on two consecutive scans (so it stopped moving).
        Anything else, including a popup already up on the first scan, runs the full `duration`.
        ponytail: watches for the station popup only, so a locked station (which shows `unlock`)
        counts as never seen; the last scan can run one capture past the deadline."""
        config = self._config.scrcpy_recovery
        interval = config.popup_probe_interval
        if interval <= 0 or duration <= 0:
            return self._sleep(duration)
        started = self._clock()
        deadline = started + duration
        watching = True
        seen = False
        fresh = False  # a scan since the click showed no popup
        last: Point | None = None  # where the previous scan saw it, None once it was missing
        closest: float | None = None  # the smallest scan-to-scan jump seen, for the moving tally
        while (remaining := deadline - self._clock()) > 0:
            if watching:
                try:
                    frame = self._vision.capture(
                        max_y=self._config.capture_regions.upgrade_station_search_y
                    )
                    best = self._vision.find_upgrade_station(frame).best
                    if best is None:
                        fresh, last = True, None
                    else:
                        if not seen:
                            seen = True
                            self._record_popup_seen(round(1000 * (self._clock() - started)))
                        jump = None if last is None else math.dist(last, best.center)
                        if not config.popup_early_exit:
                            watching = False
                        elif fresh and jump is not None and jump <= _POPUP_STILL_PX:
                            self.context.popup_early += 1
                            return True
                        else:
                            if jump is not None and (closest is None or jump < closest):
                                closest = jump
                            last = best.center
                except Exception:
                    logger.debug("Popup probe stopped", exc_info=True)
                    watching = False
                remaining = deadline - self._clock()
            nap = min(interval, remaining) if watching else remaining
            if nap > 0 and not self._sleep(nap):
                return False
        context = self.context
        if not seen and watching:
            context.popup_never += 1
        elif seen and watching:  # popup_early_exit is on here (off would have stopped watching)
            # The popup showed but the settle still ran in full. Already up from the first scan on:
            # stale (nothing the click did). Appeared after the click but never held still twice
            # in a row: moving, with how close two scans came (a few px = tolerance, tens = real).
            if not fresh:
                context.popup_stale += 1
            else:
                context.popup_moving += 1
                if closest is not None:
                    context.popup_moving_px.append(round(closest))
        return True

    def _settle_and_time_unlock(self, duration: float) -> bool:
        """_sleep(duration) that also notes when the Unlock dialog first shows (unlock_ms in the
        metrics line), to learn how much of WAIT_FOR_UNLOCK's blind settle is needed. Log-only: it
        clicks nothing, ends at the same deadline, and never raises (trouble just stops the looking
        and sleeps the rest). "Shows" = the raw best `unlock` match reaches greyed_unlock_min_conf,
        which a greyed and an enabled button both do; a first try that never gets there is
        unlock_never (the dialog did not open, or the screen is something else).
        ponytail: the early exit waits for these numbers; upgrade path: end the settle at the first
        sighting once unlock_ms p95 is well under focus_settle_delay."""
        interval = self._config.scrcpy_recovery.popup_probe_interval
        if interval <= 0 or duration <= 0:
            return self._sleep(duration)
        started = self._clock()
        deadline = started + duration
        floor = self._config.level_transition.greyed_unlock_min_conf
        watching = True
        while (remaining := deadline - self._clock()) > 0:
            if watching:
                try:
                    seen = self._vision.unlock_near_miss(self._vision.capture())
                    if seen is not None and seen.confidence >= floor:
                        watching = False
                        self.context.unlock_ms.append(round(1000 * (self._clock() - started)))
                except Exception:
                    logger.debug("Unlock probe stopped", exc_info=True)
                    watching = False  # a broken probe has no verdict: no sighting, no unlock_never
                remaining = deadline - self._clock()
            nap = min(interval, remaining) if watching else remaining
            if nap > 0 and not self._sleep(nap):
                return False
        if watching:
            self.context.unlock_never += 1
        return True

    def _capture_first_asset_frame(self, max_y: int) -> np.ndarray:
        """Reuse the popup scan only on a handler's first read before input or a wait."""
        frame = self._asset_frame
        self._asset_frame = None
        return self._vision.capture(max_y=max_y) if frame is None else frame[:max_y]

    def _handle_find_red_icons(self) -> State:
        monitoring = self._config.strategy.enabled and self.context.strategy_waiting_liveness
        if not self._config.strategy.enabled and not self._click_idle():
            return State.FIND_RED_ICONS
        regions = self._config.capture_regions
        frame = self._capture_first_asset_frame(regions.extended_search_y)
        if self._progress is not None:
            self._progress.checkpoint(boost=self._vision.boost_level(frame))
        if monitoring:
            return self._handle_strategy_find(frame)
        if self._handle_city_arrival(frame):
            return State.FIND_RED_ICONS
        # v1 checks the newLevel button against a frame limited to max_search_y, distinct from the
        # full extended_search_y frame used for the red-icon scan below.
        if self._vision.find_new_level_button(frame[: regions.max_search_y]).found:
            logger.info("New level button visible during red-icon scan")
            return flow.decide_find_red_icons(
                self.context, flow.RedIconScanObservation(True, False, ())
            )

        if self._config.strategy.enabled:
            if (renovation := self._candidate_renovation_footer(frame)) is not None:
                return renovation
            return self._handle_strategy_find(frame)

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

    def _candidate_renovation_footer(self, frame: np.ndarray) -> State | None:
        if self._config.strategy.enabled and self._vision.new_level_icon_seen(frame):
            return flow.decide_find_red_icons(
                self.context, flow.RedIconScanObservation(False, True, ())
            )
        return None

    def _handle_click_red_icon(self) -> State:
        target = flow.current_red_icon_target(self.context)
        if target is None:
            return State.OPEN_BOXES
        if self._config.strategy.enabled and self.context.target_viewport != self.context.viewport:
            return State.FIND_RED_ICONS
        offset_x, offset_y = (
            (0, 0) if self._config.strategy.enabled else self._config.red_icon.offset
        )
        x, y = target.center[0] + offset_x, target.center[1] + offset_y
        # Verified v1 behavior: a plain click() (with its post-click delay), not precise_click();
        # CHECK_UNLOCK already does the scrcpy settle when it opens.
        return flow.decide_click_red_icon(self.context, self._input.click(x, y))

    def _handle_check_unlock(self) -> State:
        if not self._settle_and_watch(self._config.scrcpy_recovery.action_settle_delay):
            return State.CHECK_UNLOCK
        frame = self._vision.capture(max_y=self._config.capture_regions.max_search_y)
        if self._config.strategy.enabled:
            read = self._vision.read_station_panel(frame, prefer_unlock=True)
            if read.state == StationPanel.UNKNOWN:
                if not self._sleep(self._config.strategy.target_observation_interval):
                    return State.CHECK_UNLOCK
                frame = self._vision.capture()
                transition = self._economy_level_transition(
                    frame, button_max_y=self._config.capture_regions.max_search_y,
                )
                if transition is not None:
                    return transition
                read = self._vision.read_station_panel(frame, prefer_unlock=True)
            panel = self._observe_strategy_panel(frame, read)
            if panel == StationPanel.UNKNOWN:
                if read.bounds is None:
                    # The tap never established a panel. Re-select without losing known MAX rows.
                    self.context.strategy_anchor = self.context.target_viewport = None
                    self.context.red_icons.clear()
                    self.context.upgrade_station_pos = None
                    self._strategy_target_frame = None
                    if (self.context.gameplay_phase == GameplayPhase.LAST
                            and self.context.strategy_endpoint is None):
                        self.context.strategy_seek = True
                return State.FIND_RED_ICONS
            if panel == StationPanel.MAX:
                return State.UPGRADE_STATS
            if panel == StationPanel.LOCKED:
                self.context.stats_from_find = True
                return State.UPGRADE_STATS
            unlock = read.unlock
            if unlock is None and read.bounds is not None:
                position = self._clickable_upgrade(
                    frame, self._config.thresholds.upgrade_station, read.upgrade,
                )
                if position is not None:
                    return flow.decide_search_upgrade_station(
                        self.context, self._config.upgrade_station,
                        flow.UpgradeStationSearchObservation(True, position, 1),
                    )
        else:
            unlock = self._vision.find_unlock_button(frame).best
        if unlock is None or not self._is_clickable(unlock.center):
            return flow.decide_check_unlock(False, None)
        clicked = self._input.click(*unlock.center)
        if clicked:
            self.context.check_unlock_clicks += 1
            self._sleep(self._config.level_transition.unlock_settle_delay)
        return flow.decide_check_unlock(True, clicked)

    def _handle_search_upgrade_station(self) -> State:
        station = self._config.upgrade_station
        attempt = self.context.state_attempt
        threshold = self._config.thresholds.upgrade_station
        if attempt > 2:
            threshold -= station.threshold_relaxation
        frame = self._capture_first_asset_frame(
            self._config.capture_regions.upgrade_station_search_y
        )
        upgrade = None
        if self._config.strategy.enabled:
            read = self._vision.read_station_panel(frame)
            panel = self._observe_strategy_panel(frame, read)
            if panel != StationPanel.AFFORDABLE:
                self.context.stats_from_find = True
                return State.UPGRADE_STATS
            upgrade = read.upgrade
        position = self._clickable_upgrade(frame, threshold, upgrade)
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

    def _clickable_upgrade(
        self, frame: np.ndarray, threshold: float, best: MatchCandidate | None = None,
    ) -> Point | None:
        if best is not None and best.confidence >= threshold and self._is_clickable(best.center):
            return best.center
        candidates = self._vision.find_upgrade_station_candidates(frame, threshold)
        return next((c.center for c in candidates if self._is_clickable(c.center)), None)

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
        panel_region = None
        transition = None
        if held:
            self._record_hold(time.monotonic() - hold_started)
            if self._config.strategy.enabled and not self._stop_requested.is_set():
                frame = self._vision.capture()
                transition = self._economy_level_transition(
                    frame, button_max_y=self._config.capture_regions.max_search_y,
                )
                read = self._vision.read_station_panel(frame)
                if read.state == StationPanel.LOCKED:
                    # A verified upgrade was affordable before the hold and is now locked.
                    self.context.station_progress_at = self._clock()
                if transition is None or read.state != StationPanel.UNKNOWN:
                    self._observe_strategy_panel(frame, read)
                panel_region = read.bounds
        post_ok = (
            held
            and (transition is not None or (
                self._click_idle(panel_region) and self._sleep(self._config.flow_timing.state_delay)
            ))
        )
        next_state = flow.decide_hold_upgrade_station(
            self.context,
            self._config.flow_timing,
            flow.UpgradeHoldObservation(True, True, held, post_ok, verified_position=target),
        )
        return transition or next_state

    def _observe_strategy_panel(
        self, frame: np.ndarray, read: StationPanelRead | None = None,
    ) -> StationPanel:
        panel = self._vision.station_panel(frame) if read is None else read.state
        context = self.context
        context.strategy_unknown_panels = (
            context.strategy_unknown_panels + 1 if panel == StationPanel.UNKNOWN else 0
        )
        if context.strategy_unknown_panels >= self._config.upgrade_station.search_attempts:
            if self._recover_stalled_mirror():
                return StationPanel.UNKNOWN
            context.invalidate_viewport()
            raise ObservationError("Station panel remained UNKNOWN; retry localization")
        if (panel in (StationPanel.AFFORDABLE, StationPanel.LOCKED)
                and context.gameplay_phase in (GameplayPhase.FIRST, GameplayPhase.LAST)
                and context.strategy_endpoint is None and context.strategy_anchor is not None
                and context.target_viewport == context.viewport
                and self._strategy_target_frame is not None):
            context.strategy_endpoint = context.strategy_anchor
            self._strategy_endpoint_frame = self._strategy_target_frame.copy()
        if (self._recovery_restart_used and context.strategy_anchor is not None
                and context.target_viewport == context.viewport):
            station = (context.viewport, context.strategy_anchor)
            if panel in (StationPanel.LOCKED, StationPanel.AFFORDABLE):
                self._recovery_station = station
            elif panel == StationPanel.MAX and self._recovery_station == station:
                self._recovery_restart_used = False
        if (panel == StationPanel.MAX and context.strategy_anchor is not None
                and context.target_viewport == context.viewport):
            context.station_progress_at = self._clock()
        flow.observe_station(context, panel)
        return panel

    def _verify_upgrade_station_round(
        self, *, frame: np.ndarray | None = None, near: Point | None = None,
    ) -> tuple[Point, float, np.ndarray, bool] | None:
        """Look near the previous position first, then widen the search if it is lost.

        Each retry uses a fresh capture and the configured relaxed threshold.
        """
        station = self._config.upgrade_station
        base = self._config.thresholds.upgrade_station
        relaxed = base - station.threshold_relaxation
        attempts = max(1, station.verify_search_attempts)
        region = self._upgrade_search_region(near) if near is not None else None
        for attempt in range(attempts):
            threshold = base if attempt == 0 else relaxed
            if frame is None:
                frame = self._vision.capture(
                    max_y=self._config.capture_regions.upgrade_station_search_y
                )
            result = self._vision.find_upgrade_station(frame, threshold, search_region=region)
            if region is not None and (
                result.best is None or not self._is_clickable(result.best.center)
            ):
                region = None  # The panel may have moved. Relocate on this frame.
                result = self._vision.find_upgrade_station(frame, threshold)
            if result.best is not None and self._is_clickable(result.best.center):
                uninterrupted = attempt == 0 and (near is None or region is not None)
                return result.best.center, relaxed, frame, uninterrupted
            frame = None  # Every retry must observe the screen again.
            # v1 doesn't sleep before giving up on the last attempt.
            if attempt < attempts - 1 and not self._sleep(station.verify_search_interval):
                return None
        return None

    def _settling_upgrade_control(
        self, frame: np.ndarray, near: Point, threshold: float,
    ) -> _SettlingControl | None:
        """Identify a single affordable control and its station name on this observation."""
        read = self._vision.read_station_panel(frame, held_position=near, prefer_unlock=True)
        if read.upgrade is None or read.bounds is None or read.state != StationPanel.AFFORDABLE:
            return None
        point, panel = read.upgrade.center, read.bounds
        if math.dist(point, near) > _POPUP_STILL_PX or not self._is_clickable(point):
            return None
        controls = self._vision.find_upgrade_station_candidates(frame, threshold, panel=panel)
        if len(controls) != 1 or math.dist(controls[0].center, point) > _POPUP_STILL_PX:
            return None
        # The calibrated popup has the station name below Level and above the stars.
        # Comparing its ink excludes the changing price, progress bar and game behind it.
        w, h = panel.x_max - panel.x_min, panel.y_max - panel.y_min
        name = frame[panel.y_min + h // 8:panel.y_min + h // 5 + 1,
                     panel.x_min + w // 10:panel.x_max - w // 5]
        ink = name.max(axis=2) < 220
        return _SettlingControl(point, panel, ink) if ink.any() and not ink.all() else None

    def _verification_cursor(self) -> Point | None:
        try:
            return self._input.get_cursor_position()
        except Exception as exc:
            logger.debug("Cursor unavailable for optional settling check: %s", exc)
            return None

    def _verify_upgrade_station(self) -> tuple[Point, float] | None:
        """Require a stable pair before holding. Legacy mode primes with its usual click."""
        station = self._config.upgrade_station
        if not self._sleep(station.verify_settle_delay):
            return None

        candidate = self._config.strategy.enabled
        frame = None
        if candidate and station.verify_settle_delay == 0:
            frame = self._capture_first_asset_frame(
                self._config.capture_regions.upgrade_station_search_y
            )
        first = self._verify_upgrade_station_round(
            frame=frame, near=self.context.upgrade_station_pos if candidate else None,
        )
        if first is None:
            logger.info("Upgrade station was not visible during verification")
            return None

        target, _, first_frame, uninterrupted = first
        context = self.context
        station_identity = (context.viewport, context.strategy_anchor, context.upgrade_station_pos)
        can_settle = (candidate and uninterrupted and context.strategy_anchor is not None
                      and context.target_viewport == context.viewport
                      and context.upgrade_station_pos is not None
                      and math.dist(target, context.upgrade_station_pos)
                      <= station.disappear_position_tolerance_px)
        cursor = self._verification_cursor() if can_settle else None
        if cursor is not None:
            first_frame = first_frame.copy()  # A later capture may overwrite its buffer.
        if not candidate and not self._input.precise_click(*target):
            logger.info("Upgrade station verification click failed at (%s, %s)", *target)
            return None
        interval = (self._config.strategy.target_observation_interval if candidate
                    else station.verify_settle_delay)
        if not self._sleep(interval):
            return None

        second = self._verify_upgrade_station_round(near=target if candidate else None)
        if second is None:
            logger.info("Upgrade station was not visible during verification")
            return None
        point, threshold, current_frame, uninterrupted = second
        if not candidate or math.dist(target, point) <= _POPUP_STILL_PX:
            return point, threshold
        if (cursor is None or not uninterrupted
                or math.dist(target, point) > station.disappear_position_tolerance_px):
            return None

        def same_station() -> bool:
            return (context.target_viewport == context.viewport
                    and station_identity == (context.viewport, context.strategy_anchor,
                                             context.upgrade_station_pos)
                    and not self._stop_requested.is_set() and self._input.is_target_foreground()
                    and self._verification_cursor() == cursor)

        if (not same_station()
                or self._vision.viewport_still(first_frame, current_frame) is not True):
            return None
        previous = self._settling_upgrade_control(first_frame, target, threshold)
        current = self._settling_upgrade_control(current_frame, point, threshold)
        if previous is None or current is None or not previous.same_panel(current):
            return None
        # Only one additional fresh capture. No input, fallback search, or retry loop.
        current_frame = current_frame.copy()
        if not self._sleep(interval) or not same_station():
            return None
        fresh = self._vision.capture(max_y=self._config.capture_regions.upgrade_station_search_y)
        if self._vision.viewport_still(current_frame, fresh) is not True:
            return None
        final = self._settling_upgrade_control(fresh, point, threshold)
        if (final is None or not current.same_panel(final)
                or math.dist(current.point, final.point) > _POPUP_STILL_PX or not same_station()):
            return None
        return final.point, threshold

    def _upgrade_search_region(self, position: Point) -> BoundingBox:
        tolerance = self._config.upgrade_station.disappear_position_tolerance_px
        return BoundingBox(
            position[0] - tolerance, position[1] - tolerance,
            position[0] + tolerance + 1, position[1] + tolerance + 1,
        )

    def _held_station_view(
        self, frame: np.ndarray, threshold: float, position: Point,
    ) -> StationPanelRead:
        """Check the current frame at the relaxed verification threshold.

        A match is only accepted if it's near `position` (the position we're actually holding).
        Live-verified this matters: at the relaxed threshold, once the real station's popup
        changes to its "MAX" state late in a hold, the template can false-positive match an
        unrelated button elsewhere on screen (observed: the footer's "Boost x12" banner, ~389px
        away, at a confidence around/above the base threshold) — high enough that tightening the
        threshold alone would not reliably exclude it. Rejecting matches far from the held
        position catches this regardless of where the false positive happens to appear, without
        assuming the real popup is confined to any particular screen region."""
        station = self._config.upgrade_station
        tolerance = station.disappear_position_tolerance_px
        region = self._upgrade_search_region(position)
        result = self._vision.find_upgrade_station(frame, threshold, search_region=region)
        if result.found and result.best is not None:
            dx = abs(result.best.center[0] - position[0])
            dy = abs(result.best.center[1] - position[1])
            box = result.best.bounding_box
            if dx <= tolerance and dy <= tolerance and (
                not self._config.strategy.enabled or
                (box.x_min <= position[0] < box.x_max and box.y_min <= position[1] < box.y_max)
            ):
                return StationPanelRead(StationPanel.AFFORDABLE, None, upgrade=result.best)
            return StationPanelRead(StationPanel.UNKNOWN, None)
        # A held upgrade resumes when income arrives; grey is not a vanished control.
        if self._config.strategy.enabled:
            read = self._vision.read_station_panel(frame, held_position=position)
            if read.state == StationPanel.LOCKED and not self._vision.stats_icon_seen(frame):
                return read
        return StationPanelRead(StationPanel.UNKNOWN, None)

    def _station_seen_near(self, frame: np.ndarray, threshold: float, position: Point) -> bool:
        return self._held_station_view(frame, threshold, position).state != StationPanel.UNKNOWN

    def _make_disappearance_check(
        self, threshold: float, position: Point, required_misses: int
    ) -> Callable[[], bool]:
        """Requires `required_misses` CONSECUTIVE misses before treating the station as gone —
        debounces a single flaky/transient miss so a real hold isn't cut short by one bad frame."""
        misses = 0
        visible_view: tuple[np.ndarray, bool, StationPanelRead] | None = None
        next_crate_check = -math.inf
        regions = self._config.capture_regions

        def station_seen() -> bool | None:
            """None requests release for visible work; ordinary misses still need confirmation."""
            nonlocal visible_view, next_crate_check
            if not self._config.strategy.enabled:
                frame = self._vision.capture(max_y=regions.upgrade_station_search_y)
                return self._station_seen_near(frame, threshold, position)
            frame = self._vision.capture(max_y=max(
                regions.upgrade_station_search_y, regions.extended_search_y, regions.box_search_y,
            ))
            fast_mode = self._vision.fast_red_icon_mode
            if (visible_view is not None and visible_view[1] == fast_mode
                    and visible_view[0].dtype == frame.dtype
                    and np.array_equal(visible_view[0], frame)):
                read = visible_view[2]
            else:
                if self._vision.new_level_icon_seen(frame[:regions.extended_search_y]):
                    return None
                read = self._held_station_view(frame[:regions.upgrade_station_search_y],
                                               threshold, position)
                if read.state == StationPanel.UNKNOWN:
                    visible_view = None
                    return False
                # The capture backend may reuse its buffer during the next poll.
                visible_view = (frame.copy(), fast_mode, read)
            if (read.state == StationPanel.LOCKED and read.bounds is not None
                    and self._clock() >= next_crate_check):
                next_crate_check = self._clock() + 1.0
                boxes = [box for box in self._vision.find_boxes(frame[:regions.box_search_y])
                         if self._is_clickable(box.center)]
                cap = self._config.box.max_clicks_per_pass
                if (not cap or len(boxes) <= cap) and any(
                    box.bounding_box.iou(read.bounds) == 0 for box in boxes
                ):
                    return None
            return True

        def check() -> bool:
            nonlocal misses
            if not self._device_available():
                return True
            seen = station_seen()
            if seen is False and self._scrcpy_recovery(self._config.scrcpy_recovery.upgrade_delay):
                seen = station_seen()
            if seen is None:
                return True
            misses = 0 if seen else misses + 1
            return misses >= max(1, required_misses)

        return check

    def _economy_level_transition(
        self, frame: np.ndarray, *, button_max_y: int | None = None,
    ) -> State | None:
        if self._vision.find_new_level_button(frame[:button_max_y]).found:
            return State.TRANSITION_LEVEL
        return self._candidate_renovation_footer(frame)

    def _handle_upgrade_stats(self) -> State:
        # Consumed on every entry, even through the idle-click bail-out below, so it can't go stale.
        from_find = self.context.stats_from_find
        self.context.stats_from_find = False
        # Verified v1 behavior: single-shot, no retry loop and no scrcpy-recovery here. This is
        # also the one state where an idle-click failure does NOT retry itself.
        if not self._click_idle():
            return State.OPEN_BOXES
        frame = self._capture_first_asset_frame(self._config.capture_regions.extended_search_y)
        transition = self._economy_level_transition(
            frame, button_max_y=self._config.capture_regions.max_search_y,
        )
        if transition is not None:
            return transition

        stats_seen = self._vision.stats_icon_seen(frame)
        if not stats_seen:
            next_state = flow.decide_upgrade_stats(
                self.context, flow.StatsIconObservation(False, False, from_find)
            )
            if next_state == State.SCROLL and self._config.strategy.enabled:
                return State.OPEN_BOXES
            return next_state

        targets = self._config.click_targets
        button_clicked = self._input.click(*targets.stats_upgrade_button_pos)
        if (button_clicked and self._sleep(self._config.flow_timing.state_delay)
                and self._wait_for_stats_panel()):
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
        return flow.decide_upgrade_stats(
            self.context, flow.StatsIconObservation(False, True, from_find)
        )

    def _wait_for_stats_panel(self) -> bool:
        deadline = self._clock() + self._config.stats_upgrade.panel_ready_timeout
        while (remaining := deadline - self._clock()) > 0:
            if self._stop_requested.is_set():
                return False
            if self._vision.stats_panel_ready(self._vision.capture()):
                return not self._stop_requested.is_set()
            interval = max(0.001, self._config.flow_timing.event_loop_interval)
            if not self._sleep(min(interval, remaining)):
                return False
        logger.warning("Stats panel did not reach its ready position; skipping burst")
        return False

    def _handle_open_boxes(self) -> State:
        if not self._click_idle():
            return State.OPEN_BOXES  # retries this same pass, so the after-scroll flag is kept
        # Consumed only now that the pass is really running (FlowContext.scan_boxes_after_scroll).
        after_scroll = self.context.scan_boxes_after_scroll
        self.context.scan_boxes_after_scroll = False
        box_search_y = self._config.capture_regions.box_search_y
        frame = self._capture_first_asset_frame(box_search_y)
        self._record_box_scan()
        self._update_box_scan_thumbnail(frame)
        config = self._config
        fast_mode = self._vision.fast_red_icon_mode if config.strategy.enabled else None
        if (transition := self._economy_level_transition(frame)) is not None:
            return transition
        boxes = self._vision.find_boxes(frame)
        if not boxes and config.scrcpy_recovery.enabled:
            # The retry capture may overwrite the first buffer. Remember only this pass's miss.
            empty = frame.copy() if config.strategy.enabled else None
            if self._scrcpy_recovery(config.scrcpy_recovery.box_delay):
                frame = self._vision.capture(max_y=box_search_y)
                if (empty is None or self._config is not config
                        or self._vision.fast_red_icon_mode != fast_mode
                        or frame.dtype != empty.dtype or not np.array_equal(frame, empty)):
                    if (transition := self._economy_level_transition(frame)) is not None:
                        return transition
                    boxes = self._vision.find_boxes(frame)

        if self._box_scan_needs_recovery(frame, len(boxes)):
            return self._queue_game_recovery()
        opened, renovation = self._open_visible_boxes(boxes, box_search_y)
        if opened:
            logger.info("Opened %s boxes", opened)
        trips = self.context.box_guard_trips
        next_state = flow.decide_open_boxes(
            self.context,
            self._config.upgrade_station,
            self._config.scroll.max_idle_pass_attempts,
            flow.BoxCycleObservation(False, opened, after_scroll),
        )
        if renovation is not None:
            return renovation
        if self.context.box_guard_trips != trips:
            logger.warning(
                "Box loop guard: %s box passes without a scroll or upgrade; %s. "
                "Last clicked positions: %s",
                self._config.upgrade_station.max_box_only_passes,
                "resuming station search" if self._config.strategy.enabled else "forcing a scroll",
                [(x, y, round(conf, 3)) for x, y, conf in self._recent_box_clicks],
            )
        if next_state == State.SCROLL and self._config.strategy.enabled:
            # Strategy localization owns camera movement. This view already had its crate pass.
            return State.FIND_RED_ICONS
        return next_state

    def _update_box_scan_thumbnail(self, frame: np.ndarray) -> None:
        thumb = _frame_thumbnail(frame)
        previous = self._latest_thumb
        self._scan_delta = None if previous is None else float(np.abs(thumb - previous).mean())
        self._latest_thumb = thumb
        if self._streak_anchor_pending:
            self._streak_anchor_pending = False
            self._streak_thumb = thumb

    def _box_scan_needs_recovery(self, frame: np.ndarray, detected: int) -> bool:
        if self._early_probe_pending:
            self._early_probe_pending = False
            self._log_early_probe(frame, detected)
            # Keep an earlier comparison frame if this scan also triggers the stall probe.
            if self.context.idle_scrolls > 0 and not self._stall_probe_pending:
                self._recent_thumb = _frame_thumbnail(frame)

        if not self._stall_probe_pending:
            return False
        self._stall_probe_pending = False
        self._log_box_near_misses(frame, detected)
        alert_thumb = _frame_thumbnail(frame)
        if self._game_looks_dead(alert_thumb):
            return True
        self._recent_thumb = alert_thumb
        return False

    def _open_visible_boxes(
        self, boxes: list[MatchCandidate], box_search_y: int,
    ) -> tuple[int, State | None]:
        clickable = [box for box in boxes if self._is_clickable(box.center)]
        self.context.box_zone_skips += len(boxes) - len(clickable)
        cap = self._config.box.max_clicks_per_pass
        if cap and len(clickable) > cap:
            # See BoxDetectionConfig.max_clicks_per_pass: not a crate scene, so click nothing. The
            # idle ladder below scrolls on, and idle_scrolls (left alone here) can arm the stall
            # checks.
            self.context.box_floods += 1
            best = sorted(clickable, key=lambda box: -box.confidence)
            logger.warning(
                "Box flood: %d clickable matches > cap %d, none clicked (conf %.3f-%.3f; top: %s)",
                len(clickable),
                cap,
                best[-1].confidence,
                best[0].confidence,
                [(*box.center, round(box.confidence, 3)) for box in best[:3]],
            )
            return 0, None
        opened = 0
        for index, box in enumerate(clickable):
            if index:
                frame = self._vision.capture(max_y=box_search_y)
                if self._config.strategy.enabled and (
                    transition := self._economy_level_transition(frame)
                ) is not None:
                    return opened, transition
                fresh = self._vision.find_boxes(frame)
                refreshed = next((item for item in fresh
                                  if item.bounding_box.iou(box.bounding_box) > 0.2
                                  and self._is_clickable(item.center)), None)
                if refreshed is None:
                    continue
                box = refreshed
            if self._input.click(*box.center):
                opened += 1
                self._recent_box_clicks.append((*box.center, box.confidence))
            else:
                self.context.box_click_fails += 1
        return opened, None

    def _handle_scroll(self) -> State:
        if self._config.strategy.enabled:
            self.context.scan_boxes_after_scroll = True
            return State.OPEN_BOXES
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
            self.context.invalidate_viewport()
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
        self.context.invalidate_viewport()
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

        candidate_checked_open = False
        if not self.context.new_level_red_icon_verified:
            if not self._perform_new_level_verification_scroll():
                logger.warning("Failed to perform verification scroll for new level red icon")
                return flow.decide_check_new_level(
                    self.context,
                    flow.NewLevelVerificationObservation(False, None, None, scroll_succeeded=False),
                )
            if self._config.strategy.enabled:
                frame = self._vision.capture()
                if self._vision.restaurant_start(frame):
                    return State.TRANSITION_LEVEL
                candidate_checked_open = True
                verified = self._vision.new_level_icon_seen(
                    frame[: self._config.capture_regions.extended_search_y]
                )
            else:
                frame = self._vision.capture(max_y=self._config.capture_regions.extended_search_y)
                _, verified, _ = self._vision.split_red_icons(self._vision.find_red_icons(frame))
            if not verified:
                return flow.decide_check_new_level(
                    self.context, flow.NewLevelVerificationObservation(False, None, None)
                )

        targets = self._config.click_targets
        level = self._config.level_transition
        # An Open dialog belongs to a restaurant already reached, including at startup.
        if not candidate_checked_open and self._vision.restaurant_start(self._vision.capture()):
            return State.TRANSITION_LEVEL
        button_ok = self._input.click(*targets.new_level_button_pos)
        if button_ok and not self._sleep(level.confirmation_delay):
            return State.CHECK_NEW_LEVEL
        transition_ok: bool | None = None
        if button_ok:
            transition_ok = self._input.click(*targets.level_transition_pos)
            if transition_ok:
                self._sleep(level.secondary_settle_delay)
        next_state = flow.decide_check_new_level(
            self.context,
            flow.NewLevelVerificationObservation(True, button_ok, transition_ok),
        )
        if next_state == State.CONFIRM_RESTAURANT:
            self.context.nl_route = "footer"
            self.context.transition_pending = True
            self.context.transition_departed = True  # footer verified; Open absent before action
            self.context.transition_confirmations = 0
            self.context.transition_started = self._clock()
            if self._progress is not None:
                self._progress.transition_attempt()
            logger.info("Restaurant transition attempted; awaiting visual confirmation")
        return next_state

    def _handle_transition_level(self, *, require_open: bool = False) -> State:
        level = self._config.level_transition
        attempt = self.context.state_attempt
        # Preserve the visible dialog. Focus and forbidden-zone checks remain in the input layer.
        # v1 captures at max_search_y for this state's search loop, not extended_search_y.
        frame = self._vision.capture(max_y=self._config.capture_regions.max_search_y)
        if require_open and not self._vision.restaurant_start(frame):
            return State.FIND_RED_ICONS
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
        # v1 logs this too. Here it also records where the Open button really is and how sure the
        # match was (the early probe compares a stuck screen's best match with it).
        logger.info(
            "New level button found at %s conf=%.3f on attempt %d",
            result.best.center,
            result.best.confidence,
            attempt,
        )
        fresh_restaurant = self._config.strategy.enabled and self._vision.restaurant_start(frame)
        if require_open and not self._recovery_input_allowed():
            return State.FIND_RED_ICONS
        clicked = self._input.click(*result.best.center)
        if clicked:
            if require_open:
                self._note_recovery_click(result.best.center)
            if fresh_restaurant:
                self.context.strategy_fresh_entry = _VIEWPORT_FRESH_ENTRY_PROBES
            self._sleep(level.settle_delay)
        next_state = flow.decide_transition_level(
            self.context, level, flow.TransitionLevelObservation(True, attempt, clicked)
        )
        if next_state == State.WAIT_FOR_UNLOCK:
            self.context.nl_route = "open"  # the Open-button route; see _end_unlock_sequence
        return next_state

    def _handle_wait_for_unlock(self) -> State:
        level = self._config.level_transition
        # Only the first try of a sequence is timed: the dialog is already up on the retries.
        settle = (
            self._settle_and_time_unlock
            if self.context.wait_for_unlock_attempts == 0
            else self._sleep
        )
        if not settle(self._config.flow_timing.focus_settle_delay):
            return State.WAIT_FOR_UNLOCK
        # v1 captures the full window height here (no max_y limit) — distinct from CHECK_UNLOCK,
        # which uses the same "unlock" template against a max_search_y-limited frame.
        frame = self._vision.capture()
        result = self._vision.find_unlock_button(frame)
        if self._verify_recovered_unlock:
            self._verify_recovered_unlock = False
            if result.best is None or not recovery_unlock_visible(
                self._vision, frame, result.best.center
            ):
                return State.FIND_RED_ICONS
        if result.best is None or not self._is_clickable(result.best.center):
            # The gate found nothing. Whether that is a greyed-out button (a raw match just under
            # the gate, so more tries cannot help) decides if the retries are worth their ~1.3 s.
            # One match only: it also feeds the miss line below.
            seen = None
            if result.best is None and level.greyed_unlock_early_out:
                seen = self._vision.unlock_near_miss(frame)
            greyed = seen is not None and seen.confidence >= level.greyed_unlock_min_conf
            if not greyed:
                self._sleep(level.unlock_search_interval)
            cut_short = greyed and self.context.wait_for_unlock_attempts + 1 < (
                level.unlock_search_attempts
            )
            missed = flow.decide_wait_for_unlock(
                self.context, level, flow.WaitForUnlockObservation(False, None, greyed)
            )
            if missed != State.WAIT_FOR_UNLOCK:  # attempts exhausted: back to the economy loop
                self.context.nl_greyed_exits += cut_short
                self._note_unlock_miss(frame, seen)
                self._end_unlock_sequence(completed=False)
            return (State.OPEN_BOXES
                    if missed == State.FIND_RED_ICONS and self.context.strategy_fresh_entry
                    else missed)

        clicked = self._input.click(*result.best.center)
        next_state = flow.decide_wait_for_unlock(
            self.context, level, flow.WaitForUnlockObservation(True, clicked)
        )
        if clicked:
            self._sleep(level.unlock_settle_delay)
        if next_state != State.WAIT_FOR_UNLOCK:
            if not clicked:
                self._note_unlock_miss(frame)
            self._end_unlock_sequence(completed=False)
        return (State.OPEN_BOXES
                if next_state == State.FIND_RED_ICONS and self.context.strategy_fresh_entry
                else next_state)

    def _handle_city_arrival(self, frame: np.ndarray) -> bool:
        if self._vision.city_arrival(frame):
            self._continue_city(frame)
            return True
        self.context.city_seen = False
        return False

    def _continue_city(self, frame: np.ndarray, *, recovering: bool = False) -> bool:
        if self._progress is not None and not self.context.city_seen:
            self._progress.city_boundary()
        self.context.city_seen = True
        result = self._vision.find_city_continue(frame)
        if (result.best is None or (recovering and not self._recovery_input_allowed())
                or not self._input.click(*result.best.center)):
            return False
        if recovering:
            self._note_recovery_click(result.best.center)
        self.context.strategy_fresh_entry = (
            _VIEWPORT_FRESH_ENTRY_PROBES if self._config.strategy.enabled else 0
        )
        self._sleep(self._config.level_transition.settle_delay)
        return True

    def _handle_confirm_restaurant(self) -> State:
        probe_started = self._clock()
        frame = self._vision.capture()
        if self._progress is not None:
            self._progress.checkpoint(boost=self._vision.boost_level(frame))
        city = self._handle_city_arrival(frame)
        if flow.confirm_restaurant(self.context, not city and self._vision.restaurant_start(frame)):
            self.context.station_progress_at = self._clock()
            self._recovery_restart_used = False
            self.context.strategy_fresh_entry = (
                _VIEWPORT_FRESH_ENTRY_PROBES if self._config.strategy.enabled else 0
            )
            if self._progress is not None:
                self._progress.confirm_transition()
            self._report_level_completion()
            self._end_unlock_sequence(completed=True)
            if self._config.strategy.enabled and self._vision.find_new_level_button(
                frame[: self._config.capture_regions.max_search_y]
            ).found:
                # The transition handler verifies the button again before clicking.
                return flow.decide_find_red_icons(
                    self.context, flow.RedIconScanObservation(True, False, ())
                )
            return State.FIND_RED_ICONS
        deadline = (
            self.context.transition_started + self._config.level_transition.confirmation_timeout
        )
        now = self._clock()
        if now >= deadline:
            logger.warning("Restaurant transition unconfirmed; no completion credited")
            self.context.transition_pending = False
            if self._progress is not None:
                self._progress.cancel_transition()
            self._end_unlock_sequence(completed=False)
            return State.FIND_RED_ICONS
        next_probe = min(probe_started + 0.2, deadline)
        if next_probe > now:
            self._sleep(next_probe - now)
        return State.CONFIRM_RESTAURANT

    def _strategy_scroll(self, direction: int, *, shorter: bool = False) -> bool | None:
        """False means unchanged views; True means changed/uncertain; None cancels the attempt."""
        context = self.context
        if context.strategy_scrolls >= self._config.strategy.viewport_limit:
            raise _StationSearchSpent("Station viewport gesture budget spent")
        context.strategy_scrolls += 1
        context.strategy_stationary_count = 0  # only a completed observation can restore evidence
        before = self._vision.capture()
        scroll = self._config.scroll
        x, y = self._config.click_targets.scroll_start_pos
        distance = round(scroll.pixel_step * scroll.distance_ratio)
        if shorter:
            distance = max(1, distance // 2)
        destination = y - direction * distance
        moved = self._input.drag(x, y, x, destination, duration=scroll.duration)
        if not moved:
            return None
        logger.info("Station scroll %s (mouse drag %s): %s/%s gestures",
                    "down" if direction > 0 else "up", "up" if direction > 0 else "down",
                    self.context.strategy_scrolls, self._config.strategy.viewport_limit)
        self.context.invalidate_viewport()
        if not self._sleep(scroll.post_scroll_settle):
            return None
        interval = self._config.strategy.target_observation_interval
        # Use the mandatory pause to separate the first pair of camera observations.
        settled = self._vision.capture() if scroll.interval_pause >= interval else None
        if not self._sleep(scroll.interval_pause):
            return None
        previous = self._vision.capture()
        still_samples = int(settled is not None and
                            self._vision.viewport_still(settled, previous) is True)
        samples = max(
            _VIEWPORT_MAX_SETTLE_SAMPLES,
            math.ceil(_VIEWPORT_SETTLE_WAIT_BUDGET / interval),
            math.ceil(_VIEWPORT_MAX_SETTLE_SAMPLES * scroll.post_scroll_settle / interval),
        )
        for _ in range(samples):
            if not self._sleep(interval):
                return None
            current = self._vision.capture()
            still_samples = (still_samples + 1
                             if self._vision.viewport_still(previous, current) is True else 0)
            if still_samples >= _VIEWPORT_REQUIRED_STILL_SAMPLES:
                # Finish the ordinary asset cycle before another camera gesture.
                self.context.stats_from_find = True
                self.context.scan_boxes_after_scroll = True
                self._station_landing = _StationLanding(
                    context.viewport, current, self._strategy_targets(current),
                )
                stationary = self._vision.viewport_still(before, current)
                if stationary is False and not context.strategy_pass_moved:
                    shift = self._vision.viewport_translation(before, current)
                    if (shift is not None and -direction * shift[1] >= max(4, 2 * abs(shift[0]))
                            and _restaurant_visible(self._vision, current, after_action=True,
                                                    for_scroll=True)):
                        context.strategy_pass_moved = True
                        self._strategy_alive_at = self._clock()
                return stationary is not True
            previous = current
        raise _ViewportError("Station viewport did not settle; recalibrate scroll pauses")

    def _handle_strategy_find(self, frame: np.ndarray) -> State:
        context = self.context
        thumb = _frame_thumbnail(frame)
        previous = self._strategy_health_thumb
        now = self._clock()
        recovery = self._config.game_recovery
        delta = 0.0 if previous is None else float(np.abs(thumb - previous).mean())
        moving = previous is not None and delta >= recovery.static_frame_delta
        if (context.strategy_waiting_liveness and moving
                and self._station_view_live(frame, moving=True)):
            return self._retry_station_pass("game activity returned")
        if self._stop_requested.is_set():
            return State.FIND_RED_ICONS
        if previous is None or delta >= recovery.static_recent_delta:
            self._strategy_alive_at = now
            self._strategy_health_thumb = thumb
        elif (now - self._strategy_alive_at >= self._config.game_recovery.stale_after_seconds
                and self._config.game_recovery.enabled and self._launcher is not None):
            return self._queue_game_recovery()
        if context.strategy_waiting_liveness:
            if self._station_recheck_at is not None and now >= self._station_recheck_at:
                self._station_recheck_at = now + RECOVERY_PROBE_SECONDS
                result = self._check_station_pass(frame, "rechecking station search", moving=moving)
                if result is not None:
                    return result
            self._sleep(self._config.scroll.post_scroll_settle)
            return State.FIND_RED_ICONS
        try:
            return self._strategy_locate(frame)
        except _ViewportError as exc:
            if not isinstance(exc, _StationSearchSpent) and not self._config.game_recovery.enabled:
                raise
            if isinstance(exc, _StationSearchSpent):
                result = self._check_station_pass(frame, str(exc), moving=moving)
                if result is not None:
                    return result
                self._station_recheck_at = self._clock() + RECOVERY_PROBE_SECONDS
            elif self._recover_stalled_mirror():
                return State.FIND_RED_ICONS
            if self._stop_requested.is_set():
                return State.FIND_RED_ICONS
            context.strategy_waiting_liveness = True
            context.invalidate_viewport()
            if self._progress is not None:
                self._progress.checkpoint("recovery")
            logger.warning("%s; checking screen liveness phase=%s gestures=%d",
                           exc, context.gameplay_phase.value, context.strategy_scrolls)
            return State.FIND_RED_ICONS

    def _check_station_pass(self, frame: np.ndarray, reason: str, *, moving: bool) -> State | None:
        """An empty pass is normal. An inconclusive screen gets read-only observations."""
        if (not self._device_available(force=True) or self._stop_requested.is_set()
                or not self._input.is_target_foreground()):
            return State.FIND_RED_ICONS
        if self.context.strategy_pass_moved:
            return self._retry_station_pass(reason)
        if (self._config.game_recovery.enabled and self._launcher is not None
                and self._launcher.package is not None):
            view = self._probe_recovery_screen()
            if (not self._device_available(force=True) or self._stop_requested.is_set()
                    or not self._input.is_target_foreground()):
                return State.FIND_RED_ICONS
            if view.status == "mirror-stale":
                return self._queue_game_recovery("phone progresses while scrcpy display is frozen")
            if view.status == "playable":
                self._strategy_alive_at = self._clock()
                return self._retry_station_pass(reason)
            if view.status == "still":
                unresponsive = self._launcher.game_unresponsive()
                if self._stop_requested.is_set():
                    return State.FIND_RED_ICONS
                if unresponsive:
                    return self._queue_game_recovery(STATION_SEARCH_ANR)
                if view.world_views is not None:
                    return self._retry_station_pass(reason)
        if self._station_view_live(frame, moving=moving):
            return self._retry_station_pass(reason)
        return None

    def _station_view_live(self, frame: np.ndarray, *, moving: bool) -> bool:
        """After a failed pass, require a fresh, foreground game view before resuming input."""
        if not self._sleep(self._config.strategy.target_observation_interval):
            return False
        fresh = self._vision.capture(max_y=frame.shape[0])
        thumb = _frame_thumbnail(fresh)
        if not moving and float(np.abs(thumb - _frame_thumbnail(frame)).mean()) < (
            self._config.game_recovery.static_frame_delta
        ):
            return False
        if (not self._device_available(force=True) or self._stop_requested.is_set()
                or not self._input.is_target_foreground()):
            return False
        if self._phone_game_foreground is False:
            return False
        if not (
            (self._phone_game_foreground is True and self._vision.viewport_still(frame, fresh)
             is True)
            or self._vision.city_arrival(fresh) or self._vision.restaurant_start(fresh)
            or self._vision.find_new_level_button(
                fresh[:self._config.capture_regions.max_search_y]).found
            or self._vision.new_level_icon_seen(fresh) or self._vision.stats_icon_seen(fresh)
            or self._vision.station_targets(fresh) or self._vision.find_boxes(fresh)
        ):
            return False
        if self._stop_requested.is_set():
            return False
        self._strategy_alive_at = self._clock()
        self._strategy_health_thumb = thumb
        return True

    def _retry_station_pass(self, reason: str) -> State:
        context = self.context
        logger.info("Station search retry phase=%s gestures=%d station_progress_age_s=%.1f "
                    "reason=%s", context.gameplay_phase.value, context.strategy_scrolls,
                    self._clock() - context.station_progress_at, reason)
        edge_count = context.strategy_stationary_count if context.strategy_seek else 0
        if not context.strategy_seek and context.strategy_stationary_count >= 2:
            edge_count = -2  # Empty inward edge. Turn back on the next drag.
        context.invalidate_viewport()
        context.strategy_waiting_liveness = False
        self._station_recheck_at = None
        context.strategy_seek = True
        context.strategy_scrolls = 0
        context.strategy_stationary_count = edge_count
        context.strategy_reacquiring = True
        context.strategy_pass_moved = False
        if self._progress is not None:
            self._progress.checkpoint("active")
        return State.FIND_RED_ICONS  # fresh New Level / city check before economy work or a probe

    def _strategy_sweep_direction(self) -> int:
        context = self.context
        if context.gameplay_phase == GameplayPhase.REVERSE_MIDDLE:
            return 1 if context.strategy_seek else -1
        remaining = context.strategy_scrolls
        scroll = self._config.scroll
        for cycle in range(1, scroll.max_cycles + 1):
            for direction in (1, -1):
                leg_length = cycle * scroll.increment_step
                if remaining < leg_length:
                    return direction
                remaining -= leg_length
        # A final moving gesture can reach an edge without proving it stationary. Preserve
        # both original cycles, then allow one largest-leg approach plus two edge checks.
        if remaining < scroll.max_cycles * scroll.increment_step + 2:
            return -1 if context.gameplay_phase == GameplayPhase.FIRST else 1
        raise _StationSearchSpent("Station search exhausted the configured down/up sweep")

    def _strategy_targets(self, frame: np.ndarray) -> list[MatchCandidate]:
        if self.context.gameplay_phase not in (GameplayPhase.FIRST, GameplayPhase.LAST):
            return [target for target in self._vision.station_targets(frame)
                    if self._is_clickable(target.center)]
        matched, positional = self._vision.endpoint_station_targets(frame)
        targets = [target for target in matched if self._is_clickable(target.center)]
        for target in positional:
            protected_last = (self.context.gameplay_phase == GameplayPhase.LAST
                              and any(zone.contains(*target.center) for zone in
                                      self._config.forbidden_zones.event_zone_options.values()))
            if (self._is_clickable(target.center) or protected_last) and not any(
                math.dist(target.center, known.center) < self._config.red_icon.fast_min_distance
                for known in targets
            ):
                targets.append(target)
        endpoint = self.context.strategy_endpoint
        if endpoint is not None and targets:
            reference = self._strategy_endpoint_frame
            if reference is None:
                return []
            height = min(reference.shape[0], frame.shape[0])
            before, after = reference[:height], frame[:height]
            if self._vision.viewport_still(before, after) is not True:
                shift = self._vision.viewport_translation(before, after)
                if shift is None or abs(shift[0]) >= ROW_DEDUPE_DISTANCE:
                    return []
                endpoint = endpoint[0] + shift[0], endpoint[1] + shift[1]
            match = flow.station_at_projection(targets, endpoint)
            return [match] if match is not None else []
        return targets

    def _localization_panel_bounds(self, frame: np.ndarray) -> BoundingBox | None:
        read = self._vision.read_station_panel(frame)
        if read.bounds is None and read.state != StationPanel.UNKNOWN:
            raise ObservationError("Station panel bounds unavailable before localization")
        return read.bounds

    def _dismiss_station_panel(self, frame: np.ndarray) -> np.ndarray | None:
        previous_panel: BoundingBox | None = None
        panel = self._localization_panel_bounds(frame)
        for _ in range(_VIEWPORT_MAX_PANEL_DISMISSALS):
            if panel is None:
                return frame
            # An outside tap can select a different station instead of clearing the overlay.
            persistent = (
                previous_panel is not None and previous_panel.iou(panel) >= _VIEWPORT_SAME_PANEL_IOU
            )
            if not self._click_idle(panel, alternate=persistent):
                if (self.context.state_attempt >= self._config.upgrade_station.search_attempts
                        and not self._stop_requested.is_set()
                        and self._input.is_target_foreground()):
                    raise ObservationError("No safe dismissal input before viewport localization")
                return None
            if not self._sleep(self._config.scrcpy_recovery.action_settle_delay):
                return None
            previous_panel = panel
            frame = self._vision.capture()
            panel = self._localization_panel_bounds(frame)
            if panel is None:
                if not self._sleep(self._config.scrcpy_recovery.action_settle_delay):
                    return None
                frame = self._vision.capture()
                panel = self._localization_panel_bounds(frame)
        if panel is not None:
            raise _ViewportError("Station panel did not close before viewport localization")
        return frame

    def _reuse_station_anchor(self, frame: np.ndarray, anchor: Point) -> State:
        context = self.context
        reference = self._strategy_target_frame
        still = self._vision.viewport_still(reference, frame) if reference is not None else None
        tolerance = self._config.red_icon.fast_min_distance
        if (still is True and anchor[1] >= _MIN_STATION_TARGET_Y + tolerance
                and self._is_clickable(anchor)):
            context.current_red_icon_index = 0
            return State.CLICK_RED_ICON
        current_target = next(
            (target for target in self._strategy_targets(frame)
             if math.dist(target.center, anchor) < tolerance),
            None,
        )
        if current_target is None:
            if still is False:
                context.invalidate_viewport()
                context.strategy_seek = True
                context.strategy_scrolls = context.strategy_stationary_count = 0
            elif still is True and self._is_clickable(anchor):
                context.current_red_icon_index = 0
                return State.CLICK_RED_ICON
            elif context.state_attempt >= self._config.upgrade_station.search_attempts:
                raise _ViewportError(
                    "Selected station remained unobservable; recalibrate localization"
                )
            else:
                self._sleep(self._config.scroll.post_scroll_settle)
            return State.FIND_RED_ICONS
        if (current_target.center[1] < _MIN_STATION_TARGET_Y
                or not self._is_clickable(current_target.center)):
            return self._reframe_strategy_target(frame, current_target.center)
        self._strategy_target_frame = frame
        context.strategy_anchor = current_target.center
        context.red_icons = [current_target]
        context.current_red_icon_index = 0
        return State.CLICK_RED_ICON

    def _observe_station_targets(
        self, frame: np.ndarray, *, fresh_entry: bool, prefer_visible: bool,
    ) -> tuple[np.ndarray, MatchCandidate | None] | State | None:
        context = self.context
        landing, self._station_landing = self._station_landing, None
        can_confirm = (
            not context.strategy_seek or fresh_entry or context.strategy_stationary_count >= 2
            or context.strategy_endpoint is not None
        )
        if not can_confirm and not prefer_visible:
            return frame, None

        observed_at = self._clock()
        if (landing is not None and landing.viewport == context.viewport
                and np.array_equal(landing.frame, frame)):
            views = [landing.targets]
        else:
            views = [self._strategy_targets(frame)]
        if not can_confirm and not views[0]:
            return frame, None

        def confirmed_target() -> MatchCandidate | None:
            y_ranges: dict[MatchCandidate, tuple[int, int]] = {}
            targets = flow.confirmed_station_targets(
                views, self._config.red_icon.fast_min_distance, y_ranges=y_ranges,
            )
            return flow.strategy_target(context, targets, y_ranges=y_ranges)

        # Two of three views reject transient effects while allowing one occluded observation.
        fresh = frame
        for _ in range(2):
            remaining = self._config.strategy.target_observation_interval - (
                self._clock() - observed_at)
            if not self._sleep(max(0.0, remaining)):
                return None
            fresh = self._vision.capture()
            observed_at = self._clock()
            if (renovation := self._candidate_renovation_footer(fresh)) is not None:
                return renovation
            if self._localization_panel_bounds(fresh) is not None:
                return None
            if len(views) == 2 and not any(views):
                return fresh, None  # A third view alone cannot confirm a station.
            views.append(self._strategy_targets(fresh))
            if len(views) != 2 or not views[0]:
                continue
            if sorted(
                (candidate.template_name, candidate.center) for candidate in views[0]
            ) == sorted((candidate.template_name, candidate.center) for candidate in views[1]):
                break
            if context.gameplay_phase == GameplayPhase.LAST:
                # Changes in other stations need no extra scan if LAST itself agrees in both.
                earlier, latest = (flow.strategy_target(context, view) for view in views)
                if (earlier is not None and latest is not None
                        and (earlier.center, earlier.template_name)
                        == (latest.center, latest.template_name)):
                    target = confirmed_target()
                    if target == latest:
                        return fresh, target
        return fresh, confirmed_target()

    def _strategy_locate(self, frame: np.ndarray) -> State:
        ready_frame = self._dismiss_station_panel(frame)
        if ready_frame is None:
            return State.FIND_RED_ICONS
        if (ready_frame is not frame
                and (renovation := self._candidate_renovation_footer(ready_frame)) is not None):
            return renovation
        context = self.context
        if (context.strategy_anchor is not None and context.target_viewport == context.viewport
                and context.red_icons):
            return self._reuse_station_anchor(ready_frame, context.strategy_anchor)

        # ponytail: only confirmed arrivals reset the camera; manual resumes retain the sweep.
        fresh_entry = (context.strategy_fresh_entry > 0
                       and context.gameplay_phase == GameplayPhase.FIRST)
        context.strategy_fresh_entry = context.strategy_fresh_entry - 1 if fresh_entry else 0
        prefer_visible = (not self._config.strategy.camera_edge_checks_first
                          and context.gameplay_phase == GameplayPhase.FIRST)
        observation = self._observe_station_targets(
            ready_frame, fresh_entry=fresh_entry, prefer_visible=prefer_visible,
        )
        if observation is None:
            return State.FIND_RED_ICONS
        if isinstance(observation, State):
            return observation
        fresh, target = observation
        if (target is None and context.strategy_reacquiring and context.strategy_seek
                and context.strategy_stationary_count >= 2
                and context.strategy_endpoint is not None):
            # The saved endpoint also missed at the proven edge. Rank fresh observations here.
            context.strategy_endpoint = None
            self._strategy_endpoint_frame = None
            return State.FIND_RED_ICONS
        if target is not None and context.strategy_endpoint is not None:
            # The saved world frame identifies the endpoint even after a recovery invalidates
            # the old camera edge. Reaching that edge again could move it out of view.
            context.strategy_seek = False
        if fresh_entry:
            if target is None:
                context.stats_from_find = True
                return State.UPGRADE_STATS  # Recheck starter crates before the next bounded probe.
            context.strategy_fresh_entry = 0
            context.strategy_seek = False
            logger.info("Fresh FIRST verified across observations; skipping camera sweep")
        if (prefer_visible and context.strategy_seek and not fresh_entry and target is not None
                and self._vision.viewport_still(ready_frame, fresh) is not True):
            target = None
            context.strategy_stationary_count = 0
            logger.info("Visible station camera not yet stable; continuing bounded sweep")
        if context.strategy_seek and not fresh_entry:
            next_state = self._seek_station_edge(prefer_visible and target is not None)
            if next_state is not None:
                return next_state
        if target is None:
            if context.strategy_expected_last is not None:
                if context.state_attempt >= self._config.upgrade_station.search_attempts:
                    return self._retry_last_reframe("projected target remained unobservable")
                return State.FIND_RED_ICONS
            return self._scroll_for_missing_station()
        if target.center[1] < _MIN_STATION_TARGET_Y or not self._is_clickable(target.center):
            # Leave room above the station for a complete Level/MAX panel below the HUD.
            return self._reframe_strategy_target(fresh, target.center)
        if context.strategy_reacquiring:
            logger.info("Station search reacquired phase=%s gestures=%d",
                        context.gameplay_phase.value, context.strategy_scrolls)
            context.strategy_reacquiring = False
        context.strategy_pass_moved = False
        context.strategy_scrolls = 0
        context.strategy_stationary_count = 0
        context.strategy_expected_last = None
        context.strategy_anchor = target.center
        context.target_viewport = context.viewport
        context.red_icons = [target]
        self._strategy_target_frame = fresh
        context.current_red_icon_index = 0
        if context.gameplay_phase == GameplayPhase.LAST:
            logger.info("Selected LAST station at %s (%s)", target.center, target.template_name)
        elif target.template_name == "stationPosition":
            logger.info("Positional %s station fallback at %s; panel verification required",
                        context.gameplay_phase.value, target.center)
        return State.CLICK_RED_ICON

    def _reframe_strategy_target(self, frame: np.ndarray, point: Point) -> State:
        context = self.context
        last = context.gameplay_phase == GameplayPhase.LAST
        if last and context.strategy_scrolls >= self._config.strategy.viewport_limit:
            return self._retry_last_reframe("reframe gesture limit reached")
        moved = self._strategy_scroll(-1, shorter=last and context.strategy_short_last_reframe)
        if moved is None:
            return State.FIND_RED_ICONS
        if last:
            if not moved:
                return self._retry_last_reframe("camera did not move")
            after = self._vision.capture(max_y=frame.shape[0])
            shift = self._vision.viewport_translation(frame, after)
            if shift is None or abs(shift[0]) >= ROW_DEDUPE_DISTANCE or shift[1] <= 0:
                return self._retry_last_reframe("camera displacement could not be verified")
            context.strategy_expected_last = (point[0] + shift[0], point[1] + shift[1])
            context.strategy_seek = False
        return State.UPGRADE_STATS

    def _retry_last_reframe(self, reason: str) -> State:
        context = self.context
        context.invalidate_viewport()
        context.strategy_seek = True
        context.strategy_scrolls = context.strategy_stationary_count = 0
        # Shorter drags preserve overlap for tracking. Retry the full drag if the short one fails.
        context.strategy_short_last_reframe = not context.strategy_short_last_reframe
        context.stats_from_find = context.scan_boxes_after_scroll = True
        logger.warning(
            "Retrying LAST from the bottom edge (%s); next reframe is %s",
            reason, "half distance" if context.strategy_short_last_reframe else "full distance",
        )
        return State.UPGRADE_STATS

    def _seek_station_edge(self, visible_target: bool) -> State | None:
        context = self.context
        if visible_target:
            context.strategy_seek = False
            logger.info("Stable visible %s station selected before camera sweep",
                        context.gameplay_phase.value)
            return None
        if context.strategy_stationary_count >= 2:
            context.strategy_seek = False
            context.strategy_scrolls = context.strategy_stationary_count = 0
            return None
        edge_direction = -1 if context.gameplay_phase == GameplayPhase.FIRST else 1
        # Verify a possible endpoint before the sweep reverses away from it.
        approach_edge = context.strategy_stationary_count in (1, -2)
        direction = edge_direction if approach_edge else self._strategy_sweep_direction()
        count = context.strategy_stationary_count
        moved = self._strategy_scroll(direction)
        if moved is not None:
            if direction == edge_direction:
                # -2 remembers the opposite edge until we reach the required endpoint.
                if not moved:
                    context.strategy_stationary_count = max(0, count) + 1
                else:
                    context.strategy_stationary_count = -2 if count == -2 else 0
            else:
                context.strategy_stationary_count = 0 if moved else max(-2, min(0, count) - 1)
        return State.UPGRADE_STATS if moved is not None else State.FIND_RED_ICONS

    def _scroll_for_missing_station(self) -> State:
        context = self.context
        if context.strategy_stationary_count >= 2:
            raise _StationSearchSpent("Verified empty end of the inward station search")
        direction = 1 if context.gameplay_phase == GameplayPhase.FIRST else -1
        count = context.strategy_stationary_count
        moved = self._strategy_scroll(direction)
        if moved is None:
            return State.FIND_RED_ICONS
        context.strategy_stationary_count = 0 if moved else count + 1
        if (context.gameplay_phase == GameplayPhase.REVERSE_MIDDLE
                and context.strategy_stationary_count >= 2
                and not context.strategy_reanchored):
            context.strategy_reanchored = True
            context.strategy_seek = True
            context.strategy_scrolls = context.strategy_stationary_count = 0
        return State.UPGRADE_STATS

    def _end_unlock_sequence(self, *, completed: bool) -> None:
        """One New Level attempt is over, completed or given up: tally it against the way it was
        entered (footer badge or Open button, set by the handler that sent it to WAIT_FOR_UNLOCK).
        """
        context = self.context
        route = context.nl_route or "other"
        context.nl_route = None
        context.nl_attempts[route] = context.nl_attempts.get(route, 0) + 1
        if completed:
            context.nl_done[route] = context.nl_done.get(route, 0) + 1

    def _note_unlock_miss(
        self, frame: np.ndarray, best: TemplateExplanation | None = None
    ) -> None:
        self.context.nl_unlock_miss += 1
        # best_conf = the raw best `unlock` match, threshold not applied: low = nothing like the
        # button on screen, 0.85-0.90 = the button there but greyed out (cash or a requirement is
        # short). via = the way in (footer badge or Open button). Diagnostics only.
        if best is None:
            best = self._vision.unlock_near_miss(frame)
        logger.info(
            "New level: unlock button not found (miss %d, levels=%d, best_conf=%s) via=%s",
            self.context.nl_unlock_miss,
            self.context.total_levels_completed,
            "-" if best is None else f"{best.confidence:.3f}",
            self.context.nl_route or "-",
        )
