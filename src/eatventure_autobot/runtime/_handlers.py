"""The eleven state handlers and the helpers only they use.

Moved out of bot.py unchanged; EatventureBot inherits it. What it reaches on self is declared
at the top of the class."""

import logging
import math
import time
from collections import deque
from collections.abc import Callable
from threading import Event
from typing import TYPE_CHECKING

import numpy as np

from eatventure_autobot.domain.config import BotConfig
from eatventure_autobot.domain.errors import ObservationError
from eatventure_autobot.domain.protocols import InputController
from eatventure_autobot.domain.state import GameplayPhase, State, StationPanel
from eatventure_autobot.domain.types import BoundingBox, MatchCandidate, Point, TemplateExplanation
from eatventure_autobot.runtime._telemetry import _frame_thumbnail
from eatventure_autobot.runtime.progress import ProgressLedger
from eatventure_autobot.runtime.vision import GameVision
from eatventure_autobot.state import transitions as flow
from eatventure_autobot.state.context import FlowContext, LevelMark

logger = logging.getLogger("eatventure_autobot.runtime.bot")

# Two consecutive scans of the station popup this close (px) count as "has stopped moving"
# (ScrcpyRecoveryConfig.popup_early_exit). The matcher reports whole-pixel positions, so a settled
# panel reads 0; 2 only absorbs a single sub-pixel rounding flip.
_POPUP_STILL_PX = 2.0
_VIEWPORT_MAX_SETTLE_SAMPLES = 12
_VIEWPORT_REQUIRED_STILL_SAMPLES = 2
_VIEWPORT_MAX_PANEL_DISMISSALS = 2
_VIEWPORT_FRESH_ENTRY_PROBES = 3
_VIEWPORT_SAME_PANEL_IOU = 0.5
_MIN_STATION_TARGET_Y = 300


class _ViewportError(ObservationError):
    """Bounded viewport readiness failure, separate from unrelated detector failures."""


class _HandlersMixin:
    # What this mixin reaches on self. EatventureBot.__init__ (bot.py) creates the attributes;
    # the methods live in bot.py or another mixin.
    _clock: Callable[[], float]
    _config: BotConfig
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
    _strategy_health_thumb: np.ndarray | None
    _strategy_target_frame: np.ndarray | None
    _strategy_alive_at: float
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

        def _relaunch_game(self) -> State:
            ...

        def _queue_game_recovery(self) -> State: ...

        def _arm_resume(self, reason: str) -> None: ...

        def stop(self, *, manual: bool = False) -> None: ...

        def _level_mark(self) -> LevelMark:
            ...

        def _report_level_completion(self) -> None:
            ...

        def _device_available(self, *, force: bool = False) -> bool: ...

        def _recover_stalled_mirror(self) -> bool: ...

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

    def _handle_find_red_icons(self) -> State:
        monitoring = self._config.strategy.enabled and self.context.strategy_waiting_liveness
        if not self._config.strategy.enabled and not self._click_idle():
            return State.FIND_RED_ICONS
        regions = self._config.capture_regions
        frame = self._vision.capture(max_y=regions.extended_search_y)
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

        candidates = self._vision.find_red_icons(frame)
        actionable, new_level_seen, _ = self._vision.split_red_icons(candidates)
        if self._config.strategy.enabled and not new_level_seen:
            return self._handle_strategy_find(frame)
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
        region = None
        if self._config.strategy.enabled:
            panel = self._observe_strategy_panel(frame)
            if panel in (StationPanel.MAX, StationPanel.UNKNOWN):
                return State.UPGRADE_STATS
            region = self._vision.station_panel_bounds(frame)
        result = (self._vision.find_unlock_button(frame, search_region=region)
                  if self._config.strategy.enabled else self._vision.find_unlock_button(frame))
        if result.best is None or not self._is_clickable(result.best.center):
            return flow.decide_check_unlock(False, None)
        clicked = self._input.click(*result.best.center)
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
        frame = self._vision.capture(max_y=self._config.capture_regions.upgrade_station_search_y)
        if self._config.strategy.enabled:
            panel = self._observe_strategy_panel(frame)
            if panel != StationPanel.AFFORDABLE:
                self.context.stats_from_find = True
                return State.UPGRADE_STATS
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
        panel_region = None
        if held:
            self._record_hold(time.monotonic() - hold_started)
            if self._config.strategy.enabled:
                frame = self._vision.capture()
                self._observe_strategy_panel(frame)
                panel_region = self._vision.station_panel_bounds(frame)
        post_ok = (
            held
            and self._click_idle(panel_region)
            and self._sleep(self._config.flow_timing.state_delay)
        )
        return flow.decide_hold_upgrade_station(
            self.context,
            self._config.flow_timing,
            flow.UpgradeHoldObservation(True, True, held, post_ok, verified_position=target),
        )

    def _observe_strategy_panel(self, frame: np.ndarray) -> StationPanel:
        panel = self._vision.station_panel(frame)
        context = self.context
        context.strategy_unknown_panels = (
            context.strategy_unknown_panels + 1 if panel == StationPanel.UNKNOWN else 0
        )
        if context.strategy_unknown_panels >= self._config.upgrade_station.search_attempts:
            if self._recover_stalled_mirror():
                return StationPanel.UNKNOWN
            context.invalidate_viewport()
            raise ObservationError("Station panel remained UNKNOWN; retry localization")
        flow.observe_station(context, panel)
        return panel

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
        """Two fresh checks before holding. Candidate checks cannot spend the available cash;
        legacy mode retains its priming click between rounds."""
        station = self._config.upgrade_station
        if not self._sleep(station.verify_settle_delay):
            return None

        first = self._verify_upgrade_station_round()
        if first is None:
            logger.info("Upgrade station was not visible during verification")
            return None

        target, _ = first
        candidate = self._config.strategy.enabled
        if not candidate and not self._input.precise_click(*target):
            logger.info("Upgrade station verification click failed at (%s, %s)", *target)
            return None
        interval = (self._config.strategy.target_observation_interval if candidate
                    else station.verify_settle_delay)
        if not self._sleep(interval):
            return None

        second = self._verify_upgrade_station_round()
        if second is None:
            logger.info("Upgrade station was not visible during verification")
            return None
        if candidate and math.dist(target, second[0]) > _POPUP_STILL_PX:
            logger.info("Upgrade control moved between verification frames; retrying")
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
        tolerance = station.disappear_position_tolerance_px
        region = BoundingBox(
            position[0] - tolerance, position[1] - tolerance,
            position[0] + tolerance + 1, position[1] + tolerance + 1,
        )
        result = self._vision.find_upgrade_station(frame, threshold, search_region=region)
        if result.found and result.best is not None:
            dx = abs(result.best.center[0] - position[0])
            dy = abs(result.best.center[1] - position[1])
            box = result.best.bounding_box
            return (dx <= tolerance and dy <= tolerance and (
                not self._config.strategy.enabled or (
                    box.x_min <= position[0] < box.x_max and box.y_min <= position[1] < box.y_max
                )
            ))
        # A held upgrade resumes when income arrives; grey is not a vanished control.
        if (
            not self._config.strategy.enabled
            or self._vision.station_panel(frame, held_position=position) != StationPanel.LOCKED
        ):
            return False
        # Release before the next stats action; the existing flow opens free worker crates too.
        return not self._vision.stats_icon_seen(frame)

    def _make_disappearance_check(
        self, threshold: float, position: Point, required_misses: int
    ) -> Callable[[], bool]:
        """Requires `required_misses` CONSECUTIVE misses before treating the station as gone —
        debounces a single flaky/transient miss so a real hold isn't cut short by one bad frame."""
        misses = 0

        def check() -> bool:
            nonlocal misses
            if not self._device_available():
                return True
            gone = self._station_disappeared(threshold, position)
            misses = misses + 1 if gone else 0
            return misses >= max(1, required_misses)

        return check

    def _handle_upgrade_stats(self) -> State:
        # Consumed on every entry, even through the idle-click bail-out below, so it can't go stale.
        from_find = self.context.stats_from_find
        self.context.stats_from_find = False
        # Verified v1 behavior: single-shot, no retry loop and no scrcpy-recovery here. This is
        # also the one state where an idle-click failure does NOT retry itself.
        if not self._click_idle():
            return State.OPEN_BOXES
        frame = self._vision.capture(max_y=self._config.capture_regions.extended_search_y)
        if self._vision.find_new_level_button(
            frame[: self._config.capture_regions.max_search_y]
        ).found:
            return flow.decide_upgrade_stats(
                self.context, flow.StatsIconObservation(True, False, from_find)
            )

        stats_seen = self._vision.stats_icon_seen(frame)
        if not stats_seen:
            return flow.decide_upgrade_stats(
                self.context, flow.StatsIconObservation(False, False, from_find)
            )

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
        frame = self._vision.capture(max_y=box_search_y)
        self._record_box_scan()
        thumb = _frame_thumbnail(frame)
        previous = self._latest_thumb
        self._scan_delta = None if previous is None else float(np.abs(thumb - previous).mean())
        self._latest_thumb = thumb
        # First scan after the first scroll of an idle streak. Armed once per streak, not keyed on
        # idle_scrolls == 1: that stays 1 for the next cycle too, whose scan would overwrite it.
        if self._streak_anchor_pending:
            self._streak_anchor_pending = False
            self._streak_thumb = thumb
        if self._vision.find_new_level_button(frame).found:
            logger.info("New level found while opening boxes")
            return flow.decide_open_boxes(
                self.context,
                self._config.upgrade_station,
                self._config.scroll.max_idle_pass_attempts,
                flow.BoxCycleObservation(True),
            )

        # These are free crates from purchases already made. Station spending is gated earlier.
        find_boxes = self._vision.find_boxes
        boxes = find_boxes(frame)
        if not boxes and self._scrcpy_recovery(self._config.scrcpy_recovery.box_delay):
            frame = self._vision.capture(max_y=box_search_y)
            if self._vision.find_new_level_button(frame).found:
                return flow.decide_open_boxes(
                    self.context,
                    self._config.upgrade_station,
                    self._config.scroll.max_idle_pass_attempts,
                    flow.BoxCycleObservation(True),
                )
            boxes = find_boxes(frame)

        if self._early_probe_pending:  # log-only: kept clear of the relaunch decision below
            self._early_probe_pending = False
            self._log_early_probe(frame, len(boxes))
            # A picture to compare the alert's scan with; a store only, never a decision. Skipped
            # when the alert is due on this very scan: it would be compared with itself.
            if self.context.idle_scrolls > 0 and not self._stall_probe_pending:
                self._recent_thumb = _frame_thumbnail(frame)

        if self._stall_probe_pending:
            self._stall_probe_pending = False
            self._log_box_near_misses(frame, len(boxes))
            alert_thumb = _frame_thumbnail(frame)
            if self._game_looks_dead(alert_thumb):
                return self._queue_game_recovery()
            self._recent_thumb = alert_thumb  # the next alert compares with this scan

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
            clickable = []
        opened = 0
        for index, box in enumerate(clickable):
            if index:
                fresh = find_boxes(self._vision.capture(max_y=box_search_y))
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
        if opened:
            logger.info("Opened %s boxes", opened)
        trips = self.context.box_guard_trips
        next_state = flow.decide_open_boxes(
            self.context,
            self._config.upgrade_station,
            self._config.scroll.max_idle_pass_attempts,
            flow.BoxCycleObservation(False, opened, after_scroll),
        )
        if self.context.box_guard_trips != trips:
            logger.warning(
                "Box loop guard: %s box passes without a scroll or upgrade; forcing a scroll. "
                "Last clicked positions: %s",
                self._config.upgrade_station.max_box_only_passes,
                [(x, y, round(conf, 3)) for x, y, conf in self._recent_box_clicks],
            )
        return next_state

    def _handle_scroll(self) -> State:
        if self._config.strategy.enabled:
            self.context.scan_boxes_after_scroll = True
            self._sleep(self._config.scroll.post_scroll_settle)
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
        # An Open dialog belongs to a restaurant already reached, including at startup.
        if self._vision.restaurant_start(self._vision.capture()):
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

    def _handle_transition_level(self) -> State:
        level = self._config.level_transition
        attempt = self.context.state_attempt
        # Preserve the visible dialog. Focus and forbidden-zone checks remain in the input layer.
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
        # v1 logs this too. Here it also records where the Open button really is and how sure the
        # match was (the early probe compares a stuck screen's best match with it).
        logger.info(
            "New level button found at %s conf=%.3f on attempt %d",
            result.best.center,
            result.best.confidence,
            attempt,
        )
        fresh_restaurant = self._config.strategy.enabled and self._vision.restaurant_start(frame)
        clicked = self._input.click(*result.best.center)
        if clicked:
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
        city = self._vision.city_arrival(frame)
        if city:
            if self._progress is not None and not self.context.city_seen:
                self._progress.city_boundary()
            result = self._vision.find_city_continue(frame)
            if result.best is not None and self._input.click(*result.best.center):
                self.context.strategy_fresh_entry = (
                    _VIEWPORT_FRESH_ENTRY_PROBES if self._config.strategy.enabled else 0
                )
                self._sleep(self._config.level_transition.settle_delay)
        self.context.city_seen = city
        return city

    def _handle_confirm_restaurant(self) -> State:
        frame = self._vision.capture()
        if self._progress is not None:
            self._progress.checkpoint(boost=self._vision.boost_level(frame))
        city = self._handle_city_arrival(frame)
        if flow.confirm_restaurant(self.context, not city and self._vision.restaurant_start(frame)):
            self.context.strategy_fresh_entry = (
                _VIEWPORT_FRESH_ENTRY_PROBES if self._config.strategy.enabled else 0
            )
            if self._progress is not None:
                self._progress.confirm_transition()
            self._report_level_completion()
            self._end_unlock_sequence(completed=True)
            return State.FIND_RED_ICONS
        if self._clock() - self.context.transition_started >= (
            self._config.level_transition.confirmation_timeout
        ):
            logger.warning("Restaurant transition unconfirmed; no completion credited")
            self.context.transition_pending = False
            if self._progress is not None:
                self._progress.cancel_transition()
            self._end_unlock_sequence(completed=False)
            return State.FIND_RED_ICONS
        self._sleep(self._config.level_transition.unlock_search_interval)
        return State.CONFIRM_RESTAURANT

    def _strategy_scroll(self, direction: int) -> bool | None:
        """False means unchanged views; True means changed/uncertain; None cancels the attempt."""
        if self.context.strategy_scrolls >= self._config.strategy.viewport_limit:
            raise _ViewportError("Station viewport search limit reached; recalibrate localization")
        self.context.strategy_scrolls += 1
        before = self._vision.capture()
        scroll = self._config.scroll
        x, y = self._config.click_targets.scroll_start_pos
        distance = round(scroll.pixel_step * scroll.distance_ratio)
        destination = y - direction * distance
        moved = self._input.drag(x, y, x, destination, duration=scroll.duration)
        if not moved:
            return None
        logger.info("Station scroll %s (mouse drag %s): %s/%s gestures",
                    "down" if direction > 0 else "up", "up" if direction > 0 else "down",
                    self.context.strategy_scrolls, self._config.strategy.viewport_limit)
        self.context.invalidate_viewport()
        if not (self._sleep(scroll.post_scroll_settle) and self._sleep(scroll.interval_pause)):
            return None
        previous = self._vision.capture()
        still_samples = 0
        interval = self._config.strategy.target_observation_interval
        samples = max(
            _VIEWPORT_MAX_SETTLE_SAMPLES,
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
                return self._vision.viewport_still(before, current) is not True
            previous = current
        raise _ViewportError("Station viewport did not settle; recalibrate scroll pauses")

    def _handle_strategy_find(self, frame: np.ndarray) -> State:
        context = self.context
        thumb = _frame_thumbnail(frame)
        previous = self._strategy_health_thumb
        now = self._clock()
        changed = previous is None or float(np.abs(thumb - previous).mean()) >= (
            self._config.game_recovery.static_recent_delta
        )
        if changed:
            self._strategy_alive_at = now
            self._strategy_health_thumb = thumb
        elif (now - self._strategy_alive_at >= self._config.game_recovery.stale_after_seconds
                and self._config.game_recovery.enabled):
            return self._queue_game_recovery()
        if context.strategy_waiting_liveness:
            if changed:
                self._arm_resume("viewport activity returned; retry localization")
                self.stop()
            else:
                self._sleep(self._config.scroll.post_scroll_settle)
            return State.FIND_RED_ICONS
        try:
            return self._strategy_locate(frame)
        except _ViewportError as exc:
            if not self._config.game_recovery.enabled:
                raise
            if self._recover_stalled_mirror():
                return State.FIND_RED_ICONS
            fresh = _frame_thumbnail(self._vision.capture())
            if float(np.abs(fresh - thumb).mean()) >= (
                self._config.game_recovery.static_recent_delta
            ):
                raise
            context.strategy_waiting_liveness = True
            context.invalidate_viewport()
            if self._progress is not None:
                self._progress.checkpoint("recovery")
            logger.warning("%s; suspending input while checking screen liveness", exc)
            return State.FIND_RED_ICONS

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
        raise _ViewportError("Station search exhausted the configured down/up sweep")

    def _strategy_targets(self, frame: np.ndarray) -> list[MatchCandidate]:
        targets = [target for target in self._vision.station_targets(frame)
                   if self._is_clickable(target.center)]
        if self.context.gameplay_phase not in (GameplayPhase.FIRST, GameplayPhase.LAST):
            return targets
        for target in self._vision.positional_station_targets(frame):
            protected_last = (self.context.gameplay_phase == GameplayPhase.LAST
                              and any(zone.contains(*target.center) for zone in
                                      self._config.forbidden_zones.event_zone_options.values()))
            if (self._is_clickable(target.center) or protected_last) and not any(
                math.dist(target.center, known.center) < self._config.red_icon.fast_min_distance
                for known in targets
            ):
                targets.append(target)
        return targets

    def _strategy_locate(self, frame: np.ndarray) -> State:
        context = self.context
        previous_panel: BoundingBox | None = None
        for dismissal_attempt in range(_VIEWPORT_MAX_PANEL_DISMISSALS):
            if self._vision.station_panel(frame) == StationPanel.UNKNOWN:
                break
            panel = self._vision.station_panel_bounds(frame)
            if panel is None:
                raise ObservationError("Station panel bounds unavailable before localization")
            # An outside tap can select a different station instead of clearing the overlay.
            persistent = bool(dismissal_attempt and (
                previous_panel is None or panel is None
                or previous_panel.iou(panel) >= _VIEWPORT_SAME_PANEL_IOU
            ))
            if not self._click_idle(panel, alternate=persistent):
                if (context.state_attempt >= self._config.upgrade_station.search_attempts
                        and not self._stop_requested.is_set()
                        and self._input.is_target_foreground()):
                    raise ObservationError("No safe dismissal input before viewport localization")
                return State.FIND_RED_ICONS
            if not self._sleep(self._config.scrcpy_recovery.action_settle_delay):
                return State.FIND_RED_ICONS
            previous_panel = panel
            frame = self._vision.capture()
            if self._vision.station_panel(frame) == StationPanel.UNKNOWN:
                if not self._sleep(self._config.scrcpy_recovery.action_settle_delay):
                    return State.FIND_RED_ICONS
                frame = self._vision.capture()
        else:
            if self._vision.station_panel(frame) != StationPanel.UNKNOWN:
                raise _ViewportError("Station panel did not close before viewport localization")
        if (context.strategy_anchor is not None and context.target_viewport == context.viewport
                and context.red_icons):
            reference = self._strategy_target_frame
            still = (self._vision.viewport_still(reference, frame)
                     if reference is not None else None)
            safe_top = _MIN_STATION_TARGET_Y + self._config.red_icon.fast_min_distance
            if (still is True and context.strategy_anchor[1] >= safe_top
                    and self._is_clickable(context.strategy_anchor)):
                context.current_red_icon_index = 0
                return State.CLICK_RED_ICON
            current_target = next((target for target in self._strategy_targets(frame)
                if math.dist(target.center, context.strategy_anchor)
                < self._config.red_icon.fast_min_distance), None)
            if current_target is None:
                if still is False:
                    context.invalidate_viewport()
                    context.strategy_seek = True
                    context.strategy_scrolls = context.strategy_stationary_count = 0
                elif still is True and self._is_clickable(context.strategy_anchor):
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
                moved = self._strategy_scroll(-1)
                return State.UPGRADE_STATS if moved is not None else State.FIND_RED_ICONS
            self._strategy_target_frame = frame
            context.strategy_anchor = current_target.center
            context.red_icons = [current_target]
            context.current_red_icon_index = 0
            return State.CLICK_RED_ICON
        # ponytail: only confirmed arrivals reset the camera; manual resumes retain the sweep.
        fresh_entry = (context.strategy_fresh_entry > 0
                       and context.gameplay_phase == GameplayPhase.FIRST)
        context.strategy_fresh_entry = context.strategy_fresh_entry - 1 if fresh_entry else 0
        observed_at = self._clock()
        views = [self._strategy_targets(frame)]
        prefer_visible = (not self._config.strategy.camera_edge_checks_first
                          and context.gameplay_phase == GameplayPhase.FIRST)
        # Two of three views reject transient effects while allowing one occluded observation.
        observe_target = (not context.strategy_seek or fresh_entry
                          or context.strategy_stationary_count >= 2
                          or (prefer_visible and bool(views[0])))
        fresh = frame
        for _ in range(2 if observe_target else 0):
            remaining = self._config.strategy.target_observation_interval - (
                self._clock() - observed_at)
            if not self._sleep(max(0.0, remaining)):
                return State.FIND_RED_ICONS
            fresh = self._vision.capture()
            observed_at = self._clock()
            if self._vision.station_panel(fresh) != StationPanel.UNKNOWN:
                return State.FIND_RED_ICONS
            views.append(self._strategy_targets(fresh))
            # Identical full target sets already satisfy two observations; ambiguity uses all three.
            if len(views) == 2 and views[0] and sorted(
                (candidate.template_name, candidate.center) for candidate in views[0]
            ) == sorted((candidate.template_name, candidate.center) for candidate in views[1]):
                break
        tolerance = self._config.red_icon.fast_min_distance
        targets: list[MatchCandidate] = []
        for index in reversed(range(len(views))):
            for candidate in views[index]:
                confirmed = any(other_index != index and any(
                    math.dist(candidate.center, other.center) < tolerance for other in other_view
                ) for other_index, other_view in enumerate(views))
                if confirmed and not any(math.dist(candidate.center, other.center) < tolerance
                                         for other in targets):
                    targets.append(candidate)
        target = flow.strategy_target(context, targets)
        if fresh_entry:
            if target is None:
                context.stats_from_find = True
                return State.UPGRADE_STATS  # Recheck starter crates before the next bounded probe.
            context.strategy_fresh_entry = 0
            context.strategy_seek = False
            logger.info("Fresh FIRST verified across observations; skipping camera sweep")
        if (prefer_visible and context.strategy_seek and not fresh_entry and target is not None
                and self._vision.viewport_still(frame, fresh) is not True):
            target = None
            context.strategy_stationary_count = 0
            logger.info("Visible station camera not yet stable; continuing bounded sweep")
        if context.strategy_seek and not fresh_entry:
            if prefer_visible and target is not None:
                context.strategy_seek = False
                logger.info("Stable visible %s station selected before camera sweep",
                            context.gameplay_phase.value)
            elif context.strategy_stationary_count >= 2:
                context.strategy_seek = False
                context.strategy_scrolls = 0
                context.strategy_stationary_count = 0
            else:
                edge_direction = -1 if context.gameplay_phase == GameplayPhase.FIRST else 1
                # Confirm a possible endpoint before the original sweep reverses away from it.
                direction = (edge_direction if context.strategy_stationary_count == 1
                             else self._strategy_sweep_direction())
                moved = self._strategy_scroll(direction)
                if moved is not None:
                    context.strategy_stationary_count = (
                        context.strategy_stationary_count + 1
                        if not moved and direction == edge_direction else 0
                    )
                return State.UPGRADE_STATS if moved is not None else State.FIND_RED_ICONS
        if target is None:
            direction = 1 if context.gameplay_phase == GameplayPhase.FIRST else -1
            moved = self._strategy_scroll(direction)
            if moved is not None:
                context.strategy_stationary_count = (
                    0 if moved else context.strategy_stationary_count + 1
                )
                if (context.gameplay_phase == GameplayPhase.REVERSE_MIDDLE
                        and context.strategy_stationary_count >= 2
                        and not context.strategy_reanchored):
                    context.strategy_reanchored = True
                    context.strategy_seek = True
                    context.strategy_scrolls = context.strategy_stationary_count = 0
            return State.UPGRADE_STATS if moved is not None else State.FIND_RED_ICONS
        if target.center[1] < _MIN_STATION_TARGET_Y or not self._is_clickable(target.center):
            # Leave room above the station for a complete Level/MAX panel below the HUD.
            moved = self._strategy_scroll(-1)
            return State.UPGRADE_STATS if moved is not None else State.FIND_RED_ICONS
        context.strategy_scrolls = 0
        context.strategy_stationary_count = 0
        context.strategy_anchor = target.center
        context.target_viewport = context.viewport
        context.red_icons = [target]
        self._strategy_target_frame = fresh
        context.current_red_icon_index = 0
        if target.template_name == "stationPosition":
            logger.info("Positional %s station fallback at %s; panel verification required",
                        context.gameplay_phase.value, target.center)
        return State.CLICK_RED_ICON

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
