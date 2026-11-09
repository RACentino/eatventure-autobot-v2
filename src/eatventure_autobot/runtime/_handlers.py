"""The eleven state handlers and the helpers only they use.

Moved out of bot.py unchanged; EatventureBot inherits it. What it reaches on self is declared
at the top of the class."""

import logging
import math
import time
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from eatventure_autobot.domain.config import BotConfig
from eatventure_autobot.domain.protocols import InputController
from eatventure_autobot.domain.state import State
from eatventure_autobot.domain.types import Point, TemplateExplanation
from eatventure_autobot.runtime._telemetry import _frame_thumbnail
from eatventure_autobot.runtime.vision import GameVision
from eatventure_autobot.state import transitions as flow
from eatventure_autobot.state.context import FlowContext, LevelMark

logger = logging.getLogger("eatventure_autobot.runtime.bot")

# Two consecutive scans of the station popup this close (px) count as "has stopped moving"
# (ScrcpyRecoveryConfig.popup_early_exit). The matcher reports whole-pixel positions, so a settled
# panel reads 0; 2 only absorbs a single sub-pixel rounding flip.
_POPUP_STILL_PX = 2.0


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

    if TYPE_CHECKING:  # annotation only: nothing here exists at runtime

        def _maybe_alert_stall(self) -> None:
            ...

        def _log_box_near_misses(self, frame: np.ndarray, detected: int) -> None:
            ...

        def _log_early_probe(self, frame: np.ndarray, detected: int) -> None:
            ...

        def _sleep(self, duration: float) -> bool:
            ...

        def _click_idle(self) -> bool:
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

        def _level_mark(self) -> LevelMark:
            ...

        def _report_level_completion(self) -> None:
            ...

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
        if not self._settle_and_watch(self._config.scrcpy_recovery.action_settle_delay):
            return State.CHECK_UNLOCK
        frame = self._vision.capture(max_y=self._config.capture_regions.max_search_y)
        result = self._vision.find_unlock_button(frame)
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

        # Reuses the shared red-icon scan; split_red_icons applies the stats-icon threshold.
        candidates = self._vision.find_red_icons(frame)
        _, _, stats_seen = self._vision.split_red_icons(candidates)
        if not stats_seen:
            return flow.decide_upgrade_stats(
                self.context, flow.StatsIconObservation(False, False, from_find)
            )

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
        return flow.decide_upgrade_stats(
            self.context, flow.StatsIconObservation(False, True, from_find)
        )

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
                return self._relaunch_game()
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
        for box in clickable:
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
        next_state = flow.decide_check_new_level(
            self.context,
            flow.NewLevelVerificationObservation(True, button_ok, transition_ok),
        )
        if next_state == State.WAIT_FOR_UNLOCK:
            self.context.nl_route = "footer"  # the badge route; see _end_unlock_sequence
        return next_state

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
        # v1 logs this too. Here it also records where the Open button really is and how sure the
        # match was (the early probe compares a stuck screen's best match with it).
        logger.info(
            "New level button found at %s conf=%.3f on attempt %d",
            result.best.center,
            result.best.confidence,
            attempt,
        )
        clicked = self._input.click(*result.best.center)
        if clicked:
            self._sleep(level.settle_delay)
        next_state = flow.decide_transition_level(
            self.context, level, flow.TransitionLevelObservation(True, attempt, clicked)
        )
        if next_state == State.WAIT_FOR_UNLOCK:
            self.context.nl_route = "open"  # the Open-button route; see _end_unlock_sequence
        return next_state

    def _handle_wait_for_unlock(self) -> State:
        if not self._click_idle():
            return State.WAIT_FOR_UNLOCK
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
            self._end_unlock_sequence(completed=False)
        if self.context.total_levels_completed > previous_total:
            self._report_level_completion()
            self._end_unlock_sequence(completed=True)
            # v1 sleeps here regardless of outcome before returning to FIND_RED_ICONS.
            self._sleep(level.unlock_settle_delay)
        return next_state

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
