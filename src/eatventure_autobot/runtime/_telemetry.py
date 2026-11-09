"""Metrics line, stall alerts and probes: everything the bot only reports.

Moved out of bot.py unchanged; EatventureBot inherits it. What it reaches on self is declared
at the top of the class."""

import logging
import math
import time
from collections.abc import Callable, Collection
from typing import TYPE_CHECKING

import numpy as np

from eatventure_autobot.domain.config import BotConfig
from eatventure_autobot.domain.protocols import Notifier
from eatventure_autobot.domain.state import State
from eatventure_autobot.domain.types import Point, TemplateExplanation
from eatventure_autobot.runtime.vision import GameVision
from eatventure_autobot.state.context import FlowContext, LevelMark

logger = logging.getLogger("eatventure_autobot.runtime.bot")

# A gap between two box scans longer than this spans a stop or a long New Level / hold stretch, not
# the cadence crates are caught at; it is left out of the box_gap_ms tally.
_BOX_GAP_MAX_SECONDS = 120.0


def _top_close_position(positions: dict[Point, int]) -> str:
    """The most-tapped X position as 'x,yxCOUNT' text (e.g. 308,250x3698), or 'none': the unattended
    logs' only clue to which popup keeps coming back."""
    if not positions:
        return "none"
    (x, y), count = max(positions.items(), key=lambda item: item[1])
    return f"{x},{y}x{count}"


def _popup_summary(samples: Collection[int]) -> str:
    """p50/p95/max in ms of the recent popup first-seen times (nearest rank), or '-' with none."""
    if not samples:
        return "-"
    ordered = sorted(samples)

    def pick(share: float) -> int:
        return ordered[round(share * (len(ordered) - 1))]

    return f"{pick(0.5)}/{pick(0.95)}/{ordered[-1]}"


def _nl_routes(attempts: dict[str, int], done: dict[str, int]) -> str:
    """'footer:12/1,open:30/2': finished New Level sequences / completions per way in, or '-'."""
    ways = sorted(attempts.items())
    return ",".join(f"{route}:{count}/{done.get(route, 0)}" for route, count in ways) or "-"


def _frame_thumbnail(frame: np.ndarray) -> np.ndarray:
    """64x36 grayscale subsample, enough to tell a frozen or dimmed view from a live one.
    ponytail: nearest-pixel sampling, not an area average; it only has to distinguish an identical
    frame (delta 0) from an animated one, so no cv2 call and any frame size works."""
    gray = frame.mean(axis=2) if frame.ndim == 3 else frame.astype(np.float64)
    rows = np.linspace(0, gray.shape[0] - 1, 36).astype(int)
    cols = np.linspace(0, gray.shape[1] - 1, 64).astype(int)
    return np.asarray(gray[rows][:, cols])


def _format_near_misses(
    seen: list[TemplateExplanation], clickable: Callable[[Point], bool] | None = None
) -> str:
    """Text for the early probe: per template its best raw match, how much of it sits inside the
    HSV gate, the median colour it had (H,S,V: judge the gate in numbers, not images) and, when a
    zone check is given, whether that spot is a forbidden zone. The alert probe keeps its own,
    test-pinned format (_log_box_near_misses)."""
    parts = []
    for item in seen:
        ratio = "n/a" if item.hsv_ratio is None else f"{item.hsv_ratio:.2f}"
        hsv = item.hsv_median
        median = "" if hsv is None else f" med_hsv={hsv[0]},{hsv[1]},{hsv[2]}"
        zone = " IN-FORBIDDEN-ZONE" if clickable and not clickable(item.center) else ""
        parts.append(
            f"{item.template_name} conf={item.confidence:.3f} at {item.center} "
            f"hsv={ratio}{median}{zone}"
        )
    return "; ".join(parts) or "none"


class _TelemetryMixin:
    # What this mixin reaches on self. EatventureBot.__init__ (bot.py) creates the attributes;
    # the methods live in bot.py or another mixin.
    _clock: Callable[[], float]
    _close_check_seconds: float
    _config: BotConfig
    _config_fingerprint: str
    _early_probe_pending: bool
    _idle_started_at: float
    _last_box_scan_at: float | None
    _last_metrics_at: float
    _last_probe_thumb: np.ndarray | None
    _latest_thumb: np.ndarray | None
    _metrics_thumb: np.ndarray | None
    _notifier: Notifier
    _recent_thumb: np.ndarray | None
    _scan_delta: float | None
    _stall_probe_pending: bool
    _state_calls: dict[State, int]
    _state_seconds: dict[State, float]
    _streak_anchor_pending: bool
    _streak_thumb: np.ndarray | None
    _vision: GameVision
    context: FlowContext

    if TYPE_CHECKING:  # annotation only: nothing here exists at runtime

        def _is_clickable(self, point: Point) -> bool:
            ...

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
            "close_suppressed=%d close_check_s=%.1f recovery_taps=%d close_top=%s "
            "box_zone_skips=%d box_click_fails=%d box_floods=%d stats_checks=%d stats_hits=%d "
            "post_scroll_boxes=%d popup_ms=%s popup_seen=%d popup_never=%d popup_early=%d "
            "check_unlock_clicks=%d popup_stale=%d popup_moving=%d popup_moving_px=%s "
            "box_gap_ms=%s unlock_ms=%s unlock_never=%d nl_greyed_exits=%d nl_routes=%s "
            "calls=%s | %s",
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
            context.box_zone_skips,
            context.box_click_fails,
            context.box_floods,
            context.stats_checks,
            context.stats_hits,
            context.post_scroll_boxes,
            _popup_summary(context.popup_ms),
            context.popup_seen,
            context.popup_never,
            context.popup_early,
            context.check_unlock_clicks,
            context.popup_stale,
            context.popup_moving,
            _popup_summary(context.popup_moving_px),
            _popup_summary(context.box_gap_ms),
            _popup_summary(context.unlock_ms),
            context.unlock_never,
            context.nl_greyed_exits,
            _nl_routes(context.nl_attempts, context.nl_done),
            ",".join(f"{state.name}:{count}" for state, count in self._state_calls.items() if count)
            or "-",
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
        Separately, at `stall_probe_scrolls` (earlier, log-only) arm the early probe, which also
        reads the red icons: the alert's probe is boxes only and a hand stop can beat it.
        ponytail: a scroll re-anchor stays deferred until a probe shows it is the problem."""
        idle = self.context.idle_scrolls
        if idle == 1:
            self._idle_started_at = time.monotonic()
            self._streak_anchor_pending = True
            self._recent_thumb = None  # a new streak never compares against an old one's picture
        early = self._config.scroll.stall_probe_scrolls
        if early > 0 and idle == early:
            self._early_probe_pending = True
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

    def _log_early_probe(self, frame: np.ndarray, detected: int) -> None:
        """One INFO line at `stall_probe_scrolls` idle scrolls: what box AND red-icon detection saw
        on this scan (best raw match per template, gates and zones not applied) and whether the last
        drag changed the picture. Box vs red-icon conf tells a crate-only miss from a miss of
        everything; a flat picture change says the drag moved nothing. Log-only: no alert, never
        reaches the dead-game check, leaves the alert probe's thumbnail alone, and a failure here
        is logged, not raised (an unexpected handler error would stop the whole bot)."""
        if self.context.idle_scrolls <= 0:  # progress ended the streak before this scan ran
            return
        try:
            context = self.context
            thumb = _frame_thumbnail(frame)
            start = self._streak_thumb
            streak = "n/a" if start is None else f"{np.abs(thumb - start).mean():.2f}"
            drag = "n/a" if self._scan_delta is None else f"{self._scan_delta:.2f}"
            quads = " ".join(
                f"{thumb[r:r + 18, c:c + 32].mean():.0f}" for r in (0, 18) for c in (0, 32)
            )
            boxes = _format_near_misses(self._vision.box_near_misses(frame), self._is_clickable)
            icons = _format_near_misses(self._vision.red_icon_near_misses(frame))
            open_seen = self._vision.new_level_near_miss(frame)
            open_button = _format_near_misses(open_seen)
            if open_seen:  # how far the best match sits from the New Level flow's blind Open click
                target = self._config.click_targets.level_transition_pos
                open_button += f" ({math.dist(open_seen[0].center, target):.0f} px from {target})"
            close_x = _format_near_misses(self._vision.close_button_near_misses(frame))
            thresholds = self._config.thresholds
            logger.info(
                "Early probe: %d idle scrolls (%.1f min), %d box(es) passed detection, "
                "box_zone_skips=%d box_click_fails=%d. Picture change across the last drag=%s, "
                "vs streak start=%s, quadrant brightness=[%s]. "
                "Boxes (need conf>=%.3f and hsv>=%.2f): %s. "
                "Red icons (need conf>=%.3f and hsv>=%.2f): %s. "
                "Open button (need conf>=%.3f): %s. Close X (need conf>=%.3f): %s",
                context.idle_scrolls,
                (time.monotonic() - self._idle_started_at) / 60,
                detected,
                context.box_zone_skips,
                context.box_click_fails,
                drag,
                streak,
                quads,
                thresholds.box,
                self._config.box.hsv.min_match_ratio,
                boxes,
                thresholds.red_icon,
                self._config.red_icon.hsv.min_match_ratio,
                icons,
                thresholds.new_level,
                open_button,
                thresholds.close_button,
                close_x,
            )
        except Exception:
            logger.exception("Early probe failed")

    def _record_box_scan(self) -> None:
        """Notes the gap since the previous OPEN_BOXES capture (box_gap_ms in the metrics line)."""
        now = self._clock()
        last, self._last_box_scan_at = self._last_box_scan_at, now
        if last is not None and now - last <= _BOX_GAP_MAX_SECONDS:
            self.context.box_gap_ms.append(round(1000 * (now - last)))

    def _record_popup_seen(self, milliseconds: int) -> None:
        context = self.context
        context.popup_seen += 1
        context.popup_ms.append(milliseconds)

    def _record_hold(self, seconds: float) -> None:
        self.context.hold_seconds_total += seconds
        # Within one poll gap of the cap = the clock ran out, not the popup vanishing.
        if seconds >= self._config.upgrade_station.click_hold_max_duration - 0.1:
            self.context.holds_capped += 1

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
