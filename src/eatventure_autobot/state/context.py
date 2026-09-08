"""Mutable per-run flow bookkeeping threaded through the pure transition functions in
transitions.py. Every field and every reset below is transcribed from a verified read of both
source repos' bot.py — in particular reset_search_cycle() clears strictly less than it looks like
it should (it does NOT clear work_done, red_icons, or the upgrade flags; only stop() does that,
via reset_run)."""

from collections import deque
from dataclasses import dataclass, field

from eatventure_autobot.domain.types import MatchCandidate, Point

# Verified from both repos' red-icon priority bookkeeping: remember at most 24 productive rows,
# treat rows within 12px as the same row when recording, and within 50px as "near" when ranking.
REMEMBERED_ROW_LIMIT = 24
ROW_DEDUPE_DISTANCE = 12
ROW_PROXIMITY_DISTANCE = 50


@dataclass
class FlowContext:
    red_icons: list[MatchCandidate] = field(default_factory=list)
    current_red_icon_index: int = 0
    successful_red_icon_rows: deque[int] = field(
        default_factory=lambda: deque(maxlen=REMEMBERED_ROW_LIMIT)
    )

    cycle_counter: int = 0
    consecutive_failed_upgrade_searches: int = 0
    upgrade_station_counter: int = 0
    upgrade_station_pos: Point | None = None
    upgrade_found_in_cycle: bool = False
    work_done: bool = False

    new_level_red_icon_verified: bool = False
    wait_for_unlock_attempts: int = 0
    total_levels_completed: int = 0
    current_level_start_time: float | None = None

    oscillation_cycle_index: int = 1
    oscillation_leg_direction: int = 1
    oscillation_leg_progress: int = 0

    # How many consecutive times the flow has re-entered the current state. Owned by the
    # orchestration loop (reset on any real state change), read by the attempt-loop states.
    state_attempt: int = 1

    def remember_successful_red_icon_row(self, y: int) -> None:
        """Records a row that produced a real upgrade so later scans revisit it first. Rows within
        ROW_DEDUPE_DISTANCE of an already-remembered one are treated as the same row."""
        y = int(y)
        if any(
            abs(existing - y) < ROW_DEDUPE_DISTANCE for existing in self.successful_red_icon_rows
        ):
            return
        self.successful_red_icon_rows.append(y)

    def reset_search_cycle(self) -> None:
        """Verified transcription of _reset_search_cycle(): the search-flow reset only. It
        deliberately leaves work_done, red_icons and the upgrade flags alone."""
        self.cycle_counter = 0
        self.wait_for_unlock_attempts = 0
        self.oscillation_cycle_index = 1
        self.oscillation_leg_direction = 1
        self.oscillation_leg_progress = 0
        self.new_level_red_icon_verified = False

    def reset_run(self) -> None:
        """The fuller reset applied when the bot stops, matching v2's stop()."""
        self.reset_search_cycle()
        self.red_icons = []
        self.current_red_icon_index = 0
        self.upgrade_station_pos = None
        self.upgrade_found_in_cycle = False
        self.work_done = False
        self.consecutive_failed_upgrade_searches = 0
        self.state_attempt = 1

    def advance_oscillation(self, increment_step: int, max_cycles: int) -> None:
        """Verified transcription of _advance_oscillation_progress(): each cycle walks a growing
        number of legs in one direction, flips direction, then widens the sweep."""
        target_steps = max(1, self.oscillation_cycle_index * increment_step)
        self.oscillation_leg_progress += 1
        if self.oscillation_leg_progress < target_steps:
            return
        self.oscillation_leg_progress = 0
        if self.oscillation_leg_direction > 0:
            self.oscillation_leg_direction = -1
            return
        self.oscillation_leg_direction = 1
        self.oscillation_cycle_index += 1
        if self.oscillation_cycle_index > max_cycles:
            self.oscillation_cycle_index = 1
