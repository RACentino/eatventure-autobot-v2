"""Pure per-state decision functions: given the current FlowContext and an already-gathered
observation (capture + template matching + a single click/hold/scroll attempt — all real I/O
performed by the orchestration layer, never here), decide the next State and update the
counters that belong to the flow, not to any single handler.

Every function below is a direct, branch-for-branch transcription of v1's bot.py (see
../eatventure-autobot-v1/bot.py), by deliberate decision: this codebase's state-handler flow and
sequence follows v1's exactly, including branches where an earlier revision of this file
deliberately synthesized a different (sometimes improved) behavior. Attempt caps v1 never had
(on CHECK_UNLOCK, UPGRADE_STATS, SCROLL, and both CHECK_NEW_LEVEL branches) have been removed to
match — those states are unbounded self-loops or single-shot misses in v1, relying solely on the
same-state watchdog (resilience/watchdog.py) as the backstop.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from eatventure_autobot.domain.config import (
    FlowTimingConfig,
    LevelTransitionConfig,
    UpgradeStationConfig,
)
from eatventure_autobot.domain.state import State
from eatventure_autobot.domain.types import MatchCandidate, Point
from eatventure_autobot.state.context import ROW_PROXIMITY_DISTANCE, FlowContext

# --- FIND_RED_ICONS -----------------------------------------------------------------------


@dataclass(frozen=True)
class RedIconScanObservation:
    new_level_button_found: bool
    new_level_footer_icon_found: bool
    red_icons: tuple[MatchCandidate, ...]


def red_icon_priority_key(
    context: FlowContext, candidate: MatchCandidate
) -> tuple[int, int, float]:
    """Verified ordering: rows that previously produced a real upgrade come first, then
    top-to-bottom, then by descending confidence."""
    y = candidate.center[1]
    near_productive_row = any(
        abs(y - row) < ROW_PROXIMITY_DISTANCE for row in context.successful_red_icon_rows
    )
    return (0 if near_productive_row else 1, y, -candidate.confidence)


def sort_red_icons_by_priority(
    context: FlowContext, candidates: Iterable[MatchCandidate]
) -> list[MatchCandidate]:
    return sorted(candidates, key=lambda candidate: red_icon_priority_key(context, candidate))


def decide_find_red_icons(context: FlowContext, obs: RedIconScanObservation) -> State:
    # Verified v1 behavior: work_done resets unconditionally on every FIND_RED_ICONS entry, before
    # any branch below — it tracks "did this find->...->open_boxes cycle do anything productive,"
    # not a run-lifetime flag.
    context.work_done = False
    if obs.new_level_button_found:
        context.cycle_counter = 0
        return State.TRANSITION_LEVEL
    if obs.new_level_footer_icon_found:
        return State.CHECK_NEW_LEVEL
    if not obs.red_icons:
        return State.OPEN_BOXES
    context.red_icons = sort_red_icons_by_priority(context, obs.red_icons)
    context.current_red_icon_index = 0
    context.cycle_counter = 0
    # Verified v1 behavior: work_done is set True the moment icons are queued, before any click is
    # attempted — not only after a click succeeds.
    context.work_done = True
    return State.CLICK_RED_ICON


# --- CLICK_RED_ICON ------------------------------------------------------------------------


def has_more_red_icons(context: FlowContext) -> bool:
    return context.current_red_icon_index < len(context.red_icons)


def current_red_icon_target(context: FlowContext) -> MatchCandidate | None:
    if not has_more_red_icons(context):
        return None
    return context.red_icons[context.current_red_icon_index]


def decide_click_red_icon(context: FlowContext, click_succeeded: bool) -> State:
    if click_succeeded:
        return State.CHECK_UNLOCK
    context.current_red_icon_index += 1
    if not has_more_red_icons(context):
        return State.OPEN_BOXES
    return State.CLICK_RED_ICON


# --- CHECK_UNLOCK ----------------------------------------------------------------------------


def decide_check_unlock(unlock_button_found: bool, click_succeeded: bool | None) -> State:
    # Verified v1 behavior: no attempt cap here — a stuck click-retry loop is bounded only by the
    # same-state watchdog, matching every other unbounded branch v1 relies on it for.
    if not unlock_button_found:
        return State.SEARCH_UPGRADE_STATION
    if click_succeeded:
        return State.SEARCH_UPGRADE_STATION
    return State.CHECK_UNLOCK


# --- SEARCH_UPGRADE_STATION ------------------------------------------------------------------


@dataclass(frozen=True)
class UpgradeStationSearchObservation:
    found: bool
    position: Point | None
    attempt_number: int  # 1-based


def decide_search_upgrade_station(
    context: FlowContext, config: UpgradeStationConfig, obs: UpgradeStationSearchObservation
) -> State:
    if obs.found:
        context.upgrade_station_pos = obs.position
        context.upgrade_found_in_cycle = True
        # Deliberately NOT resetting consecutive_failed_upgrade_searches here (moved to
        # decide_hold_upgrade_station's genuine-completion branch, see comment there): a station
        # that is found every cycle but never successfully held would otherwise re-zero this
        # counter on every pass, before the hold below ever gets a chance to increment it past 1.
        context.cycle_counter = 0
        return State.HOLD_UPGRADE_STATION
    if obs.attempt_number >= config.search_attempts:
        context.consecutive_failed_upgrade_searches += 1
        return State.OPEN_BOXES
    return State.SEARCH_UPGRADE_STATION


# --- HOLD_UPGRADE_STATION ----------------------------------------------------------------------


@dataclass(frozen=True)
class UpgradeHoldObservation:
    target_available: bool  # stored position exists and is not inside a forbidden zone
    verification_passed: bool  # the station was still visible on a fresh verification frame
    hold_completed: bool  # hold_at() returned True (covers both timeout and early interrupt)
    post_hold_actions_succeeded: bool = True  # the post-hold idle click and settle delay
    verified_position: Point | None = None  # the freshly re-detected position, when verified


def decide_hold_upgrade_station(
    context: FlowContext, config: FlowTimingConfig, obs: UpgradeHoldObservation
) -> State:
    # Intentional deviation from literal v1 parity (v1 has this exact gap too, unfixed): a red
    # icon whose station is found every SEARCH but never actually holds/clicks successfully would
    # otherwise loop FIND_RED_ICONS<->OPEN_BOXES forever, since nothing here used to touch this
    # counter and the same-state watchdog never fires (the state keeps changing every tick).
    # Reusing consecutive_failed_upgrade_searches feeds the existing OPEN_BOXES->SCROLL escape
    # (failed_searches_before_scroll) without adding a new counter or config value -- but it only
    # resets on a genuinely COMPLETED hold below, not on decide_search_upgrade_station's "found"
    # branch, so repeated hold failures actually accumulate across cycles instead of being
    # re-zeroed by the very next search success.
    if not obs.target_available:
        context.consecutive_failed_upgrade_searches += 1
        return State.OPEN_BOXES
    if not obs.verification_passed:
        # Verified: the verification helper itself clears both of these before bailing.
        context.upgrade_station_pos = None
        context.upgrade_found_in_cycle = False
        context.consecutive_failed_upgrade_searches += 1
        return State.OPEN_BOXES
    # Replace the stored position with the freshly-verified one: if the hold below fails without
    # clearing it (next branch), the next attempt should retry against the accurate position.
    context.upgrade_station_pos = obs.verified_position
    if not obs.hold_completed:
        # Verified: a failed hold returns without clearing the stored position.
        context.consecutive_failed_upgrade_searches += 1
        return State.OPEN_BOXES
    # The hold actually completed: real progress, so clear the streak here rather than at
    # decide_search_upgrade_station's "found" branch (see comment there).
    context.consecutive_failed_upgrade_searches = 0
    current = current_red_icon_target(context)
    if current is not None:
        context.remember_successful_red_icon_row(current.center[1])
    context.upgrade_station_pos = None
    if not obs.post_hold_actions_succeeded:
        return State.OPEN_BOXES
    context.upgrade_station_counter += 1
    if context.upgrade_station_counter >= config.upgrades_before_stats:
        context.upgrade_station_counter = 0
        return State.UPGRADE_STATS
    return State.OPEN_BOXES


# --- UPGRADE_STATS ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatsIconObservation:
    new_level_button_found: bool
    stats_icon_found: bool


def decide_upgrade_stats(obs: StatsIconObservation) -> State:
    # Verified v1 behavior: single-shot, no retry — a miss goes straight to SCROLL.
    if obs.new_level_button_found:
        return State.TRANSITION_LEVEL
    if obs.stats_icon_found:
        return State.OPEN_BOXES
    return State.SCROLL


# --- OPEN_BOXES ------------------------------------------------------------------------------


@dataclass(frozen=True)
class BoxCycleObservation:
    new_level_button_found: bool
    boxes_opened: int = 0


def decide_open_boxes(
    context: FlowContext,
    config: UpgradeStationConfig,
    max_idle_pass_attempts: int,
    obs: BoxCycleObservation,
) -> State:
    if obs.new_level_button_found:
        return State.TRANSITION_LEVEL
    if obs.boxes_opened > 0:
        context.work_done = True
        context.cycle_counter = 0

    # Verified transcription of _next_state_after_box_cycle. Each branch's counter reset is
    # load-bearing: without them the triggering condition stays true and the bot re-enters the
    # same branch on every subsequent box cycle.
    if context.consecutive_failed_upgrade_searches >= config.failed_searches_before_scroll:
        context.consecutive_failed_upgrade_searches = 0
        context.cycle_counter = 0
        return State.SCROLL
    if context.upgrade_found_in_cycle:
        context.upgrade_found_in_cycle = False
        context.cycle_counter = 0
        return State.FIND_RED_ICONS
    if context.work_done:
        context.cycle_counter = 0
        return State.FIND_RED_ICONS
    context.cycle_counter += 1
    if context.cycle_counter >= max(1, max_idle_pass_attempts):
        context.cycle_counter = 0
        return State.SCROLL
    return State.FIND_RED_ICONS


# --- SCROLL ------------------------------------------------------------------------------------


def decide_scroll(context: FlowContext, scroll_succeeded: bool) -> State:
    # Verified v1 behavior: no attempt cap on a drag failure — the same-state watchdog is the only
    # backstop, matching v1's unbounded self-loop here.
    if not scroll_succeeded:
        return State.SCROLL
    context.cycle_counter = 0
    return State.FIND_RED_ICONS


# --- CHECK_NEW_LEVEL -----------------------------------------------------------------------


@dataclass(frozen=True)
class NewLevelVerificationObservation:
    verified: bool
    button_click_succeeded: bool | None
    transition_click_succeeded: bool | None
    # None when no verification scroll was attempted this cycle (already verified from an earlier
    # pass, so the scroll+rescan branch was skipped); False when the restored verification-scroll
    # drag (or one of its settle sleeps) failed — distinct from a completed rescan finding no
    # icon, and distinct from the click-retry tail's own False fields below.
    scroll_succeeded: bool | None = None


def decide_check_new_level(context: FlowContext, obs: NewLevelVerificationObservation) -> State:
    # Verified v1 behavior: the verify phase gives up on the very first miss (no retry loop), and
    # the click-retry tail loops unboundedly (watchdog-only) — neither branch counts attempts.
    # A failed verification-scroll drag self-loops the same unbounded way, without touching
    # new_level_red_icon_verified or the search cycle.
    if obs.scroll_succeeded is False:
        return State.CHECK_NEW_LEVEL
    if obs.verified:
        context.new_level_red_icon_verified = True
    if not obs.verified:
        context.reset_search_cycle()
        return State.FIND_RED_ICONS
    if obs.button_click_succeeded is False or obs.transition_click_succeeded is False:
        return State.CHECK_NEW_LEVEL
    return State.WAIT_FOR_UNLOCK


# --- TRANSITION_LEVEL ----------------------------------------------------------------------


@dataclass(frozen=True)
class TransitionLevelObservation:
    found: bool
    attempt_number: int
    click_succeeded: bool | None


def decide_transition_level(
    context: FlowContext, config: LevelTransitionConfig, obs: TransitionLevelObservation
) -> State:
    if not obs.found:
        if obs.attempt_number >= config.search_attempts:
            context.reset_search_cycle()
            return State.FIND_RED_ICONS
        return State.TRANSITION_LEVEL
    if obs.click_succeeded is False:
        return State.CHECK_NEW_LEVEL
    return State.WAIT_FOR_UNLOCK


# --- WAIT_FOR_UNLOCK -----------------------------------------------------------------------


@dataclass(frozen=True)
class WaitForUnlockObservation:
    unlock_found: bool
    click_succeeded: bool | None


def decide_wait_for_unlock(
    context: FlowContext, config: LevelTransitionConfig, obs: WaitForUnlockObservation
) -> State:
    # Success is checked before the attempt cap below: the cap exists to bound consecutive
    # failures, not to discard a real success that happens to land on the last allowed attempt.
    if obs.unlock_found and obs.click_succeeded:
        context.total_levels_completed += 1
        context.wait_for_unlock_attempts = 0
        context.reset_search_cycle()
        return State.FIND_RED_ICONS

    context.wait_for_unlock_attempts += 1
    if context.wait_for_unlock_attempts >= config.unlock_search_attempts:
        context.wait_for_unlock_attempts = 0
        context.reset_search_cycle()
        return State.FIND_RED_ICONS
    return State.WAIT_FOR_UNLOCK
