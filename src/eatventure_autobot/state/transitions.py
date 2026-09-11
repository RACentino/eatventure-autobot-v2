"""Pure per-state decision functions: given the current FlowContext and an already-gathered
observation (capture + template matching + a single click/hold/scroll attempt — all real I/O
performed by the orchestration layer, never here), decide the next State and update the
counters that belong to the flow, not to any single handler.

Sourced from a verified read of both source repos' bot.py (see GREENFIELD_PLAN.md's "state-logic
research" section). Every function below is now a direct transcription of verified source behavior:
decide_open_boxes and decide_hold_upgrade_station were re-checked line-by-line against v2's
handle_open_boxes / _next_state_after_box_cycle / handle_hold_upgrade_station (confirmed identical
to v1's) at the start of Stage 3, and both were corrected — the counter resets in
_next_state_after_box_cycle are load-bearing, not incidental.

One deliberate behavioral choice, not a transcription: decide_check_new_level follows v1's fuller
_reset_search_cycle() on verification failure rather than v2's partial reset, matching the
fail-closed-leaning posture locked in decision 3.
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
        context.work_done = True
        return State.CHECK_UNLOCK
    context.current_red_icon_index += 1
    if not has_more_red_icons(context):
        return State.OPEN_BOXES
    return State.CLICK_RED_ICON


# --- CHECK_UNLOCK ----------------------------------------------------------------------------


def decide_check_unlock(
    unlock_button_found: bool,
    click_succeeded: bool | None,
    attempt_number: int,
    max_attempts: int,
) -> State:
    if not unlock_button_found:
        return State.SEARCH_UPGRADE_STATION
    if click_succeeded:
        return State.SEARCH_UPGRADE_STATION
    if attempt_number >= max_attempts:
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
        context.consecutive_failed_upgrade_searches = 0
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
    if not obs.target_available:
        return State.OPEN_BOXES
    if not obs.verification_passed:
        # Verified: the verification helper itself clears both of these before bailing.
        context.upgrade_station_pos = None
        context.upgrade_found_in_cycle = False
        return State.OPEN_BOXES
    # Replace the stored position with the freshly-verified one: if the hold below fails without
    # clearing it (next branch), the next attempt should retry against the accurate position.
    context.upgrade_station_pos = obs.verified_position
    if not obs.hold_completed:
        # Verified: a failed hold returns without clearing the stored position.
        return State.OPEN_BOXES
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
    attempt_number: int


def decide_upgrade_stats(obs: StatsIconObservation, max_attempts: int = 2) -> State:
    if obs.new_level_button_found:
        return State.TRANSITION_LEVEL
    if obs.stats_icon_found:
        return State.OPEN_BOXES
    if obs.attempt_number >= max_attempts:
        return State.SCROLL
    return State.UPGRADE_STATS


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


def decide_scroll(
    context: FlowContext, scroll_succeeded: bool, attempt_number: int, max_attempts: int
) -> State:
    if not scroll_succeeded:
        if attempt_number >= max_attempts:
            context.cycle_counter = 0
            return State.FIND_RED_ICONS
        return State.SCROLL
    context.cycle_counter = 0
    return State.FIND_RED_ICONS


# --- CHECK_NEW_LEVEL -----------------------------------------------------------------------


@dataclass(frozen=True)
class NewLevelVerificationObservation:
    verified: bool
    attempt_number: int
    button_click_succeeded: bool | None
    transition_click_succeeded: bool | None


def decide_check_new_level(
    context: FlowContext,
    max_attempts: int,
    click_max_attempts: int,
    obs: NewLevelVerificationObservation,
) -> State:
    if obs.verified:
        context.new_level_red_icon_verified = True
    if not obs.verified:
        if obs.attempt_number >= max_attempts:
            context.reset_search_cycle()
            return State.FIND_RED_ICONS
        return State.CHECK_NEW_LEVEL
    if obs.button_click_succeeded is False or obs.transition_click_succeeded is False:
        if obs.attempt_number >= click_max_attempts:
            context.reset_search_cycle()
            return State.FIND_RED_ICONS
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
