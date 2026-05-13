from collections.abc import Callable

from bot.state_machine import State
from bot.types import StateResult


class StateRegistrationMixin:
    def _state_handler_pairs(self) -> tuple[tuple[State, Callable[[State], StateResult]], ...]:
        return (
            (State.FIND_RED_ICONS, self.handle_find_red_icons),
            (State.CLICK_RED_ICON, self.handle_click_red_icon),
            (State.CHECK_UNLOCK, self.handle_check_unlock),
            (State.SEARCH_UPGRADE_STATION, self.handle_search_upgrade_station),
            (State.HOLD_UPGRADE_STATION, self.handle_hold_upgrade_station),
            (State.OPEN_BOXES, self.handle_open_boxes),
            (State.UPGRADE_STATS, self.handle_upgrade_stats),
            (State.SCROLL, self.handle_scroll),
            (State.CHECK_NEW_LEVEL, self.handle_check_new_level),
            (State.TRANSITION_LEVEL, self.handle_transition_level),
            (State.WAIT_FOR_UNLOCK, self.handle_wait_for_unlock),
        )

    def register_states(self) -> None:
        for state, handler in self._state_handler_pairs():
            self.state_machine.register_handler(state, handler)
