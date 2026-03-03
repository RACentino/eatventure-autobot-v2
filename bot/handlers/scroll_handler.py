"""Handler for searching for icons and stations by scrolling."""

import logging
from typing import List, Tuple, Optional

from bot.handlers.base import BaseHandler
from bot.state_machine import State
from core import config
from core.logger import setup_logger

logger = setup_logger("bot.handlers.scroll_handler")

class ScrollHandler(BaseHandler):
    """
    Tactical Scroll Handler.
    Orchestrates searching for assets by scrolling through the world.
    """
    def __init__(self, bot: 'EatventureBot'):
        super().__init__(bot)

    def handle(self, current_state: State) -> State:
        """Branches based on the active FSM state."""
        if current_state == State.SCROLL:
            return self._handle_scroll()
        return State.FIND_RED_ICONS

    def _handle_scroll(self) -> State:
        """
        Execution Phase: Scrolls through the world using the OscillatingSearcher.
        """
        logger.info("Initiating oscillating search cycle...")
        
        target_state = self.bot.searcher.execute_cycle(
            check_priority=self.bot.check_priority_targets,
            check_main_target=self.bot.check_main_success,
            check_fallbacks=self.bot.check_fallbacks
        )
        
        if target_state:
            return target_state
            
        logger.info("Oscillating search exhausted. Returning to base scanning.")
        return State.FIND_RED_ICONS
