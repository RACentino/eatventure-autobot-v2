"""Handler for searching and upgrading stations."""

import logging
from typing import List, Tuple, Optional

from bot.handlers.base import BaseHandler
from bot.state_machine import State
from core import config
from core.logger import setup_logger

logger = setup_logger("bot.handlers.upgrade_handler")

class UpgradeHandler(BaseHandler):
    """
    Tactical Upgrade Handler.
    Orchestrates finding and clicking 'Upgrade Station' templates.
    """
    def __init__(self, bot: 'EatventureBot'):
        super().__init__(bot)
        self.upgrade_station_pos: Optional[Tuple[int, int]] = None

    def handle(self, current_state: State) -> State:
        """Branches based on the active FSM state."""
        if current_state == State.SEARCH_UPGRADE_STATION:
            return self._handle_search_upgrade_station()
        elif current_state == State.HOLD_UPGRADE_STATION:
            return self._handle_hold_upgrade_station()
        return State.FIND_RED_ICONS

    def _handle_search_upgrade_station(self) -> State:
        """
        Discovery Phase: Scans for safe upgrade stations with retry logic.
        Uses consolidated scanning service from orchestrator.
        """
        max_attempts = config.UPGRADE_STATION_SEARCH_MAX_ATTEMPTS
        
        for attempt in range(max_attempts):
            logger.info(f"Scanning for upgrade station (Attempt {attempt+1}/{max_attempts})...")
            
            # Simple Scan via Orchestrator
            pos = self.bot.scan_for_upgrade_station()
            if pos:
                logger.info(f"Found safe upgrade station at {pos}.")
                self.upgrade_station_pos = pos
                if hasattr(self.bot, 'tuner'):
                    self.bot.tuner.record_search_result(True)
                return State.HOLD_UPGRADE_STATION
            
            if attempt < max_attempts - 1:
                self.bot.sleep(config.UPGRADE_SEARCH_INTERVAL)

        logger.info("No safe upgrade stations detected. Initiating scroll.")
        if hasattr(self.bot, 'tuner'):
            self.bot.tuner.record_search_result(False)
        return State.SCROLL

    def _handle_hold_upgrade_station(self) -> State:
        """
        Execution Phase: Holds the upgrade station click with duration and settle padding.
        """
        if not self.upgrade_station_pos:
            return State.FIND_RED_ICONS

        x, y = self.upgrade_station_pos
        logger.info(f"Holding upgrade station at ({x}, {y}) for {config.UPGRADE_HOLD_DURATION}s")

        self.check_interrupts()
        
        # Use tuner values if available for adaptive pacing
        search_interval = getattr(self.bot.tuner, "search_interval", config.UPGRADE_SEARCH_INTERVAL)
        
        success = self.mouse.hold(x, y, duration=config.UPGRADE_HOLD_DURATION)
        if success:
            logger.info("Upgrade hold complete. Settling...")
            self.bot.sleep(search_interval)
            return State.UPGRADE_STATS
        else:
            logger.warning("Hold action failed or interrupted.")
            return State.FIND_RED_ICONS
