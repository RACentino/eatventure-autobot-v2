"""Handler for miscellaneous bot states (boxes, stats, unlocks)."""

import logging
import time
from typing import Optional

from bot.handlers.base import BaseHandler
from bot.state_machine import State
from core import config
from core.logger import setup_logger

logger = setup_logger("bot.handlers.misc_handler")

class MiscHandler(BaseHandler):
    """
    Tactical Miscellaneous Handler.
    Handles boxes, stats upgrades, and unlock checks.
    """
    def __init__(self, bot: 'EatventureBot'):
        super().__init__(bot)

    def handle(self, current_state: State) -> State:
        """Branches based on the active FSM state."""
        if current_state == State.OPEN_BOXES:
            return self._handle_open_boxes()
        elif current_state == State.UPGRADE_STATS:
            return self._handle_upgrade_stats()
        elif current_state == State.CHECK_UNLOCK:
            return self._handle_check_unlock()
        elif current_state == State.WAIT_FOR_UNLOCK:
            return self._handle_wait_for_unlock()
        elif current_state == State.RECOVERY:
            return self._handle_recovery()
        return State.FIND_RED_ICONS

    def _handle_open_boxes(self) -> State:
        """Executes box opening logic for all box variations."""
        logger.info("Scanning for boxes via consolidated service...")
        
        if self.bot.scan_and_click_boxes():
            # Settle after opening
            self.bot.sleep(config.STATE_DELAY)
            return State.FIND_RED_ICONS
            
        return State.SEARCH_UPGRADE_STATION

    def _handle_upgrade_stats(self) -> State:
        """Executes stats upgrade logic with high-frequency click loop."""
        logger.info("Performing stats upgrade sequence...")
        
        # 1. Click stats icon
        logger.info(f"Opening stats menu at {config.STATS_UPGRADE_BUTTON_POS}")
        if self.mouse.click(config.STATS_UPGRADE_BUTTON_POS[0], config.STATS_UPGRADE_BUTTON_POS[1]):
            self.bot.sleep(config.STATE_DELAY)
            
            # 2. Execute rapid click loop
            logger.info(f"Rapid clicking stats for {config.STATS_UPGRADE_CLICK_DURATION}s")
            start = time.monotonic()
            while time.monotonic() - start < config.STATS_UPGRADE_CLICK_DURATION:
                self.check_interrupts()
                # Rapid tap
                self.mouse.click(
                    config.STATS_UPGRADE_POS[0], 
                    config.STATS_UPGRADE_POS[1], 
                    wait_after=False
                )
                time.sleep(config.STATS_UPGRADE_CLICK_DELAY)
        
        # FIX: [Architectural adjustment] Force idle click after stats sequence to guarantee UI clearance.
        logger.info("Stats upgrade complete. Clicking idle to ensure UI is clear.")
        self.mouse.click(config.IDLE_CLICK_POS[0], config.IDLE_CLICK_POS[1])
        self.bot.sleep(config.IDLE_CLICK_SETTLE_DELAY)
            
        return State.CHECK_NEW_LEVEL

    def _handle_check_unlock(self) -> State:
        """Checks for any available unlocks."""
        logger.info("Checking for unlocks...")
        screenshot = self.capture.capture(max_y=config.MAX_SEARCH_Y)
        
        if "unlock" in self.bot.templates:
            template, mask = self.bot.templates["unlock"]
            found, conf, x, y = self.matcher.find_template(
                screenshot, template, mask=mask, threshold=config.UNLOCK_THRESHOLD
            )
            
            if found and not self.mouse.is_in_forbidden_zone(x, y):
                logger.info(f"Found unlock button at ({x}, {y}). Clicking.")
                self.check_interrupts()
                if self.mouse.click(x, y):
                    self.bot.sleep(config.UNLOCK_POST_CLICK_DELAY)
                    return State.FIND_RED_ICONS
                    
        return State.SEARCH_UPGRADE_STATION

    def _handle_wait_for_unlock(self) -> State:
        """
        New City Synchronization: High-frequency polling for the first unlock button.
        Executes a 50ms hot loop as per architecture.
        """
        logger.info("Waiting for initial station unlock in new city...")
        start_time = time.monotonic()
        timeout = config.WAIT_FOR_UNLOCK_TIMEOUT
        
        while time.monotonic() - start_time < timeout:
            # Mechanical Guard: Check bot status before interaction
            if not self.bot.running:
                return State.FIND_RED_ICONS

            screenshot = self.capture.capture(max_y=config.WINDOW_HEIGHT)
            if "unlock" in self.bot.templates:
                template, mask = self.bot.templates["unlock"]
                found, conf, x, y = self.matcher.find_template(
                    screenshot, template, mask=mask, threshold=config.UNLOCK_THRESHOLD
                )
                
                if found:
                    logger.info(f"First station unlocked at ({x}, {y})!")
                    if self.mouse.click(x, y):
                        self.bot.sleep(config.UNLOCK_POST_CLICK_DELAY)
                        return State.FIND_RED_ICONS
            
            # Hot loop wait
            time.sleep(config.WAIT_UNLOCK_HOT_LOOP)

        logger.warning("Wait for unlock timed out. Returning to discovery.")
        return State.FIND_RED_ICONS

    def _handle_recovery(self) -> State:
        """Recovery state to clear unexpected modals or reset positioning."""
        logger.warning("Bot is in RECOVERY state. Attempting to clear modals...")
        # Click a known safe 'neutral' position to clear modals
        self.mouse.click(config.IDLE_CLICK_POS[0], config.IDLE_CLICK_POS[1]) # Neutral idle click
        time.sleep(1.0)
        return State.FIND_RED_ICONS
