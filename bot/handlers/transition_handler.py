"""Handler for checking and transitioning between levels."""

import logging
import cv2
import numpy as np
from typing import List, Tuple, Optional

from bot.handlers.base import BaseHandler
from bot.state_machine import State
from core import config
from core.logger import setup_logger

logger = setup_logger("bot.handlers.transition_handler")

class TransitionHandler(BaseHandler):
    """
    Tactical Transition Handler.
    Orchestrates checking for level completion and initiating the transition.
    """
    def __init__(self, bot: 'EatventureBot'):
        super().__init__(bot)

    def handle(self, current_state: State) -> State:
        """Branches based on the active FSM state."""
        if current_state == State.CHECK_NEW_LEVEL:
            return self._handle_check_new_level()
        elif current_state == State.TRANSITION_LEVEL:
            return self._handle_transition_level()
        return State.FIND_RED_ICONS

    def _handle_check_new_level(self) -> State:
        """
        Discovery Phase: Scans for safe new level icons.
        Excludes coordinates within forbidden zones.
        """
        logger.info("Scanning for new level...")
        screenshot = self.capture.capture(max_y=config.WINDOW_HEIGHT)
        
        template, mask = self.matcher.load_template(f"{config.ASSETS_DIR}/newLevel.png")
        
        matches = self.matcher.find_all_templates(
            screenshot, 
            template, 
            mask=mask, 
            threshold=config.MATCH_THRESHOLD,
            template_name="NewLevel"
        )
        
        # Find the first safe match
        safe_new_level = None
        for conf, x, y in matches:
            try:
                self.mouse._validate_bounds(x, y)
                safe_new_level = (x, y)
                break
            except Exception:
                logger.debug(f"New level at ({x}, {y}) is in forbidden zone. Skipping.")
                continue

        if not safe_new_level:
            logger.info("No safe new levels detected. Returning to discovery.")
            return State.FIND_RED_ICONS

        logger.info(f"Found safe new level at {safe_new_level}.")
        return State.TRANSITION_LEVEL

    def _find_and_click_largest_red_button(self) -> bool:
        """Dynamically finds and clicks the largest red UI element (e.g. Renovate/Fly buttons)."""
        screenshot = self.capture.capture(max_y=config.WINDOW_HEIGHT)
        
        # Color Gate for Red Buttons
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array(config.RED_HSV_LOWER1), np.array(config.RED_HSV_UPPER1))
        mask2 = cv2.inRange(hsv, np.array(config.RED_HSV_LOWER2), np.array(config.RED_HSV_UPPER2))
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphological operations to group button pixels
        kernel = np.ones((15, 15), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
            
        # Find largest contour (likely the primary action button)
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 500: # Minimum size for a button
            return False
            
        # Get center of bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        center_x = x + (w // 2)
        center_y = y + (h // 2)
        
        # Click the center of the detected button
        if not self.mouse.is_in_forbidden_zone(center_x, center_y):
            return self.mouse.click(center_x, center_y)
            
        return False

    def _handle_transition_level(self) -> State:
        """
        Execution Phase: Initiates the level transition sequence via Computer Vision.
        """
        logger.info("Initiating level transition sequence...")
        
        self.check_interrupts()
        
        # 1. Click New Level Button (Primary Renovate)
        logger.info("Searching for New Level icon/button via CV...")
        screenshot = self.capture.capture(max_y=config.WINDOW_HEIGHT)
        if "newLevel" in self.bot.templates:
            template, mask = self.bot.templates["newLevel"]
            found, conf, x, y = self.matcher.find_template(
                screenshot, template, mask=mask, threshold=config.NEW_LEVEL_THRESHOLD
            )
            if found:
                logger.info(f"Clicking New Level icon at ({x}, {y})")
                if self.mouse.click(x, y):
                    self.bot.sleep(config.NEW_LEVEL_BUTTON_DELAY)
                    
                    # 2. Click New Level Confirmation (Dynamically find red Renovate button)
                    logger.info("Dynamically searching for Renovate button...")
                    if self._find_and_click_largest_red_button():
                        self.bot.sleep(config.NEW_LEVEL_FOLLOWUP_DELAY)
                        
                        # 3. Click Final Transition (Dynamically find red Fly button)
                        logger.info("Dynamically searching for Fly button...")
                        if self._find_and_click_largest_red_button():
                            logger.info("Level transition sequence successfully executed.")
                            
                            # Flowchart: Send Telegram Msg -> Reset Cycle Stats
                            self._execute_post_transition_tasks()
                            
                            # Settle after level reset
                            self.bot.sleep(config.UI_TRANSITION_PADDING)
                            return State.FIND_RED_ICONS
            
        logger.warning("Level transition sequence failed or incomplete. Returning to discovery.")
        return State.FIND_RED_ICONS

    def _execute_post_transition_tasks(self) -> None:
        """Executes notification and stat resets per the flowchart."""
        # 1. Notify Telegram
        if self.bot.telegram.enabled:
            # Placeholder level number (would be tracked in bot state)
            self.bot.telegram.notify_new_level(1, 0.0) 
            
        # 2. Reset Cycle Stats (Historical Learner)
        logger.info("Resetting cycle statistics for new level.")
        self.bot.historical_learner.reset()
