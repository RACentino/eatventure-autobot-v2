"""Handler for detecting and clicking red icons."""

import logging
import time
from typing import List, Tuple, Optional, Dict, Any

from bot.handlers.base import BaseHandler
from bot.state_machine import State
from core import config
from core.logger import setup_logger

logger = setup_logger("bot.handlers.icon_handler")

class IconHandler(BaseHandler):
    """
    Tactical Red Icon Handler.
    Encapsulates logic for finding and interacting with red icons.
    """
    def __init__(self, bot: 'EatventureBot'):
        super().__init__(bot)
        self.red_icons: List[Tuple[float, int, int]] = []
        self.current_icon_index: int = 0
        self._recent_red_icon_history: List[Dict[str, Any]] = []
        # Blackout cache: { (world_x, world_y): timestamp }
        self._blackout_cache: Dict[Tuple[int, int], float] = {}

    def handle(self, current_state: State) -> State:
        """Branches based on the active FSM state."""
        if current_state == State.FIND_RED_ICONS:
            return self._handle_find_red_icons()
        elif current_state == State.CLICK_RED_ICON:
            return self._handle_click_red_icon()
        return State.FIND_RED_ICONS

    def _partition_red_icons(self, icons: List[Tuple[float, int, int]]) -> Tuple[List[Tuple[float, int, int]], List[Tuple[float, int, int]]]:
        """
        Gatekeeper Phase: Splits detections into safe and forbidden lists based on coordinates.
        Also applies the blackout cache for world-space persistence.
        """
        safe = []
        forbidden = []
        now = time.monotonic()
        
        # 1. Purge expired blackout entries
        self._blackout_cache = {
            pos: ts for pos, ts in self._blackout_cache.items() 
            if now - ts <= config.FORBIDDEN_BLACKOUT_DURATION
        }

        for conf, x, y in icons:
            # 2. Convert to world-space for blackout check
            # world_y = screen_y + (scroll_offset * pixels_per_unit)
            # For simplicity, we use the units directly if precision allows
            world_y = int(y + (self.bot.scroll_offset_units * config.SCROLL_PIXEL_STEP))
            world_pos = (x, world_y)

            # 3. Check Blackout Cache
            is_blacked_out = False
            for (bx, by), ts in self._blackout_cache.items():
                if abs(bx - x) <= config.RED_ICON_STABILITY_RADIUS and abs(by - world_y) <= config.RED_ICON_STABILITY_RADIUS:
                    is_blacked_out = True
                    break
            
            if is_blacked_out:
                continue

            # 4. Standard Forbidden Zone Check
            click_x = x + config.RED_ICON_OFFSET_X
            click_y = y + config.RED_ICON_OFFSET_Y
            if not self.mouse.is_in_forbidden_zone(click_x, click_y):
                safe.append((conf, x, y))
            else:
                forbidden.append((conf, x, y))
                # Add to blackout cache to prevent immediate re-trigger during scroll
                self._blackout_cache[world_pos] = now

        return safe, forbidden

    def _handle_find_red_icons(self, interrupt: bool = False) -> Optional[State]:
        """
        Discovery Phase: Scans for stable red icons across all templates.
        """
        if not interrupt:
            logger.info("Scanning for stable red icons...")
        screenshot = self.capture.capture(max_y=config.MAX_SEARCH_Y)
        
        all_matches = []
        for name, template, mask in self.bot.iter_red_icon_templates():
            matches = self.matcher.find_all_templates(
                screenshot, 
                template, 
                mask=mask, 
                threshold=config.RED_ICON_THRESHOLD,
                min_distance=config.RED_ICON_MIN_DISTANCE,
                template_name=name
            )
            
            for conf, x, y in matches:
                # Color Gate Check
                red_pixels = self.matcher.count_red_pixels(screenshot, x, y)
                if red_pixels >= config.RED_ICON_PIXEL_THRESHOLD:
                    all_matches.append((conf, x, y, red_pixels))

        # Stability Debouncing (Slow is Smooth)
        stable_icons = self._stable_red_icons(all_matches) if not interrupt else [m[:3] for m in all_matches if m[3] >= config.RED_ICON_PIXEL_THRESHOLD * 1.5]
        
        # Guard Layer: Partition Red Icons (Safe vs Forbidden)
        safe_icons, forbidden_icons = self._partition_red_icons(stable_icons)

        if not safe_icons:
            if forbidden_icons and not interrupt:
                logger.info(f"Detected {len(forbidden_icons)} red icons in forbidden zones. Scrolling to clear them.")
                return State.SCROLL
            elif not interrupt:
                logger.info("No stable red icons detected. Initiating fallback search.")
                return State.OPEN_BOXES
            return None

        if interrupt:
            logger.info(f"[ScrollInterrupt] Found {len(safe_icons)} safe red icons.")
        else:
            logger.info(f"Locked on {len(safe_icons)} stable red icons.")
        self.red_icons = sorted(safe_icons, key=lambda i: i[1]) # Sort by horizontal position
        self.current_icon_index = 0
        return State.CLICK_RED_ICON

    def _stable_red_icons(self, current_matches: List[Tuple[float, int, int, int]]) -> List[Tuple[float, int, int]]:
        """Temporal consistency check to prevent transient detections."""
        now = time.monotonic()
        ttl = config.RED_ICON_STABILITY_CACHE_TTL
        radius = config.RED_ICON_STABILITY_RADIUS
        min_hits = config.RED_ICON_STABILITY_MIN_HITS
        
        # Purge history
        self._recent_red_icon_history = [
            h for h in self._recent_red_icon_history if now - h["timestamp"] <= ttl
        ]
        
        # Add current frame
        self._recent_red_icon_history.append({"timestamp": now, "icons": current_matches})
        
        stable = []
        for conf, x, y, px_count in current_matches:
            # Immediate trigger for high density
            if px_count >= config.RED_ICON_PIXEL_THRESHOLD * 1.5:
                stable.append((conf, x, y))
                continue
                
            hits = 0
            for entry in self._recent_red_icon_history:
                for h_conf, hx, hy, hpx in entry["icons"]:
                    if abs(hx - x) <= radius and abs(hy - y) <= radius:
                        hits += 1
                        break
            
            if hits >= min_hits:
                stable.append((conf, x, y))
        
        return stable

    def _handle_click_red_icon(self) -> State:
        """
        Execution Phase: Interacts with the prioritized red icon queue with offsets.
        """
        if self.current_icon_index >= len(self.red_icons):
            return State.OPEN_BOXES

        conf, x, y = self.red_icons[self.current_icon_index]
        
        # Re-verify presence before clicking (Slow is Smooth)
        # Using a slight threshold drop for verification as in legacy
        verify_screenshot = self.capture.capture(max_y=config.MAX_SEARCH_Y)
        red_pixels = self.matcher.count_red_pixels(verify_screenshot, x, y)
        if red_pixels < config.RED_ICON_PIXEL_THRESHOLD * 0.8:
            logger.info(f"Icon at ({x}, {y}) vanished. Skipping.")
            self.current_icon_index += 1
            return State.CLICK_RED_ICON if self.current_icon_index < len(self.red_icons) else State.OPEN_BOXES

        click_x = x + config.RED_ICON_OFFSET_X
        click_y = y + config.RED_ICON_OFFSET_Y
        
        logger.info(f"Interacting with red icon at ({click_x}, {click_y})")
        self.check_interrupts()
        
        # Use tuner values if available for adaptive pacing
        click_delay = getattr(self.bot.tuner, "click_delay", config.CLICK_DELAY)
        
        start_time = time.monotonic()
        success = self.mouse.click(click_x, click_y, wait_after=False)
        exec_time = time.monotonic() - start_time
        
        # Learning Loop (Parallel) -> LogTime -> Tuner
        if hasattr(self.bot, 'tuner'):
            self.bot.tuner.record_click_result(success, exec_time)
            
        if success:
            self.bot.sleep(click_delay)
            self.current_icon_index += 1
            return State.CHECK_UNLOCK
        
        self.current_icon_index += 1
        return State.CLICK_RED_ICON if self.current_icon_index < len(self.red_icons) else State.OPEN_BOXES
