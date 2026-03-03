import time
import logging
from typing import Optional, Callable, Any

from core import config

logger = logging.getLogger(__name__)

class OscillatingSearcher:
    """
    Refactored Algorithm Engine: Implements the strictly incremental Oscillating Search.
    Follows a multi-step pattern (UP then DOWN) for each widening cycle with 
    precise settle-and-scan synchronization.
    """

    def __init__(self, bot: Any):
        self.bot = bot
        self.max_cycles = config.MAX_SCROLL_CYCLES
        self.scroll_increment = config.SCROLL_INCREMENT_STEP
        # Baseline settle time used for synchronization
        self.settle_duration = config.POST_SCROLL_SETTLE

    def execute_cycle(self, 
                      check_priority: Callable, 
                      check_main_target: Callable, 
                      check_fallbacks: Optional[Callable] = None) -> Optional[Any]:
        """
        The orchestrator for the widening incremental search pattern.
        """
        logger.info(f"[Search] Initializing Incremental Search (Limit: {self.max_cycles} cycles)")

        # Baseline: Verify current area before moving
        initial_hit = self._perform_vision_pass(check_priority, check_main_target, check_fallbacks)
        if initial_hit:
            return initial_hit

        # Multi-step loop: Follows the exact incrementing pattern up to MAX_SCROLL_CYCLES
        for cycle_index in range(1, self.max_cycles + 1):
            # Calculate steps for this specific cycle
            steps_in_leg = cycle_index * self.scroll_increment
            
            # --- PHASE 1: DOWN LEG ---
            logger.info(f"[Search] Cycle {cycle_index}: Starting DOWN leg ({steps_in_leg} steps)")
            target_found = self._run_step_sequence(steps_in_leg, 1, 
                                                  check_priority, check_main_target, check_fallbacks)
            if target_found:
                return target_found

            # Boundary Scan: Ensure the area is clear before reversing direction
            self.bot.sleep(config.CYCLE_PAUSE_DURATION)
            boundary_hit = self._perform_vision_pass(check_priority, check_main_target, check_fallbacks)
            if boundary_hit:
                return boundary_hit

            # --- PHASE 2: UP LEG ---
            logger.info(f"[Search] Cycle {cycle_index}: Starting UP leg ({steps_in_leg} steps)")
            target_found = self._run_step_sequence(steps_in_leg, -1, 
                                                  check_priority, check_main_target, check_fallbacks)
            if target_found:
                return target_found

            # Inter-Cycle Transition Scan: Stabilization before next cycle expansion
            self.bot.sleep(config.CYCLE_PAUSE_DURATION)
            cycle_hit = self._perform_vision_pass(check_priority, check_main_target, check_fallbacks)
            if cycle_hit:
                return cycle_hit

        logger.warning(f"[Search] Logic exhausted after {self.max_cycles} cycles.")
        return None

    def _run_step_sequence(self, count: int, direction: int, p_check: Callable, m_check: Callable, f_check: Optional[Callable]) -> Optional[Any]:
        """Executes a sequence of individual scroll-and-scan steps with mandatory settle padding."""
        for _ in range(count):
            # Mechanical Guard: Check bot status before interaction
            if not self.bot.running:
                return None

            # 1. Action: Execute the individual scroll step
            self.perform_scroll(direction)

            # 2. Mandatory settle: Padded duration ensures frame stability before scan.
            # Synchronizes SCROLL_INTERVAL_PAUSE and POST_SCROLL_SETTLE to ensure slow/smooth execution.
            settle_wait = self.settle_duration + config.SCROLL_INTERVAL_PAUSE
            self.bot.sleep(settle_wait)

            # 3. Intra-loop vision interrupt (High-frequency Red Icon scan)
            # Delegates back to the orchestrator to check for interrupts
            red_interrupt = self.bot.check_intra_scroll_red_interrupt()
            if red_interrupt:
                return red_interrupt

            # 4. Standard layered scan pipeline (Priority -> Main -> Fallback)
            hit = self._perform_vision_pass(p_check, m_check, f_check)
            if hit:
                return hit
        return None

    def _perform_vision_pass(self, p_check: Callable, m_check: Callable, f_check: Optional[Callable]) -> Optional[Any]:
        """Atomic vision check following the Priority -> Main -> Fallback protocol."""
        priority_hit = p_check()
        if priority_hit:
            return priority_hit

        main_hit = m_check()
        if main_hit:
            return main_hit

        if f_check:
            fallback_hit = f_check()
            if fallback_hit:
                return fallback_hit
            
        return None

    def perform_scroll(self, direction: Any, distance_ratio: Optional[float] = None, duration: Optional[float] = None):
        """Standardized mechanical drag interface with config-driven timing."""
        dir_int = self._map_direction(direction)
        start_x, start_y = config.SCROLL_START_POS
        
        # Distance calculation
        ratio = distance_ratio or config.SCROLL_DISTANCE_RATIO
        pixel_distance = int(config.SCROLL_PIXEL_STEP * ratio)
        end_y = start_y - (pixel_distance * dir_int)

        # Timing: Use SCROLL_DURATION from config for smooth, non-ballistic movement.
        scroll_duration = duration if duration is not None else config.SCROLL_DURATION

        self.bot.mouse.drag(
            start_x, start_y, start_x, end_y, 
            duration=scroll_duration
        )
        
        if hasattr(self.bot, 'scroll_offset_units'):
            self.bot.scroll_offset_units -= (ratio * dir_int)

    def _map_direction(self, direction: Any) -> int:
        """Standardizes input directions into integers (1 or -1)."""
        if isinstance(direction, int): 
            return direction
        return {"DOWN": -1, "UP": 1}.get(str(direction).upper(), 1)
