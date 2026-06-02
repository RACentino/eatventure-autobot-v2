import logging

from core import bounded_float, bounded_int, config
from bot.state_machine import State
from bot.types import StateResult

logger = logging.getLogger(__name__)


class ScrollHandlerMixin:
    def _reset_search_cycle(self) -> None:
        self.cycle_counter = 0
        self.wait_for_unlock_attempts = 0
        self._oscillation_cycle_index = 1
        self._oscillation_leg_direction = 1
        self._oscillation_leg_progress = 0
        self._new_level_red_icon_verified = False

    def _advance_oscillation_progress(self) -> None:
        increment_step = bounded_int(config.SCROLL_INCREMENT_STEP, 1, minimum=1)
        max_scroll_cycles = bounded_int(config.MAX_SCROLL_CYCLES, 1, minimum=1)
        target_steps = max(1, int(self._oscillation_cycle_index) * increment_step)
        self._oscillation_leg_progress += 1
        if self._oscillation_leg_progress < target_steps:
            return
        self._oscillation_leg_progress = 0
        if self._oscillation_leg_direction > 0:
            self._oscillation_leg_direction = -1
            return
        self._oscillation_leg_direction = 1
        self._oscillation_cycle_index += 1
        if self._oscillation_cycle_index > max_scroll_cycles:
            self._oscillation_cycle_index = 1

    def _scroll_distance(self) -> tuple[int, int, int]:
        pixel_step = bounded_float(config.SCROLL_PIXEL_STEP, 90.0, minimum=1.0)
        distance_ratio = bounded_float(config.SCROLL_DISTANCE_RATIO, 1.0, minimum=0.1)
        distance = int(round(pixel_step * distance_ratio))
        start_x, start_y = config.SCROLL_START_POS
        return distance, start_x, start_y

    def _perform_oscillating_scroll_step(self) -> bool:
        distance, start_x, start_y = self._scroll_distance()
        direction = 1 if self._oscillation_leg_direction > 0 else -1
        target_y = start_y - distance if direction > 0 else start_y + distance
        logger.info(
            "Oscillating scroll step: cycle=%s direction=%s progress=%s",
            self._oscillation_cycle_index,
            "down" if direction > 0 else "up",
            self._oscillation_leg_progress + 1,
        )
        moved = self.mouse_controller.drag(
            start_x,
            start_y,
            start_x,
            target_y,
            duration=config.SCROLL_DURATION,
            relative=True,
        )
        if moved:
            self._sleep(config.POST_SCROLL_SETTLE)
            self._sleep(config.SCROLL_INTERVAL_PAUSE)
            self._advance_oscillation_progress()
            self._click_idle()
        return bool(moved)

    def _perform_single_down_scroll(self) -> bool:
        distance, start_x, start_y = self._scroll_distance()
        target_y = start_y - distance
        logger.info("Verification scroll down before confirming new level red icon")
        moved = self.mouse_controller.drag(
            start_x,
            start_y,
            start_x,
            target_y,
            duration=config.SCROLL_DURATION,
            relative=True,
        )
        if moved:
            self._sleep(config.POST_SCROLL_SETTLE)
            self._sleep(config.SCROLL_INTERVAL_PAUSE)
        return bool(moved)

    def handle_scroll(self, current_state: State) -> StateResult:
        self._click_idle()
        self._perform_oscillating_scroll_step()
        self.cycle_counter = 0
        return State.FIND_RED_ICONS
