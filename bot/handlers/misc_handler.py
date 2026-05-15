import logging

from core import config
from bot.state_machine import State
from bot.types import BoxCandidate, StateResult

logger = logging.getLogger(__name__)


class MiscHandlerMixin:
    def _click_box_candidates(self, merged_boxes: list[BoxCandidate]) -> tuple[int, float]:
        boxes_found = 0
        best_box_confidence = 0.0
        for confidence, x, y, _, _, _ in merged_boxes:
            if self.mouse_controller.is_in_forbidden_zone(x, y, relative=True):
                logger.debug("Box candidate is in a forbidden zone")
                continue
            if self.mouse_controller.click(x, y, relative=True):
                boxes_found += 1
                best_box_confidence = max(best_box_confidence, confidence)
        return boxes_found, best_box_confidence

    def _next_state_after_box_cycle(self) -> State:
        if self.consecutive_failed_cycles >= 3:
            self.consecutive_failed_cycles = 0
            self.cycle_counter = 0
            logger.info("Repeated search failures reached threshold, forcing scroll")
            return State.SCROLL

        if self.upgrade_found_in_cycle:
            self.upgrade_found_in_cycle = False
            self.cycle_counter = 0
            logger.info("Upgrade found in cycle, staying in current area")
            return State.FIND_RED_ICONS

        if self.work_done:
            self.cycle_counter = 0
            logger.info("Work completed in current area, rescanning before scrolling")
            return State.FIND_RED_ICONS

        self.cycle_counter += 1
        logger.info("No work detected in current area (idle pass %s/2)", self.cycle_counter)
        if self.cycle_counter >= 2:
            self.cycle_counter = 0
            return State.SCROLL

        return State.FIND_RED_ICONS

    def handle_open_boxes(self, current_state: State) -> StateResult:
        self._click_idle()

        limited_screenshot = self.window_capture.capture(
            max_y=getattr(config, "BOX_SEARCH_Y", config.MAX_SEARCH_Y)
        )

        found, confidence, _, _ = self._find_new_level_button(limited_screenshot)
        if found:
            self.vision_optimizer.update_new_level_confidence(confidence)
            logger.info("New level found while opening boxes")
            return State.TRANSITION_LEVEL

        box_threshold = self._box_threshold()
        box_candidates = self._collect_box_candidates(limited_screenshot, box_threshold)
        if not box_candidates and self._scrcpy_miss_recovery_sleep(getattr(config, "SCRCPY_BOX_MISS_RECOVERY_DELAY", 0.0)):
            limited_screenshot = self.window_capture.capture(
                max_y=getattr(config, "BOX_SEARCH_Y", config.MAX_SEARCH_Y)
            )
            found, confidence, _, _ = self._find_new_level_button(limited_screenshot)
            if found:
                self.vision_optimizer.update_new_level_confidence(confidence)
                logger.info("New level found while opening boxes after SCRCPY recovery")
                return State.TRANSITION_LEVEL
            box_candidates = self._collect_box_candidates(limited_screenshot, box_threshold)

        merged_boxes = self._merge_box_candidates_with_supervision(box_candidates)
        boxes_found, best_box_confidence = self._click_box_candidates(merged_boxes)

        if boxes_found > 0:
            self.work_done = True
            self.cycle_counter = 0
            self.vision_optimizer.update_box_confidence(best_box_confidence)
            logger.info("Opened %s boxes", boxes_found)
        else:
            self.vision_optimizer.update_box_miss()

        return self._next_state_after_box_cycle()
