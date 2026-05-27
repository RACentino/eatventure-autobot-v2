import logging
from datetime import datetime

from core import config
from bot.state_machine import State
from bot.types import StateResult

logger = logging.getLogger(__name__)
MAX_TRANSITION_LEVEL_ATTEMPTS = 10


class TransitionHandlerMixin:
    def _record_level_completion(self) -> float:
        self.total_levels_completed += 1
        elapsed = 0.0
        if self.current_level_start_time is not None:
            elapsed = (datetime.now() - self.current_level_start_time).total_seconds()
        self.current_level_start_time = datetime.now()
        self._reset_search_cycle()
        self.telegram.notify_new_level(self.total_levels_completed, elapsed)
        self.historical_learner.record_completion(elapsed, "transition")
        return elapsed

    def handle_check_new_level(self, current_state: State) -> StateResult:
        if not self._click_idle():
            logger.warning("Failed to clear focus before confirming the new level")
            return State.CHECK_NEW_LEVEL
        self._sleep(config.IDLE_SETTLE_DELAY)
        if not self._new_level_red_icon_verified:
            if not self._perform_single_down_scroll():
                logger.warning("Failed to perform verification scroll for new level red icon")
                return State.CHECK_NEW_LEVEL

            confirmed_icon = self._find_new_level_red_icon()
            if confirmed_icon is None:
                logger.info("New level red icon disappeared after verification scroll; resuming main flow")
                self._new_level_red_icon_verified = False
                self.vision_optimizer.update_new_level_red_icon_miss()
                self._reset_search_cycle()
                return State.FIND_RED_ICONS

            self._new_level_red_icon_verified = True
            self.vision_optimizer.update_new_level_red_icon_confidence(confirmed_icon[0])
            logger.info(
                "New level red icon confirmed after verification scroll at (%s, %s) [%.3f]",
                confirmed_icon[1],
                confirmed_icon[2],
                confirmed_icon[0],
            )

        opened = self.mouse_controller.click(
            config.NEW_LEVEL_BUTTON_POS[0],
            config.NEW_LEVEL_BUTTON_POS[1],
            relative=True,
        )
        if not opened:
            logger.warning("Failed to click the new level button at %s", config.NEW_LEVEL_BUTTON_POS)
            return State.CHECK_NEW_LEVEL
        self._sleep(config.NEW_LEVEL_CLICK_SETTLE_DELAY)
        advanced = self.mouse_controller.click(
            config.LEVEL_TRANSITION_POS[0],
            config.LEVEL_TRANSITION_POS[1],
            relative=True,
        )
        if not advanced:
            logger.warning("Failed to click the level transition button at %s", config.LEVEL_TRANSITION_POS)
            return State.CHECK_NEW_LEVEL
        if not self._sleep(config.LEVEL_TRANSITION_CONFIRM_DELAY):
            return State.OPEN_BOXES
        elapsed = self._record_level_completion()
        logger.info(
            "Level %s completed via verified red-icon path. Time spent: %.1fs",
            self.total_levels_completed,
            elapsed,
        )
        return State.WAIT_FOR_UNLOCK

    def handle_transition_level(self, current_state: State) -> StateResult:
        self._click_idle()

        max_attempts = min(MAX_TRANSITION_LEVEL_ATTEMPTS, max(1, int(config.TRANSITION_LEVEL_MAX_ATTEMPTS)))
        for attempt in range(max_attempts):
            limited_screenshot = self.window_capture.capture(max_y=config.MAX_SEARCH_Y)

            found, confidence, x, y = self._find_new_level_button(limited_screenshot)
            if found:
                self.vision_optimizer.update_new_level_confidence(confidence)
                logger.info("New level button found at (%s, %s) on attempt %s", x, y, attempt + 1)
                clicked = self.mouse_controller.click(x, y, relative=True)
                if not clicked:
                    logger.warning("New level button click failed at (%s, %s)", x, y)
                    return State.CHECK_NEW_LEVEL
                self._sleep(config.LEVEL_TRANSITION_COMPLETE_DELAY)
                elapsed = self._record_level_completion()
                logger.info(
                    "Level %s completed. Time spent: %.1fs",
                    self.total_levels_completed,
                    elapsed,
                )
                return State.WAIT_FOR_UNLOCK

            if attempt < max_attempts - 1:
                self._sleep(config.LEVEL_TRANSITION_RETRY_INTERVAL)

        self.vision_optimizer.update_new_level_miss()
        logger.warning("New level button not found after %s attempts", max_attempts)
        self._reset_search_cycle()
        return State.FIND_RED_ICONS

    def handle_wait_for_unlock(self, current_state: State) -> StateResult:
        if not self._click_idle():
            logger.warning("Failed to clear focus while waiting for the next unlock")
            return State.WAIT_FOR_UNLOCK
        self._sleep(config.IDLE_SETTLE_DELAY)

        self.wait_for_unlock_attempts += 1
        if self.wait_for_unlock_attempts > self.max_wait_for_unlock_attempts:
            logger.warning(
                "Unlock button not found after %s attempts, resetting",
                self.max_wait_for_unlock_attempts,
            )
            self.wait_for_unlock_attempts = 0
            self._reset_search_cycle()
            return State.FIND_RED_ICONS

        screenshot = self.window_capture.capture()
        template_pair = self._template("unlock")
        if template_pair is None:
            self._sleep(config.WAIT_FOR_UNLOCK_RETRY_INTERVAL)
            return State.WAIT_FOR_UNLOCK

        template, mask = template_pair
        found, confidence, x, y = self.image_matcher.find_template(
            screenshot,
            template,
            mask=mask,
            threshold=config.UNLOCK_THRESHOLD,
            template_name="unlock",
        )
        if not found:
            self._sleep(config.WAIT_FOR_UNLOCK_RETRY_INTERVAL)
            return State.WAIT_FOR_UNLOCK

        logger.info("Unlock button found at (%s, %s) [%.3f]", x, y, confidence)
        if self.mouse_controller.is_in_forbidden_zone(x, y, relative=True):
            logger.warning("Unlock button found in forbidden zone at (%s, %s)", x, y)
            self._sleep(config.WAIT_FOR_UNLOCK_RETRY_INTERVAL)
            return State.WAIT_FOR_UNLOCK
        if not self.mouse_controller.click(x, y, relative=True):
            logger.warning("Unlock button click failed at (%s, %s)", x, y)
            return State.WAIT_FOR_UNLOCK

        self._sleep(config.WAIT_FOR_UNLOCK_SETTLE_DELAY)
        self.wait_for_unlock_attempts = 0
        self._reset_search_cycle()
        return State.FIND_RED_ICONS
