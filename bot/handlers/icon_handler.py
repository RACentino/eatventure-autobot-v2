import logging
from collections.abc import Iterable
from typing import Any

from core import config
from bot.state_machine import State
from bot.types import RedIcon, StateResult

logger = logging.getLogger(__name__)


class IconHandlerMixin:
    def _remember_successful_red_icon_position(self, y_value: Any) -> None:
        y_value = int(y_value)
        for existing_y in self.successful_red_icon_positions:
            if abs(existing_y - y_value) < 12:
                return
        self.successful_red_icon_positions.append(y_value)

    def _clickable_red_icons(self, red_icons: Iterable[RedIcon]) -> list[RedIcon]:
        return [
            (confidence, x, y)
            for confidence, x, y in red_icons
            if not self.mouse_controller.is_in_forbidden_zone(
                x + config.RED_ICON_OFFSET_X,
                y + config.RED_ICON_OFFSET_Y,
                relative=True,
            )
        ]

    def _red_icon_priority_key(self, icon: RedIcon) -> tuple[int, int, float]:
        confidence, _, y = icon
        for success_y in self.successful_red_icon_positions:
            if abs(y - success_y) < 50:
                return (0, y, -confidence)
        return (1, y, -confidence)

    def handle_find_red_icons(self, current_state: State) -> StateResult:
        self._click_idle()

        self.work_done = False

        screenshot = self.window_capture.capture(max_y=config.EXTENDED_SEARCH_Y)
        limited_screenshot = screenshot[: config.MAX_SEARCH_Y, :]

        has_new_level_template = self._template("newLevel") is not None
        found, confidence, x, y = self._find_new_level_button(limited_screenshot)
        if found:
            self.cycle_counter = 0
            self.vision_optimizer.update_new_level_confidence(confidence)
            logger.info("newLevel.png found at (%s, %s)", x, y)
            return State.TRANSITION_LEVEL
        if has_new_level_template:
            self.vision_optimizer.update_new_level_miss()

        scan_threshold = self._red_icon_scan_threshold()

        min_matches = self._red_icon_min_matches()
        self.red_icons, valid_red_icon_confidences, best_new_level_icon = self._scan_red_icon_frame(
            screenshot,
            limited_screenshot,
            scan_threshold,
            min_matches,
        )

        self.vision_optimizer.update_red_icon_scan(valid_red_icon_confidences)

        if (
            not self.red_icons
            and best_new_level_icon is None
            and self._scrcpy_miss_recovery_sleep(config.SCRCPY_RED_ICON_MISS_RECOVERY_DELAY)
        ):
            screenshot = self.window_capture.capture(max_y=config.EXTENDED_SEARCH_Y)
            limited_screenshot = screenshot[: config.MAX_SEARCH_Y, :]
            found, confidence, x, y = self._find_new_level_button(limited_screenshot)
            if found:
                self.cycle_counter = 0
                self.vision_optimizer.update_new_level_confidence(confidence)
                logger.info("newLevel.png found at (%s, %s) after SCRCPY recovery", x, y)
                return State.TRANSITION_LEVEL
            self.red_icons, valid_red_icon_confidences, best_new_level_icon = self._scan_red_icon_frame(
                screenshot,
                limited_screenshot,
                scan_threshold,
                min_matches,
            )
            self.vision_optimizer.update_red_icon_scan(valid_red_icon_confidences)

        if best_new_level_icon is not None:
            self.vision_optimizer.update_new_level_red_icon_confidence(best_new_level_icon[0])
            logger.info(
                "New level red icon detected at (%s, %s) [%.3f]",
                best_new_level_icon[1],
                best_new_level_icon[2],
                best_new_level_icon[0],
            )
            self._new_level_red_icon_verified = False
            return State.CHECK_NEW_LEVEL
        self.vision_optimizer.update_new_level_red_icon_miss()

        if self.red_icons:
            filtered_icons = self._clickable_red_icons(self.red_icons)

            if not filtered_icons:
                logger.info("No valid red icons after forbidden-zone filtering")
                return State.OPEN_BOXES

            self.red_icons = sorted(filtered_icons, key=self._red_icon_priority_key)
            self.current_red_icon_index = 0
            self.cycle_counter = 0
            self.work_done = True
            logger.info("%s red icons ready to process", len(self.red_icons))
            return State.CLICK_RED_ICON

        return State.OPEN_BOXES

    def handle_click_red_icon(self, current_state: State) -> StateResult:
        if self.current_red_icon_index >= len(self.red_icons):
            logger.info("All red icons processed, continuing cycle")
            return State.OPEN_BOXES

        confidence, x, y = self.red_icons[self.current_red_icon_index]
        click_x = x + config.RED_ICON_OFFSET_X
        click_y = y + config.RED_ICON_OFFSET_Y

        clicked = self.mouse_controller.click(click_x, click_y, relative=True)
        self.tuner.record_click_result(clicked)
        self._apply_tuning()
        if not clicked:
            logger.warning("Red icon click failed at (%s, %s)", click_x, click_y)
            self.current_red_icon_index += 1
            if self.current_red_icon_index < len(self.red_icons):
                return State.CLICK_RED_ICON
            return State.OPEN_BOXES

        logger.info(
            "Clicked red icon %s/%s at (%s, %s) [%.3f]",
            self.current_red_icon_index + 1,
            len(self.red_icons),
            click_x,
            click_y,
            confidence,
        )
        return State.CHECK_UNLOCK

    def handle_check_unlock(self, current_state: State) -> StateResult:
        limited_screenshot = self.window_capture.capture(max_y=config.MAX_SEARCH_Y)

        template_pair = self._template("unlock")
        if template_pair is None:
            return State.SEARCH_UPGRADE_STATION

        template, mask = template_pair
        found, confidence, x, y = self.image_matcher.find_template(
            limited_screenshot,
            template,
            mask=mask,
            threshold=config.UNLOCK_THRESHOLD,
            template_name="unlock",
        )
        if not found or self.mouse_controller.is_in_forbidden_zone(x, y, relative=True):
            return State.SEARCH_UPGRADE_STATION

        logger.info("Unlock found at (%s, %s) [%.3f]", x, y, confidence)
        if not self.mouse_controller.click(x, y, relative=True):
            logger.warning("Unlock click failed at (%s, %s)", x, y)
            return State.CHECK_UNLOCK

        return State.SEARCH_UPGRADE_STATION
