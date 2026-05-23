import logging
import math
import time

from core import config
from interaction.mouse import precise_sleep
from bot.state_machine import State
from bot.types import RedIcon, StateResult

logger = logging.getLogger(__name__)


class UpgradeHandlerMixin:
    @staticmethod
    def _hold_max_duration() -> float:
        hold_max_duration = float(config.CLICK_HOLD_MAX_DURATION)
        if not math.isfinite(hold_max_duration):
            return 0.0
        return max(0.0, hold_max_duration)

    def _position_cursor_for_upgrade_hold(self, x: int, y: int) -> tuple[int, int] | None:
        screen_pos = self.mouse_controller._resolve_screen_position(x, y, relative=True)
        if screen_pos is None:
            logger.warning("Upgrade station hold position could not be resolved at (%s, %s)", x, y)
            return None

        screen_x, screen_y = screen_pos
        if self.mouse_controller._set_cursor_pos(screen_x, screen_y):
            if self.mouse_controller.move_delay > 0:
                precise_sleep(self.mouse_controller.move_delay)
            return screen_x, screen_y

        self.tuner.record_click_result(False)
        self._apply_tuning()
        logger.warning("Failed to position cursor for Upgrade Station hold at (%s, %s)", x, y)
        return None

    def _find_current_upgrade_hold_match(
        self,
        base_threshold: float,
        relaxed_threshold: float,
    ) -> RedIcon | None:
        current_match = self._find_upgrade_station_match(base_threshold)
        if current_match is not None:
            return current_match
        current_match = self._find_upgrade_station_match(relaxed_threshold)
        if current_match is not None:
            return current_match
        if self._scrcpy_miss_recovery_sleep(config.SCRCPY_UPGRADE_MISS_RECOVERY_DELAY):
            return self._find_upgrade_station_match(relaxed_threshold)
        return None

    def _reposition_held_upgrade_station(
        self,
        screen_position: tuple[int, int],
        x: int,
        y: int,
    ) -> tuple[int, int] | None:
        next_screen_pos = self.mouse_controller._resolve_screen_position(x, y, relative=True)
        if next_screen_pos is None:
            logger.warning("Upgrade station hold target became invalid at (%s, %s)", x, y)
            return None
        if next_screen_pos == screen_position:
            return screen_position
        if self.mouse_controller._set_cursor_pos(next_screen_pos[0], next_screen_pos[1]):
            return next_screen_pos
        logger.warning("Failed to reposition held cursor to (%s, %s)", x, y)
        return None

    @staticmethod
    def _hold_duration_reached(hold_max_duration: float, hold_elapsed: float) -> bool:
        return hold_max_duration > 0.0 and hold_elapsed >= hold_max_duration

    def _monitor_upgrade_station_hold(
        self,
        screen_position_holder: list[tuple[int, int]],
        base_threshold: float,
        relaxed_threshold: float,
        hold_check_interval: float,
        hold_max_duration: float,
        hold_started_at: float,
    ) -> tuple[bool, bool, float]:
        while True:
            hold_elapsed = time.monotonic() - hold_started_at
            if self._stop_requested.is_set():
                logger.warning("Upgrade station hold interrupted after %.2fs", hold_elapsed)
                return False, False, hold_elapsed
            if self._hold_duration_reached(hold_max_duration, hold_elapsed):
                logger.warning(
                    "Upgrade station hold max duration %.2fs reached after %.2fs; releasing hold",
                    hold_max_duration,
                    hold_elapsed,
                )
                return True, True, hold_elapsed
            if not self._sleep(hold_check_interval):
                return False, False, hold_elapsed

            current_match = self._find_current_upgrade_hold_match(base_threshold, relaxed_threshold)
            if current_match is None:
                return True, False, time.monotonic() - hold_started_at

            confidence, x, y = current_match
            self.upgrade_station_pos = (x, y)
            self.vision_optimizer.update_upgrade_station_confidence(confidence)

            next_position = self._reposition_held_upgrade_station(screen_position_holder[0], x, y)
            if next_position is None:
                return False, False, time.monotonic() - hold_started_at
            screen_position_holder[0] = next_position

    def _hold_upgrade_station_until_complete(
        self,
        screen_position: tuple[int, int],
        base_threshold: float,
        relaxed_threshold: float,
        hold_check_interval: float,
        hold_max_duration: float,
    ) -> tuple[bool, bool, float]:
        screen_x, screen_y = screen_position
        hold_started_at = time.monotonic()
        screen_position_holder = [screen_position]
        holding = False

        try:
            if not self.mouse_controller._left_down_at_screen(
                screen_x,
                screen_y,
                interrupt_check=self._stop_requested.is_set,
            ):
                self.tuner.record_click_result(False)
                self._apply_tuning()
                logger.warning("Upgrade station hold press failed at (%s, %s)", screen_x, screen_y)
                return False, False, time.monotonic() - hold_started_at

            holding = True
            self.tuner.record_click_result(True)
            self._apply_tuning()
            return self._monitor_upgrade_station_hold(
                screen_position_holder,
                base_threshold,
                relaxed_threshold,
                hold_check_interval,
                hold_max_duration,
                hold_started_at,
            )
        finally:
            if holding:
                release_x, release_y = screen_position_holder[0]
                self.mouse_controller._left_up_at_screen(release_x, release_y)

    def handle_search_upgrade_station(self, current_state: State) -> StateResult:
        base_threshold = self._upgrade_station_threshold()
        relaxed_threshold = max(0.0, base_threshold - 0.05)
        max_attempts = int(config.UPGRADE_STATION_SEARCH_MAX_ATTEMPTS)

        for attempt in range(max_attempts):
            if "upgradeStation" not in self.templates:
                break

            current_threshold = base_threshold if attempt < 2 else relaxed_threshold
            match = self._find_upgrade_station_match(current_threshold)
            if match is not None:
                confidence, x, y = match
                logger.info("Upgrade station found at (%s, %s) on attempt %s", x, y, attempt + 1)
                self.upgrade_station_pos = (x, y)
                self.upgrade_found_in_cycle = True
                self.consecutive_failed_cycles = 0
                self.cycle_counter = 0
                self.vision_optimizer.update_upgrade_station_confidence(confidence)
                self.tuner.record_search_result(True)
                self._apply_tuning()
                return State.HOLD_UPGRADE_STATION

            if attempt < max_attempts - 1:
                if not self._sleep(self.tuner.search_interval):
                    return State.OPEN_BOXES

        self.vision_optimizer.update_upgrade_station_miss()
        self.tuner.record_search_result(False)
        self._apply_tuning()
        self.consecutive_failed_cycles += 1
        logger.info("Upgrade station not found, returning to OPEN_BOXES")
        return State.OPEN_BOXES

    def handle_hold_upgrade_station(self, current_state: State) -> StateResult:
        if not self.upgrade_station_pos:
            return State.OPEN_BOXES

        x, y = self.upgrade_station_pos
        if self.mouse_controller.is_in_forbidden_zone(x, y, relative=True):
            logger.warning("Upgrade station blocked by forbidden zone at (%s, %s)", x, y)
            return State.OPEN_BOXES

        logger.info("Single-clicking upgrade station before verification at (%s, %s)", x, y)
        clicked = self.mouse_controller.precise_click(x, y, relative=True)
        self.tuner.record_click_result(clicked)
        self._apply_tuning()
        if not clicked:
            logger.warning("Upgrade station verification click failed at (%s, %s)", x, y)
            return State.OPEN_BOXES

        if not self._sleep(config.UPGRADE_STATION_VERIFY_SETTLE_DELAY):
            return State.OPEN_BOXES

        base_threshold = self._upgrade_station_threshold()
        relaxed_threshold = max(0.0, base_threshold - 0.05)
        verified_match, verification_completed = self._find_verified_upgrade_station_match(
            base_threshold,
            relaxed_threshold,
            (x, y),
        )
        if not verification_completed:
            return State.OPEN_BOXES

        if verified_match is None:
            logger.info("Upgrade station disappeared after verification click; continuing main flow")
            self.upgrade_station_pos = None
            self.upgrade_found_in_cycle = False
            self.vision_optimizer.update_upgrade_station_miss()
            self.tuner.record_search_result(False)
            self._apply_tuning()
            return State.OPEN_BOXES

        confidence, x, y = verified_match
        self.upgrade_station_pos = (x, y)
        self.vision_optimizer.update_upgrade_station_confidence(confidence)
        self.tuner.record_search_result(True)
        self._apply_tuning()
        logger.info("Upgrade station verified active at (%s, %s) [%.3f]", x, y, confidence)
        if self.current_red_icon_index < len(self.red_icons):
            _, _, red_y = self.red_icons[self.current_red_icon_index]
            self._remember_successful_red_icon_position(red_y)

        hold_check_interval = max(0.05, min(0.20, float(config.UPGRADE_STATION_VERIFY_SEARCH_INTERVAL)))
        hold_max_duration = self._hold_max_duration()
        screen_position = self._position_cursor_for_upgrade_hold(x, y)
        if screen_position is None:
            return State.OPEN_BOXES

        logger.info("Press-and-holding upgrade station at (%s, %s)", x, y)
        hold_completed, hold_stopped_by_max_duration, hold_elapsed = self._hold_upgrade_station_until_complete(
            screen_position,
            base_threshold,
            relaxed_threshold,
            hold_check_interval,
            hold_max_duration,
        )
        if not hold_completed:
            return State.OPEN_BOXES

        if hold_stopped_by_max_duration:
            logger.warning(
                "Upgrade station still detected after %.2fs hold; treating verification as failed",
                hold_elapsed,
            )
            self.upgrade_station_pos = None
            self.vision_optimizer.update_upgrade_station_miss()
            self.tuner.record_search_result(False)
            self._apply_tuning()
            return State.SEARCH_UPGRADE_STATION

        logger.info("Upgrade station no longer detected after %.2fs hold", hold_elapsed)
        self.upgrade_station_pos = None

        self._click_idle()
        self._sleep(config.STATE_DELAY)
        self.upgrade_station_counter += 1
        if self.upgrade_station_counter >= int(config.UPGRADE_STATION_STATS_THRESHOLD):
            self.upgrade_station_counter = 0
            logger.info("Upgrade counter reached stats threshold")
            return State.UPGRADE_STATS

        return State.OPEN_BOXES

    def handle_upgrade_stats(self, current_state: State) -> StateResult:
        self._click_idle()

        screenshot = self.window_capture.capture(max_y=config.EXTENDED_SEARCH_Y)
        limited_screenshot = screenshot[: config.MAX_SEARCH_Y, :]

        found, confidence, _, _ = self._find_new_level_button(limited_screenshot)
        if found:
            self.vision_optimizer.update_new_level_confidence(confidence)
            return State.TRANSITION_LEVEL

        best_stats_match = self._find_best_zone_red_icon(
            screenshot,
            self._stats_upgrade_threshold(),
            config.UPGRADE_RED_ICON_X_MIN,
            config.UPGRADE_RED_ICON_X_MAX,
            config.UPGRADE_RED_ICON_Y_MIN,
            config.UPGRADE_RED_ICON_Y_MAX,
            min_distance=80,
        )

        if best_stats_match is None:
            self.vision_optimizer.update_stats_upgrade_miss()
            logger.info("No stats icon detected")
            return State.SCROLL

        best_stats_confidence, _, _ = best_stats_match
        self.vision_optimizer.update_stats_upgrade_confidence(best_stats_confidence)
        self.cycle_counter = 0
        logger.info("Stats icon found, upgrading")
        opened = self.mouse_controller.click(
            config.STATS_UPGRADE_BUTTON_POS[0],
            config.STATS_UPGRADE_BUTTON_POS[1],
            relative=True,
        )
        if not opened:
            return State.OPEN_BOXES

        self._sleep(config.STATE_DELAY)
        clicked = self.mouse_controller.spam_click_at(
            config.STATS_UPGRADE_POS[0],
            config.STATS_UPGRADE_POS[1],
            duration=config.STATS_UPGRADE_CLICK_DURATION,
            click_delay=config.STATS_UPGRADE_CLICK_DELAY,
            relative=True,
            interrupt_check=self._stop_requested.is_set,
        )
        if not clicked:
            logger.warning("Stats upgrade spam-click failed at %s", config.STATS_UPGRADE_POS)
            return State.OPEN_BOXES

        self._click_idle()
        logger.info("Stats upgrade completed")
        return State.OPEN_BOXES
