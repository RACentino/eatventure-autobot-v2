import logging
import queue
import threading
from collections import deque
from datetime import datetime
from typing import Any

from core import config
from vision.matcher import ImageMatcher
from interaction.mouse import MouseController, precise_sleep, wait_event
from bot.handlers.base import StateRegistrationMixin
from bot.handlers.icon_handler import IconHandlerMixin
from bot.handlers.misc_handler import MiscHandlerMixin
from bot.handlers.scroll_handler import ScrollHandlerMixin
from bot.handlers.transition_handler import TransitionHandlerMixin
from bot.handlers.upgrade_handler import UpgradeHandlerMixin
from bot.optimization import AdaptiveTuner, HistoricalLearner, VisionOptimizer, VisionPersistence
from bot.state_machine import State, StateMachine
from core.notifications import TelegramNotifier
from vision.capture import (
    ForbiddenAreaOverlay,
    WindowCapture,
    WindowCaptureError,
    WindowNotAvailableError,
)
from vision.scanner import VisionScannerMixin

logger = logging.getLogger(__name__)
PENDING_LEARNED_BEHAVIOR_QUEUE_MAXSIZE = 16
PENDING_LEARNED_BEHAVIOR_DRAIN_LIMIT = 64
BOT_RUN_LOOP_ITERATION_LIMIT = 2_147_483_647











class EatventureBot(
    StateRegistrationMixin,
    IconHandlerMixin,
    UpgradeHandlerMixin,
    MiscHandlerMixin,
    ScrollHandlerMixin,
    TransitionHandlerMixin,
    VisionScannerMixin,
):
    def __init__(self) -> None:
        logger.info("Initializing Eatventure Bot")
        self._stop_requested = threading.Event()
        self._step_active = threading.Event()
        self._pending_learned_behaviors: queue.Queue[dict[str, float]] = queue.Queue(
            maxsize=PENDING_LEARNED_BEHAVIOR_QUEUE_MAXSIZE
        )
        self._applied_runtime_behavior: tuple[float, float, float] | None = None

        self._initialize_runtime_services()
        self._initialize_learning_services()
        self._initialize_state_handlers_and_templates()
        self._initialize_runtime_state()

        logger.info("Bot initialized successfully")

    def _initialize_runtime_services(self) -> None:
        self.window_capture = WindowCapture(config.WINDOW_TITLE, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        self.image_matcher = ImageMatcher(config.MATCH_THRESHOLD)
        self.mouse_controller = MouseController(
            self.window_capture.get_window_rect,
            config.CLICK_DELAY,
            config.MOUSE_MOVE_DELAY,
        )
        self.state_machine = StateMachine(State.FIND_RED_ICONS)
        self.telegram = TelegramNotifier(
            config.TELEGRAM_BOT_TOKEN,
            config.TELEGRAM_CHAT_ID,
            config.TELEGRAM_ENABLED,
        )

    def _initialize_learning_services(self) -> None:
        self.tuner = AdaptiveTuner()
        self._runtime_behavior_snapshot = self._runtime_behavior_from_tuner()
        self.vision_persistence = VisionPersistence(config.AI_VISION_STATE_FILE, config.AI_VISION_SAVE_INTERVAL)
        self.vision_optimizer = VisionOptimizer(self.vision_persistence)
        self.vision_optimizer.apply_persisted_state(self.vision_persistence.load())
        self.learning_persistence = VisionPersistence(
            config.AI_LEARNING_STATE_FILE,
            config.AI_LEARNING_SAVE_INTERVAL,
        )
        self.historical_learner = HistoricalLearner(self, self.learning_persistence)
        self._apply_pending_learned_behavior_updates()

    def _initialize_state_handlers_and_templates(self) -> None:
        self.register_states()
        self.templates = self.load_templates()
        self._red_icon_template_names_cache = self._red_icon_template_names()
        self._box_template_names_cache = self._box_template_names()
        self._red_icon_max_width, self._red_icon_max_height = self._red_icon_template_span()
        self.ready = self._validate_required_templates()

    def _initialize_runtime_state(self) -> None:
        self._successful_red_icon_history_limit = 24
        self.running = False
        self.red_icons = []
        self.current_red_icon_index = 0
        self.wait_for_unlock_attempts = 0
        self.max_wait_for_unlock_attempts = 4
        self.work_done = False
        self.cycle_counter = 0
        self.upgrade_station_counter = 0
        self.successful_red_icon_positions = deque(maxlen=self._successful_red_icon_history_limit)
        self.upgrade_found_in_cycle = False
        self.consecutive_failed_cycles = 0
        self.total_levels_completed = 0
        self.current_level_start_time = None
        self.upgrade_station_pos = None
        self.overlay = None
        self._oscillation_cycle_index = 1
        self._oscillation_leg_direction = 1
        self._oscillation_leg_progress = 0
        self._new_level_red_icon_verified = False
        self.forbidden_zones = self._configured_forbidden_zones()

    @staticmethod
    def _configured_forbidden_zones() -> list[tuple[int, int, int, int]]:
        return config.numbered_forbidden_zone_bounds()

    def _sleep(self, duration: Any) -> bool:
        return wait_event(self._stop_requested, duration)

    def _apply_tuning(self) -> None:
        runtime_behavior = (
            float(self.tuner.click_delay),
            float(self.tuner.move_delay),
            float(self.tuner.search_interval),
        )
        if runtime_behavior == self._applied_runtime_behavior:
            return
        click_delay, move_delay, _ = runtime_behavior
        self.mouse_controller.click_delay = click_delay
        self.mouse_controller.move_delay = move_delay
        self._applied_runtime_behavior = runtime_behavior
        self._runtime_behavior_snapshot = self._runtime_behavior_from_tuner()

    def _runtime_behavior_from_tuner(self) -> dict[str, float]:
        return {
            "click_delay": float(self.tuner.click_delay),
            "move_delay": float(self.tuner.move_delay),
            "search_interval": float(self.tuner.search_interval),
        }

    def _next_pending_learned_behavior(self) -> dict[str, float] | None:
        latest_behavior = None
        for _ in range(PENDING_LEARNED_BEHAVIOR_DRAIN_LIMIT):
            try:
                latest_behavior = self._pending_learned_behaviors.get_nowait()
            except queue.Empty:
                return latest_behavior
            self._pending_learned_behaviors.task_done()
        logger.debug(
            "Deferred learned-behavior drain after %s updates",
            PENDING_LEARNED_BEHAVIOR_DRAIN_LIMIT,
        )
        return latest_behavior

    def _discard_pending_learned_behavior_updates(self) -> None:
        for _ in range(PENDING_LEARNED_BEHAVIOR_DRAIN_LIMIT):
            try:
                self._pending_learned_behaviors.get_nowait()
            except queue.Empty:
                return
            self._pending_learned_behaviors.task_done()
        logger.debug(
            "Stopped learned-behavior discard after %s updates",
            PENDING_LEARNED_BEHAVIOR_DRAIN_LIMIT,
        )

    def _apply_pending_learned_behavior_updates(self) -> None:
        learned_behavior = self._next_pending_learned_behavior()
        if learned_behavior is None:
            return
        self.tuner.apply_runtime_behavior(learned_behavior)
        self._apply_tuning()


    def _click_idle(self) -> bool:
        idle_x, idle_y = config.IDLE_CLICK_POS
        return self.mouse_controller.click(idle_x, idle_y, relative=True)






    def _scrcpy_miss_recovery_sleep(self, duration: Any) -> bool:
        if not bool(config.SCRCPY_MISS_RECOVERY_ENABLED):
            return False
        try:
            delay = max(0.0, float(duration))
        except (TypeError, ValueError):
            delay = 0.0
        if delay <= 0:
            return True
        return self._sleep(delay)



    def get_runtime_behavior_snapshot(self) -> dict[str, float]:
        return dict(self._runtime_behavior_snapshot)

    def apply_learned_behavior(self, learned: dict[str, Any]) -> None:
        if hasattr(self, "historical_learner"):
            learned = self.historical_learner._sanitize_behavior(learned)
        if not learned:
            return
        learned_behavior = dict(learned)
        try:
            self._pending_learned_behaviors.put_nowait(learned_behavior)
            return
        except queue.Full:
            self._discard_pending_learned_behavior_updates()
        try:
            self._pending_learned_behaviors.put_nowait(learned_behavior)
        except queue.Full:
            logger.warning("Dropping learned behavior update because the queue is full")

    def wipe_memory(self) -> None:
        self._discard_pending_learned_behavior_updates()
        self.tuner.reset()
        self.vision_optimizer.reset()
        self.historical_learner.reset()
        self.successful_red_icon_positions = deque(maxlen=self._successful_red_icon_history_limit)
        self.current_level_start_time = datetime.now() if self.running else None
        self._apply_tuning()
        self._discard_pending_learned_behavior_updates()






































    def request_stop(self) -> None:
        self._stop_requested.set()

    def start(self) -> bool:
        if self.running:
            return True
        if self._step_active.is_set():
            logger.warning("Cannot start bot while a previous state step is still stopping")
            return False
        if not self.ready:
            logger.error("Cannot start bot because required templates are missing")
            return False
        try:
            self.window_capture.ensure_window(resize=True)
        except WindowCaptureError as exc:
            logger.error("Cannot start bot: %s", exc)
            self.running = False
            return False
        self._stop_requested.clear()
        self.running = True
        if self.current_level_start_time is None:
            self.current_level_start_time = datetime.now()
        self.historical_learner.start()
        if config.SHOW_FORBIDDEN_AREA and self.overlay is None:
            self.overlay = ForbiddenAreaOverlay(self.window_capture, self.forbidden_zones)
            self.overlay.start()
        return True

    def stop(self) -> None:
        self._stop_requested.set()
        if not self.running and self.overlay is None:
            self.historical_learner.stop()
            return
        self.running = False
        self.historical_learner.stop()
        if self.overlay is not None:
            self.overlay.stop()
            self.overlay = None

    def step(self) -> bool:
        if self._step_active.is_set():
            logger.warning("Ignoring reentrant bot step")
            return False
        self._step_active.set()
        try:
            if self._stop_requested.is_set():
                self.stop()
                return False
            if not self.window_capture.is_window_active():
                logger.error("Window '%s' is not available", config.WINDOW_TITLE)
                self.stop()
                return False
            self._apply_pending_learned_behavior_updates()
            self._apply_tuning()
            updated = bool(self.state_machine.update())
            if not updated:
                logger.error(
                    "State machine update failed in state %s; stopping bot",
                    self.state_machine.get_state_name(),
                )
                self.stop()
                return False
            return True
        except (WindowNotAvailableError, WindowCaptureError) as exc:
            logger.error("Stopping bot: %s", exc)
            self.stop()
            return False
        except Exception:
            logger.exception("Stopping bot due to unexpected state-handler failure")
            self.stop()
            return False
        finally:
            self._step_active.clear()

    def run(self) -> None:
        if not self.start():
            return
        try:
            for _ in range(BOT_RUN_LOOP_ITERATION_LIMIT):
                if not self.running:
                    return
                if not self.window_capture.is_window_active():
                    logger.error("Window '%s' is no longer active", config.WINDOW_TITLE)
                    break
                self.step()
                precise_sleep(0.1)
            else:
                logger.error("Bot run loop reached iteration limit")
        finally:
            self.stop()
