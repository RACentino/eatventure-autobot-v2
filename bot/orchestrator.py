"""The core bot orchestrator. Manages high-level bot execution and lifecycle."""

import logging
import time
import threading
from typing import Optional, Dict, List, Any, Type, Callable

from core import config
from core.exceptions import LevelCompleteInterrupt, BotStoppedInterrupt
from core.logger import setup_logger
from vision.capture import WindowCapture
from vision.matcher import ImageMatcher
from vision.scanner import AssetScanner
from interaction.mouse import MouseController
from bot.state_machine import StateMachine, State
from bot.searcher import OscillatingSearcher
from bot.optimization import AdaptiveTuner, VisionOptimizer, VisionPersistence, HistoricalLearner
from core.notifications import TelegramNotifier

logger = setup_logger("bot.orchestrator")

class EatventureBot:
    """
    Principal Orchestrator.
    Manages centralized resources (Vision, Mouse, FSM) and delegates specific
    logic to discrete Tactical Handlers.
    """
    def __init__(self):
        logger.info("Initializing Orchestrator (Eatventure Bot Refactored)...")
        
        # 1. Core Services Initialization
        self.capture = WindowCapture(config.WINDOW_TITLE, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        self.capture.set_cache_ttl(config.CAPTURE_CACHE_TTL)
        self.matcher = ImageMatcher(config.MATCH_THRESHOLD)
        self.mouse = MouseController(self.capture.hwnd)
        self.state_machine = StateMachine(State.FIND_RED_ICONS)
        self.scanner = AssetScanner(self.matcher)
        self.searcher = OscillatingSearcher(self)
        
        # 2. Optimization & Persistence
        self.tuner = AdaptiveTuner()
        self.vision_persistence = VisionPersistence(config.AI_VISION_STATE_FILE, config.AI_VISION_SAVE_INTERVAL)
        self.vision_optimizer = VisionOptimizer(self.vision_persistence)
        self.vision_optimizer.apply_persisted_state(self.vision_persistence.load())
        self.learning_persistence = VisionPersistence(config.AI_LEARNING_STATE_FILE, config.AI_LEARNING_SAVE_INTERVAL)
        self.historical_learner = HistoricalLearner(self, self.learning_persistence)
        
        # 3. Communication
        self.telegram = TelegramNotifier(
            config.TELEGRAM_BOT_TOKEN, 
            config.TELEGRAM_CHAT_ID, 
            config.TELEGRAM_ENABLED
        )
        
        # 4. Asset Management
        self.templates: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
        self.load_assets()
        
        # Guard Injection: Inject global interrupt hook into mouse layer
        self.mouse.interrupt_callback = self.check_interrupts_status

        # 5. State & Data Persistence
        self.running = False
        self.scroll_offset_units = 0.0
        self._interrupt_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._handlers: Dict[str, Any] = {}
        
        self.register_tactical_handlers()
        self.state_machine.set_priority_resolver(self.resolve_priority_state)

        logger.info("Orchestrator successfully initialized.")

    def load_assets(self) -> None:
        """Loads all required game templates into memory."""
        self.templates = self.scanner.scan(config.ASSETS_DIR, required_templates=config.REQUIRED_TEMPLATES)
        logger.info(f"Loaded {len(self.templates)} templates.")

    def _monitor_new_level(self) -> None:
        """Background thread that scans for level transition triggers while the main thread is busy."""
        logger.info("New Level Monitor Thread started.")
        while self.running:
            try:
                # 1. Yield priority if main thread is in critical interaction
                # We check a 'quiet' period if necessary, or just sleep
                time.sleep(config.NEW_LEVEL_MONITOR_INTERVAL)

                # 2. Check for new level icon
                screenshot = self.capture.capture(max_y=config.WINDOW_HEIGHT)
                if "newLevel" in self.templates:
                    template, mask = self.templates["newLevel"]
                    found, conf, x, y = self.matcher.find_template(
                        screenshot, 
                        template, 
                        mask=mask, 
                        threshold=config.NEW_LEVEL_THRESHOLD
                    )
                    
                    if found:
                        # Ensure it's not in a forbidden zone before interrupting
                        if not self.mouse.is_in_forbidden_zone(x, y):
                            logger.info(f"Monitor: New level detected (conf: {conf:.2f}). Triggering interrupt.")
                            self._interrupt_event.set()
                            # Once set, we wait for main thread to clear it or for bot to stop
                            while self.running and self._interrupt_event.is_set():
                                time.sleep(0.1)

            except Exception as exc:
                logger.error(f"Monitor thread encountered an error: {exc}")
                time.sleep(1.0)
        logger.info("New Level Monitor Thread stopped.")

    def iter_red_icon_templates(self):
        """Generator for iterating through loaded red icon templates."""
        for name in config.RED_ICON_TEMPLATES:
            if name in self.templates:
                yield name, self.templates[name][0], self.templates[name][1]

    def get_runtime_behavior_snapshot(self) -> Dict[str, float]:
        """Returns current EMA-tuned timing parameters for historical learning."""
        return {
            "click_delay": float(self.tuner.click_delay),
            "move_delay": float(self.tuner.move_delay),
            "upgrade_click_interval": float(self.tuner.upgrade_click_interval),
            "search_interval": float(self.tuner.search_interval),
        }

    def apply_learned_behavior(self, learned: Dict[str, Any], reason: str = "historical", best_time: float = 0.0) -> None:
        """Applies timing profile optimizations derived from historical performance."""
        if not learned:
            return
        self.tuner.click_delay = float(learned.get("click_delay", self.tuner.click_delay))
        self.tuner.move_delay = float(learned.get("move_delay", self.tuner.move_delay))
        self.tuner.upgrade_click_interval = float(learned.get("upgrade_click_interval", self.tuner.upgrade_click_interval))
        self.tuner.search_interval = float(learned.get("search_interval", self.tuner.search_interval))
        logger.info(f"Optimization applied ({reason}): Timings adjusted based on {best_time:.2f}s run.")

    def check_priority_targets(self) -> Optional[State]:
        """STEP A: Priority Scan. Checks for Red Icons and Level Transitions."""
        return self.resolve_priority_state(self.state_machine.current_state)

    def check_main_success(self) -> Optional[State]:
        """STEP B: Main Target Scan. Reserved for specific success conditions."""
        return None

    def scan_and_click_boxes(self) -> bool:
        """Consolidated box opening scan. Returns True if any box was clicked."""
        screenshot = self.capture.capture(max_y=config.MAX_SEARCH_Y)
        found_any = False
        
        for i in range(1, 6):
            name = f"box{i}"
            if name in self.templates:
                template, mask = self.templates[name]
                found, conf, x, y = self.matcher.find_template(screenshot, template, mask=mask, threshold=config.BOX_THRESHOLD)
                if found and not self.mouse.is_in_forbidden_zone(x, y):
                    logger.info(f"Consolidated Scan: Found {name}, clicking.")
                    if self.mouse.click(x, y):
                        found_any = True
                        # No sleep here, let the caller handle pacing
        return found_any

    def scan_and_click_upgrade_station(self) -> bool:
        """Consolidated upgrade station scan and click. Returns True if station was clicked."""
        pos = self.scan_for_upgrade_station()
        if pos:
            logger.info("Consolidated Scan: Found upgrade station, clicking.")
            return self.mouse.click(pos[0], pos[1])
        return False

    def scan_for_upgrade_station(self) -> Optional[Tuple[int, int]]:
        """Consolidated upgrade station detection. Returns (x, y) if found and safe."""
        screenshot = self.capture.capture(max_y=config.MAX_SEARCH_Y)
        if "upgradeStation" in self.templates:
            template, mask = self.templates["upgradeStation"]
            found, conf, x, y = self.matcher.find_template(
                screenshot, template, mask=mask, threshold=config.UPGRADE_STATION_THRESHOLD
            )
            if found and not self.mouse.is_in_forbidden_zone(x, y):
                return (x, y)
        return None

    def check_fallbacks(self) -> Optional[State]:
        """STEP C: Fallback Scan. Clicks boxes and stations."""
        # 1. Boxes
        if self.scan_and_click_boxes():
            return State.FIND_RED_ICONS
        
        # 2. Upgrade Station
        if self.scan_and_click_upgrade_station():
            return State.FIND_RED_ICONS
                    
        return None

    def check_intra_scroll_red_interrupt(self) -> Optional[State]:
        """High-frequency scan for red icons during active scrolls."""
        handler = self._handlers.get("icon")
        if handler:
            # We reuse the icon handler's logic but in a non-state-changing way if possible
            # Or just let it return State.CLICK_RED_ICON
            return handler._handle_find_red_icons(interrupt=True)
        return None

    def sleep(self, duration: float) -> None:
        """Centralized sleep that is aware of high-priority interrupts."""
        start = time.monotonic()
        while time.monotonic() - start < duration:
            self.check_interrupts()
            time.sleep(min(0.01, duration - (time.monotonic() - start)))

    def register_tactical_handlers(self) -> None:
        """Instantiates and registers all discrete state handlers."""
        from bot.handlers.icon_handler import IconHandler
        from bot.handlers.upgrade_handler import UpgradeHandler
        from bot.handlers.transition_handler import TransitionHandler
        from bot.handlers.scroll_handler import ScrollHandler
        from bot.handlers.misc_handler import MiscHandler
        
        # Instantiate Handlers
        self._handlers = {
            "icon": IconHandler(self),
            "upgrade": UpgradeHandler(self),
            "transition": TransitionHandler(self),
            "scroll": ScrollHandler(self),
            "misc": MiscHandler(self)
        }

        # Register Handlers to FSM
        self.state_machine.register_handler(State.FIND_RED_ICONS, self._handlers["icon"].handle)
        self.state_machine.register_handler(State.CLICK_RED_ICON, self._handlers["icon"].handle)
        self.state_machine.register_handler(State.SEARCH_UPGRADE_STATION, self._handlers["upgrade"].handle)
        self.state_machine.register_handler(State.HOLD_UPGRADE_STATION, self._handlers["upgrade"].handle)
        self.state_machine.register_handler(State.CHECK_NEW_LEVEL, self._handlers["transition"].handle)
        self.state_machine.register_handler(State.TRANSITION_LEVEL, self._handlers["transition"].handle)
        self.state_machine.register_handler(State.SCROLL, self._handlers["scroll"].handle)
        self.state_machine.register_handler(State.OPEN_BOXES, self._handlers["misc"].handle)
        self.state_machine.register_handler(State.UPGRADE_STATS, self._handlers["misc"].handle)
        self.state_machine.register_handler(State.CHECK_UNLOCK, self._handlers["misc"].handle)
        self.state_machine.register_handler(State.WAIT_FOR_UNLOCK, self._handlers["misc"].handle)
        self.state_machine.register_handler(State.RECOVERY, self._handlers["misc"].handle)

    def check_interrupts_status(self) -> bool:
        """
        Returns True if a critical interrupt is pending.
        Used by the action layer (MouseController) to refuse dispatch.
        """
        if not self.running:
            return True
        if self._interrupt_event.is_set():
            return True
        return False

    def check_interrupts(self) -> None:
        """
        The Global Safety Check.
        Raises an exception if a critical interrupt (e.g. New Level) is pending.
        """
        if not self.running:
            raise BotStoppedInterrupt("Execution stopped by user.")
        if self._interrupt_event.is_set():
            logger.info("Critical Interrupt: New level detected. Halting tactical handler.")
            self._interrupt_event.clear()
            raise LevelCompleteInterrupt("New level reached.")

    def start(self) -> None:
        """Starts the main bot execution loop."""
        if not self.running:
            logger.info("Bot execution started.")
            self.running = True
            self._interrupt_event.clear()
            
            # Start Monitor Thread
            self._monitor_thread = threading.Thread(target=self._monitor_new_level, daemon=True)
            self._monitor_thread.start()

    def stop(self) -> None:
        """Gracefully halts the bot execution."""
        if self.running:
            logger.info("Bot execution halted.")
            self.running = False
            # We don't need to join the monitor thread specifically since it's daemon 
            # and we set self.running = False, but we wait a moment.
            if self._monitor_thread:
                self._monitor_thread.join(timeout=2.0)
                self._monitor_thread = None

    def step(self) -> None:
        """Executes a single FSM update cycle."""
        if self.running:
            try:
                self.state_machine.update()
            except LevelCompleteInterrupt:
                logger.info("Interrupt caught in step loop. Transitioning to transition sequence.")
                self.state_machine.transition(State.TRANSITION_LEVEL)
            except BotStoppedInterrupt:
                logger.info("Bot stopped by interrupt.")
                self.stop()
            except Exception as exc:
                logger.error(f"Critical error in step: {exc}", exc_info=True)
                self.stop()

    def resolve_priority_state(self, current_state: State) -> Optional[State]:
        """
        Priority Guard Layer.
        Checks for high-priority transitions (e.g. New Level detection) 
        before the FSM executes normal state handlers.
        """
        if current_state in (State.CHECK_NEW_LEVEL, State.TRANSITION_LEVEL):
            return None

        # 1. New Level Check (Vision-Based)
        # Scan for the new level icon in the background
        screenshot = self.capture.capture(max_y=config.WINDOW_HEIGHT)
        
        if "newLevel" in self.templates:
            template, mask = self.templates["newLevel"]
            found, conf, x, y = self.matcher.find_template(
                screenshot, 
                template, 
                mask=mask, 
                threshold=config.NEW_LEVEL_THRESHOLD,
                template_name="NewLevel"
            )
            
            if found:
                logger.info(f"Priority interrupt: New level detected (conf: {conf:.2f})!")
                return State.TRANSITION_LEVEL

        return None
