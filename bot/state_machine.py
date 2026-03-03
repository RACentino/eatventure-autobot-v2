"""Stateless Finite State Machine (FSM) engine."""

import logging
import time
from enum import Enum, auto
from typing import Callable, Dict, Optional, Any, Tuple

from core import config
from core.logger import setup_logger

logger = setup_logger("bot.state_machine")

class State(Enum):
    """Enumeration of all legal bot states."""
    FIND_RED_ICONS = auto()
    CLICK_RED_ICON = auto()
    CHECK_UNLOCK = auto()
    SEARCH_UPGRADE_STATION = auto()
    HOLD_UPGRADE_STATION = auto()
    OPEN_BOXES = auto()
    UPGRADE_STATS = auto()
    SCROLL = auto()
    CHECK_NEW_LEVEL = auto()
    TRANSITION_LEVEL = auto()
    WAIT_FOR_UNLOCK = auto()
    RECOVERY = auto() # New recovery state for error handling

class StateMachine:
    """
    Pure, stateless FSM engine.
    Orchestrates transitions between states based on handler return values.
    """
    def __init__(self, initial_state: State = State.FIND_RED_ICONS):
        self.current_state = initial_state
        self.previous_state: Optional[State] = None
        self._handlers: Dict[State, Callable[[State], State]] = {}
        self._priority_resolver: Optional[Callable[[State], Optional[State]]] = None
        
        logger.info(f"State machine initialized in: {initial_state.name}")

    def register_handler(self, state: State, handler: Callable[[State], State]) -> None:
        """Registers a handler function for a specific state."""
        self._handlers[state] = handler
        logger.debug(f"Registered handler for state: {state.name}")

    def set_priority_resolver(self, resolver: Callable[[State], Optional[State]]) -> None:
        """Registers a priority resolver to check for interrupts before handler execution."""
        self._priority_resolver = resolver
        logger.debug("Priority resolver registered.")

    def transition(self, new_state: State) -> None:
        """Performs a transition to a new state if it differs from the current one."""
        if new_state != self.current_state:
            logger.info(f"Transition: {self.current_state.name} -> {new_state.name}")
            self.previous_state = self.current_state
            self.current_state = new_state

    def update(self) -> bool:
        """
        Executes a single update cycle.
        1. Checks for priority overrides (e.g. New Level).
        2. Executes the current state's handler.
        3. Transitions to the next state returned by the handler.
        """
        # 1. Check for priority interrupts (The 'Guard' Layer)
        if self._priority_resolver:
            try:
                priority_state = self._priority_resolver(self.current_state)
                if priority_state and isinstance(priority_state, State):
                    self.transition(priority_state)
            except Exception as exc:
                logger.error(f"Priority resolver failed: {exc}", exc_info=True)

        # 2. Execute State Handler
        handler = self._handlers.get(self.current_state)
        if not handler:
            logger.warning(f"No handler registered for state: {self.current_state.name}")
            return False

        try:
            next_state = handler(self.current_state)
            
            # Enforce execution pacing based on configuration
            tick_delay = config.FSM_TICK_DELAY
            if tick_delay > 0:
                time.sleep(tick_delay)

            if next_state and isinstance(next_state, State):
                self.transition(next_state)
            return True
        except (LevelCompleteInterrupt, BotStoppedInterrupt):
            # Propagate critical interrupts to the orchestrator/main loop
            raise
        except Exception as exc:
            logger.error(f"Handler for state {self.current_state.name} failed: {exc}", exc_info=True)
            # Default to recovery state on unhandled handler errors
            self.transition(State.RECOVERY)
            return False

    def get_state_name(self) -> str:
        """Returns the human-readable name of the current state."""
        return self.current_state.name
