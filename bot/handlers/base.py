"""Base class for all tactical bot handlers."""

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

from bot.state_machine import State
from core.logger import setup_logger

if TYPE_CHECKING:
    from bot.orchestrator import EatventureBot

logger = setup_logger("bot.handlers.base")

class BaseHandler(ABC):
    """
    Abstract Tactical Handler.
    Each discrete state in the FSM is encapsulated within a subclass of BaseHandler.
    """
    def __init__(self, bot: 'EatventureBot'):
        self.bot = bot
        self.mouse = bot.mouse
        self.capture = bot.capture
        self.matcher = bot.matcher

    @abstractmethod
    def handle(self, current_state: State) -> State:
        """Executes the specific handler logic for this state."""
        pass

    def check_interrupts(self) -> None:
        """
        Delegates to the orchestrator to check for global interrupts.
        Ensures handlers can be safely halted by high-priority state changes.
        """
        self.bot.check_interrupts()
