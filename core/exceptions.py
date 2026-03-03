"""Core custom exceptions for the Eatventure automation bot."""

class BotBaseException(Exception):
    """Base exception for all bot-related errors."""
    pass

class VisionError(BotBaseException):
    """Raised when an unexpected vision or OpenCV error occurs."""
    pass

class ScreenCaptureError(VisionError):
    """Raised when the screen cannot be captured safely."""
    pass

class TargetOutOfBoundsError(BotBaseException):
    """Raised when a click target is outside the safe client boundary."""
    pass

class StateTransitionError(BotBaseException):
    """Raised when the FSM encounters an illegal transition."""
    pass

class LevelCompleteInterrupt(BotBaseException):
    """Raised when a new level is detected to immediately halt standard gameplay."""
    pass

class BotStoppedInterrupt(BotBaseException):
    """Raised when the bot is intentionally stopped by the user."""
    pass
