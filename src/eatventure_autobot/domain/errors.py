class AutobotError(RuntimeError):
    """Base for every error raised by the bot's own code."""


class ConfigError(AutobotError):
    """Raised when configuration values fail validation."""


class CaptureError(AutobotError):
    """Raised when the screen-capture backend fails."""


class WindowNotAvailableError(CaptureError):
    """Raised when the target window cannot be found or has become invalid."""


class InputError(AutobotError):
    """Raised when the input backend fails to move the cursor or send a click."""


class DetectionError(AutobotError):
    """Raised when template loading or matching fails outside of a normal no-match result."""
