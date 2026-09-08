import threading

from eatventure_autobot.domain.config import InputTimingConfig
from eatventure_autobot.domain.protocols import InputController, ScreenCapture
from eatventure_autobot.domain.types import Zone
from eatventure_autobot.input.pynput_controller import PynputInputController


def create_input_controller(
    screen_capture: ScreenCapture,
    timing: InputTimingConfig,
    static_zones: tuple[Zone, ...] = (),
    stop_event: threading.Event | None = None,
) -> InputController:
    return PynputInputController(screen_capture, timing, static_zones, stop_event=stop_event)
