import sys
import threading

from eatventure_autobot.domain.errors import CaptureError
from eatventure_autobot.domain.protocols import ScreenCapture


def create_screen_capture(
    title: str,
    target_width: int,
    target_height: int,
    stop_event: threading.Event | None = None,
) -> ScreenCapture:
    if sys.platform.startswith("linux"):
        from eatventure_autobot.capture.linux import LinuxWindowCapture

        return LinuxWindowCapture(title, target_width, target_height, stop_event=stop_event)
    if sys.platform == "win32":
        from eatventure_autobot.capture.windows import WindowsWindowCapture

        return WindowsWindowCapture(title, target_width, target_height, stop_event=stop_event)

    raise CaptureError(f"Unsupported platform: {sys.platform}")
