"""Best-effort desktop popup through notify-send. Where the tool is missing (or fails) it does
nothing: the self-stop is already in bot.log and, if configured, in Telegram."""

import contextlib
import shutil
import subprocess


def notify_desktop(title: str, body: str) -> None:
    executable = shutil.which("notify-send")
    if executable is None:
        return
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        subprocess.run([executable, "-u", "critical", title, body], timeout=3, check=False)
