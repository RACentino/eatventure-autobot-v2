"""Revive a dead game over adb. The runner is injectable so tests never touch a phone; the real
one runs adb without a shell, with a timeout, and never raises."""

import logging
import subprocess
from collections.abc import Callable, Sequence

from eatventure_autobot.domain.config import GameRecoveryConfig

logger = logging.getLogger(__name__)

Runner = Callable[[Sequence[str], float], tuple[int, str]]


def run_adb(argv: Sequence[str], timeout: float) -> tuple[int, str]:
    try:
        done = subprocess.run(
            list(argv), capture_output=True, text=True, timeout=timeout, check=False
        )  # fixed argv list, never a shell
    except (OSError, subprocess.SubprocessError) as exc:  # missing adb, timeout, no device
        return 127, str(exc)
    return done.returncode, done.stdout + done.stderr


class GameLauncher:
    def __init__(self, config: GameRecoveryConfig, runner: Runner = run_adb) -> None:
        self._config = config
        self._run = runner
        self.package: str | None = None

    def detect_package(self) -> bool:
        """Pick the one installed package matching package_hint. Ambiguity disables relaunch."""
        code, output = self._run(
            [self._config.adb_path, "shell", "pm", "list", "packages"], self._config.command_timeout
        )
        if code != 0:
            logger.error(
                "Game relaunch unavailable: 'adb shell pm list packages' failed: %s",
                output.strip(),
            )
            return False
        hint = self._config.package_hint.lower()
        found = sorted(
            {
                line.removeprefix("package:").strip()
                for line in output.splitlines()
                if line.startswith("package:") and hint in line.lower()
            }
        )
        if len(found) != 1:
            logger.error(
                "Game relaunch disabled: %d packages match '%s': %s",
                len(found),
                self._config.package_hint,
                ", ".join(found) or "none",
            )
            return False
        self.package = found[0]
        logger.info("Game relaunch armed for package %s", self.package)
        return True

    def relaunch(self) -> bool:
        """Force-stop then launch. A frozen app needs the stop; a crashed one ignores it."""
        if self.package is None:
            return False
        adb, timeout = self._config.adb_path, self._config.command_timeout
        steps = (
            [adb, "shell", "am", "force-stop", self.package],
            [adb, "shell", "monkey", "-p", self.package]
            + ["-c", "android.intent.category.LAUNCHER", "1"],
        )
        for argv in steps:
            code, output = self._run(argv, timeout)
            if code != 0:
                logger.error(
                    "Game relaunch step failed (%s): %s", " ".join(argv[1:]), output.strip()
                )
                return False
        return True
