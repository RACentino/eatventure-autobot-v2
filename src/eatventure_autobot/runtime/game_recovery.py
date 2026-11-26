"""Manage the selected Android device, scrcpy mirror, and launch of a closed game."""

import logging
import re
import subprocess
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from eatventure_autobot.domain.config import GameRecoveryConfig
from eatventure_autobot.domain.errors import CaptureError

logger = logging.getLogger(__name__)

Runner = Callable[[Sequence[str], float], tuple[int, str]]

# `adb shell` joins its arguments and hands them to the device's own shell, so a package name read
# back from the device has to be shell-inert before it is passed on. This is Android's own rule for
# package names (letter-led dotted segments of letters, digits and underscores).
_PACKAGE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*")
_PROC_ROOT = Path("/proc")


def _primary_window_focus(dump: str) -> str | None:
    """Read display 0 only when every secondary display explicitly has no focused window."""
    display: str | None = None  # Older dumps have a single, unlabelled focus.
    focus: dict[str | None, str | None] = {}
    for raw in dump.splitlines():
        line = raw.strip()
        if line.startswith("Display:"):
            header = re.fullmatch(
                r"Display:\s+mDisplayId=(0|[1-9][0-9]*)(?:\s+\(organized\))?", line,
            )
            if header is None or header[1] in focus:
                return None
            display = header[1]
            focus[display] = None
        elif line.startswith("mCurrentFocus="):
            if focus.get(display) is not None:
                return None
            focus[display] = line.removeprefix("mCurrentFocus=")

    if None in focus:
        return focus[None] if len(focus) == 1 else None
    if any(value != "null" for key, value in focus.items() if key != "0"):
        return None
    return focus.get("0")


def _scrcpy_device(process_id: int) -> str | None:
    """Read the live adb server child's serial, including children of scrcpy's worker threads."""
    process = _PROC_ROOT / str(process_id)
    try:
        if process_id <= 0 or (process / "exe").resolve(strict=True).name != "scrcpy":
            return None
        children = {child for task in (process / "task").glob("*/children")
                    for child in task.read_text().split() if child.isdecimal()}
    except OSError:
        return None
    devices = set()
    for child in children:
        try:
            child_path = _PROC_ROOT / child
            if (child_path / "exe").resolve(strict=True).name != "adb":
                continue
            arguments = (child_path / "cmdline").read_bytes().decode(
                errors="surrogateescape").rstrip("\0").split("\0")
        except OSError:
            continue
        if (len(arguments) >= 8 and arguments[1] == "-s" and arguments[3] == "shell"
                and "app_process" in arguments and "com.genymobile.scrcpy.Server" in arguments
                and not {"control=false", "video=false", "video_source=camera"} & set(arguments)):
            devices.add(arguments[2])
    return next(iter(devices)) if len(devices) == 1 else None


@dataclass(frozen=True)
class DeviceIdentity:
    serial: str
    transport_id: str


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
        self.serial: str | None = config.device_serial
        self.identity: DeviceIdentity | None = None
        self._mirror: subprocess.Popen[bytes] | None = None
        self._mirror_identity: DeviceIdentity | None = None
        self._external_mirror_pid: int | None = None
        self._external_mirror_identity: DeviceIdentity | None = None

    def mirror_bound(self) -> bool:
        if self._external_mirror_pid is not None:
            return (self.identity is not None and self._external_mirror_identity == self.identity
                    and _scrcpy_device(self._external_mirror_pid) == self.identity.serial)
        return (self._mirror is not None and self._mirror.poll() is None
                and self._mirror_identity == self.identity)

    @property
    def mirror_pid(self) -> int | None:
        if self._external_mirror_pid is not None:
            return self._external_mirror_pid
        return self._mirror.pid if self._mirror is not None else None

    def bind_existing_mirror(self, windows: list[tuple[int, bool]]) -> int | None:
        if self.identity is None:
            return None
        owned_pid = (self._mirror.pid if self._mirror is not None
                     and self._mirror.poll() is None else None)
        verified = {process_id: active for process_id, active in windows
                    if (process_id == owned_pid and self._mirror_identity == self.identity)
                    or _scrcpy_device(process_id) == self.identity.serial}
        # Prefer the user's mirror over a still-open mirror from an earlier bot session.
        candidates = {pid: active for pid, active in verified.items() if pid != owned_pid}
        candidates = candidates or verified
        if len(candidates) > 1:
            candidates = {pid: active for pid, active in candidates.items() if active}
            if len(candidates) != 1:
                raise CaptureError("Several selected-phone scrcpy windows are open; focus one")
        if not candidates:
            return None
        process_id = next(iter(candidates))
        self._external_mirror_pid = None if process_id == owned_pid else process_id
        self._external_mirror_identity = self.identity if process_id != owned_pid else None
        logger.info("Bound %s scrcpy pid=%d to selected device %s",
                    "managed" if process_id == owned_pid else "external", process_id, self.serial)
        return process_id

    @property
    def owns_mirror(self) -> bool:
        return self._mirror is not None

    def select_device(self) -> bool:
        devices = self._devices()
        if devices is None:
            logger.warning("Could not read adb device list")
            return False
        selected = [d for d in devices if self.serial is None or d.serial == self.serial]
        if len(selected) != 1:
            logger.error("Select one connected device explicitly; found %d", len(devices))
            return False
        self.identity = selected[0]
        self.serial = self.identity.serial
        return True

    def _devices(self) -> list[DeviceIdentity] | None:
        code, output = self._run(
            [self._config.adb_path, "devices", "-l"], self._config.command_timeout
        )
        if code or not output.strip():
            return None
        devices: list[DeviceIdentity] = []
        listed = False
        serials: set[str] = set()
        for line in output.splitlines():
            parts = line.split()
            if not parts or line.startswith("* daemon "):
                continue
            if line.strip() == "List of devices attached":
                listed = True
                continue
            if (len(parts) < 2 or parts[0] in serials or parts[1] not in
                    {"device", "offline", "unauthorized", "recovery", "sideload", "bootloader"}):
                return None
            serials.add(parts[0])
            listed = True
            if parts[1] == "device":
                ids = [p.removeprefix("transport_id:") for p in parts[2:]
                       if p.startswith("transport_id:")]
                if len(ids) > 1 or (ids and not ids[0].isdecimal()):
                    return None
                devices.append(DeviceIdentity(parts[0], ids[0] if ids else ""))
        return devices if listed else None

    def connected(self) -> bool | None:
        """Distinguish a confirmed loss of our transport from an unreadable adb reply."""
        if self.identity is None:
            return False
        devices = self._devices()
        if devices is None:
            return None
        for device in devices:
            if device.serial == self.identity.serial:
                if bool(device.transport_id) != bool(self.identity.transport_id):
                    return None
                return device == self.identity
        return False

    def restart_mirror(self, title: str, width: int, height: int,
                       cancelled: threading.Event) -> bool:
        """Only terminate the process we launched. Never kill the shared adb server."""
        if cancelled.is_set() or not self.connected():
            return False
        self._external_mirror_pid = None
        self._external_mirror_identity = None
        if self._mirror is not None and self._mirror.poll() is None:
            self._mirror.terminate()
            try:
                self._mirror.wait(timeout=self._config.command_timeout)
            except subprocess.TimeoutExpired:
                self._mirror.kill()
                self._mirror.wait(timeout=self._config.command_timeout)
        if cancelled.is_set():
            return False
        try:
            self._mirror = subprocess.Popen([
                self._config.scrcpy_path, "--serial", str(self.serial), "--window-title", title,
                f"--window-width={width}", f"--window-height={height}", "--no-audio",
                "--video-buffer=0", "--video-bit-rate=12M", "--max-size=1024",
                "--video-codec=h264", "--max-fps=144", "--stay-awake",
                "--always-on-top", "--turn-screen-off",
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._mirror_identity = self.identity
        except OSError:
            logger.exception("Could not start selected-device scrcpy")
            return False
        return True

    def close(self) -> None:
        if self._mirror is not None and self._mirror.poll() is None:
            self._mirror.terminate()
            try:
                self._mirror.wait(timeout=self._config.command_timeout)
            except subprocess.TimeoutExpired:
                self._mirror.kill()
                self._mirror.wait(timeout=self._config.command_timeout)

    def _adb(self, *args: str) -> list[str]:
        if self.serial is None:
            raise RuntimeError("No device selected")
        return [self._config.adb_path, "-s", self.serial, *args]

    def detect_package(self) -> bool:
        """Pick the one installed package matching package_hint. Ambiguity disables relaunch."""
        if self.serial is None and not self.select_device():
            return False
        code, output = self._run(self._adb("shell", "pm", "list", "packages"),
                                 self._config.command_timeout)
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
        if not _PACKAGE_NAME.fullmatch(found[0]):
            logger.error("Game relaunch disabled: %r is not a valid Android package name", found[0])
            return False
        self.package = found[0]
        logger.info("Game relaunch armed for package %s", self.package)
        return True

    def game_foreground(self) -> bool | None:
        """Check Android independently of a mirror that can still show an old game frame."""
        if self.package is None or self.serial is None:
            return None
        code, output = self._run(self._adb("shell", "dumpsys", "window", "displays"),
                                 self._config.command_timeout)
        # Some older Android releases expose the focus only in the full window dump.
        if code == 0 and "mCurrentFocus=" not in output:
            code, output = self._run(self._adb("shell", "dumpsys", "window"),
                                     self._config.command_timeout)
        if code:
            return None
        focused = _primary_window_focus(output)
        if focused is None:
            return None
        component = re.fullmatch(
            r"Window\{[a-fA-F0-9]+ u\d+ ([\w.]+)/[\w.$]+\}", focused,
        )
        if component is None or not _PACKAGE_NAME.fullmatch(component[1]):
            return None
        return component[1] == self.package

    def game_exited(self) -> bool:
        """True only if pidof ran on the phone and found no instance of the main process."""
        if self.package is None or self.serial is None or not _PACKAGE_NAME.fullmatch(self.package):
            return False
        # A remote status marker distinguishes a real empty process lookup from a broken ADB pipe.
        code, output = self._run(self._adb("shell", f"pidof {self.package}; echo pidof-status:$?"),
                                 self._config.command_timeout)
        return code == 0 and output.strip() == "pidof-status:1"

    def launch(self, cancelled: threading.Event | None = None) -> bool:
        """Bring the selected package to the foreground without stopping its process."""
        if self.package is None or self.serial is None or (cancelled and cancelled.is_set()):
            return False
        argv = self._adb("shell", "monkey", "-p", self.package, "-c",
                         "android.intent.category.LAUNCHER", "1")
        code, output = self._run(argv, self._config.command_timeout)
        if code:
            logger.error("Game launch failed: %s", output.strip())
        return code == 0
