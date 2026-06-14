"""Atomic JSON persistence with write-throttling.

Extracted from bot.py to keep RuntimePersistence self-contained and
independently testable.  No dependency on EatventureBot or any v2 subsystem.
"""

import json
import logging
import os
import tempfile
import threading
import time

logger = logging.getLogger(__name__)


def _remove_temp_file(path: str | None) -> None:
    """Remove a temporary file, suppressing any OSError."""
    if not path:
        return
    try:
        os.remove(path)
    except OSError as exc:
        logger.debug("Failed to remove temporary file %s: %s", path, exc)


class RuntimePersistence:
    """Thread-safe atomic JSON state persistence with save-throttling.

    Writes are performed via a temp-file + os.replace() to avoid
    partial-write corruption.  Saves are throttled to at most one
    write per *save_interval* seconds unless *force=True*.
    """

    def __init__(self, path: str, save_interval: float) -> None:
        self.path = path
        self.save_interval = max(0.0, float(save_interval))
        self._last_save_time = 0.0
        self._lock = threading.RLock()

    def load(self) -> dict:
        """Return the persisted state dict, or {} on any failure."""
        if not self.path:
            return {}
        with self._lock:
            return self._load_locked()

    def _load_locked(self) -> dict:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(
                "Failed to load persisted state from %s: %s", self.path, exc
            )
            return {}
        return data if isinstance(data, dict) else {}

    def save(self, state: dict, force: bool = False) -> bool:
        """Atomically persist *state* to disk.

        Returns True if the write occurred, False if throttled or on error.
        """
        if not self.path:
            return False
        with self._lock:
            now = time.monotonic()
            if self._save_is_throttled(now, force):
                return False
            temp_path = None
            try:
                temp_path = self._write_temp_state_file(state)
                os.replace(temp_path, self.path)
                self._last_save_time = now
                return True
            except (OSError, TypeError, ValueError) as exc:
                logger.error(
                    "Failed to persist state to %s: %s", self.path, exc
                )
                _remove_temp_file(temp_path)
                return False

    def _save_is_throttled(self, now: float, force: bool) -> bool:
        return (
            not force
            and self.save_interval > 0
            and now - self._last_save_time < self.save_interval
        )

    def _write_temp_state_file(self, state: dict) -> str:
        directory = os.path.dirname(self.path)
        target_dir = directory or "."
        if directory:
            os.makedirs(directory, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=target_dir, delete=False
        ) as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
            return handle.name
