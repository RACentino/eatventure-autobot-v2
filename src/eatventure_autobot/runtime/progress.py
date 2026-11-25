"""Small atomic summary. Only observed runtime is credited; process gaps never are."""

import hashlib
import importlib
import importlib.metadata
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from eatventure_autobot.domain.config import BotConfig
from eatventure_autobot.domain.state import BoostLevel

logger = logging.getLogger(__name__)
CATEGORIES = ("active", "recovery", "manual", "disconnected", "error")


def candidate_identity(config: BotConfig, fingerprint: str) -> tuple[str, str]:
    root = config.paths.project_root
    digest = hashlib.sha256(fingerprint.encode())
    digest.update(f"{sys.implementation.name}:{sys.version}:{sys.platform}".encode())
    for distribution in ("opencv-python", "numpy", "pynput", "requests", "PyWinCtl",
                         "python-xlib", "mss"):
        try:
            installed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            installed = "missing"
        digest.update(f"{distribution}=={installed}\n".encode())
    for path in sorted([*root.glob("src/**/*.py"), *config.paths.assets_dir.glob("*.png"),
                        root / "pyproject.toml"]):
        if path.is_file():
            digest.update(os.path.relpath(path, root).encode())
            digest.update(path.read_bytes())
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=False)
        revision = result.stdout.strip() if result.returncode == 0 else "unversioned"
    except OSError:
        revision = "unversioned"
    return digest.hexdigest()[:20], revision


class ProgressLedger:
    def __init__(
        self, path: Path, identity: str, *, revision: str, config: str,
        loadout: str, candidate: bool, clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time, detailed: bool = True,
    ) -> None:
        self.path, self.identity = path, identity
        self.detailed = detailed
        self._clock, self._wall_clock = clock, wall_clock
        self._at, self._wall = clock(), wall_clock()
        self._saved_at = self._at
        self.category = "manual"
        self.boost = BoostLevel.UNKNOWN
        self._closed = False
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = path.with_suffix(".lock").open("a+b")
        try:
            if os.name == "nt":
                msvcrt = importlib.import_module("msvcrt")
                self._lock_file.write(b"\0")
                self._lock_file.flush()
                self._lock_file.seek(0)
                msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock_file.close()
            raise RuntimeError(f"Another bot owns progress summary: {path}") from None
        self.data: dict[str, Any] = {"version": 1, "candidates": {}}
        try:
            if path.exists():
                self.data = json.loads(path.read_text())
                self._validate_summary()
        except Exception:
            self._lock_file.close()
            raise
        self.record = self.data["candidates"].setdefault(identity, {
            "revision": revision, "config": config, "loadout": loadout, "candidate": candidate,
            "seconds": dict.fromkeys(CATEGORIES, 0.0), "completions": 0, "days": {},
            "boosts": {}, "city_boundaries": 0, "last_city_active_s": None,
            "complete_city_active_s": [],
        })
        if not detailed:
            self.record["detailed"] = False
        self.record["last_city_active_s"] = None
        self.record["last_city_boost_active_s"] = None
        self.record["pending_transition"] = False  # never infer an unseen process-gap transition

    def _validate_summary(self) -> None:
        if self.data.get("version") != 1 or not isinstance(self.data.get("candidates"), dict):
            raise ValueError(f"Invalid progress summary: {self.path}")
        for record in self.data["candidates"].values():
            for category in CATEGORIES:
                value = record["seconds"][category]
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid {category} runtime in {self.path}")
            if not isinstance(record["completions"], int) or record["completions"] < 0:
                raise ValueError(f"Invalid completion count in {self.path}")

    def checkpoint(self, category: str | None = None, boost: BoostLevel | None = None) -> None:
        if category is not None and category not in CATEGORIES:
            raise ValueError(f"Unknown runtime category: {category}")
        category_changed = category is not None and category != self.category
        changed = category_changed or boost is not None and boost != self.boost
        now, wall = self._clock(), self._wall_clock()
        seconds = max(0.0, now - self._at)
        self.record["seconds"][self.category] += seconds
        if self.detailed and self.category in ("active", "recovery"):
            bucket = self.record["boosts"].setdefault(self.boost.value, {
                "active": 0.0, "recovery": 0.0, "completions": 0,
            })
            bucket[self.category] += seconds
        self._at, self._wall = now, wall
        if category is not None:
            self.category = category
        if boost is not None:
            self.boost = boost
        if changed:
            if category_changed and category in ("manual", "disconnected", "error"):
                self.record["last_city_active_s"] = None
                self.record["last_city_boost_active_s"] = None
            logger.info("Runtime category=%s candidate=%s boost=%s loadout=%s",
                        self.category, self.identity, self.boost.value, self.record["loadout"])
        if changed or (self.detailed and now - self._saved_at >= 5):
            self.save()

    def transition_attempt(self) -> None:
        self.record["pending_transition"] = True
        self.save()

    def cancel_transition(self) -> None:
        self.record["pending_transition"] = False
        self.save()

    def confirm_transition(self) -> bool:
        if not self.record["pending_transition"]:
            return False
        self.checkpoint()
        self.record["pending_transition"] = False
        self.record["completions"] += 1
        day = datetime.fromtimestamp(self._wall_clock()).date().isoformat()
        self.record["days"][day] = self.record["days"].get(day, 0) + 1
        if self.detailed:
            bucket = self.record["boosts"].setdefault(self.boost.value, {
                "active": 0.0, "recovery": 0.0, "completions": 0,
            })
            bucket["completions"] += 1
        self.save()
        return True

    def city_boundary(self) -> None:
        self.checkpoint()
        self.record["city_boundaries"] += 1
        if not self.detailed:
            self.save()
            return
        active = self.record["seconds"]["active"]
        previous = self.record["last_city_active_s"]
        boosts = {name: bucket["active"] for name, bucket in self.record["boosts"].items()}
        if previous is not None:
            self.record["complete_city_active_s"].append(active - previous)
            self.record["complete_city_active_s"] = self.record["complete_city_active_s"][-100:]
            start = self.record["last_city_boost_active_s"] or {}
            samples = self.record.setdefault("complete_city_samples", [])
            samples.append({"active_s": active - previous, "boost_active_s": {
                name: value - start.get(name, 0.0) for name, value in boosts.items()
                if value > start.get(name, 0.0)
            }})
            self.record["complete_city_samples"] = samples[-100:]
        self.record["last_city_active_s"] = active
        self.record["last_city_boost_active_s"] = boosts
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=self.path.parent, delete=False) as out:
                temporary = out.name
                json.dump(self.data, out, sort_keys=True, allow_nan=False)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temporary, self.path)
            self._saved_at = self._clock()
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    def close(self) -> None:
        if not self._closed:
            try:
                self.checkpoint()
                self.save()
            finally:
                self._closed = True
                self._lock_file.close()
