from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

from eatventure_autobot.domain.types import (
    HsvGate,
    MatchCandidate,
    MatchResult,
    Point,
    WindowBounds,
    Zone,
)


class ScreenCapture(Protocol):
    def ensure_window(self, resize: bool = False) -> None: ...

    def get_window_rect(self) -> WindowBounds: ...

    def is_window_active(self) -> bool: ...

    def capture(self, max_y: int | None = None) -> np.ndarray: ...

    def close(self) -> None: ...


class InputController(Protocol):
    def is_target_foreground(self) -> bool: ...

    def get_cursor_position(self) -> Point: ...

    def is_in_forbidden_zone(self, x: int, y: int, relative: bool = True) -> bool: ...

    def set_event_forbidden_zone(self, zone: Zone) -> None: ...

    def click(self, x: int, y: int, relative: bool = True, delay: float | None = None) -> bool: ...

    def precise_click(
        self, x: int, y: int, relative: bool = True, delay: float | None = None
    ) -> bool: ...

    def spam_click_at(
        self,
        x: int,
        y: int,
        duration: float,
        click_delay: float,
        relative: bool = True,
    ) -> bool: ...

    def drag(
        self,
        from_x: int,
        from_y: int,
        to_x: int,
        to_y: int,
        duration: float | None = None,
        relative: bool = True,
    ) -> bool: ...

    def hold_at(
        self,
        x: int,
        y: int,
        duration: float,
        check_interval: float,
        interrupt_check: Callable[[], bool],
        relative: bool = True,
    ) -> bool: ...

    def release_left_button(self) -> bool: ...


class TemplateMatcher(Protocol):
    def load_template(self, template_path: Path) -> None: ...

    def find_template(
        self,
        frame: np.ndarray,
        template_name: str,
        threshold: float,
        hsv_gate: HsvGate | None = None,
    ) -> MatchResult: ...

    def find_template_candidates(
        self,
        frame: np.ndarray,
        template_name: str,
        threshold: float,
        min_distance: int = 15,
        hsv_gate: HsvGate | None = None,
    ) -> list[MatchCandidate]: ...

    def find_all_templates(
        self,
        frame: np.ndarray,
        template_names: tuple[str, ...],
        threshold: float,
        min_distance: int = 15,
        hsv_gate: HsvGate | None = None,
    ) -> list[MatchCandidate]: ...

    def find_candidates_across_templates(
        self,
        frame: np.ndarray,
        template_names: tuple[str, ...],
        threshold: float,
        min_distance: int,
        hsv_gate: HsvGate | None,
    ) -> list[MatchCandidate]: ...

    def suppress_overlaps(
        self, candidates: list[MatchCandidate], iou_threshold: float
    ) -> list[MatchCandidate]: ...


class Notifier(Protocol):
    def notify_bot_started(self) -> None: ...

    def notify_bot_stopped(self) -> None: ...

    def notify_new_level(self, level_number: int, time_spent: float) -> None: ...

    def close(self) -> None: ...


class HotkeyPoint(Protocol):
    """Cursor-position readout used by the 'X' debug hotkey, relative to the target window."""

    def __call__(self) -> Point: ...
