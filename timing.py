"""Timing and sleep utilities shared across bot modules.

Extracted from mouse_controller.py so that bot.py, learner.py, and
mouse_controller.py all consume a single source of truth for
interruptible sleep primitives.

No RGB gate code.  No Windows-specific imports.  Fully OS-agnostic.
"""

import math
import time
from collections.abc import Callable
from typing import Protocol

MIN_SLEEP_SLICE = 0.001
MAX_SLEEP_ITERATIONS = 120_000


class StopEventLike(Protocol):
    """Duck-typed interface compatible with threading.Event."""

    def is_set(self) -> bool: ...

    def wait(self, timeout: float) -> bool: ...


def _duration(value: object, default: float = 0.0) -> float:
    """Parse *value* as a non-negative float, falling back to *default*."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        number = float(default)
    if not math.isfinite(number):
        number = float(default)
    return max(0.0, number)


def _sleep_iterations(deadline: float) -> int:
    remaining = max(0.0, deadline - time.perf_counter())
    return min(
        MAX_SLEEP_ITERATIONS, max(1, int(math.ceil(remaining / MIN_SLEEP_SLICE)) + 3)
    )


def precise_sleep(duration: object) -> None:
    """Busy-accurate sleep with no stop-event check."""
    wait_event(None, duration)


def sleep_until(
    deadline: float, stop_event: StopEventLike | None = None
) -> bool:
    """Sleep until *deadline* (perf_counter seconds), or until *stop_event* fires.

    Returns True if the deadline was reached without interruption,
    False if *stop_event* was set before the deadline.
    """
    for _ in range(_sleep_iterations(deadline)):
        if stop_event is not None and stop_event.is_set():
            return False
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return stop_event is None or not stop_event.is_set()
        if stop_event is None:
            time.sleep(min(remaining, 0.05))
        elif stop_event.wait(min(remaining, 0.05)):
            return False
    return stop_event is None or not stop_event.is_set()


def wait_event(
    stop_event: StopEventLike | None, duration: object
) -> bool:
    """Sleep for *duration* seconds, honouring *stop_event* if given.

    Returns True if the sleep completed without interruption.
    """
    secs = _duration(duration)
    if secs <= 0:
        return stop_event is None or not stop_event.is_set()
    return sleep_until(time.perf_counter() + secs, stop_event)


class _InterruptAdapter:
    """Wraps a ``Callable[[], bool]`` as a StopEventLike for wait_event."""

    def __init__(self, interrupt_check: Callable[[], bool]) -> None:
        self._interrupt_check = interrupt_check

    def is_set(self) -> bool:
        return bool(self._interrupt_check())

    def wait(self, timeout: float) -> bool:
        end_time = time.perf_counter() + max(0.0, timeout)
        for _ in range(_sleep_iterations(end_time)):
            if self.is_set():
                return True
            remaining = end_time - time.perf_counter()
            if remaining <= 0:
                return self.is_set()
            time.sleep(min(remaining, 0.01))
        return self.is_set()
