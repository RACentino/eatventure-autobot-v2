from typing import Any

from bot.state_machine import State

TemplatePair = tuple[Any, Any]
MatchCandidate = tuple[float, int, int, int, int]
BoxCandidate = tuple[float, int, int, int, int, str]
RedIcon = tuple[float, int, int]
StateResult = State | None

__all__ = [
    "BoxCandidate",
    "MatchCandidate",
    "RedIcon",
    "StateResult",
    "TemplatePair",
]
