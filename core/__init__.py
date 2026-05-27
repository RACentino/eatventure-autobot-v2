import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

__all__ = ["bounded_float", "bounded_int", "project_path"]


def project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def bounded_float(
    value: object,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        number = float(default)
    if not math.isfinite(number):
        number = float(default)
    if minimum is not None:
        try:
            minimum_value = float(minimum)
        except (TypeError, ValueError, OverflowError):
            minimum_value = None
        if minimum_value is not None and math.isfinite(minimum_value):
            number = max(minimum_value, number)
    if maximum is not None:
        try:
            maximum_value = float(maximum)
        except (TypeError, ValueError, OverflowError):
            maximum_value = None
        if maximum_value is not None and math.isfinite(maximum_value):
            number = min(maximum_value, number)
    return number


def bounded_int(
    value: object,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        number = int(default)
    if minimum is not None:
        try:
            minimum_value = int(minimum)
        except (TypeError, ValueError, OverflowError):
            minimum_value = None
        if minimum_value is not None:
            number = max(minimum_value, number)
    if maximum is not None:
        try:
            maximum_value = int(maximum)
        except (TypeError, ValueError, OverflowError):
            maximum_value = None
        if maximum_value is not None:
            number = min(maximum_value, number)
    return number
