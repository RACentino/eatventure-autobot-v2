from dataclasses import dataclass

Point = tuple[int, int]


@dataclass(frozen=True, slots=True)
class WindowBounds:
    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"WindowBounds must have positive size, got {self.width}x{self.height}"
            )


@dataclass(frozen=True, slots=True)
class Zone:
    """A rectangular region in window-relative coordinates. y_max=None means unbounded downward."""

    x_min: int
    x_max: int
    y_min: int
    y_max: int | None = None

    def __post_init__(self) -> None:
        if self.x_min > self.x_max:
            raise ValueError(f"Zone x_min ({self.x_min}) must be <= x_max ({self.x_max})")
        if self.y_max is not None and self.y_min > self.y_max:
            raise ValueError(f"Zone y_min ({self.y_min}) must be <= y_max ({self.y_max})")

    def contains(self, x: int, y: int) -> bool:
        if not (self.x_min <= x <= self.x_max):
            return False
        if self.y_max is None:
            return y >= self.y_min
        return self.y_min <= y <= self.y_max


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __post_init__(self) -> None:
        if self.x_min > self.x_max or self.y_min > self.y_max:
            raise ValueError(f"Invalid bounding box: {self}")

    def area(self) -> int:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def iou(self, other: "BoundingBox") -> float:
        overlap_x = max(0, min(self.x_max, other.x_max) - max(self.x_min, other.x_min))
        overlap_y = max(0, min(self.y_max, other.y_max) - max(self.y_min, other.y_min))
        overlap_area = overlap_x * overlap_y
        if overlap_area == 0:
            return 0.0
        union_area = self.area() + other.area() - overlap_area
        return overlap_area / union_area if union_area > 0 else 0.0


@dataclass(frozen=True, slots=True)
class HsvRange:
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]

    def __post_init__(self) -> None:
        for name, triplet in (("lower", self.lower), ("upper", self.upper)):
            hue, sat, val = triplet
            if not (0 <= hue <= 179):
                raise ValueError(f"HsvRange.{name} hue {hue} out of range [0, 179]")
            if not (0 <= sat <= 255) or not (0 <= val <= 255):
                raise ValueError(
                    f"HsvRange.{name} saturation/value out of range [0, 255]: {triplet}"
                )


@dataclass(frozen=True, slots=True)
class HsvGate:
    ranges: tuple[HsvRange, ...]
    min_match_ratio: float

    def __post_init__(self) -> None:
        if not self.ranges:
            raise ValueError("HsvGate requires at least one HsvRange")
        if not (0.0 < self.min_match_ratio <= 1.0):
            raise ValueError(
                f"HsvGate.min_match_ratio must be in (0, 1], got {self.min_match_ratio}"
            )


@dataclass(frozen=True, slots=True)
class MatchCandidate:
    """A single template-match candidate found in a captured frame."""

    template_name: str
    confidence: float
    center: Point
    bounding_box: BoundingBox

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"MatchCandidate.confidence must be in [0, 1], got {self.confidence}")


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Outcome of a single-template search: whether it passed threshold, and the best candidate."""

    found: bool
    best: MatchCandidate | None

    def __post_init__(self) -> None:
        if self.found and self.best is None:
            raise ValueError("MatchResult.found=True requires a best candidate")
