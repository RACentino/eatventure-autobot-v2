from dataclasses import dataclass

import config

FORBIDDEN_CLICK_ZONE_NAME = "FORBIDDEN_CLICK"
NUMBERED_FORBIDDEN_ZONE_PREFIX = "FORBIDDEN_ZONE_"


@dataclass(frozen=True)
class ForbiddenZone:
    name: str
    x_min: int
    x_max: int
    y_min: int
    y_max: int | None


def configured_forbidden_zones() -> tuple[ForbiddenZone, ...]:
    base_zone = ForbiddenZone(
        FORBIDDEN_CLICK_ZONE_NAME,
        int(config.FORBIDDEN_CLICK_X_MIN),
        int(config.FORBIDDEN_CLICK_X_MAX),
        int(config.FORBIDDEN_CLICK_Y_MIN),
        None,
    )
    numbered_zones = tuple(
        ForbiddenZone(
            f"{NUMBERED_FORBIDDEN_ZONE_PREFIX}{zone_index}",
            int(x_min),
            int(x_max),
            int(y_min),
            int(y_max),
        )
        for zone_index, (x_min, x_max, y_min, y_max) in enumerate(
            config.NUMBERED_FORBIDDEN_ZONE_BOUNDS, start=1
        )
    )
    return (base_zone, *numbered_zones)


def bounded_forbidden_zone(
    zone: ForbiddenZone, frame_width: int, frame_height: int
) -> tuple[int, int, int, int] | None:
    if frame_width <= 0 or frame_height <= 0:
        return None
    raw_y_max = frame_height - 1 if zone.y_max is None else int(zone.y_max)
    outside_frame = (
        zone.x_max < 0
        or zone.x_min >= frame_width
        or raw_y_max < 0
        or zone.y_min >= frame_height
    )
    if outside_frame:
        return None
    return (
        max(0, int(zone.x_min)),
        min(frame_width - 1, int(zone.x_max)),
        max(0, int(zone.y_min)),
        min(frame_height - 1, raw_y_max),
    )


def point_inside_bounds(x: int, y: int, bounds: tuple[int, int, int, int]) -> bool:
    x_min, x_max, y_min, y_max = bounds
    return x_min <= x <= x_max and y_min <= y <= y_max


def point_inside_forbidden_zone(x: int, y: int, zone: ForbiddenZone) -> bool:
    if zone.y_max is None:
        return int(y) >= zone.y_min and zone.x_min <= int(x) <= zone.x_max
    return zone.x_min <= int(x) <= zone.x_max and zone.y_min <= int(y) <= zone.y_max


def first_forbidden_zone_containing_point(
    x: int, y: int, zones: tuple[ForbiddenZone, ...] | list[ForbiddenZone]
) -> ForbiddenZone | None:
    for zone in zones:
        if point_inside_forbidden_zone(x, y, zone):
            return zone
    return None


def point_blocked_by_forbidden_zones(
    x: int,
    y: int,
    zones: tuple[ForbiddenZone, ...],
    frame_width: int,
    frame_height: int,
) -> bool:
    if not (0 <= int(x) < frame_width and 0 <= int(y) < frame_height):
        return True
    for zone in zones:
        bounds = bounded_forbidden_zone(zone, frame_width, frame_height)
        if bounds is not None and point_inside_bounds(int(x), int(y), bounds):
            return True
    return False


def box_intersects_forbidden_zones(
    box: tuple[float, float, float, float],
    zones: tuple[ForbiddenZone, ...],
    frame_width: int,
    frame_height: int,
) -> bool:
    left, top, right, bottom = box
    for zone in zones:
        bounds = bounded_forbidden_zone(zone, frame_width, frame_height)
        if bounds is not None and box_intersects_bounds(
            left, top, right, bottom, bounds
        ):
            return True
    return False


def box_intersects_bounds(
    left: float,
    top: float,
    right: float,
    bottom: float,
    bounds: tuple[int, int, int, int],
) -> bool:
    x_min, x_max, y_min, y_max = bounds
    return left <= x_max and right >= x_min and top <= y_max and bottom >= y_min
