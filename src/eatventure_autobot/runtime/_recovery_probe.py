"""Read-only phone and scrcpy observations made only during suspected stalls."""

import threading
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from eatventure_autobot.domain.config import BotConfig
from eatventure_autobot.domain.errors import CaptureError
from eatventure_autobot.domain.state import BoostLevel
from eatventure_autobot.domain.types import Point
from eatventure_autobot.runtime.game_recovery import GameLauncher
from eatventure_autobot.runtime.vision import GameVision

type Dialog = Literal["welcome", "open", "unlock"]


@dataclass(frozen=True)
class ScreenProbe:
    status: Literal["playable", "mirror-stale", "still", "unknown"]
    screen: Dialog | None = None
    signature: np.ndarray | None = None
    static: bool = False
    world_views: tuple[np.ndarray, np.ndarray] | None = None  # normalized phone, then mirror


def same_game_area(previous: np.ndarray | None, current: np.ndarray | None, limit: float) -> bool:
    return (previous is not None and current is not None and previous.shape == current.shape
            and float(np.abs(previous - current).mean()) < limit)


def _content_rows(frame: np.ndarray) -> slice:
    # The native 1080x2400 recording has a 95px solid top inset, scrcpy has 4px.
    # Strip only shallow, horizontally uniform borders when the aspect ratios differ.
    columns = frame[:, ::max(1, frame.shape[1] // 64)]
    visible = np.flatnonzero(np.ptp(columns, axis=1).max(axis=1) > 12)
    if visible.size and visible[0] < len(frame) // 10 and visible[-1] >= len(frame) * .9:
        return slice(int(visible[0]), int(visible[-1]) + 1)
    return slice(None)


def _phone_at_mirror_size(phone: np.ndarray, mirror: np.ndarray) -> np.ndarray | None:
    height, width = mirror.shape[:2]
    ratio = (width * phone.shape[0]) / (height * phone.shape[1])
    if not .9 < ratio < 1.1:
        return None
    # Linear scaling retains the Open lettering used by the normal restaurant detector.
    scaled = cv2.resize(phone, (width, height), interpolation=cv2.INTER_LINEAR)
    if abs(ratio - 1) > .01:
        target = scaled[_content_rows(mirror)]
        target[:] = cv2.resize(
            phone[_content_rows(phone)], (width, len(target)), interpolation=cv2.INTER_LINEAR,
        )
    return scaled


def _signature(frame: np.ndarray, config: BotConfig) -> np.ndarray:
    # Compare the restaurant, excluding the Android clock, money counter and footer.
    height = frame.shape[0]
    bottom = min(height, config.capture_regions.max_search_y * height // config.window.height)
    world = frame[height // 10:bottom]
    return cv2.resize(world, (36, 64), interpolation=cv2.INTER_AREA).astype(np.float32)


def _actionable(vision: GameVision, frame: np.ndarray, config: BotConfig,
                expected_unlock: bool) -> tuple[Dialog, Point] | None:
    if vision.city_arrival(frame):
        button = vision.find_city_continue(frame).best
        return ("welcome", button.center) if button else None
    if vision.restaurant_start(frame):
        button = vision.find_new_level_button(frame[:config.capture_regions.max_search_y]).best
        return ("open", button.center) if button else None
    if expected_unlock:
        button = vision.find_unlock_button(frame).best
        if button and recovery_unlock_visible(vision, frame, button.center):
            return "unlock", button.center
    return None


def recovery_unlock_visible(vision: GameVision, frame: np.ndarray, point: Point) -> bool:
    # The blue template also matches Upgrade and Collect x2. Only Unlock has this compact panel.
    bounds = vision.station_panel_bounds(frame)
    return bool(bounds and 0 < bounds.y_min < bounds.y_max < frame.shape[0]
                and bounds.x_max - bounds.x_min > bounds.y_max - bounds.y_min
                and bounds.x_min <= point[0] < bounds.x_max
                and bounds.y_min <= point[1] < bounds.y_max
                and not vision.find_close_buttons(frame))


def _world_obscured(vision: GameVision, frame: np.ndarray) -> bool:
    return bool(vision.find_close_buttons(frame) or vision.station_panel_bounds(frame)
                or vision.restaurant_start(frame) or vision.city_arrival(frame))


def _restaurant_visible(vision: GameVision, frame: np.ndarray, *, after_action: bool,
                        for_scroll: bool = False) -> bool:
    if after_action and _world_obscured(vision, frame):
        return False
    hud = (vision.boost_level(frame) != BoostLevel.UNKNOWN or vision.stats_icon_seen(frame)
           or vision.new_level_icon_seen(frame))
    if after_action and not hud:
        # Quiet restaurants can have an unrecognized boost and no HUD notification.
        if not for_scroll:
            return False
        targets = vision.station_targets(frame)
        return (any(t.template_name.startswith(("stationMarker", "stationBuilt")) for t in targets)
                and (len(targets) > 1 or bool(vision.find_boxes(frame))))
    if hud and not after_action:
        return True
    return bool(vision.station_targets(frame) or vision.find_boxes(frame))


def _moving(span: np.ndarray, limit: float) -> bool:
    # A moving cook may occupy only one tile in an otherwise quiet restaurant.
    tiles = cv2.resize(span, (6, 8), interpolation=cv2.INTER_AREA)
    return bool(span.mean() >= limit or tiles.mean(axis=2).max() >= 4 * limit)


def scroll_surface_current(vision: GameVision, config: BotConfig, before: ScreenProbe) -> bool:
    if before.world_views is None or before.signature is None:
        return False
    fresh = vision.capture()
    return (fresh.shape == before.world_views[1].shape
            and same_game_area(before.signature, _signature(fresh, config),
                               config.game_recovery.static_recent_delta)
            and _restaurant_visible(vision, fresh, after_action=True, for_scroll=True))


def probe_screen(
    vision: GameVision, launcher: GameLauncher, config: BotConfig, cancelled: threading.Event, *,
    expected_unlock: bool = False, expected_gameplay: bool = False, check_scroll: bool = False,
    checked_world: np.ndarray | None = None, after_scroll: bool = False,
) -> ScreenProbe:
    """Sample both views afresh. The caller verifies selected-device identity and foreground."""
    phone_samples, mirror_samples = [], []
    interval = config.game_recovery.recovery_poll_seconds
    for index in range(3):
        if cancelled.is_set() or (index and cancelled.wait(interval)):
            return ScreenProbe("unknown")
        encoded = launcher.capture_device_png()
        if not encoded or cancelled.is_set():
            return ScreenProbe("unknown")
        try:
            phone = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
            if phone is None:
                return ScreenProbe("unknown")
            mirror = vision.capture()
            if mirror.ndim != 3 or mirror.shape[2] != 3 or min(mirror.shape[:2]) < 2:
                return ScreenProbe("unknown")
            normalized = _phone_at_mirror_size(phone, mirror)
            if normalized is None or cancelled.is_set():
                return ScreenProbe("unknown")
            phone_samples.append(_signature(normalized, config))
            mirror_samples.append(_signature(mirror, config))
        except (CaptureError, cv2.error):
            return ScreenProbe("unknown")

    assert normalized is not None  # every completed sample has matching geometry
    phone_motion = np.ptp(phone_samples, axis=0)
    mirror_motion = np.ptp(mirror_samples, axis=0)
    limit = config.game_recovery.static_frame_delta
    phone_moves = _moving(phone_motion, limit)
    mirror_moves = _moving(mirror_motion, limit)
    signature = mirror_samples[-1]
    if phone_moves and not mirror_moves:
        return ScreenProbe("mirror-stale", signature=signature)
    # Allow compression and slight sampling skew. An unrelated frozen image proves nothing.
    alignment_limit = 3 * config.game_recovery.static_recent_delta
    static = bool(not phone_moves and not mirror_moves)
    if np.abs(phone_samples[-1] - signature).mean() > alignment_limit:
        # A phone that panned to a bare edge can now disagree with an entirely frozen mirror.
        # This pair is useful for measuring a previous drag; "unknown" cannot initiate one.
        worlds = (normalized, mirror.copy()) if static and after_scroll else None
        return ScreenProbe("unknown", world_views=worlds)
    control = _actionable(vision, mirror, config, expected_unlock)
    phone_control = _actionable(vision, normalized, config, expected_unlock)
    if control or phone_control:
        if control and phone_control and control[0] == phone_control[0]:
            dx, dy = np.subtract(control[1], phone_control[1])
            if max(abs(dx), abs(dy)) <= config.upgrade_station.disappear_position_tolerance_px:
                return ScreenProbe("playable", control[0], signature, static)
        return ScreenProbe("unknown")
    matching_motion = _moving(np.minimum(phone_motion, mirror_motion), limit)
    if ((matching_motion or (static and expected_gameplay))
            and _restaurant_visible(vision, mirror, after_action=expected_gameplay)
            and _restaurant_visible(vision, normalized, after_action=expected_gameplay)):
        return ScreenProbe("playable", signature=signature, static=static)
    if static:
        worlds = None
        if after_scroll:
            # A verified drag can move every visible station off screen at a camera edge.
            if not (_world_obscured(vision, mirror) or _world_obscured(vision, normalized)):
                worlds = (normalized, mirror.copy())
        elif (check_scroll and not same_game_area(checked_world, signature, alignment_limit)
              and _restaurant_visible(vision, mirror, after_action=True, for_scroll=True)
              and _restaurant_visible(vision, normalized, after_action=True, for_scroll=True)):
            worlds = (normalized, mirror.copy())
        return ScreenProbe("still", signature=signature, static=True, world_views=worlds)
    return ScreenProbe("unknown")


def scroll_response(vision: GameVision, before: ScreenProbe, after: ScreenProbe,
                    direction: int) -> ScreenProbe:
    """A settled pan can be static within each sample burst but displaced between bursts."""
    if after.status in ("playable", "mirror-stale"):
        return after
    if before.world_views is None or after.world_views is None:
        return ScreenProbe("unknown")
    motion = []
    for old, new in zip(before.world_views, after.world_views, strict=True):
        if vision.viewport_still(old, new) is True:
            motion.append("still")
            continue
        shift = vision.viewport_translation(old, new)
        # A vertical drag must move world features in the drag direction, beyond pixel jitter.
        moved = shift is not None and direction * shift[1] >= max(4, 2 * abs(shift[0]))
        motion.append("moved" if moved else "unknown")
    if motion == ["moved", "still"]:
        return ScreenProbe("mirror-stale")
    if after.status == "still":
        if motion == ["moved", "moved"]:
            return ScreenProbe("playable")
        if motion == ["still", "still"]:
            return after
    return ScreenProbe("unknown")
