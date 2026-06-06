import logging
import threading
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

import config
from window_capture import configure_overlay_canvas, configure_overlay_root, destroy_overlay_root, position_overlay_over_rect, set_overlay_visible_regions

logger = logging.getLogger(__name__)

try:
    import supervision as sv
except Exception as exc:
    sv = None
    _SUPERVISION_IMPORT_ERROR = exc
else:
    _SUPERVISION_IMPORT_ERROR = None

ASSET_TRACKING_LOOP_ITERATION_LIMIT = 2_147_483_647
ASSET_TRACKING_OVERLAY_LOOP_ITERATION_LIMIT = 2_147_483_647
ASSET_CLASS_IDS = {"red_icon": 0, "upgrade_station": 1, "box": 2}
ASSET_CLASS_NAMES = {value: key for key, value in ASSET_CLASS_IDS.items()}


@dataclass(frozen=True)
class ForbiddenZone:
    name: str
    x_min: int
    x_max: int
    y_min: int
    y_max: int | None


@dataclass(frozen=True)
class AssetDetection:
    asset_type: str
    template_name: str
    confidence: float
    center_x: int
    center_y: int
    width: int
    height: int
    target_x: int | None = None
    target_y: int | None = None


@dataclass(frozen=True)
class TrackedAsset:
    asset_type: str
    template_name: str
    tracker_id: int
    confidence: float
    center_x: int
    center_y: int
    width: int
    height: int
    target_x: int | None = None
    target_y: int | None = None


@dataclass(frozen=True)
class AssetTrackingSnapshot:
    frame_number: int
    timestamp: float
    width: int
    height: int
    assets: tuple[TrackedAsset, ...]


DetectionProvider = Callable[[np.ndarray], list[AssetDetection]]


def _configured_forbidden_zones() -> tuple[ForbiddenZone, ...]:
    zones = [
        ForbiddenZone("FORBIDDEN_CLICK", config.FORBIDDEN_CLICK_X_MIN, config.FORBIDDEN_CLICK_X_MAX, config.FORBIDDEN_CLICK_Y_MIN, None),
    ]
    for index, (x_min, x_max, y_min, y_max) in enumerate(config.NUMBERED_FORBIDDEN_ZONE_BOUNDS, start=1):
        zones.append(ForbiddenZone(f"FORBIDDEN_ZONE_{index}", int(x_min), int(x_max), int(y_min), int(y_max)))
    return tuple(zones)


def _candidate_xyxy(candidate: AssetDetection) -> tuple[float, float, float, float]:
    return _centered_xyxy(candidate.center_x, candidate.center_y, candidate.width, candidate.height)


def _asset_xyxy(asset: TrackedAsset) -> tuple[float, float, float, float]:
    return _centered_xyxy(asset.center_x, asset.center_y, asset.width, asset.height)


def _centered_xyxy(center_x: int, center_y: int, width: int, height: int) -> tuple[float, float, float, float]:
    half_width = max(1.0, float(width)) / 2.0
    half_height = max(1.0, float(height)) / 2.0
    return (
        float(center_x) - half_width,
        float(center_y) - half_height,
        float(center_x) + half_width,
        float(center_y) + half_height,
    )


def _valid_detection(candidate: AssetDetection) -> bool:
    values = (candidate.confidence, candidate.center_x, candidate.center_y, candidate.width, candidate.height)
    return all(np.isfinite(float(value)) for value in values) and candidate.width > 0 and candidate.height > 0


def _valid_asset(asset: TrackedAsset) -> bool:
    values = (asset.confidence, asset.center_x, asset.center_y, asset.width, asset.height)
    return all(np.isfinite(float(value)) for value in values) and asset.width > 0 and asset.height > 0


def _sequence_value(values: Any, index: int) -> Any:
    if values is None:
        return None
    try:
        return values[index]
    except (IndexError, TypeError, KeyError):
        return None


def _string_value(value: Any, default: str) -> str:
    if value is None:
        return default
    decoded = value.item() if hasattr(value, "item") else value
    return str(decoded)


def _int_value(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return int(default)
    return number


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _target_coordinate(value: int | None, fallback: int) -> int:
    return int(fallback if value is None else value)


def _bounded_forbidden_zone(zone: ForbiddenZone, frame_width: int, frame_height: int) -> tuple[int, int, int, int] | None:
    if frame_width <= 0 or frame_height <= 0:
        return None
    raw_y_max = frame_height - 1 if zone.y_max is None else int(zone.y_max)
    if zone.x_max < 0 or zone.x_min >= frame_width or raw_y_max < 0 or zone.y_min >= frame_height:
        return None
    return max(0, zone.x_min), min(frame_width - 1, zone.x_max), max(0, zone.y_min), min(frame_height - 1, raw_y_max)


def _point_blocked_by_forbidden_zones(
    x: int,
    y: int,
    zones: tuple[ForbiddenZone, ...],
    frame_width: int,
    frame_height: int,
) -> bool:
    if not (0 <= int(x) < frame_width and 0 <= int(y) < frame_height):
        return True
    for zone in zones:
        bounds = _bounded_forbidden_zone(zone, frame_width, frame_height)
        if bounds is not None and _point_inside_bounds(int(x), int(y), bounds):
            return True
    return False


def _point_inside_bounds(x: int, y: int, bounds: tuple[int, int, int, int]) -> bool:
    x_min, x_max, y_min, y_max = bounds
    return x_min <= x <= x_max and y_min <= y <= y_max


def _box_intersects_forbidden_zones(
    box: tuple[float, float, float, float],
    zones: tuple[ForbiddenZone, ...],
    frame_width: int,
    frame_height: int,
) -> bool:
    left, top, right, bottom = box
    for zone in zones:
        bounds = _bounded_forbidden_zone(zone, frame_width, frame_height)
        if bounds is not None and _box_intersects_bounds(left, top, right, bottom, bounds):
            return True
    return False


def _box_intersects_bounds(left: float, top: float, right: float, bottom: float, bounds: tuple[int, int, int, int]) -> bool:
    x_min, x_max, y_min, y_max = bounds
    return left <= x_max and right >= x_min and top <= y_max and bottom >= y_min


class AssetTracker:
    def __init__(self, window_capture: Any, detection_provider: DetectionProvider) -> None:
        self.window_capture = window_capture
        self.detection_provider = detection_provider
        self.interval = max(0.01, float(config.ASSET_TRACKING_INTERVAL))
        self.max_detections = max(1, int(config.ASSET_TRACKING_MAX_DETECTIONS))
        self._tracker = self._create_tracker()
        self._forbidden_zones = _configured_forbidden_zones()
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._snapshot = AssetTrackingSnapshot(0, 0.0, 0, 0, ())

    @staticmethod
    def available() -> bool:
        return sv is not None and hasattr(sv, "ByteTrack")

    def _create_tracker(self) -> Any | None:
        if not self.available():
            if _SUPERVISION_IMPORT_ERROR is not None:
                logger.warning("Supervision ByteTrack unavailable: %s", _SUPERVISION_IMPORT_ERROR)
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", FutureWarning)
            supervision_module = sv
            if supervision_module is None:
                return None
            return supervision_module.ByteTrack(
                track_activation_threshold=float(config.ASSET_TRACKING_TRACK_ACTIVATION_THRESHOLD),
                lost_track_buffer=int(config.ASSET_TRACKING_LOST_TRACK_BUFFER),
                minimum_matching_threshold=float(config.ASSET_TRACKING_MINIMUM_MATCHING_THRESHOLD),
                frame_rate=float(config.ASSET_TRACKING_FRAME_RATE),
                minimum_consecutive_frames=int(config.ASSET_TRACKING_MINIMUM_CONSECUTIVE_FRAMES),
            )

    def start(self) -> bool:
        if not config.ASSET_TRACKING_ENABLED:
            return False
        if self._thread is not None and self._thread.is_alive():
            return True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="asset_tracker", daemon=True)
        self._thread.start()
        logger.info("Asset tracker started")
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=float(config.ASSET_TRACKING_THREAD_JOIN_TIMEOUT))
        logger.info("Asset tracker stopped")

    def reset(self) -> None:
        tracker_reset = getattr(self._tracker, "reset", None)
        if callable(tracker_reset):
            tracker_reset()
        with self._lock:
            self._snapshot = AssetTrackingSnapshot(0, 0.0, 0, 0, ())

    def assets(self, asset_type: str | None = None) -> list[TrackedAsset]:
        with self._lock:
            snapshot = self._snapshot
        if snapshot.timestamp <= 0 or time.time() - snapshot.timestamp > float(config.ASSET_TRACKING_MAX_SNAPSHOT_AGE):
            return []
        assets = self._filter_trackable_assets(list(snapshot.assets), snapshot.width, snapshot.height)
        if asset_type is None:
            return assets
        return [asset for asset in assets if asset.asset_type == asset_type]

    def snapshot(self) -> AssetTrackingSnapshot:
        with self._lock:
            snapshot = self._snapshot
        assets = tuple(self._filter_trackable_assets(list(snapshot.assets), snapshot.width, snapshot.height))
        if len(assets) == len(snapshot.assets):
            return snapshot
        return AssetTrackingSnapshot(snapshot.frame_number, snapshot.timestamp, snapshot.width, snapshot.height, assets)

    def _run(self) -> None:
        for _ in range(ASSET_TRACKING_LOOP_ITERATION_LIMIT):
            if self._stop.is_set():
                return
            try:
                self._process_frame()
            except Exception:
                logger.exception("Asset tracking frame failed")
            if self._stop.wait(self.interval):
                return
        logger.error("Asset tracker loop reached iteration limit")

    def _process_frame(self) -> None:
        screenshot = self.window_capture.capture(max_y=config.ASSET_TRACKING_CAPTURE_Y)
        height, width = screenshot.shape[:2]
        detections = self._filter_trackable_detections(self.detection_provider(screenshot), width, height)
        detections = sorted(detections, key=lambda item: item.confidence, reverse=True)[: self.max_detections]
        assets = tuple(self._filter_trackable_assets(self._track_detections(detections), width, height))
        with self._lock:
            self._snapshot = AssetTrackingSnapshot(self._snapshot.frame_number + 1, time.time(), width, height, assets)

    def _filter_trackable_detections(self, detections: list[AssetDetection], frame_width: int, frame_height: int) -> list[AssetDetection]:
        return [candidate for candidate in detections if self._detection_is_trackable(candidate, frame_width, frame_height)]

    def _detection_is_trackable(self, candidate: AssetDetection, frame_width: int, frame_height: int) -> bool:
        if not _valid_detection(candidate):
            return False
        target_x = _target_coordinate(candidate.target_x, candidate.center_x)
        target_y = _target_coordinate(candidate.target_y, candidate.center_y)
        if _point_blocked_by_forbidden_zones(candidate.center_x, candidate.center_y, self._forbidden_zones, frame_width, frame_height):
            return False
        if _point_blocked_by_forbidden_zones(target_x, target_y, self._forbidden_zones, frame_width, frame_height):
            return False
        return not _box_intersects_forbidden_zones(_candidate_xyxy(candidate), self._forbidden_zones, frame_width, frame_height)

    def _filter_trackable_assets(self, assets: list[TrackedAsset], frame_width: int, frame_height: int) -> list[TrackedAsset]:
        return [asset for asset in assets if self._asset_is_trackable(asset, frame_width, frame_height)]

    def _asset_is_trackable(self, asset: TrackedAsset, frame_width: int, frame_height: int) -> bool:
        if not _valid_asset(asset):
            return False
        target_x = _target_coordinate(asset.target_x, asset.center_x)
        target_y = _target_coordinate(asset.target_y, asset.center_y)
        if _point_blocked_by_forbidden_zones(asset.center_x, asset.center_y, self._forbidden_zones, frame_width, frame_height):
            return False
        if _point_blocked_by_forbidden_zones(target_x, target_y, self._forbidden_zones, frame_width, frame_height):
            return False
        return not _box_intersects_forbidden_zones(_asset_xyxy(asset), self._forbidden_zones, frame_width, frame_height)

    def _track_detections(self, detections: list[AssetDetection]) -> list[TrackedAsset]:
        if self._tracker is None:
            return self._fallback_assets(detections)
        supervision_detections = self._to_supervision_detections(detections)
        try:
            tracked = self._tracker.update_with_detections(supervision_detections)
        except Exception as exc:
            logger.debug("ByteTrack update failed: %s", exc)
            return self._fallback_assets(detections)
        return self._from_supervision_detections(tracked)

    def _to_supervision_detections(self, detections: list[AssetDetection]) -> Any:
        supervision_module = sv
        if supervision_module is None:
            raise RuntimeError("supervision is unavailable")
        valid = [candidate for candidate in detections if _valid_detection(candidate)]
        boxes = np.asarray([_candidate_xyxy(candidate) for candidate in valid], dtype=np.float32).reshape((-1, 4))
        confidences = np.asarray([candidate.confidence for candidate in valid], dtype=np.float32)
        class_ids = np.asarray([ASSET_CLASS_IDS.get(candidate.asset_type, 0) for candidate in valid], dtype=np.int32)
        data = self._supervision_data(valid)
        return supervision_module.Detections(xyxy=boxes, confidence=confidences, class_id=class_ids, data=data)

    @staticmethod
    def _supervision_data(detections: list[AssetDetection]) -> dict[str, Any]:
        return {
            "asset_type": np.asarray([candidate.asset_type for candidate in detections], dtype=object),
            "template_name": np.asarray([candidate.template_name for candidate in detections], dtype=object),
            "center_x": np.asarray([candidate.center_x for candidate in detections], dtype=np.int32),
            "center_y": np.asarray([candidate.center_y for candidate in detections], dtype=np.int32),
            "width": np.asarray([candidate.width for candidate in detections], dtype=np.int32),
            "height": np.asarray([candidate.height for candidate in detections], dtype=np.int32),
            "target_x": np.asarray([_target_coordinate(candidate.target_x, candidate.center_x) for candidate in detections], dtype=np.int32),
            "target_y": np.asarray([_target_coordinate(candidate.target_y, candidate.center_y) for candidate in detections], dtype=np.int32),
        }

    def _from_supervision_detections(self, detections: Any) -> list[TrackedAsset]:
        data = getattr(detections, "data", {}) or {}
        confidence_values = getattr(detections, "confidence", None)
        class_values = getattr(detections, "class_id", None)
        tracker_values = getattr(detections, "tracker_id", None)
        tracked_assets = []
        for index in range(min(len(detections), self.max_detections)):
            tracked_assets.append(self._tracked_asset_from_detection(detections, data, confidence_values, class_values, tracker_values, index))
        return sorted(tracked_assets, key=lambda item: item.confidence, reverse=True)

    def _tracked_asset_from_detection(self, detections: Any, data: dict[str, Any], confidences: Any, class_ids: Any, tracker_ids: Any, index: int) -> TrackedAsset:
        xyxy = detections.xyxy[index]
        class_id = _int_value(_sequence_value(class_ids, index), 0)
        asset_type = _string_value(_sequence_value(data.get("asset_type"), index), ASSET_CLASS_NAMES.get(class_id, "red_icon"))
        template_name = _string_value(_sequence_value(data.get("template_name"), index), asset_type)
        width = _int_value(_sequence_value(data.get("width"), index), int(xyxy[2] - xyxy[0]))
        height = _int_value(_sequence_value(data.get("height"), index), int(xyxy[3] - xyxy[1]))
        center_x = _int_value(_sequence_value(data.get("center_x"), index), int((xyxy[0] + xyxy[2]) / 2))
        center_y = _int_value(_sequence_value(data.get("center_y"), index), int((xyxy[1] + xyxy[3]) / 2))
        target_x = _int_value(_sequence_value(data.get("target_x"), index), center_x)
        target_y = _int_value(_sequence_value(data.get("target_y"), index), center_y)
        return TrackedAsset(
            asset_type=asset_type,
            template_name=template_name,
            tracker_id=_int_value(_sequence_value(tracker_ids, index), -1),
            confidence=_float_value(_sequence_value(confidences, index)),
            center_x=center_x,
            center_y=center_y,
            width=max(1, width),
            height=max(1, height),
            target_x=target_x,
            target_y=target_y,
        )

    def _fallback_assets(self, detections: list[AssetDetection]) -> list[TrackedAsset]:
        assets = []
        for candidate in detections[: self.max_detections]:
            if _valid_detection(candidate):
                assets.append(
                    TrackedAsset(
                        candidate.asset_type,
                        candidate.template_name,
                        -1,
                        float(candidate.confidence),
                        int(candidate.center_x),
                        int(candidate.center_y),
                        int(candidate.width),
                        int(candidate.height),
                        _target_coordinate(candidate.target_x, candidate.center_x),
                        _target_coordinate(candidate.target_y, candidate.center_y),
                    )
                )
        return assets


class AssetTrackingOverlay:
    def __init__(self, window_capture: Any, tracker: AssetTracker) -> None:
        self.window_capture = window_capture
        self.tracker = tracker
        self.running = False
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, name="asset_tracking_overlay", daemon=True)
        self.thread.start()
        logger.info("Asset tracking overlay started")

    def stop(self) -> None:
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        logger.info("Asset tracking overlay stopped")

    def _run(self) -> None:
        try:
            import tkinter as tk

            root = tk.Tk()
            background = configure_overlay_root(root, "Asset Tracking Overlay")
            canvas = tk.Canvas(root, highlightthickness=0, background=background)
            configure_overlay_canvas(canvas, background)
            canvas.pack(fill="both", expand=True)
            self._overlay_loop(root, canvas)
        except Exception as exc:
            logger.error("Failed to create asset tracking overlay: %s", exc)
        finally:
            self.running = False

    def _overlay_loop(self, root: Any, canvas: Any) -> None:
        refresh_seconds = max(0.02, float(config.ASSET_TRACKING_OVERLAY_REFRESH_MS) / 1000.0)
        for _ in range(ASSET_TRACKING_OVERLAY_LOOP_ITERATION_LIMIT):
            if not self.running:
                break
            if not self._draw(root, canvas):
                break
            root.update_idletasks()
            root.update()
            time.sleep(refresh_seconds)
        self._destroy(root)

    def _draw(self, root: Any, canvas: Any) -> bool:
        try:
            if not position_overlay_over_rect(root, canvas, self.window_capture.get_window_rect()):
                return False
            canvas.delete("asset")
            assets = self.tracker.assets()[: int(config.ASSET_TRACKING_OVERLAY_MAX_ITEMS)]
            set_overlay_visible_regions(root, _asset_visible_regions(assets))
            for asset in assets:
                self._draw_asset(canvas, asset)
            return True
        except Exception as exc:
            logger.error("Error in asset tracking overlay loop: %s", exc)
            return False

    @staticmethod
    def _draw_asset(canvas: Any, asset: TrackedAsset) -> None:
        color = _asset_color(asset.asset_type)
        x1, y1, x2, y2 = _asset_bounds(asset)
        label = _asset_label(asset)
        canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags="asset")
        canvas.create_oval(asset.center_x - 2, asset.center_y - 2, asset.center_x + 2, asset.center_y + 2, fill=color, outline=color, tags="asset")
        canvas.create_text(x1 + 3, max(8, y1 - 8), text=label, anchor="w", fill=color, tags="asset")

    @staticmethod
    def _destroy(root: Any) -> None:
        destroy_overlay_root(root)


def _asset_bounds(asset: TrackedAsset) -> tuple[int, int, int, int]:
    half_width = max(1, int(asset.width)) // 2
    half_height = max(1, int(asset.height)) // 2
    return (
        int(asset.center_x) - half_width,
        int(asset.center_y) - half_height,
        int(asset.center_x) + half_width,
        int(asset.center_y) + half_height,
    )


def _asset_visible_regions(assets: list[TrackedAsset]) -> list[tuple[int, int, int, int]]:
    regions = []
    for asset in assets:
        x1, y1, x2, y2 = _asset_bounds(asset)
        width = max(1, int(x2 - x1))
        height = max(1, int(y2 - y1))
        regions.extend(_outlined_regions(x1, y1, width, height, 3))
        regions.append((int(asset.center_x) - 3, int(asset.center_y) - 3, 6, 6))
        regions.append((x1, max(0, y1 - 18), min(180, max(48, width + 80)), 18))
    return regions


def _outlined_regions(x: int, y: int, width: int, height: int, border: int) -> list[tuple[int, int, int, int]]:
    border = max(1, int(border))
    return [
        (x, y, width, border),
        (x, y + height - border, width, border),
        (x, y, border, height),
        (x + width - border, y, border, height),
    ]


def _asset_color(asset_type: str) -> str:
    colors = {
        "red_icon": config.ASSET_TRACKING_RED_ICON_COLOR,
        "upgrade_station": config.ASSET_TRACKING_UPGRADE_STATION_COLOR,
        "box": config.ASSET_TRACKING_BOX_COLOR,
    }
    return str(colors.get(asset_type, "#ffffff"))


def _asset_label(asset: TrackedAsset) -> str:
    track_label = f"#{asset.tracker_id}" if asset.tracker_id >= 0 else "#raw"
    return f"{asset.asset_type} {track_label} {asset.confidence:.2f}"
