import logging
import queue
import threading
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

import config
from domain import (
    ASSET_CLASS_IDS as DOMAIN_ASSET_CLASS_IDS,
    ASSET_CLASS_NAMES as DOMAIN_ASSET_CLASS_NAMES,
    AssetType,
    asset_class_id_for,
    asset_class_name_for,
    asset_type_value,
)
from forbidden_zones import (
    ForbiddenZone,
    bounded_forbidden_zone,
    box_intersects_bounds,
    box_intersects_forbidden_zones,
    configured_forbidden_zones,
    point_blocked_by_forbidden_zones,
    point_inside_bounds,
)

logger = logging.getLogger(__name__)

_SUPERVISION_IMPORT_ERROR: Exception | None

try:
    import supervision as supervision_module
except Exception as supervision_import_exception:
    supervision_module = None  # type: ignore[assignment]
    _SUPERVISION_IMPORT_ERROR = supervision_import_exception
else:
    _SUPERVISION_IMPORT_ERROR = None

ASSET_TRACKING_LOOP_ITERATION_LIMIT = 2_147_483_647
FRAME_MIN_INTERVAL_SECONDS = 1.0 / 60.0
DETECTION_COOLDOWN_MAX_FRAME_INTERVALS = 60.0

DETECTION_RESULT_DRAIN_LIMIT = 8
IMAGE_MIN_CHANNELS = 3

ASSET_CLASS_IDS = DOMAIN_ASSET_CLASS_IDS
ASSET_CLASS_NAMES = DOMAIN_ASSET_CLASS_NAMES

AssetOverlayItem = tuple[str, int, float, int, int, int, int]


@dataclass(frozen=True)
class AssetDetection:
    asset_type: AssetType | str
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
    asset_type: AssetType | str
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
    return configured_forbidden_zones()


def _candidate_xyxy(candidate: AssetDetection) -> tuple[float, float, float, float]:
    return _centered_xyxy(
        candidate.center_x, candidate.center_y, candidate.width, candidate.height
    )


def _asset_xyxy(asset: TrackedAsset) -> tuple[float, float, float, float]:
    return _centered_xyxy(asset.center_x, asset.center_y, asset.width, asset.height)


def _centered_xyxy(
    center_x: int, center_y: int, width: int, height: int
) -> tuple[float, float, float, float]:
    half_width = max(1.0, float(width)) / 2.0
    half_height = max(1.0, float(height)) / 2.0
    return (
        float(center_x) - half_width,
        float(center_y) - half_height,
        float(center_x) + half_width,
        float(center_y) + half_height,
    )


def _has_finite_positive_dimensions(
    confidence: float,
    width: int,
    height: int,
    center_x: int,
    center_y: int,
) -> bool:
    numeric_values = (confidence, center_x, center_y, width, height)
    return (
        all(np.isfinite(float(numeric_value)) for numeric_value in numeric_values)
        and width > 0
        and height > 0
    )


def _valid_detection(candidate: AssetDetection) -> bool:
    return _has_finite_positive_dimensions(
        candidate.confidence,
        candidate.width,
        candidate.height,
        candidate.center_x,
        candidate.center_y,
    )


def _valid_asset(asset: TrackedAsset) -> bool:
    return _has_finite_positive_dimensions(
        asset.confidence,
        asset.width,
        asset.height,
        asset.center_x,
        asset.center_y,
    )


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
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _target_coordinate(value: int | None, fallback: int) -> int:
    return int(fallback if value is None else value)


def _bounded_forbidden_zone(
    zone: ForbiddenZone, frame_width: int, frame_height: int
) -> tuple[int, int, int, int] | None:
    return bounded_forbidden_zone(zone, frame_width, frame_height)


def _point_blocked_by_forbidden_zones(
    x: int,
    y: int,
    zones: tuple[ForbiddenZone, ...],
    frame_width: int,
    frame_height: int,
) -> bool:
    return point_blocked_by_forbidden_zones(x, y, zones, frame_width, frame_height)


def _point_inside_bounds(x: int, y: int, bounds: tuple[int, int, int, int]) -> bool:
    return point_inside_bounds(x, y, bounds)


def _box_intersects_forbidden_zones(
    box: tuple[float, float, float, float],
    zones: tuple[ForbiddenZone, ...],
    frame_width: int,
    frame_height: int,
) -> bool:
    return box_intersects_forbidden_zones(box, zones, frame_width, frame_height)


def _box_intersects_bounds(
    left: float,
    top: float,
    right: float,
    bottom: float,
    bounds: tuple[int, int, int, int],
) -> bool:
    return box_intersects_bounds(left, top, right, bottom, bounds)


def _detection_passes_zone_check(
    center_x: int,
    center_y: int,
    target_x: int,
    target_y: int,
    bounding_box: tuple[float, float, float, float],
    forbidden_zones: tuple[ForbiddenZone, ...],
    frame_width: int,
    frame_height: int,
) -> bool:
    center_blocked = _point_blocked_by_forbidden_zones(
        center_x, center_y, forbidden_zones, frame_width, frame_height
    )
    if center_blocked:
        return False
    target_blocked = _point_blocked_by_forbidden_zones(
        target_x, target_y, forbidden_zones, frame_width, frame_height
    )
    if target_blocked:
        return False
    return not _box_intersects_forbidden_zones(
        bounding_box, forbidden_zones, frame_width, frame_height
    )


def _drop_and_replace_frame_queue(
    frame_queue: queue.Queue,  # type: ignore[type-arg]
    frame: np.ndarray,
) -> None:
    for _ in range(DETECTION_RESULT_DRAIN_LIMIT):
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            break
    try:
        frame_queue.put_nowait(frame)
    except queue.Full:
        pass


class AssetTracker:
    def __init__(
        self, window_capture: Any, detection_provider: DetectionProvider
    ) -> None:
        self.window_capture = window_capture
        self.detection_provider = detection_provider
        self.max_detections = max(1, int(config.ASSET_TRACKING_MAX_DETECTIONS))
        self._byte_tracker = self._create_byte_tracker()
        self._forbidden_zones = _configured_forbidden_zones()
        self._stop_flag = threading.Event()
        self._snapshot_lock = threading.RLock()
        self._frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._capture_thread: threading.Thread | None = None
        self._detection_thread: threading.Thread | None = None
        self._snapshot = AssetTrackingSnapshot(0, 0.0, 0, 0, ())

    @staticmethod
    def available() -> bool:
        return supervision_module is not None and hasattr(
            supervision_module, "ByteTrack"
        )

    def _create_byte_tracker(self) -> Any | None:
        if not self.available():
            if _SUPERVISION_IMPORT_ERROR is not None:
                logger.warning(
                    "Supervision ByteTrack unavailable: %s", _SUPERVISION_IMPORT_ERROR
                )
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", FutureWarning)
            if supervision_module is None:
                return None
            return supervision_module.ByteTrack(
                track_activation_threshold=float(
                    config.ASSET_TRACKING_TRACK_ACTIVATION_THRESHOLD
                ),
                lost_track_buffer=int(config.ASSET_TRACKING_LOST_TRACK_BUFFER),
                minimum_matching_threshold=float(
                    config.ASSET_TRACKING_MINIMUM_MATCHING_THRESHOLD
                ),
                frame_rate=float(config.ASSET_TRACKING_FRAME_RATE),
                minimum_consecutive_frames=int(
                    config.ASSET_TRACKING_MINIMUM_CONSECUTIVE_FRAMES
                ),
            )

    def start(self) -> bool:
        if not config.ASSET_TRACKING_ENABLED:
            return False
        if self._capture_thread is not None and self._capture_thread.is_alive():
            return True
        self._stop_flag.clear()
        self._capture_thread = threading.Thread(
            target=self._run_capture_thread,
            name="asset_capture",
            daemon=True,
        )
        self._detection_thread = threading.Thread(
            target=self._run_detection_thread,
            name="asset_detection",
            daemon=True,
        )
        self._capture_thread.start()
        self._detection_thread.start()
        logger.info("Asset tracker started (dual-thread pipeline)")
        return True

    def stop(self) -> None:
        self._stop_flag.set()
        join_timeout = float(config.ASSET_TRACKING_THREAD_JOIN_TIMEOUT)
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=join_timeout)
        if self._detection_thread is not None and self._detection_thread.is_alive():
            self._detection_thread.join(timeout=join_timeout)
        logger.info("Asset tracker stopped")

    def reset(self) -> None:
        byte_tracker_reset = getattr(self._byte_tracker, "reset", None)
        if callable(byte_tracker_reset):
            byte_tracker_reset()
        with self._snapshot_lock:
            self._snapshot = AssetTrackingSnapshot(0, 0.0, 0, 0, ())

    def assets(self, asset_type: AssetType | str | None = None) -> list[TrackedAsset]:
        with self._snapshot_lock:
            snapshot = self._snapshot
        snapshot_is_stale = (
            snapshot.timestamp <= 0
            or time.time() - snapshot.timestamp
            > float(config.ASSET_TRACKING_MAX_SNAPSHOT_AGE)
        )
        if snapshot_is_stale:
            return []
        filtered_assets = self._filter_trackable_assets(
            list(snapshot.assets), snapshot.width, snapshot.height
        )
        if asset_type is None:
            return filtered_assets
        requested_asset_type = asset_type_value(asset_type)
        return [
            asset
            for asset in filtered_assets
            if asset_type_value(asset.asset_type) == requested_asset_type
        ]

    def snapshot(self) -> AssetTrackingSnapshot:
        with self._snapshot_lock:
            snapshot = self._snapshot
        filtered_assets = tuple(
            self._filter_trackable_assets(
                list(snapshot.assets), snapshot.width, snapshot.height
            )
        )
        if len(filtered_assets) == len(snapshot.assets):
            return snapshot
        return AssetTrackingSnapshot(
            snapshot.frame_number,
            snapshot.timestamp,
            snapshot.width,
            snapshot.height,
            filtered_assets,
        )

    def _run_capture_thread(self) -> None:
        screenshotter = self._create_private_screenshotter()
        if screenshotter is None:
            logger.error(
                "Asset capture thread could not initialise screenshotter; exiting"
            )
            return
        try:
            frame_interval = max(
                FRAME_MIN_INTERVAL_SECONDS, float(config.ASSET_TRACKING_INTERVAL)
            )
            last_capture_time = 0.0
            for _ in range(ASSET_TRACKING_LOOP_ITERATION_LIMIT):
                if self._stop_flag.is_set():
                    return
                now = time.monotonic()
                elapsed = now - last_capture_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                    continue
                frame = self._capture_frame(screenshotter)
                if frame is not None:
                    _drop_and_replace_frame_queue(self._frame_queue, frame)
                    last_capture_time = time.monotonic()
                else:
                    time.sleep(frame_interval)
            logger.error("Asset capture thread reached iteration limit")
        finally:
            self._close_private_screenshotter(screenshotter)

    @staticmethod
    def _close_private_screenshotter(screenshotter: Any) -> None:
        close_screenshotter = getattr(screenshotter, "close", None)
        if not callable(close_screenshotter):
            return
        try:
            close_screenshotter()
        except Exception as close_error:
            logger.debug("Asset tracker screenshotter close failed: %s", close_error)

    def _create_private_screenshotter(self) -> Any | None:
        try:
            import mss as mss_module

            return mss_module.mss()
        except Exception as screenshotter_error:
            logger.error(
                "Asset tracker cannot create screenshotter: %s", screenshotter_error
            )
            return None

    def _capture_frame(self, screenshotter: Any) -> np.ndarray | None:
        try:
            window_x, window_y, window_width, window_height = (
                self.window_capture.get_window_rect()
            )
        except Exception as bounds_error:
            logger.debug("Asset capture could not read window bounds: %s", bounds_error)
            return None
        capture_height = min(window_height, int(config.ASSET_TRACKING_CAPTURE_Y))
        if window_width <= 0 or capture_height <= 0:
            return None
        try:
            raw_capture = screenshotter.grab(
                {
                    "left": window_x,
                    "top": window_y,
                    "width": window_width,
                    "height": capture_height,
                }
            )
            image_array = np.asarray(raw_capture)
        except Exception as grab_error:
            logger.debug("Asset capture frame grab failed: %s", grab_error)
            return None
        if image_array.ndim != IMAGE_MIN_CHANNELS:
            return None
        if image_array.shape[2] < IMAGE_MIN_CHANNELS:
            return None
        bgr_frame = image_array[:, :, :IMAGE_MIN_CHANNELS].astype(np.uint8, copy=False)
        if not bgr_frame.flags.c_contiguous:
            bgr_frame = np.ascontiguousarray(bgr_frame)
        return bgr_frame

    def _run_detection_thread(self) -> None:
        frame_interval = max(
            FRAME_MIN_INTERVAL_SECONDS, float(config.ASSET_TRACKING_INTERVAL)
        )
        for _ in range(ASSET_TRACKING_LOOP_ITERATION_LIMIT):
            if self._stop_flag.is_set():
                return
            frame = self._read_latest_frame(frame_interval)
            if frame is None:
                continue
            frame_height, frame_width = frame.shape[:2]
            detection_started_at = time.perf_counter()
            raw_detections = self._safe_call_detection_provider(frame)
            self._apply_detections(raw_detections, frame_width, frame_height)
            detection_duration = time.perf_counter() - detection_started_at
            if not self._wait_after_detection(detection_duration, frame_interval):
                return
        logger.error("Asset detection thread reached iteration limit")

    def _wait_after_detection(
        self, detection_duration: float, frame_interval: float
    ) -> bool:
        cooldown_duration = self._detection_cooldown_duration(
            detection_duration, frame_interval
        )
        if cooldown_duration <= 0.0:
            return True
        return not self._stop_flag.wait(cooldown_duration)

    @staticmethod
    def _detection_cooldown_duration(
        detection_duration: float, frame_interval: float
    ) -> float:
        if detection_duration <= frame_interval:
            return 0.0
        maximum_cooldown = frame_interval * DETECTION_COOLDOWN_MAX_FRAME_INTERVALS
        return min(detection_duration - frame_interval, maximum_cooldown)

    def _read_latest_frame(self, timeout: float) -> np.ndarray | None:
        latest_frame: np.ndarray | None = None
        try:
            latest_frame = self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        for _ in range(DETECTION_RESULT_DRAIN_LIMIT):
            try:
                latest_frame = self._frame_queue.get_nowait()
            except queue.Empty:
                break
        return latest_frame

    def _safe_call_detection_provider(self, frame: np.ndarray) -> list[AssetDetection]:
        try:
            result = self.detection_provider(frame)
            return result if isinstance(result, list) else []
        except Exception as provider_error:
            logger.debug("Detection provider raised an exception: %s", provider_error)
            return []

    def _apply_detections(
        self,
        raw_detections: list[AssetDetection],
        frame_width: int,
        frame_height: int,
    ) -> None:
        filtered_detections = self._filter_trackable_detections(
            raw_detections, frame_width, frame_height
        )
        prioritized_detections = sorted(
            filtered_detections,
            key=lambda detection: detection.confidence,
            reverse=True,
        )[: self.max_detections]
        tracked_assets = tuple(
            self._filter_trackable_assets(
                self._track_detections(prioritized_detections),
                frame_width,
                frame_height,
            )
        )
        with self._snapshot_lock:
            new_frame_number = self._snapshot.frame_number + 1
            self._snapshot = AssetTrackingSnapshot(
                new_frame_number, time.time(), frame_width, frame_height, tracked_assets
            )

    def _filter_trackable_detections(
        self, detections: list[AssetDetection], frame_width: int, frame_height: int
    ) -> list[AssetDetection]:
        return [
            candidate
            for candidate in detections
            if self._detection_is_trackable(candidate, frame_width, frame_height)
        ]

    def _detection_is_trackable(
        self, candidate: AssetDetection, frame_width: int, frame_height: int
    ) -> bool:
        if not _valid_detection(candidate):
            return False
        target_x = _target_coordinate(candidate.target_x, candidate.center_x)
        target_y = _target_coordinate(candidate.target_y, candidate.center_y)
        return _detection_passes_zone_check(
            candidate.center_x,
            candidate.center_y,
            target_x,
            target_y,
            _candidate_xyxy(candidate),
            self._forbidden_zones,
            frame_width,
            frame_height,
        )

    def _filter_trackable_assets(
        self, assets: list[TrackedAsset], frame_width: int, frame_height: int
    ) -> list[TrackedAsset]:
        return [
            asset
            for asset in assets
            if self._asset_is_trackable(asset, frame_width, frame_height)
        ]

    def _asset_is_trackable(
        self, asset: TrackedAsset, frame_width: int, frame_height: int
    ) -> bool:
        if not _valid_asset(asset):
            return False
        target_x = _target_coordinate(asset.target_x, asset.center_x)
        target_y = _target_coordinate(asset.target_y, asset.center_y)
        return _detection_passes_zone_check(
            asset.center_x,
            asset.center_y,
            target_x,
            target_y,
            _asset_xyxy(asset),
            self._forbidden_zones,
            frame_width,
            frame_height,
        )

    def _track_detections(self, detections: list[AssetDetection]) -> list[TrackedAsset]:
        if self._byte_tracker is None:
            return self._build_fallback_assets(detections)
        supervision_detections = self._to_supervision_detections(detections)
        try:
            tracked = self._byte_tracker.update_with_detections(supervision_detections)
        except Exception as tracking_error:
            logger.debug("ByteTrack update failed: %s", tracking_error)
            return self._build_fallback_assets(detections)
        return self._from_supervision_detections(tracked)

    def _to_supervision_detections(self, detections: list[AssetDetection]) -> Any:
        if supervision_module is None:
            raise RuntimeError("supervision is unavailable")
        valid_detections = [
            candidate for candidate in detections if _valid_detection(candidate)
        ]
        bounding_boxes = np.asarray(
            [_candidate_xyxy(candidate) for candidate in valid_detections],
            dtype=np.float32,
        ).reshape((-1, 4))
        confidence_values = np.asarray(
            [candidate.confidence for candidate in valid_detections], dtype=np.float32
        )
        class_id_values = np.asarray(
            [
                asset_class_id_for(candidate.asset_type)
                for candidate in valid_detections
            ],
            dtype=np.int32,
        )
        supervision_data = self._build_supervision_data(valid_detections)
        return supervision_module.Detections(
            xyxy=bounding_boxes,
            confidence=confidence_values,
            class_id=class_id_values,
            data=supervision_data,
        )

    @staticmethod
    def _build_supervision_data(detections: list[AssetDetection]) -> dict[str, Any]:
        return {
            "asset_type": np.asarray(
                [asset_type_value(candidate.asset_type) for candidate in detections],
                dtype=object,
            ),
            "template_name": np.asarray(
                [candidate.template_name for candidate in detections], dtype=object
            ),
            "center_x": np.asarray(
                [candidate.center_x for candidate in detections], dtype=np.int32
            ),
            "center_y": np.asarray(
                [candidate.center_y for candidate in detections], dtype=np.int32
            ),
            "width": np.asarray(
                [candidate.width for candidate in detections], dtype=np.int32
            ),
            "height": np.asarray(
                [candidate.height for candidate in detections], dtype=np.int32
            ),
            "target_x": np.asarray(
                [
                    _target_coordinate(candidate.target_x, candidate.center_x)
                    for candidate in detections
                ],
                dtype=np.int32,
            ),
            "target_y": np.asarray(
                [
                    _target_coordinate(candidate.target_y, candidate.center_y)
                    for candidate in detections
                ],
                dtype=np.int32,
            ),
        }

    def _from_supervision_detections(self, detections: Any) -> list[TrackedAsset]:
        detection_data = getattr(detections, "data", {}) or {}
        confidence_array = getattr(detections, "confidence", None)
        class_id_array = getattr(detections, "class_id", None)
        tracker_id_array = getattr(detections, "tracker_id", None)
        tracked_assets = [
            self._build_tracked_asset(
                detections,
                detection_data,
                confidence_array,
                class_id_array,
                tracker_id_array,
                detection_index,
            )
            for detection_index in range(min(len(detections), self.max_detections))
        ]
        return sorted(tracked_assets, key=lambda asset: asset.confidence, reverse=True)

    def _build_tracked_asset(
        self,
        detections: Any,
        detection_data: dict[str, Any],
        confidence_array: Any,
        class_id_array: Any,
        tracker_id_array: Any,
        detection_index: int,
    ) -> TrackedAsset:
        xyxy_box = detections.xyxy[detection_index]
        class_id = _int_value(_sequence_value(class_id_array, detection_index), 0)
        asset_type = _string_value(
            _sequence_value(detection_data.get("asset_type"), detection_index),
            asset_class_name_for(class_id),
        )
        template_name = _string_value(
            _sequence_value(detection_data.get("template_name"), detection_index),
            asset_type,
        )
        width = _int_value(
            _sequence_value(detection_data.get("width"), detection_index),
            int(xyxy_box[2] - xyxy_box[0]),
        )
        height = _int_value(
            _sequence_value(detection_data.get("height"), detection_index),
            int(xyxy_box[3] - xyxy_box[1]),
        )
        center_x = _int_value(
            _sequence_value(detection_data.get("center_x"), detection_index),
            int((xyxy_box[0] + xyxy_box[2]) / 2),
        )
        center_y = _int_value(
            _sequence_value(detection_data.get("center_y"), detection_index),
            int((xyxy_box[1] + xyxy_box[3]) / 2),
        )
        target_x = _int_value(
            _sequence_value(detection_data.get("target_x"), detection_index), center_x
        )
        target_y = _int_value(
            _sequence_value(detection_data.get("target_y"), detection_index), center_y
        )
        return TrackedAsset(
            asset_type=asset_type,
            template_name=template_name,
            tracker_id=_int_value(
                _sequence_value(tracker_id_array, detection_index), -1
            ),
            confidence=_float_value(_sequence_value(confidence_array, detection_index)),
            center_x=center_x,
            center_y=center_y,
            width=max(1, width),
            height=max(1, height),
            target_x=target_x,
            target_y=target_y,
        )

    def _build_fallback_assets(
        self, detections: list[AssetDetection]
    ) -> list[TrackedAsset]:
        fallback_assets = []
        for candidate in detections[: self.max_detections]:
            if not _valid_detection(candidate):
                continue
            fallback_assets.append(
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
        return fallback_assets
