import logging
import multiprocessing
import multiprocessing.queues
import queue
import threading
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

import config
from window_capture import (
    configure_overlay_canvas,
    configure_overlay_root,
    create_overlay_queue,
    destroy_overlay_root,
    position_overlay_over_rect,
    position_overlay_over_target,
    replace_queue_latest,
    set_overlay_visible_regions,
    start_overlay_process,
    stop_overlay_process,
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
ASSET_TRACKING_OVERLAY_LOOP_ITERATION_LIMIT = 2_147_483_647
ASSET_TRACKING_OVERLAY_QUEUE_DRAIN_LIMIT = 32

# Minimum inter-frame interval for the detection worker (caps at ~30 FPS).
# Prevents runaway CPU consumption when ASSET_TRACKING_INTERVAL is near zero.
DETECTION_WORKER_MIN_FRAME_INTERVAL_SECONDS = 0.033

# Seconds to wait for the detection worker process to stop gracefully.
DETECTION_WORKER_STOP_TIMEOUT_SECONDS = 2.0

# Maximum results drained from the detection result queue per tracker tick.
DETECTION_RESULT_DRAIN_LIMIT = 8

ASSET_CLASS_IDS = {"red_icon": 0, "upgrade_station": 1, "box": 2}
ASSET_CLASS_NAMES = {class_id: name for name, class_id in ASSET_CLASS_IDS.items()}

AssetOverlayItem = tuple[str, int, float, int, int, int, int]


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
    base_zone = ForbiddenZone(
        "FORBIDDEN_CLICK",
        config.FORBIDDEN_CLICK_X_MIN,
        config.FORBIDDEN_CLICK_X_MAX,
        config.FORBIDDEN_CLICK_Y_MIN,
        None,
    )
    numbered_zones = [
        ForbiddenZone(
            f"FORBIDDEN_ZONE_{zone_index}",
            int(x_min),
            int(x_max),
            int(y_min),
            int(y_max),
        )
        for zone_index, (x_min, x_max, y_min, y_max) in enumerate(
            config.NUMBERED_FORBIDDEN_ZONE_BOUNDS, start=1
        )
    ]
    return (base_zone, *numbered_zones)


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
        max(0, zone.x_min),
        min(frame_width - 1, zone.x_max),
        max(0, zone.y_min),
        min(frame_height - 1, raw_y_max),
    )


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


def _box_intersects_bounds(
    left: float,
    top: float,
    right: float,
    bottom: float,
    bounds: tuple[int, int, int, int],
) -> bool:
    x_min, x_max, y_min, y_max = bounds
    return left <= x_max and right >= x_min and top <= y_max and bottom >= y_min


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
    """Return True when none of the three zone checks block this detection."""
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


# ---------------------------------------------------------------------------
# Detection worker subprocess — runs vision pipeline in an isolated process
# to bypass the Python GIL entirely.
# ---------------------------------------------------------------------------

def _detection_worker_process_main(
    window_title: str,
    capture_max_y: int,
    frame_queue: multiprocessing.queues.Queue,
    result_queue: multiprocessing.queues.Queue,
    stop_event: Any,
    min_frame_interval: float,
    detection_provider_function: Any,
) -> None:
    """Entry point for the isolated detection subprocess.

    Captures frames from the target window, runs the detection pipeline, and
    enqueues results. Frames are captured at most once per *min_frame_interval*
    seconds. A maxsize=1 *frame_queue* ensures only the latest frame is passed
    to the detection pipeline; stale frames are silently dropped. Results are
    placed on *result_queue* for the tracker thread to consume.
    """
    try:
        import mss as mss_module
        import pywinctl as pywinctl_module
    except Exception as import_error:
        logger.error("Detection worker cannot import capture dependencies: %s", import_error)
        return

    try:
        screenshotter = mss_module.mss()
    except Exception as init_error:
        logger.error("Detection worker cannot initialise mss: %s", init_error)
        return

    last_frame_time = 0.0

    for _ in range(ASSET_TRACKING_LOOP_ITERATION_LIMIT):
        if stop_event.is_set():
            return

        now = time.monotonic()
        elapsed_since_last_frame = now - last_frame_time
        if elapsed_since_last_frame < min_frame_interval:
            time.sleep(min_frame_interval - elapsed_since_last_frame)
            continue

        screenshot = _capture_worker_frame(
            screenshotter, pywinctl_module, window_title, capture_max_y
        )
        if screenshot is None:
            time.sleep(min_frame_interval)
            continue

        last_frame_time = time.monotonic()

        detections = _safe_run_detection_provider(detection_provider_function, screenshot)
        frame_height, frame_width = screenshot.shape[:2]

        payload = (detections, frame_width, frame_height)
        _drop_and_enqueue(result_queue, payload)

    logger.error("Detection worker loop reached iteration limit")


def _capture_worker_frame(
    screenshotter: Any,
    pywinctl_module: Any,
    window_title: str,
    capture_max_y: int,
) -> np.ndarray | None:
    """Capture a single frame in the worker process. Returns None on failure."""
    try:
        windows = pywinctl_module.getWindowsWithTitle(window_title) or []
    except Exception as capture_error:
        logger.debug("Detection worker window lookup failed: %s", capture_error)
        return None

    live_windows = [window for window in windows if _worker_window_alive(window)]
    if not live_windows:
        return None

    target_window = None
    for window in live_windows:
        if getattr(window, "title", None) == window_title:
            target_window = window
            break
    if target_window is None:
        target_window = live_windows[0]

    try:
        box = target_window.box
        window_left = int(box.left if hasattr(box, "left") else box[0])
        window_top = int(box.top if hasattr(box, "top") else box[1])
        window_width = int(box.width if hasattr(box, "width") else box[2])
        window_height = int(box.height if hasattr(box, "height") else box[3])
    except Exception as bounds_error:
        logger.debug("Detection worker cannot read window bounds: %s", bounds_error)
        return None

    capture_height = min(window_height, int(capture_max_y))
    if window_width <= 0 or capture_height <= 0:
        return None

    try:
        raw_screenshot = screenshotter.grab({
            "left": window_left,
            "top": window_top,
            "width": window_width,
            "height": capture_height,
        })
        image_array = np.asarray(raw_screenshot)
    except Exception as grab_error:
        logger.debug("Detection worker frame grab failed: %s", grab_error)
        return None

    if image_array.ndim != 3 or image_array.shape[2] < 3:
        return None

    bgr_image = image_array[:, :, :3].astype(np.uint8, copy=False)
    if not bgr_image.flags.c_contiguous:
        bgr_image = np.ascontiguousarray(bgr_image)
    return bgr_image


def _worker_window_alive(window: Any) -> bool:
    alive_attribute = getattr(window, "isAlive", None)
    try:
        if callable(alive_attribute):
            return bool(alive_attribute())
        return alive_attribute is None or bool(alive_attribute)
    except Exception:
        return False


def _safe_run_detection_provider(
    detection_provider: Any, screenshot: np.ndarray
) -> list[AssetDetection]:
    try:
        result = detection_provider(screenshot)
        return result if isinstance(result, list) else []
    except Exception as provider_error:
        logger.debug("Detection worker provider failed: %s", provider_error)
        return []


def _drop_and_enqueue(target_queue: multiprocessing.queues.Queue, payload: Any) -> None:
    """Drain the queue and enqueue only the freshest payload."""
    for _ in range(DETECTION_RESULT_DRAIN_LIMIT):
        try:
            target_queue.get_nowait()
        except queue.Empty:
            break
    try:
        target_queue.put_nowait(payload)
    except queue.Full:
        pass


# ---------------------------------------------------------------------------
# AssetTracker — orchestrates the worker process and applies ByteTrack locally
# ---------------------------------------------------------------------------

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
        self._consumer_thread: threading.Thread | None = None
        self._worker_process: Any | None = None
        self._worker_stop_event: Any | None = None
        self._result_queue: multiprocessing.queues.Queue | None = None
        self._snapshot = AssetTrackingSnapshot(0, 0.0, 0, 0, ())

    @staticmethod
    def available() -> bool:
        return supervision_module is not None and hasattr(supervision_module, "ByteTrack")

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
        if self._consumer_thread is not None and self._consumer_thread.is_alive():
            return True
        self._stop_flag.clear()
        self._result_queue = self._spawn_detection_worker()
        self._consumer_thread = threading.Thread(
            target=self._consume_detection_results,
            name="asset_tracker_consumer",
            daemon=True,
        )
        self._consumer_thread.start()
        logger.info("Asset tracker started (multiprocessing detection worker)")
        return True

    def _spawn_detection_worker(self) -> multiprocessing.queues.Queue:
        spawn_context = multiprocessing.get_context("spawn")
        result_queue: multiprocessing.queues.Queue = spawn_context.Queue(maxsize=1)
        self._worker_stop_event = spawn_context.Event()
        min_frame_interval = max(
            DETECTION_WORKER_MIN_FRAME_INTERVAL_SECONDS,
            float(config.ASSET_TRACKING_INTERVAL),
        )
        self._worker_process = spawn_context.Process(
            target=_detection_worker_process_main,
            name="asset_detection_worker",
            args=(
                self.window_capture.window_title,
                int(config.ASSET_TRACKING_CAPTURE_Y),
                spawn_context.Queue(maxsize=1),
                result_queue,
                self._worker_stop_event,
                min_frame_interval,
                self.detection_provider,
            ),
            daemon=True,
        )
        self._worker_process.start()
        return result_queue

    def stop(self) -> None:
        self._stop_flag.set()
        self._stop_detection_worker()
        if self._consumer_thread is not None and self._consumer_thread.is_alive():
            self._consumer_thread.join(
                timeout=float(config.ASSET_TRACKING_THREAD_JOIN_TIMEOUT)
            )
        logger.info("Asset tracker stopped")

    def _stop_detection_worker(self) -> None:
        if self._worker_stop_event is not None:
            self._worker_stop_event.set()
        if self._worker_process is not None and self._worker_process.is_alive():
            self._worker_process.join(timeout=DETECTION_WORKER_STOP_TIMEOUT_SECONDS)
            if self._worker_process.is_alive():
                self._worker_process.terminate()
                self._worker_process.join(timeout=DETECTION_WORKER_STOP_TIMEOUT_SECONDS)
        self._worker_process = None
        self._worker_stop_event = None

    def reset(self) -> None:
        byte_tracker_reset = getattr(self._byte_tracker, "reset", None)
        if callable(byte_tracker_reset):
            byte_tracker_reset()
        with self._snapshot_lock:
            self._snapshot = AssetTrackingSnapshot(0, 0.0, 0, 0, ())

    def assets(self, asset_type: str | None = None) -> list[TrackedAsset]:
        with self._snapshot_lock:
            snapshot = self._snapshot
        snapshot_is_stale = (
            snapshot.timestamp <= 0
            or time.time() - snapshot.timestamp > float(config.ASSET_TRACKING_MAX_SNAPSHOT_AGE)
        )
        if snapshot_is_stale:
            return []
        filtered_assets = self._filter_trackable_assets(
            list(snapshot.assets), snapshot.width, snapshot.height
        )
        if asset_type is None:
            return filtered_assets
        return [asset for asset in filtered_assets if asset.asset_type == asset_type]

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

    def _consume_detection_results(self) -> None:
        """Consumer thread: drains the result queue and applies ByteTrack."""
        for _ in range(ASSET_TRACKING_LOOP_ITERATION_LIMIT):
            if self._stop_flag.is_set():
                return
            self._drain_and_apply_latest_result()
            if self._stop_flag.wait(timeout=0.005):
                return
        logger.error("Asset tracker consumer loop reached iteration limit")

    def _drain_and_apply_latest_result(self) -> None:
        latest_payload: tuple[list[AssetDetection], int, int] | None = None
        for _ in range(DETECTION_RESULT_DRAIN_LIMIT):
            try:
                candidate = self._result_queue.get_nowait()  # type: ignore[union-attr]
                latest_payload = candidate
            except queue.Empty:
                break
        if latest_payload is None:
            return
        raw_detections, frame_width, frame_height = latest_payload
        self._apply_detections(raw_detections, frame_width, frame_height)

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
            filtered_detections, key=lambda detection: detection.confidence, reverse=True
        )[: self.max_detections]
        tracked_assets = tuple(
            self._filter_trackable_assets(
                self._track_detections(prioritized_detections), frame_width, frame_height
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
            [_candidate_xyxy(candidate) for candidate in valid_detections], dtype=np.float32
        ).reshape((-1, 4))
        confidence_values = np.asarray(
            [candidate.confidence for candidate in valid_detections], dtype=np.float32
        )
        class_id_values = np.asarray(
            [ASSET_CLASS_IDS.get(candidate.asset_type, 0) for candidate in valid_detections],
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
                [candidate.asset_type for candidate in detections], dtype=object
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
                detections, detection_data, confidence_array,
                class_id_array, tracker_id_array, detection_index,
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
            ASSET_CLASS_NAMES.get(class_id, "red_icon"),
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


# ---------------------------------------------------------------------------
# Overlay subsystems (unchanged API; internal cleanup only)
# ---------------------------------------------------------------------------

class AssetTrackingOverlay:
    def __init__(self, window_capture: Any, tracker: AssetTracker) -> None:
        self.window_capture = window_capture
        self.tracker = tracker
        self.running = False
        self.thread: threading.Thread | None = None
        self.process: Any | None = None
        self.stop_event: Any | None = None
        self.payload_queue: Any | None = None

    def start(self) -> None:
        if self.running:
            return
        try:
            self.payload_queue = create_overlay_queue(maxsize=2)
            refresh_milliseconds = max(20, int(config.ASSET_TRACKING_OVERLAY_REFRESH_MS))
            self.process, self.stop_event = start_overlay_process(
                "asset_tracking_overlay",
                _run_asset_tracking_overlay_process,
                self.window_capture.window_title,
                self.payload_queue,
                refresh_milliseconds,
            )
            self.running = True
            self.thread = threading.Thread(
                target=self._feed_overlay_queue,
                name="asset_tracking_overlay_feeder",
                daemon=True,
            )
            self.thread.start()
        except Exception as startup_error:
            self.running = False
            logger.error(
                "Failed to start asset tracking overlay process: %s", startup_error
            )
            self.stop()
            return
        logger.info("Asset tracking overlay started")

    def stop(self) -> None:
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        stop_overlay_process(self.process, self.stop_event)
        self.process = None
        self.stop_event = None
        self.payload_queue = None
        self.thread = None
        logger.info("Asset tracking overlay stopped")

    def _feed_overlay_queue(self) -> None:
        refresh_seconds = max(
            0.02, float(config.ASSET_TRACKING_OVERLAY_REFRESH_MS) / 1000.0
        )
        for _ in range(ASSET_TRACKING_OVERLAY_LOOP_ITERATION_LIMIT):
            if not self.running:
                return
            if self.payload_queue is not None:
                overlay_payload = _asset_overlay_payload(self.tracker.assets())
                replace_queue_latest(self.payload_queue, overlay_payload)
            if self.stop_event is not None and self.stop_event.is_set():
                return
            time.sleep(refresh_seconds)
        logger.error("Asset tracking overlay feeder reached iteration limit")

    def _run(self) -> None:
        if self.payload_queue is None or self.stop_event is None:
            return
        _ = (configure_overlay_root, configure_overlay_canvas)
        refresh_milliseconds = max(20, int(config.ASSET_TRACKING_OVERLAY_REFRESH_MS))
        _run_asset_tracking_overlay_process(
            self.window_capture.window_title,
            self.payload_queue,
            refresh_milliseconds,
            self.stop_event,
        )

    def _draw(self, root: Any, canvas: Any) -> bool:
        try:
            _ = (position_overlay_over_rect, set_overlay_visible_regions)
            overlay_items = _asset_overlay_payload(self.tracker.assets())
            _draw_asset_overlay(
                root, canvas, self.window_capture.window_title, overlay_items
            )
            return True
        except Exception as draw_error:
            logger.error("Error in asset tracking overlay loop: %s", draw_error)
            return False


def _run_asset_tracking_overlay_process(
    window_title: str, payload_queue: Any, refresh_ms: int, stop_event: Any
) -> None:
    try:
        import tkinter as tk

        root = tk.Tk()
        background = configure_overlay_root(root, "Asset Tracking Overlay")
        canvas = tk.Canvas(root, highlightthickness=0, background=background)
        configure_overlay_canvas(canvas, background)
        canvas.pack(fill="both", expand=True)
        latest_items: tuple[AssetOverlayItem, ...] = ()
        _schedule_asset_overlay_draw(
            root,
            canvas,
            window_title,
            payload_queue,
            latest_items,
            max(20, int(refresh_ms)),
            stop_event,
        )
        root.mainloop()
    except Exception as overlay_error:
        logger.error("Failed to create asset tracking overlay: %s", overlay_error)
    finally:
        if "root" in locals():
            destroy_overlay_root(root)


def _schedule_asset_overlay_draw(
    root: Any,
    canvas: Any,
    window_title: str,
    payload_queue: Any,
    latest_items: tuple[AssetOverlayItem, ...],
    refresh_ms: int,
    stop_event: Any,
) -> None:
    if stop_event.is_set():
        root.quit()
        return
    next_items = _latest_asset_overlay_payload(payload_queue, latest_items)
    _draw_asset_overlay(root, canvas, window_title, next_items)
    root.after(
        refresh_ms,
        _schedule_asset_overlay_draw,
        root,
        canvas,
        window_title,
        payload_queue,
        next_items,
        refresh_ms,
        stop_event,
    )


def _latest_asset_overlay_payload(
    payload_queue: Any, fallback: tuple[AssetOverlayItem, ...]
) -> tuple[AssetOverlayItem, ...]:
    latest_items = fallback
    for _ in range(ASSET_TRACKING_OVERLAY_QUEUE_DRAIN_LIMIT):
        try:
            latest_items = tuple(payload_queue.get_nowait())
        except queue.Empty:
            break
    return latest_items


def _draw_asset_overlay(
    root: Any, canvas: Any, window_title: str, items: tuple[AssetOverlayItem, ...]
) -> None:
    if not position_overlay_over_target(root, canvas, window_title):
        return
    canvas.delete("asset")
    set_overlay_visible_regions(root, _asset_item_visible_regions(items))
    for overlay_item in items:
        _draw_asset_item(canvas, overlay_item)


def _draw_asset_item(canvas: Any, item: AssetOverlayItem) -> None:
    asset_type, tracker_id, confidence, center_x, center_y, width, height = item
    color = _asset_color(asset_type)
    x1, y1, x2, y2 = _asset_item_bounds(item)
    label = _asset_item_label(asset_type, tracker_id, confidence)
    canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags="asset")
    canvas.create_oval(
        center_x - 2,
        center_y - 2,
        center_x + 2,
        center_y + 2,
        fill=color,
        outline=color,
        tags="asset",
    )
    canvas.create_text(
        x1 + 3, max(8, y1 - 8), text=label, anchor="w", fill=color, tags="asset"
    )


def _asset_overlay_payload(assets: list[TrackedAsset]) -> tuple[AssetOverlayItem, ...]:
    max_items = max(1, int(config.ASSET_TRACKING_OVERLAY_MAX_ITEMS))
    return tuple(_asset_overlay_item(asset) for asset in assets[:max_items])


def _asset_overlay_item(asset: TrackedAsset) -> AssetOverlayItem:
    return (
        str(asset.asset_type),
        int(asset.tracker_id),
        float(asset.confidence),
        int(asset.center_x),
        int(asset.center_y),
        max(1, int(asset.width)),
        max(1, int(asset.height)),
    )


def _asset_bounds(asset: TrackedAsset) -> tuple[int, int, int, int]:
    half_width = max(1, int(asset.width)) // 2
    half_height = max(1, int(asset.height)) // 2
    return (
        int(asset.center_x) - half_width,
        int(asset.center_y) - half_height,
        int(asset.center_x) + half_width,
        int(asset.center_y) + half_height,
    )


def _asset_visible_regions(
    assets: list[TrackedAsset],
) -> list[tuple[int, int, int, int]]:
    regions = []
    for asset in assets:
        x1, y1, x2, y2 = _asset_bounds(asset)
        region_width = max(1, int(x2 - x1))
        region_height = max(1, int(y2 - y1))
        regions.extend(_outlined_regions(x1, y1, region_width, region_height, 3))
        regions.append((int(asset.center_x) - 3, int(asset.center_y) - 3, 6, 6))
        regions.append((x1, max(0, y1 - 18), min(180, max(48, region_width + 80)), 18))
    return regions


def _asset_item_bounds(item: AssetOverlayItem) -> tuple[int, int, int, int]:
    _, _, _, center_x, center_y, width, height = item
    half_width = max(1, int(width)) // 2
    half_height = max(1, int(height)) // 2
    return (
        int(center_x) - half_width,
        int(center_y) - half_height,
        int(center_x) + half_width,
        int(center_y) + half_height,
    )


def _asset_item_visible_regions(
    items: tuple[AssetOverlayItem, ...],
) -> list[tuple[int, int, int, int]]:
    regions = []
    for overlay_item in items:
        _, _, _, center_x, center_y, _, _ = overlay_item
        x1, y1, x2, y2 = _asset_item_bounds(overlay_item)
        region_width = max(1, int(x2 - x1))
        region_height = max(1, int(y2 - y1))
        regions.extend(_outlined_regions(x1, y1, region_width, region_height, 3))
        regions.append((int(center_x) - 3, int(center_y) - 3, 6, 6))
        regions.append(
            (x1, max(0, y1 - 18), min(180, max(48, region_width + 80)), 18)
        )
    return regions


def _outlined_regions(
    x: int, y: int, width: int, height: int, border: int
) -> list[tuple[int, int, int, int]]:
    border = max(1, int(border))
    return [
        (x, y, width, border),
        (x, y + height - border, width, border),
        (x, y, border, height),
        (x + width - border, y, border, height),
    ]


def _asset_color(asset_type: str) -> str:
    color_map = {
        "red_icon": config.ASSET_TRACKING_RED_ICON_COLOR,
        "upgrade_station": config.ASSET_TRACKING_UPGRADE_STATION_COLOR,
        "box": config.ASSET_TRACKING_BOX_COLOR,
    }
    return str(color_map.get(asset_type, "#ffffff"))


def _asset_label(asset: TrackedAsset) -> str:
    track_label = f"#{asset.tracker_id}" if asset.tracker_id >= 0 else "#raw"
    return f"{asset.asset_type} {track_label} {asset.confidence:.2f}"


def _asset_item_label(asset_type: str, tracker_id: int, confidence: float) -> str:
    track_label = f"#{tracker_id}" if tracker_id >= 0 else "#raw"
    return f"{asset_type} {track_label} {confidence:.2f}"
