import logging
import threading
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

import config

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
class AssetDetection:
    asset_type: str
    template_name: str
    confidence: float
    center_x: int
    center_y: int
    width: int
    height: int


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


@dataclass(frozen=True)
class AssetTrackingSnapshot:
    frame_number: int
    timestamp: float
    width: int
    height: int
    assets: tuple[TrackedAsset, ...]


DetectionProvider = Callable[[np.ndarray], list[AssetDetection]]


def _candidate_xyxy(candidate: AssetDetection) -> tuple[float, float, float, float]:
    half_width = max(1.0, float(candidate.width)) / 2.0
    half_height = max(1.0, float(candidate.height)) / 2.0
    return (
        float(candidate.center_x) - half_width,
        float(candidate.center_y) - half_height,
        float(candidate.center_x) + half_width,
        float(candidate.center_y) + half_height,
    )


def _valid_detection(candidate: AssetDetection) -> bool:
    values = (candidate.confidence, candidate.center_x, candidate.center_y, candidate.width, candidate.height)
    return all(np.isfinite(float(value)) for value in values) and candidate.width > 0 and candidate.height > 0


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


class AssetTracker:
    def __init__(self, window_capture: Any, detection_provider: DetectionProvider) -> None:
        self.window_capture = window_capture
        self.detection_provider = detection_provider
        self.interval = max(0.01, float(config.ASSET_TRACKING_INTERVAL))
        self.max_detections = max(1, int(config.ASSET_TRACKING_MAX_DETECTIONS))
        self._tracker = self._create_tracker()
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
        assets = list(snapshot.assets)
        if asset_type is None:
            return assets
        return [asset for asset in assets if asset.asset_type == asset_type]

    def snapshot(self) -> AssetTrackingSnapshot:
        with self._lock:
            return self._snapshot

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
        detections = self.detection_provider(screenshot)
        detections = sorted(detections, key=lambda item: item.confidence, reverse=True)[: self.max_detections]
        assets = tuple(self._track_detections(detections))
        height, width = screenshot.shape[:2]
        with self._lock:
            self._snapshot = AssetTrackingSnapshot(self._snapshot.frame_number + 1, time.time(), width, height, assets)

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
        return TrackedAsset(
            asset_type=asset_type,
            template_name=template_name,
            tracker_id=_int_value(_sequence_value(tracker_ids, index), -1),
            confidence=_float_value(_sequence_value(confidences, index)),
            center_x=center_x,
            center_y=center_y,
            width=max(1, width),
            height=max(1, height),
        )

    def _fallback_assets(self, detections: list[AssetDetection]) -> list[TrackedAsset]:
        assets = []
        for candidate in detections[: self.max_detections]:
            if _valid_detection(candidate):
                assets.append(TrackedAsset(candidate.asset_type, candidate.template_name, -1, float(candidate.confidence), int(candidate.center_x), int(candidate.center_y), int(candidate.width), int(candidate.height)))
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
            root.title("Asset Tracking Overlay")
            root.resizable(False, False)
            self._set_topmost(root)
            canvas = tk.Canvas(root, highlightthickness=0, background="#101010")
            canvas.pack(fill="both", expand=True)
            self._overlay_loop(root, canvas)
        except Exception as exc:
            logger.error("Failed to create asset tracking overlay: %s", exc)
        finally:
            self.running = False

    @staticmethod
    def _set_topmost(root: Any) -> None:
        try:
            root.attributes("-topmost", True)
        except Exception:
            logger.debug("Topmost overlay attribute is unavailable")

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
            x, y, width, height = self.window_capture.get_window_rect()
            root.geometry(f"{width}x{height}+{x + width + 16}+{max(0, y)}")
            canvas.config(width=width, height=height)
            canvas.delete("asset")
            for asset in self.tracker.snapshot().assets[: int(config.ASSET_TRACKING_OVERLAY_MAX_ITEMS)]:
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
        try:
            root.destroy()
        except Exception:
            logger.debug("Asset tracking overlay root was already closed")


def _asset_bounds(asset: TrackedAsset) -> tuple[int, int, int, int]:
    half_width = max(1, int(asset.width)) // 2
    half_height = max(1, int(asset.height)) // 2
    return (
        int(asset.center_x) - half_width,
        int(asset.center_y) - half_height,
        int(asset.center_x) + half_width,
        int(asset.center_y) + half_height,
    )


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
