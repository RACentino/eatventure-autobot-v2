"""OpenCV masked-template matching + HSV pixel-ratio gate + IoU-based NMS. This is a faithful,
verified port of v1's image_matcher.py core algorithm (read in full — TM_SQDIFF_NORMED, erosion-
based local-minima candidate search via connected components, HSV gate with hue-wraparound
handling), not a guess. Per GREENFIELD_PLAN.md decision 6 (the detection spike): this keeps the
NMS/dedup step (it measurably cut duplicate candidates ~2x with no shown latency cost) but does
not carry over v2's extra region-merging/component-region search — that wasn't justified by the
spike's evidence either way. One deliberate simplification, not a transcription: NMS here is pure
IoU (via BoundingBox.iou), dropping v1's extra ad-hoc pixel-distance pre-filter — a more principled
and equally effective way to express "these two boxes overlap."
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from eatventure_autobot.domain.errors import DetectionError
from eatventure_autobot.domain.types import BoundingBox, HsvGate, MatchCandidate, MatchResult, Point

logger = logging.getLogger(__name__)

_DEFAULT_NMS_IOU_THRESHOLD = 0.20


def _as_bgr(image: np.ndarray, label: str) -> np.ndarray:
    if image is None or not hasattr(image, "shape") or image.size == 0:
        raise DetectionError(f"{label} is empty")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise DetectionError(f"{label} has unsupported shape {image.shape}")


def _match_template(
    screenshot: np.ndarray, template: np.ndarray, mask: np.ndarray | None, template_name: str
) -> np.ndarray | None:
    match_mask = None if mask is not None and np.all(mask) else mask
    try:
        result = cv2.matchTemplate(screenshot, template, cv2.TM_SQDIFF_NORMED, mask=match_mask)
    except cv2.error as exc:
        logger.warning("[%s] Template matching failed: %s", template_name, exc)
        return None
    if result.size == 0:
        return None
    np.nan_to_num(result, copy=False, nan=1.0, posinf=1.0, neginf=1.0)
    np.clip(result, 0.0, 1.0, out=result)
    return result


def _local_minima_candidates(
    result: np.ndarray, max_score: float, min_distance: int
) -> list[Point]:
    if result.size == 0 or result.ndim != 2:
        return []
    window = max(3, min(int(result.shape[0]), int(result.shape[1]), max(3, min_distance)))
    if window % 2 == 0:
        window = max(3, window - 1)
    kernel = np.ones((window, window), dtype=np.float32)
    local_min = cv2.erode(result, kernel)
    candidate_mask = (result <= max_score) & (result <= local_min + 1e-6)
    if not np.any(candidate_mask):
        return []
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_mask.astype(np.uint8), connectivity=8
    )
    candidates: list[Point] = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area <= 0:
            continue
        region = result[y : y + h, x : x + w]
        # Two distinct components can have overlapping bounding rectangles even when their pixels
        # are disjoint (non-convex shapes); mask minMaxLoc to this component's own labeled pixels
        # so a nearby component's lower score can't get attributed to the wrong candidate.
        label_region = labels[y : y + h, x : x + w]
        masked_region = np.where(label_region == index, region, 1.0)
        min_value, _, min_location, _ = cv2.minMaxLoc(masked_region)
        if min_value <= max_score:
            candidates.append((int(x + min_location[0]), int(y + min_location[1])))
    return candidates


def _hsv_range_mask(
    hsv_region: np.ndarray, lower: tuple[int, int, int], upper: tuple[int, int, int]
) -> np.ndarray:
    lower_arr = np.array(lower, dtype=np.uint8)
    upper_arr = np.array(upper, dtype=np.uint8)
    if lower_arr[0] <= upper_arr[0]:
        return cv2.inRange(hsv_region, lower_arr, upper_arr)
    lower_wrap, upper_wrap = lower_arr.copy(), upper_arr.copy()
    lower_wrap[0], upper_wrap[0] = 0, 179
    return cv2.bitwise_or(
        cv2.inRange(hsv_region, lower_arr, upper_wrap),
        cv2.inRange(hsv_region, lower_wrap, upper_arr),
    )


def _check_hsv_gate(
    screenshot: np.ndarray,
    template: np.ndarray,
    location: Point,
    mask: np.ndarray | None,
    hsv_gate: HsvGate,
    hsv_frame: np.ndarray | None = None,
) -> bool:
    x, y = location
    height, width = template.shape[:2]
    if hsv_frame is not None:
        # Caller already converted the whole frame to HSV once (see
        # _find_candidates_across_templates) — slice instead of re-converting per candidate.
        hsv_region = hsv_frame[y : y + height, x : x + width]
        if hsv_region.shape[:2] != (height, width):
            return False
    else:
        roi = screenshot[y : y + height, x : x + width]
        if roi.shape[:2] != (height, width):
            return False
        try:
            hsv_region = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        except cv2.error as exc:
            logger.debug("HSV gate conversion failed: %s", exc)
            return False
    active_mask = np.ones((height, width), dtype=bool) if mask is None else mask > 0
    active_count = int(np.count_nonzero(active_mask))
    if active_count <= 0:
        return False
    combined: np.ndarray = np.zeros((height, width), dtype=np.uint8)
    for hsv_range in hsv_gate.ranges:
        combined = cv2.bitwise_or(
            combined, _hsv_range_mask(hsv_region, hsv_range.lower, hsv_range.upper)
        )
    matched_count = int(np.count_nonzero((combined > 0) & active_mask))
    return (matched_count / active_count) >= hsv_gate.min_match_ratio


def filter_by_template_consensus(
    candidates: list[MatchCandidate],
    min_matches: int,
    iou_threshold: float = _DEFAULT_NMS_IOU_THRESHOLD,
) -> list[MatchCandidate]:
    """Keeps only spots where at least min_matches DISTINCT templates agree, returning the
    highest-confidence candidate per surviving cluster. This is v2's cross-template agreement
    requirement (verified: used for both box and normal-mode red-icon detection), which v1 lacks —
    it suppresses spots that only one template variant happens to like."""
    if min_matches <= 1:
        return sorted(candidates, key=lambda candidate: candidate.confidence, reverse=True)

    clusters: list[list[MatchCandidate]] = []
    for candidate in sorted(candidates, key=lambda candidate: candidate.confidence, reverse=True):
        for cluster in clusters:
            if cluster[0].bounding_box.iou(candidate.bounding_box) > iou_threshold:
                cluster.append(candidate)
                break
        else:
            clusters.append([candidate])

    return [
        cluster[0]
        for cluster in clusters
        if len({member.template_name for member in cluster}) >= min_matches
    ]


class OpenCvTemplateMatcher:
    def __init__(self) -> None:
        self._templates: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}

    def load_template(self, template_path: Path) -> None:
        raw = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise DetectionError(f"Template not found: {template_path}")
        mask = None
        if raw.ndim == 3 and raw.shape[2] == 4:
            alpha = raw[:, :, 3]
            if not np.any(alpha > 0):
                raise DetectionError(f"Template has no visible pixels: {template_path}")
            mask = np.zeros_like(alpha)
            mask[alpha > 0] = 255
        self._templates[template_path.stem] = (_as_bgr(raw, str(template_path)), mask)

    def _lookup(self, template_name: str) -> tuple[np.ndarray, np.ndarray | None]:
        if template_name not in self._templates:
            raise DetectionError(f"Template not loaded: {template_name}")
        return self._templates[template_name]

    def find_template(
        self,
        frame: np.ndarray,
        template_name: str,
        threshold: float,
        hsv_gate: HsvGate | None = None,
    ) -> MatchResult:
        template, mask = self._lookup(template_name)
        frame = _as_bgr(frame, "frame")
        if template.shape[0] > frame.shape[0] or template.shape[1] > frame.shape[1]:
            return MatchResult(found=False, best=None)
        result = _match_template(frame, template, mask, template_name)
        if result is None:
            return MatchResult(found=False, best=None)
        min_value, _, min_location, _ = cv2.minMaxLoc(result)
        confidence = float(1.0 - min_value)
        if not np.isfinite(confidence) or confidence < threshold:
            return MatchResult(found=False, best=None)
        location = (int(min_location[0]), int(min_location[1]))
        if hsv_gate is not None and not _check_hsv_gate(frame, template, location, mask, hsv_gate):
            return MatchResult(found=False, best=None)
        height, width = template.shape[:2]
        center = (location[0] + width // 2, location[1] + height // 2)
        box = BoundingBox(location[0], location[1], location[0] + width, location[1] + height)
        candidate = MatchCandidate(template_name, confidence, center, box)
        return MatchResult(found=True, best=candidate)

    def find_template_candidates(
        self,
        frame: np.ndarray,
        template_name: str,
        threshold: float,
        min_distance: int = 15,
        hsv_gate: HsvGate | None = None,
        hsv_frame: np.ndarray | None = None,
    ) -> list[MatchCandidate]:
        template, mask = self._lookup(template_name)
        frame = _as_bgr(frame, "frame")
        if template.shape[0] > frame.shape[0] or template.shape[1] > frame.shape[1]:
            return []
        result = _match_template(frame, template, mask, template_name)
        if result is None:
            return []
        height, width = template.shape[:2]
        candidates: list[MatchCandidate] = []
        for x, y in _local_minima_candidates(result, 1.0 - threshold, min_distance):
            confidence = float(1.0 - result[y, x])
            if not np.isfinite(confidence):
                continue
            if hsv_gate is not None and not _check_hsv_gate(
                frame, template, (x, y), mask, hsv_gate, hsv_frame=hsv_frame
            ):
                continue
            center = (x + width // 2, y + height // 2)
            box = BoundingBox(x, y, x + width, y + height)
            candidates.append(MatchCandidate(template_name, confidence, center, box))
        candidates.sort(key=lambda candidate: candidate.confidence, reverse=True)
        return candidates

    def _find_candidates_across_templates(
        self,
        frame: np.ndarray,
        template_names: tuple[str, ...],
        threshold: float,
        min_distance: int,
        hsv_gate: HsvGate | None,
    ) -> list[MatchCandidate]:
        """Converts the frame to HSV at most once (only if hsv_gate is set) and reuses it across
        every template — the per-frame-shared-mask optimization GREENFIELD_PLAN.md documents
        keeping from v2, previously only claimed in this module's docstring, never actually built:
        each template independently re-converted its own per-candidate ROI to HSV."""
        frame_bgr = _as_bgr(frame, "frame")
        hsv_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV) if hsv_gate is not None else None
        all_candidates: list[MatchCandidate] = []
        for name in template_names:
            all_candidates.extend(
                self.find_template_candidates(
                    frame_bgr, name, threshold, min_distance, hsv_gate, hsv_frame=hsv_frame
                )
            )
        return all_candidates

    def find_all_templates(
        self,
        frame: np.ndarray,
        template_names: tuple[str, ...],
        threshold: float,
        min_distance: int = 15,
        hsv_gate: HsvGate | None = None,
    ) -> list[MatchCandidate]:
        candidates = self._find_candidates_across_templates(
            frame, template_names, threshold, min_distance, hsv_gate
        )
        return self.suppress_overlaps(candidates, _DEFAULT_NMS_IOU_THRESHOLD)

    def find_candidates_across_templates(
        self,
        frame: np.ndarray,
        template_names: tuple[str, ...],
        threshold: float,
        min_distance: int,
        hsv_gate: HsvGate | None,
    ) -> list[MatchCandidate]:
        return self._find_candidates_across_templates(
            frame, template_names, threshold, min_distance, hsv_gate
        )

    def suppress_overlaps(
        self, candidates: list[MatchCandidate], iou_threshold: float = _DEFAULT_NMS_IOU_THRESHOLD
    ) -> list[MatchCandidate]:
        if not candidates:
            return []
        ordered = sorted(candidates, key=lambda candidate: candidate.confidence, reverse=True)
        kept: list[MatchCandidate] = []
        for candidate in ordered:
            if not any(
                candidate.bounding_box.iou(existing.bounding_box) > iou_threshold
                for existing in kept
            ):
                kept.append(candidate)
        return kept
