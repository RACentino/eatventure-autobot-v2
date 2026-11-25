"""OpenCV masked-template matching + HSV pixel-ratio gate + IoU-based NMS. This is a faithful,
verified port of v1's image_matcher.py core algorithm (read in full — TM_SQDIFF_NORMED, erosion-
based local-minima candidate search via connected components, HSV gate with hue-wraparound
handling), not a guess. An empirical detection spike against live frames confirmed this keeps the
NMS/dedup step (it measurably cut duplicate candidates ~2x with no shown latency cost) but does
not carry over an earlier, unjustified extra region-merging/component-region search. One deliberate
simplification, not a transcription: NMS here is pure IoU (via BoundingBox.iou), dropping v1's
extra ad-hoc pixel-distance pre-filter — a more principled and equally effective way to express
"these two boxes overlap."
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from eatventure_autobot.domain.errors import DetectionError
from eatventure_autobot.domain.types import (
    BoundingBox,
    HsvGate,
    HsvRange,
    MatchCandidate,
    MatchResult,
    Point,
    TemplateExplanation,
)

logger = logging.getLogger(__name__)

_DEFAULT_NMS_IOU_THRESHOLD = 0.20
_MEAN_BGR = np.full((1, 3), 1 / 3, dtype=np.float32)
_PANEL_SCORE_ROUNDOFF = 2e-5


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


def _search_crop(
    frame: np.ndarray, size: tuple[int, int], region: BoundingBox | None
) -> tuple[np.ndarray, int, int]:
    """Bound candidate centers (half-open), retaining the whole template at frame edges."""
    if region is None:
        return frame, 0, 0
    height, width = size
    left = max(0, region.x_min - width // 2)
    top = max(0, region.y_min - height // 2)
    right = min(frame.shape[1], region.x_max + width - width // 2 - 1)
    bottom = min(frame.shape[0], region.y_max + height - height // 2 - 1)
    if region.area() == 0 or right <= left or bottom <= top:
        return frame[:0, :0], left, top
    return frame[top:bottom, left:right], left, top


class _Pattern(NamedTuple):
    pixels: np.ndarray
    mask: np.ndarray | None
    left: int
    top: int
    gray: np.ndarray | None
    header_mean: np.ndarray | None


def _prepare_pattern(template: np.ndarray, mask: np.ndarray | None) -> _Pattern:
    left = top = 0
    pixels = template
    header_mean = None
    if mask is not None and mask.dtype == np.uint8:
        if mask.ndim == 2:
            x, y, width, height = cv2.boundingRect(mask)
            if width and height:
                left, top = x, y
                pixels = template[y:y + height, x:x + width]
                mask = mask[y:y + height, x:x + width]
                if np.all(mask):
                    mask = None
                    if pixels.ndim == 3 and pixels.shape[2] == 3:
                        header_mean = cv2.transform(pixels, _MEAN_BGR)
        elif np.all(mask):
            mask = None
    gray = None
    if (pixels.ndim == 3 and pixels.shape[2] == 3 and (mask is None or mask.ndim == 2)
            and all(np.array_equal(pixels[..., 0], pixels[..., ch]) for ch in (1, 2))):
        gray = np.ascontiguousarray(pixels[..., 0])
    return _Pattern(pixels, mask, left, top, gray, header_mean)


def _panel_score_crop(
    frame: np.ndarray, template: np.ndarray, pattern: _Pattern, threshold: float,
) -> tuple[np.ndarray, int, int]:
    """Bound all possible full-colour matches using the panel's opaque rectangular header."""
    if pattern.header_mean is None or frame.dtype != np.uint8 or threshold <= 0:
        return frame, 0, 0
    left, top = pattern.left, pattern.top
    height, width = pattern.header_mean.shape[:2]
    h, w = template.shape[:2]
    pixels = frame[top:frame.shape[0] - h + top + height, left:frame.shape[1] - w + left + width]
    # Averaging BGR can only reduce RMS error (Cauchy–Schwarz). A passing normalized SQDIFF
    # has mean squared channel error <= (1-threshold)*255² on uint8 pixels. Two intensity
    # levels cover uint8 rounding and OpenCV correlation round-off.
    coarse = cv2.matchTemplate(cv2.transform(pixels, _MEAN_BGR), pattern.header_mean,
                               cv2.TM_SQDIFF)
    limit = width * height * (255 * np.sqrt(1 - threshold) + 2) ** 2
    x, y, cols, rows = cv2.boundingRect((coarse <= limit).astype(np.uint8))
    if not cols or not rows:
        return frame[:0, :0], 0, 0
    return frame[y:y + rows + h - 1, x:x + cols + w - 1], x, y


def _match_template(
    screenshot: np.ndarray, template: np.ndarray, mask: np.ndarray | None, template_name: str,
    pattern: _Pattern | None = None,
) -> np.ndarray | None:
    try:
        if pattern is None:
            pattern = _prepare_pattern(template, mask)
        height, width = pattern.pixels.shape[:2]
        top, left = pattern.top, pattern.left
        # Preserve the original placement grid when dropping transparent template borders.
        screenshot = screenshot[top:screenshot.shape[0] - template.shape[0] + top + height,
                                left:screenshot.shape[1] - template.shape[1] + left + width]
        pixels = pattern.pixels
        if (pattern.gray is not None and screenshot.ndim == 3 and screenshot.shape[2] == 3
                and all(np.array_equal(screenshot[..., 0], screenshot[..., ch]) for ch in (1, 2))):
            screenshot = np.ascontiguousarray(screenshot[..., 0])
            pixels = pattern.gray
        result = cv2.matchTemplate(screenshot, pixels, cv2.TM_SQDIFF_NORMED, mask=pattern.mask)
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
    candidate_mask = result <= max_score
    if not np.any(candidate_mask):
        return []
    window = max(3, min(int(result.shape[0]), int(result.shape[1]), max(3, min_distance)))
    if window % 2 == 0:
        window = max(3, window - 1)
    kernel = np.ones((window, window), dtype=np.float32)
    local_min = cv2.erode(result, kernel)
    candidate_mask &= result <= local_min + 1e-6
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


def hsv_ranges_mask(hsv_region: np.ndarray, ranges: tuple[HsvRange, ...]) -> np.ndarray:
    """Mark pixels covered by any of the configured HSV ranges."""
    combined = np.zeros(hsv_region.shape[:2], dtype=np.uint8)
    for hsv_range in ranges:
        combined |= _hsv_range_mask(hsv_region, hsv_range.lower, hsv_range.upper)
    return combined


def _hsv_match_ratio(
    screenshot: np.ndarray,
    template: np.ndarray,
    location: Point,
    mask: np.ndarray | None,
    hsv_gate: HsvGate,
    hsv_frame: np.ndarray | None = None,
) -> float | None:
    """Share of the template's visible pixels whose colour is inside the HSV gate at `location`;
    None when it can't be measured (window falls off the frame, conversion failed, no pixels)."""
    x, y = location
    height, width = template.shape[:2]
    if hsv_frame is not None:
        # The caller supplies a converted frame to avoid converting each candidate.
        hsv_region = hsv_frame[y : y + height, x : x + width]
        if hsv_region.shape[:2] != (height, width):
            return None
    else:
        roi = screenshot[y : y + height, x : x + width]
        if roi.shape[:2] != (height, width):
            return None
        try:
            hsv_region = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        except cv2.error as exc:
            logger.debug("HSV gate conversion failed: %s", exc)
            return None
    active_mask = np.ones((height, width), dtype=bool) if mask is None else mask > 0
    active_count = int(np.count_nonzero(active_mask))
    if active_count <= 0:
        return None
    combined = hsv_ranges_mask(hsv_region, hsv_gate.ranges)
    matched_count = int(np.count_nonzero((combined > 0) & active_mask))
    return matched_count / active_count


def _hsv_median(
    screenshot: np.ndarray, template: np.ndarray, location: Point, mask: np.ndarray | None
) -> tuple[int, int, int] | None:
    """Median (H, S, V) of the template's visible pixels at `location`; None when the window falls
    off the frame, has no visible pixels or can't be converted. Diagnostics only (stall probe)."""
    x, y = location
    height, width = template.shape[:2]
    roi = screenshot[y : y + height, x : x + width]
    if roi.shape[:2] != (height, width):
        return None
    active = np.ones((height, width), dtype=bool) if mask is None else mask > 0
    if not active.any():
        return None
    try:
        hsv_region = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    except cv2.error as exc:
        logger.debug("HSV median conversion failed: %s", exc)
        return None
    hue, saturation, value = (int(np.median(hsv_region[:, :, c][active])) for c in range(3))
    return hue, saturation, value


def _check_hsv_gate(
    screenshot: np.ndarray,
    template: np.ndarray,
    location: Point,
    mask: np.ndarray | None,
    hsv_gate: HsvGate,
    hsv_frame: np.ndarray | None = None,
) -> bool:
    ratio = _hsv_match_ratio(screenshot, template, location, mask, hsv_gate, hsv_frame)
    return ratio is not None and ratio >= hsv_gate.min_match_ratio


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
    def __init__(self, threads: int = 1) -> None:
        self._templates: dict[str, tuple[np.ndarray, np.ndarray | None, _Pattern]] = {}
        # cv2.matchTemplate releases the GIL, so a multi-template scan runs its templates side by
        # side (BoxDetectionConfig.match_threads). 1 = no pool and the old sequential loop. The pool
        # starts its threads on first use and lives as long as the matcher (the whole process).
        self._pool = (
            ThreadPoolExecutor(threads, thread_name_prefix="match") if threads > 1 else None
        )

    def load_template(self, template_path: Path) -> tuple[int, int]:
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
        pixels = _as_bgr(raw, str(template_path))
        self._templates[template_path.stem] = pixels, mask, _prepare_pattern(pixels, mask)
        return int(raw.shape[1]), int(raw.shape[0])

    def _lookup(self, template_name: str) -> tuple[np.ndarray, np.ndarray | None, _Pattern]:
        if template_name not in self._templates:
            raise DetectionError(f"Template not loaded: {template_name}")
        return self._templates[template_name]

    def _panel_location(
        self, scores: np.ndarray, offset: Point, name: str, threshold: float,
        original: np.ndarray, centers: BoundingBox, padding: int = 0,
    ) -> Point | None:
        value, _, (dx, dy), _ = cv2.minMaxLoc(scores)
        cutoff = 1.0 - threshold
        if value > cutoff + _PANEL_SCORE_ROUNDOFF:
            return None
        # OpenCV correlation round-off depends on the image extent. Resolve a close threshold
        # decision or competing minima on the original search grid to preserve its chosen box.
        if (abs(value - cutoff) <= _PANEL_SCORE_ROUNDOFF
                or np.count_nonzero(scores <= value + _PANEL_SCORE_ROUNDOFF) > 1):
            match = self.find_template(original, name, threshold, search_region=centers).best
            if match is None:
                return None
            return match.bounding_box.x_min - padding, match.bounding_box.y_min - padding
        return offset[0] + dx, offset[1] + dy

    def find_panel_bounds(
        self, frame: np.ndarray, names: tuple[str, ...], threshold: float,
        centers: BoundingBox, padding: int,
    ) -> BoundingBox | None:
        """Prefer any fully visible panel to a clipped one, in configured template order."""
        if not names:
            return None
        height, width = frame.shape[:2]
        radius = max(max(self._lookup(name)[0].shape[:2]) // 2 for name in names)
        outside = max(0, -centers.x_min, -centers.y_min, centers.x_max - width,
                      centers.y_max - height)
        padding = min(padding, radius + outside)
        extended = cv2.copyMakeBorder(
            _as_bgr(frame, "frame"), padding, padding, padding, padding, cv2.BORDER_CONSTANT,
        )
        region = BoundingBox(centers.x_min + padding, centers.y_min + padding,
                             centers.x_max + padding, centers.y_max + padding)
        clipped = None
        for name in names:
            template, mask, pattern = self._lookup(name)
            h, w = template.shape[:2]
            image, x, y = _search_crop(extended, template.shape[:2], region)
            if image.shape[0] < h or image.shape[1] < w:
                continue
            image, dx, dy = _panel_score_crop(image, template, pattern, threshold)
            if not image.size:
                continue
            scores = _match_template(image, template, mask, name, pattern)
            if scores is None:
                continue
            # Each score denotes the top-left of the original, untrimmed panel template.
            x, y = x + dx - padding, y + dy - padding
            left, top = max(0, -x), max(0, -y)
            right, bottom = width - w - x + 1, height - h - y + 1
            if right > left and bottom > top:
                interior = scores[top:bottom, left:right]
                if interior.size:
                    position = self._panel_location(interior, (x + left, y + top), name,
                                                    threshold, frame, centers)
                    if position is not None:
                        x0, y0 = position
                        return BoundingBox(x0, y0, x0 + w, y0 + h)
            if clipped is None:
                position = self._panel_location(scores, (x, y), name, threshold,
                                                extended, region, padding)
                if position is not None:
                    x0, y0 = position
                    left, top = max(0, x0), max(0, y0)
                    right, bottom = min(width, x0 + w), min(height, y0 + h)
                    if left < right and top < bottom:
                        clipped = BoundingBox(left, top, right, bottom)
        return clipped

    def find_template(
        self,
        frame: np.ndarray,
        template_name: str,
        threshold: float,
        hsv_gate: HsvGate | None = None,
        search_region: BoundingBox | None = None,
    ) -> MatchResult:
        template, mask, pattern = self._lookup(template_name)
        frame = _as_bgr(frame, "frame")
        frame, offset_x, offset_y = _search_crop(frame, template.shape[:2], search_region)
        if template.shape[0] > frame.shape[0] or template.shape[1] > frame.shape[1]:
            return MatchResult(found=False, best=None)
        result = _match_template(frame, template, mask, template_name, pattern)
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
        location = (location[0] + offset_x, location[1] + offset_y)
        center = (location[0] + width // 2, location[1] + height // 2)
        box = BoundingBox(location[0], location[1], location[0] + width, location[1] + height)
        candidate = MatchCandidate(template_name, confidence, center, box)
        return MatchResult(found=True, best=candidate)

    def explain_template(
        self, frame: np.ndarray, template_name: str, hsv_gate: HsvGate | None = None
    ) -> TemplateExplanation | None:
        """The single best raw match of a template with no threshold or gate applied, plus how
        much of it sits inside `hsv_gate` — what the detector saw when it reported no match.
        Diagnostics only (the stall probe); never used to decide a click."""
        template, mask, pattern = self._lookup(template_name)
        frame = _as_bgr(frame, "frame")
        if template.shape[0] > frame.shape[0] or template.shape[1] > frame.shape[1]:
            return None
        result = _match_template(frame, template, mask, template_name, pattern)
        if result is None:
            return None
        min_value, _, min_location, _ = cv2.minMaxLoc(result)
        confidence = float(1.0 - min_value)
        if not np.isfinite(confidence):
            return None
        location = (int(min_location[0]), int(min_location[1]))
        ratio = (
            None
            if hsv_gate is None
            else _hsv_match_ratio(frame, template, location, mask, hsv_gate)
        )
        height, width = template.shape[:2]
        center = (location[0] + width // 2, location[1] + height // 2)
        median = _hsv_median(frame, template, location, mask)
        return TemplateExplanation(template_name, confidence, center, ratio, median)

    def find_template_candidates(
        self,
        frame: np.ndarray,
        template_name: str,
        threshold: float,
        min_distance: int = 15,
        hsv_gate: HsvGate | None = None,
        hsv_frame: np.ndarray | None = None,
        search_region: BoundingBox | None = None,
        hsv_origin: Point = (0, 0),
    ) -> list[MatchCandidate]:
        template, mask, pattern = self._lookup(template_name)
        frame = _as_bgr(frame, "frame")
        frame, offset_x, offset_y = _search_crop(frame, template.shape[:2], search_region)
        if hsv_frame is not None:
            x, y = offset_x - hsv_origin[0], offset_y - hsv_origin[1]
            hsv_frame = hsv_frame[y:y + frame.shape[0], x:x + frame.shape[1]]
        if template.shape[0] > frame.shape[0] or template.shape[1] > frame.shape[1]:
            return []
        result = _match_template(frame, template, mask, template_name, pattern)
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
            x, y = x + offset_x, y + offset_y
            center = (x + width // 2, y + height // 2)
            box = BoundingBox(x, y, x + width, y + height)
            candidates.append(MatchCandidate(template_name, confidence, center, box))
        candidates.sort(key=lambda candidate: candidate.confidence, reverse=True)
        return candidates

    def find_candidates_across_templates(
        self,
        frame: np.ndarray,
        template_names: tuple[str, ...],
        threshold: float,
        min_distance: int,
        hsv_gate: HsvGate | None,
        *,
        search_region: BoundingBox | None = None,
    ) -> list[MatchCandidate]:
        """Convert the frame to HSV once when gating, and reuse it across templates."""
        frame_bgr = _as_bgr(frame, "frame")
        hsv_frame = None
        hsv_origin = (0, 0)
        if hsv_gate is not None and template_names:
            pixels = frame_bgr
            if search_region is not None:
                sizes = [self._lookup(name)[0].shape for name in template_names]
                support = (max(s[0] for s in sizes), max(s[1] for s in sizes))
                pixels, x, y = _search_crop(frame_bgr, support, search_region)
                if not pixels.size:
                    return []
                hsv_origin = (x, y)
            hsv_frame = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV)

        def one(name: str) -> list[MatchCandidate]:
            return self.find_template_candidates(
                frame_bgr, name, threshold, min_distance, hsv_gate, hsv_frame=hsv_frame,
                search_region=search_region, hsv_origin=hsv_origin,
            )

        # map() yields in template order and re-raises a worker's exception here, so the result and
        # the failure behaviour are exactly those of the sequential loop.
        per_template = (
            list(self._pool.map(one, template_names))
            if self._pool is not None and len(template_names) > 1
            else [one(name) for name in template_names]
        )
        return [candidate for found in per_template for candidate in found]

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
