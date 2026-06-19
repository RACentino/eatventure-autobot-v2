import logging
from typing import Any, cast

import cv2
import numpy as np

_SUPERVISION_IMPORT_ERROR: Exception | None
sv: Any

try:
    import supervision as sv
except Exception as exc:
    sv = None
    _SUPERVISION_IMPORT_ERROR = exc
else:
    _SUPERVISION_IMPORT_ERROR = None

logger = logging.getLogger(__name__)

MatchResult = tuple[bool, float, int, int]
MatchCandidate = tuple[float, int, int, int, int]
MatchRegion = tuple[int, int, int, int]
Point = tuple[int, int]
MAX_TEMPLATE_CANDIDATES = 400
HSV_REGION_RANGE_LIMIT = 32
HSV_REGION_COMPONENT_LIMIT = 96
HSV_REGION_MAX_COVERAGE_RATIO = 0.45
HSV_REGION_MAX_TOTAL_AREA_RATIO = 0.70
HSV_REGION_MERGE_PADDING_PIXELS = 2
HSV_REGION_MINIMUM_COMPONENT_AREA = 1
MIN_SCALED_TEMPLATE_DIMENSION = 1


def _threshold(value: Any, default: float = 0.85) -> float:
    try:
        fallback = float(default)
    except (TypeError, ValueError):
        fallback = 0.85
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = fallback
    if not np.isfinite(number):
        number = fallback
    return max(0.0, min(1.0, number))


def _as_bgr(image: Any, label: str) -> np.ndarray:
    if image is None or not hasattr(image, "shape") or image.size == 0:
        raise ValueError(f"{label} is empty")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"{label} has unsupported shape {image.shape}")


def _normalized_mask(
    mask: Any, template_shape: tuple[int, ...], label: str
) -> np.ndarray | None:
    if mask is None:
        return None
    if not hasattr(mask, "shape") or mask.size == 0:
        return None
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.ndim != 2 or mask.shape[:2] != template_shape[:2]:
        logger.warning("[%s] Ignoring incompatible mask", label)
        return None
    normalized = np.zeros(mask.shape[:2], dtype=np.uint8)
    normalized[mask > 0] = 255
    return normalized if np.any(normalized) else None


def _match_template(
    screenshot: np.ndarray, template: np.ndarray, mask: np.ndarray | None, label: str
) -> np.ndarray | None:
    if (
        template.shape[0] > screenshot.shape[0]
        or template.shape[1] > screenshot.shape[1]
    ):
        logger.debug("[%s] Template larger than screenshot", label)
        return None
    try:
        result = cv2.matchTemplate(
            screenshot, template, cv2.TM_SQDIFF_NORMED, mask=mask
        )
    except cv2.error as exc:
        logger.warning("[%s] Template matching failed: %s", label, exc)
        return None
    if result.size == 0:
        return None
    return np.clip(np.nan_to_num(result, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 1.0)


def _center(location: Point, template: np.ndarray) -> Point:
    height, width = template.shape[:2]
    return location[0] + width // 2, location[1] + height // 2


def _candidate_box(
    candidate: tuple[Any, ...],
) -> tuple[float, float, float, float] | None:
    try:
        _, center_x, center_y, width, height = candidate[:5]
        center_x, center_y, width, height = (
            float(center_x),
            float(center_y),
            float(width),
            float(height),
        )
    except (TypeError, ValueError):
        return None
    if (
        not all(np.isfinite(value) for value in (center_x, center_y, width, height))
        or width <= 0
        or height <= 0
    ):
        return None
    return (
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2,
    )


class ImageMatcher:
    def __init__(self, threshold: float = 0.85) -> None:
        self.threshold = _threshold(threshold)

    @staticmethod
    def supervision_available() -> bool:
        return sv is not None

    def load_template(self, template_path: Any) -> tuple[np.ndarray, np.ndarray | None]:
        template = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
        if template is None:
            raise FileNotFoundError(f"Template not found: {template_path}")
        mask = None
        if template.ndim == 3 and template.shape[2] == 4:
            alpha = template[:, :, 3]
            if not np.any(alpha > 0):
                raise ValueError(f"Template has no visible pixels: {template_path}")
            mask = np.zeros_like(alpha)
            mask[alpha > 0] = 255
        return _as_bgr(template, str(template_path)), mask

    def find_template(
        self,
        screenshot: np.ndarray,
        template: np.ndarray,
        mask: np.ndarray | None = None,
        threshold: float | None = None,
        template_name: str = "Unknown",
        hsv_ranges: Any = None,
        hsv_match_threshold: float = 0.9,
    ) -> MatchResult:
        try:
            screenshot = _as_bgr(screenshot, "screenshot")
            template = _as_bgr(template, template_name)
            mask = _normalized_mask(mask, template.shape, template_name)
        except (ValueError, cv2.error) as exc:
            logger.warning("[%s] Invalid match input: %s", template_name, exc)
            return False, 0.0, 0, 0
        result = _match_template(screenshot, template, mask, template_name)
        if result is None:
            return False, 0.0, 0, 0
        min_value, _, raw_min_location, _ = cv2.minMaxLoc(result)
        min_location = (int(raw_min_location[0]), int(raw_min_location[1]))
        confidence = float(1.0 - min_value)
        if not np.isfinite(confidence) or confidence < _threshold(
            threshold, self.threshold
        ):
            return False, confidence if np.isfinite(confidence) else 0.0, 0, 0
        center_x, center_y = _center(min_location, template)
        if hsv_ranges and not self._check_hsv_gate(
            screenshot, template, min_location, mask, hsv_ranges, hsv_match_threshold
        ):
            return False, confidence, 0, 0
        return True, confidence, center_x, center_y

    def find_all_templates(
        self,
        screenshot: np.ndarray,
        template: np.ndarray,
        mask: np.ndarray | None = None,
        threshold: float | None = None,
        min_distance: int = 15,
        scales: list[float] | None = None,
        template_name: str = "Unknown",
        use_supervision_nms: bool = False,
        supervision_iou_threshold: float = 0.5,
        supervision_class_agnostic: bool = True,
    ) -> list[tuple[float, int, int]]:
        matches = self.find_template_candidates(
            screenshot, template, mask, threshold, min_distance, scales, template_name
        )
        matches = self._non_max_suppression(matches, min_distance)
        if use_supervision_nms:
            filtered = self.filter_candidates_with_supervision_nms(
                matches,
                iou_threshold=supervision_iou_threshold,
                class_agnostic=supervision_class_agnostic,
            )
            if filtered is not None:
                matches = filtered
        return [(confidence, x, y) for confidence, x, y, _, _ in matches]

    def find_all_color_gated_templates(
        self,
        screenshot: np.ndarray,
        template: np.ndarray,
        mask: np.ndarray | None = None,
        threshold: float | None = None,
        min_distance: int = 15,
        scales: list[float] | None = None,
        template_name: str = "Unknown",
        hsv_ranges: Any = None,
        use_supervision_nms: bool = False,
        supervision_iou_threshold: float = 0.5,
        supervision_class_agnostic: bool = True,
    ) -> list[tuple[float, int, int]]:
        matches = self.find_color_gated_template_candidates(
            screenshot,
            template,
            mask,
            threshold,
            min_distance,
            scales,
            template_name,
            hsv_ranges,
        )
        matches = self._non_max_suppression(matches, min_distance)
        if use_supervision_nms:
            filtered = self.filter_candidates_with_supervision_nms(
                matches,
                iou_threshold=supervision_iou_threshold,
                class_agnostic=supervision_class_agnostic,
            )
            if filtered is not None:
                matches = filtered
        return [(confidence, x, y) for confidence, x, y, _, _ in matches]

    def find_template_candidates(
        self,
        screenshot: np.ndarray,
        template: np.ndarray,
        mask: np.ndarray | None = None,
        threshold: float | None = None,
        min_distance: int = 15,
        scales: list[float] | None = None,
        template_name: str = "Unknown",
    ) -> list[MatchCandidate]:
        try:
            screenshot = _as_bgr(screenshot, "screenshot")
            template = _as_bgr(template, template_name)
            mask = _normalized_mask(mask, template.shape, template_name)
        except (ValueError, cv2.error) as exc:
            logger.warning("[%s] Invalid multi-match input: %s", template_name, exc)
            return []
        matches: list[MatchCandidate] = []
        max_score = 1.0 - _threshold(threshold, self.threshold)
        for scale in self._valid_scales(scales, template.shape):
            matches.extend(
                self._template_candidates_for_scale(
                    screenshot,
                    template,
                    mask,
                    scale,
                    max_score,
                    min_distance,
                    template_name,
                )
            )
        return sorted(matches, key=lambda match: match[0], reverse=True)[
            :MAX_TEMPLATE_CANDIDATES
        ]

    def find_color_gated_template_candidates(
        self,
        screenshot: np.ndarray,
        template: np.ndarray,
        mask: np.ndarray | None = None,
        threshold: float | None = None,
        min_distance: int = 15,
        scales: list[float] | None = None,
        template_name: str = "Unknown",
        hsv_ranges: Any = None,
    ) -> list[MatchCandidate]:
        try:
            screenshot = _as_bgr(screenshot, "screenshot")
            template = _as_bgr(template, template_name)
            mask = _normalized_mask(mask, template.shape, template_name)
        except (ValueError, cv2.error) as exc:
            logger.warning(
                "[%s] Invalid color-gated match input: %s", template_name, exc
            )
            return []
        matches_by_location: dict[tuple[int, int, int, int], MatchCandidate] = {}
        max_score = 1.0 - _threshold(threshold, self.threshold)
        for region in self._color_candidate_regions(screenshot, template, hsv_ranges):
            for candidate in self._region_template_candidates(
                screenshot,
                template,
                mask,
                region,
                scales,
                max_score,
                min_distance,
                template_name,
            ):
                key = self._candidate_location_key(candidate)
                if (
                    key not in matches_by_location
                    or candidate[0] > matches_by_location[key][0]
                ):
                    matches_by_location[key] = candidate
        return sorted(
            matches_by_location.values(), key=lambda match: match[0], reverse=True
        )[:MAX_TEMPLATE_CANDIDATES]

    def _region_template_candidates(
        self,
        screenshot: np.ndarray,
        template: np.ndarray,
        mask: np.ndarray | None,
        region: MatchRegion,
        scales: list[float] | None,
        max_score: float,
        min_distance: int,
        template_name: str,
    ) -> list[MatchCandidate]:
        left, top, right, bottom = region
        regional_screenshot = screenshot[top:bottom, left:right]
        matches: list[MatchCandidate] = []
        for scale in self._valid_scales(scales, template.shape):
            for candidate in self._template_candidates_for_scale(
                regional_screenshot,
                template,
                mask,
                scale,
                max_score,
                min_distance,
                template_name,
            ):
                matches.append(self._offset_candidate(candidate, left, top))
        return matches

    @staticmethod
    def _candidate_location_key(candidate: MatchCandidate) -> tuple[int, int, int, int]:
        _, center_x, center_y, width, height = candidate
        return int(center_x), int(center_y), int(width), int(height)

    @staticmethod
    def _offset_candidate(
        candidate: MatchCandidate, offset_x: int, offset_y: int
    ) -> MatchCandidate:
        confidence, center_x, center_y, width, height = candidate
        return confidence, center_x + offset_x, center_y + offset_y, width, height

    def _color_candidate_regions(
        self, screenshot: np.ndarray, template: np.ndarray, hsv_ranges: Any
    ) -> list[MatchRegion]:
        full_frame_region = self._full_frame_region(screenshot)
        combined_mask = self._combined_hsv_mask(screenshot, hsv_ranges)
        if combined_mask is None:
            return [full_frame_region]
        coverage_ratio = self._mask_coverage_ratio(combined_mask)
        if coverage_ratio <= 0.0 or coverage_ratio > HSV_REGION_MAX_COVERAGE_RATIO:
            return [full_frame_region]
        component_regions = self._component_match_regions(
            combined_mask, screenshot.shape, template.shape
        )
        if not component_regions or len(component_regions) > HSV_REGION_COMPONENT_LIMIT:
            return [full_frame_region]
        merged_regions = self._merge_match_regions(component_regions)
        if self._regions_cover_too_much_area(merged_regions, screenshot.shape):
            return [full_frame_region]
        return merged_regions or [full_frame_region]

    def _combined_hsv_mask(
        self, screenshot: np.ndarray, hsv_ranges: Any
    ) -> np.ndarray | None:
        try:
            hsv_range_values = list(hsv_ranges)[:HSV_REGION_RANGE_LIMIT]
        except TypeError:
            return None
        if not hsv_range_values:
            return None
        hsv_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros(screenshot.shape[:2], dtype=np.uint8)
        for hsv_range in hsv_range_values:
            normalized_range = self._normalize_hsv_range(hsv_range)
            if normalized_range is None:
                continue
            lower, upper = normalized_range
            combined_mask = cast(
                np.ndarray,
                cv2.bitwise_or(
                    combined_mask, self._hsv_mask(hsv_screenshot, lower, upper)
                ),
            )
        return combined_mask

    @staticmethod
    def _mask_coverage_ratio(mask: np.ndarray) -> float:
        if mask.size == 0:
            return 1.0
        return float(np.count_nonzero(mask)) / float(mask.size)

    @staticmethod
    def _full_frame_region(screenshot: np.ndarray) -> MatchRegion:
        height, width = screenshot.shape[:2]
        return 0, 0, int(width), int(height)

    def _component_match_regions(
        self,
        combined_mask: np.ndarray,
        screenshot_shape: tuple[int, ...],
        template_shape: tuple[int, ...],
    ) -> list[MatchRegion]:
        _, _, stats, _ = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
        if stats.shape[0] - 1 > HSV_REGION_COMPONENT_LIMIT:
            return []
        regions: list[MatchRegion] = []
        for component_index in range(1, stats.shape[0]):
            component_area = int(stats[component_index, cv2.CC_STAT_AREA])
            if component_area < HSV_REGION_MINIMUM_COMPONENT_AREA:
                continue
            region = self._expanded_component_region(
                stats[component_index], screenshot_shape, template_shape
            )
            if region is not None:
                regions.append(region)
        return regions

    @staticmethod
    def _expanded_component_region(
        component_stats: np.ndarray,
        screenshot_shape: tuple[int, ...],
        template_shape: tuple[int, ...],
    ) -> MatchRegion | None:
        screenshot_height, screenshot_width = screenshot_shape[:2]
        template_height, template_width = template_shape[:2]
        left = int(component_stats[cv2.CC_STAT_LEFT])
        top = int(component_stats[cv2.CC_STAT_TOP])
        width = int(component_stats[cv2.CC_STAT_WIDTH])
        height = int(component_stats[cv2.CC_STAT_HEIGHT])
        region_left = max(0, left - template_width)
        region_top = max(0, top - template_height)
        region_right = min(int(screenshot_width), left + width + template_width)
        region_bottom = min(int(screenshot_height), top + height + template_height)
        if region_right - region_left < template_width:
            return None
        if region_bottom - region_top < template_height:
            return None
        return region_left, region_top, region_right, region_bottom

    def _merge_match_regions(self, regions: list[MatchRegion]) -> list[MatchRegion]:
        merged_regions: list[MatchRegion] = []
        for region in sorted(regions, key=self._region_sort_key):
            self._append_or_merge_region(merged_regions, region)
        return merged_regions

    @staticmethod
    def _region_sort_key(region: MatchRegion) -> tuple[int, int, int, int]:
        left, top, right, bottom = region
        return top, left, bottom, right

    def _append_or_merge_region(
        self, merged_regions: list[MatchRegion], region: MatchRegion
    ) -> None:
        for index, existing_region in enumerate(merged_regions):
            if self._regions_overlap(existing_region, region):
                merged_regions[index] = self._combined_region(existing_region, region)
                return
        merged_regions.append(region)

    @staticmethod
    def _regions_overlap(first: MatchRegion, second: MatchRegion) -> bool:
        first_left, first_top, first_right, first_bottom = first
        second_left, second_top, second_right, second_bottom = second
        return not (
            first_right + HSV_REGION_MERGE_PADDING_PIXELS < second_left
            or second_right + HSV_REGION_MERGE_PADDING_PIXELS < first_left
            or first_bottom + HSV_REGION_MERGE_PADDING_PIXELS < second_top
            or second_bottom + HSV_REGION_MERGE_PADDING_PIXELS < first_top
        )

    @staticmethod
    def _combined_region(first: MatchRegion, second: MatchRegion) -> MatchRegion:
        return (
            min(first[0], second[0]),
            min(first[1], second[1]),
            max(first[2], second[2]),
            max(first[3], second[3]),
        )

    @staticmethod
    def _regions_cover_too_much_area(
        regions: list[MatchRegion], screenshot_shape: tuple[int, ...]
    ) -> bool:
        screenshot_height, screenshot_width = screenshot_shape[:2]
        full_area = max(1, int(screenshot_width) * int(screenshot_height))
        region_area = sum(
            max(0, right - left) * max(0, bottom - top)
            for left, top, right, bottom in regions
        )
        return float(region_area) / float(full_area) > HSV_REGION_MAX_TOTAL_AREA_RATIO

    def _template_candidates_for_scale(
        self,
        screenshot: np.ndarray,
        template: np.ndarray,
        mask: np.ndarray | None,
        scale: float,
        max_score: float,
        min_distance: int,
        template_name: str,
    ) -> list[MatchCandidate]:
        scaled_template, scaled_mask = self._scaled(template, mask, scale)
        result = _match_template(
            screenshot, scaled_template, scaled_mask, template_name
        )
        if result is None:
            return []
        height, width = scaled_template.shape[:2]
        candidates = []
        for x, y in self._candidate_points(result, max_score, min_distance):
            confidence = float(1.0 - result[y, x])
            if np.isfinite(confidence):
                candidates.append(
                    (confidence, x + width // 2, y + height // 2, width, height)
                )
        return candidates

    def filter_candidates_with_supervision_nms(
        self,
        candidates: list[tuple[Any, ...]],
        iou_threshold: float = 0.5,
        class_agnostic: bool = True,
        class_ids: list[int] | tuple[int, ...] | None = None,
    ) -> list[tuple[Any, ...]] | None:
        if not candidates:
            return []
        if sv is None:
            if _SUPERVISION_IMPORT_ERROR is not None:
                logger.debug("Supervision unavailable: %s", _SUPERVISION_IMPORT_ERROR)
            return None
        arrays = self._supervision_arrays(candidates, class_ids)
        if arrays is None:
            return []
        boxes, confidences, normalized_class_ids, indexes = arrays
        try:
            detections = sv.Detections(
                xyxy=boxes, confidence=confidences, class_id=normalized_class_ids
            )
            detections["candidate_index"] = indexes
            retained = detections.with_nms(
                threshold=_threshold(iou_threshold, 0.5),
                class_agnostic=bool(class_agnostic),
            )
        except Exception as exc:
            logger.debug("Supervision NMS failed: %s", exc)
            return None
        data = getattr(retained, "data", None)
        if isinstance(data, dict) and "candidate_index" in data:
            retained_indexes = [int(index) for index in data["candidate_index"]]
            return sorted(
                (candidates[index] for index in retained_indexes),
                key=lambda item: item[0],
                reverse=True,
            )
        return None

    @staticmethod
    def _valid_scales(
        scales: list[float] | None, template_shape: tuple[int, ...]
    ) -> list[float]:
        values = [1.0] if scales is None else scales[:32]
        normalized = []
        template_height, template_width = template_shape[:2]
        for scale in values:
            try:
                scale = float(scale)
            except (TypeError, ValueError):
                continue
            scaled_width = int(round(float(template_width) * scale))
            scaled_height = int(round(float(template_height) * scale))
            if (
                np.isfinite(scale)
                and scale > 0
                and scaled_width >= MIN_SCALED_TEMPLATE_DIMENSION
                and scaled_height >= MIN_SCALED_TEMPLATE_DIMENSION
            ):
                normalized.append(scale)
        return normalized or [1.0]

    @staticmethod
    def _scaled(
        template: np.ndarray, mask: np.ndarray | None, scale: float
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if scale == 1.0:
            return template, mask
        scaled_template = cv2.resize(
            template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        if mask is None:
            return scaled_template, None
        scaled_mask = cv2.resize(
            mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
        )
        scaled_mask[scaled_mask > 0] = 255
        return scaled_template, scaled_mask

    @staticmethod
    def _candidate_points(
        result: np.ndarray, max_score: float, min_distance: int
    ) -> list[Point]:
        window = max(3, int(min_distance))
        if window % 2 == 0:
            window += 1
        local_min = cv2.erode(result, np.ones((window, window), dtype=np.float32))
        points = np.argwhere((result <= max_score) & (result <= local_min + 1e-6))
        if points.size == 0:
            return []
        scored = sorted(
            ((float(result[y, x]), int(x), int(y)) for y, x in points),
            key=lambda item: item[0],
        )
        return [(x, y) for _, x, y in scored[:MAX_TEMPLATE_CANDIDATES]]

    @staticmethod
    def _iou(first: MatchCandidate, second: MatchCandidate) -> float:
        _, ax, ay, aw, ah = first
        _, bx, by, bw, bh = second
        left = max(ax - aw / 2, bx - bw / 2)
        top = max(ay - ah / 2, by - bh / 2)
        right = min(ax + aw / 2, bx + bw / 2)
        bottom = min(ay + ah / 2, by + bh / 2)
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        union = aw * ah + bw * bh - intersection
        return intersection / union if union > 0 else 0.0

    def _non_max_suppression(
        self, matches: list[MatchCandidate], min_distance: int
    ) -> list[MatchCandidate]:
        filtered: list[MatchCandidate] = []
        for candidate in sorted(matches, key=lambda match: match[0], reverse=True):
            if all(
                abs(candidate[1] - kept[1]) >= min_distance
                or abs(candidate[2] - kept[2]) >= min_distance
                or self._iou(candidate, kept) <= 0.2
                for kept in filtered
            ):
                filtered.append(candidate)
        return filtered

    @staticmethod
    def _supervision_arrays(
        candidates: list[tuple[Any, ...]], class_ids: list[int] | tuple[int, ...] | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        boxes = []
        confidences = []
        normalized_class_ids = []
        indexes = []
        for index, candidate in enumerate(candidates[:MAX_TEMPLATE_CANDIDATES]):
            box = _candidate_box(candidate)
            if box is None:
                continue
            try:
                confidence = float(candidate[0])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(confidence):
                continue
            boxes.append(box)
            confidences.append(max(0.0, min(1.0, confidence)))
            try:
                class_id = 0 if class_ids is None else int(class_ids[index])
            except (IndexError, TypeError, ValueError):
                class_id = 0
            normalized_class_ids.append(class_id)
            indexes.append(index)
        if not boxes:
            return None
        return (
            np.asarray(boxes, dtype=np.float32),
            np.asarray(confidences, dtype=np.float32),
            np.asarray(normalized_class_ids, dtype=np.int32),
            np.asarray(indexes, dtype=np.int32),
        )

    @staticmethod
    def _normalize_hsv_range(hsv_range: Any) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            lower, upper = hsv_range
            lower = np.asarray(lower, dtype=np.int16)
            upper = np.asarray(upper, dtype=np.int16)
        except (TypeError, ValueError):
            return None
        if lower.shape != (3,) or upper.shape != (3,):
            return None
        lower = np.array(
            [
                max(0, min(179, int(lower[0]))),
                max(0, min(255, int(lower[1]))),
                max(0, min(255, int(lower[2]))),
            ],
            dtype=np.uint8,
        )
        upper = np.array(
            [
                max(0, min(179, int(upper[0]))),
                max(0, min(255, int(upper[1]))),
                max(0, min(255, int(upper[2]))),
            ],
            dtype=np.uint8,
        )
        return lower, upper

    @staticmethod
    def _hsv_mask(
        hsv_region: np.ndarray, lower: np.ndarray, upper: np.ndarray
    ) -> np.ndarray:
        if int(lower[0]) <= int(upper[0]):
            return cv2.inRange(hsv_region, lower, upper)
        lower_wrap = lower.copy()
        upper_wrap = upper.copy()
        lower_wrap[0] = 0
        upper_wrap[0] = 179
        return cv2.bitwise_or(
            cv2.inRange(hsv_region, lower, upper_wrap),
            cv2.inRange(hsv_region, lower_wrap, upper),
        )

    def _check_hsv_gate(
        self,
        screenshot: np.ndarray,
        template: np.ndarray,
        location: Point,
        mask: np.ndarray | None,
        hsv_ranges: Any,
        hsv_match_threshold: float,
    ) -> bool:
        x, y = location
        height, width = template.shape[:2]
        if x < 0 or y < 0:
            return False
        region = screenshot[y : y + height, x : x + width]
        if region.shape[:2] != template.shape[:2]:
            return False
        active_mask = np.ones((height, width), dtype=bool) if mask is None else mask > 0
        active_count = int(np.count_nonzero(active_mask))
        if active_count <= 0:
            return False
        ranges = [
            item
            for value in hsv_ranges
            if (item := self._normalize_hsv_range(value)) is not None
        ]
        if not ranges:
            return False
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        combined = np.zeros((height, width), dtype=np.uint8)
        for lower, upper in ranges[:32]:
            combined = cast(
                np.ndarray,
                cv2.bitwise_or(combined, self._hsv_mask(hsv_region, lower, upper)),
            )
        matched_count = int(np.count_nonzero((combined > 0) & active_mask))
        return matched_count / active_count >= _threshold(hsv_match_threshold, 0.9)
