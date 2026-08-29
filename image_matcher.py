import logging
from itertools import islice
from typing import Any, cast

import cv2
import numpy as np

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


def _bounded_hsv_range_values(hsv_ranges: Any) -> tuple[Any, ...]:
    try:
        return tuple(islice(iter(hsv_ranges), HSV_REGION_RANGE_LIMIT))
    except TypeError:
        return ()


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
        return cast(np.ndarray, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    if image.ndim == 3 and image.shape[2] == 3:
        return cast(np.ndarray, image)
    if image.ndim == 3 and image.shape[2] == 4:
        return cast(np.ndarray, cv2.cvtColor(image, cv2.COLOR_BGRA2BGR))
    raise ValueError(f"{label} has unsupported shape {image.shape}")


def _normalized_mask(
    mask: Any, template_shape: tuple[int, ...], label: str
) -> np.ndarray | None:
    if mask is None:
        return None
    if not hasattr(mask, "shape") or mask.size == 0:
        raise ValueError(f"{label} mask is empty")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.ndim != 2 or mask.shape[:2] != template_shape[:2]:
        raise ValueError(f"{label} mask is incompatible with the template")
    normalized = np.zeros(mask.shape[:2], dtype=np.uint8)
    normalized[mask > 0] = 255
    if not np.any(normalized):
        raise ValueError(f"{label} mask has no active pixels")
    return normalized


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

    def find_template_candidates(
        self,
        screenshot: np.ndarray,
        template: np.ndarray,
        mask: np.ndarray | None = None,
        threshold: float | None = None,
        min_distance: int = 15,
        template_name: str = "Unknown",
        hsv_ranges: Any = None,
        hsv_match_threshold: float = 0.9,
    ) -> list[MatchCandidate]:
        candidates = self.find_color_gated_template_candidates(
            screenshot,
            template,
            mask,
            threshold,
            min_distance,
            template_name,
            hsv_ranges,
            hsv_match_threshold,
        )
        if hsv_ranges:
            candidates = self.filter_candidates_by_hsv(
                screenshot,
                candidates,
                template,
                mask,
                hsv_ranges,
                hsv_match_threshold,
            )
        return candidates

    def find_all_templates(
        self,
        screenshot: np.ndarray,
        template: np.ndarray,
        mask: np.ndarray | None = None,
        threshold: float | None = None,
        min_distance: int = 15,
        template_name: str = "Unknown",
        hsv_ranges: Any = None,
        hsv_match_threshold: float = 0.9,
    ) -> list[tuple[float, int, int]]:
        candidates = self.find_template_candidates(
            screenshot,
            template,
            mask,
            threshold,
            min_distance,
            template_name,
            hsv_ranges,
            hsv_match_threshold,
        )
        return [
            (confidence, x, y)
            for confidence, x, y, _, _ in self.suppress_overlaps(candidates, 0.20)
        ]

    def suppress_overlaps(
        self, candidates: list[tuple[Any, ...]], iou_threshold: float = 0.20
    ) -> list[tuple[Any, ...]]:
        return self.filter_candidates_with_supervision_nms(candidates, iou_threshold)

    def find_color_gated_template_candidates(
        self,
        screenshot: np.ndarray,
        template: np.ndarray,
        mask: np.ndarray | None = None,
        threshold: float | None = None,
        min_distance: int = 15,
        template_name: str = "Unknown",
        hsv_ranges: Any = None,
        hsv_match_threshold: float | None = None,
        hsv_mask: np.ndarray | None = None,
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
        for region in self._color_candidate_regions(
            screenshot,
            template,
            hsv_ranges,
            mask,
            hsv_match_threshold,
            min_distance,
            hsv_mask,
        ):
            for candidate in self._region_template_candidates(
                screenshot,
                template,
                mask,
                region,
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

    def filter_candidates_by_hsv(
        self,
        screenshot: np.ndarray,
        candidates: list[tuple[Any, ...]],
        template: np.ndarray,
        mask: np.ndarray | None = None,
        hsv_ranges: Any = None,
        hsv_match_threshold: float = 0.9,
        hsv_mask: np.ndarray | None = None,
    ) -> list[tuple[Any, ...]]:
        if not candidates:
            return []
        try:
            screenshot = _as_bgr(screenshot, "screenshot")
            template = _as_bgr(template, "template")
            mask = _normalized_mask(mask, template.shape, "template")
            combined_mask = (
                self._validated_hsv_mask(hsv_mask, screenshot.shape)
                if hsv_mask is not None
                else self._combined_hsv_mask(screenshot, hsv_ranges)
            )
        except (ValueError, cv2.error) as exc:
            logger.warning("Invalid HSV candidate input: %s", exc)
            return []
        if combined_mask is None or not np.any(combined_mask):
            return []

        screenshot_height, screenshot_width = screenshot.shape[:2]
        minimum_ratio = _threshold(hsv_match_threshold, 0.9)
        scaled_masks: dict[tuple[int, int], np.ndarray] = {}
        return [
            candidate
            for candidate in candidates
            if self._candidate_passes_hsv_gate(
                candidate,
                combined_mask,
                mask,
                scaled_masks,
                screenshot_width,
                screenshot_height,
                minimum_ratio,
            )
        ]

    @staticmethod
    def _candidate_passes_hsv_gate(
        candidate: tuple[Any, ...],
        combined_mask: np.ndarray,
        template_mask: np.ndarray | None,
        scaled_masks: dict[tuple[int, int], np.ndarray],
        screenshot_width: int,
        screenshot_height: int,
        minimum_ratio: float,
    ) -> bool:
        geometry = ImageMatcher._candidate_geometry(
            candidate, screenshot_width, screenshot_height
        )
        if geometry is None:
            return False
        left, top, right, bottom, width, height = geometry
        region_mask = combined_mask[top:bottom, left:right] > 0
        if template_mask is None:
            active_count = width * height
            matched_count = int(np.count_nonzero(region_mask))
        else:
            mask_key = width, height
            active_mask = scaled_masks.get(mask_key)
            if active_mask is None:
                active_mask = (
                    template_mask > 0
                    if template_mask.shape == (height, width)
                    else cv2.resize(
                        template_mask,
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    > 0
                )
                scaled_masks[mask_key] = active_mask
            active_count = int(np.count_nonzero(active_mask))
            if active_count <= 0:
                return False
            matched_count = int(np.count_nonzero(region_mask & active_mask))
        return matched_count / active_count >= minimum_ratio

    @staticmethod
    def _candidate_geometry(
        candidate: tuple[Any, ...], screenshot_width: int, screenshot_height: int
    ) -> tuple[int, int, int, int, int, int] | None:
        try:
            _, center_x, center_y, width, height = candidate[:5]
            numeric_values = tuple(
                float(value) for value in (center_x, center_y, width, height)
            )
        except (TypeError, ValueError):
            return None
        if not all(np.isfinite(value) for value in numeric_values):
            return None
        center_x, center_y, width, height = map(int, numeric_values)
        if width <= 0 or height <= 0:
            return None
        left, top = center_x - width // 2, center_y - height // 2
        right, bottom = left + width, top + height
        if left < 0 or top < 0:
            return None
        if right > screenshot_width or bottom > screenshot_height:
            return None
        return left, top, right, bottom, width, height

    def _region_template_candidates(
        self,
        screenshot: np.ndarray,
        template: np.ndarray,
        mask: np.ndarray | None,
        region: MatchRegion,
        max_score: float,
        min_distance: int,
        template_name: str,
    ) -> list[MatchCandidate]:
        left, top, right, bottom = region
        regional_screenshot = screenshot[top:bottom, left:right]
        return [
            self._offset_candidate(candidate, left, top)
            for candidate in self._template_candidates_for_scale(
                regional_screenshot,
                template,
                mask,
                max_score,
                min_distance,
                template_name,
            )
        ]

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
        self,
        screenshot: np.ndarray,
        template: np.ndarray,
        hsv_ranges: Any,
        mask: np.ndarray | None = None,
        hsv_match_threshold: float | None = None,
        min_distance: int = 0,
        hsv_mask: np.ndarray | None = None,
    ) -> list[MatchRegion]:
        full_frame_region = self._full_frame_region(screenshot)
        if hsv_ranges is None:
            return [full_frame_region]
        combined_mask = (
            self._validated_hsv_mask(hsv_mask, screenshot.shape)
            if hsv_mask is not None
            else self._combined_hsv_mask(screenshot, hsv_ranges)
        )
        if combined_mask is None or not np.any(combined_mask):
            return []
        if hsv_match_threshold is not None:
            regions = self._hsv_gate_match_regions(
                combined_mask,
                screenshot.shape,
                template.shape,
                mask,
                hsv_match_threshold,
                min_distance,
            )
            if not regions:
                return []
            if len(
                regions
            ) > HSV_REGION_COMPONENT_LIMIT or self._regions_cover_too_much_area(
                regions, screenshot.shape
            ):
                return [full_frame_region]
            return regions
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

    def _hsv_gate_match_regions(
        self,
        combined_mask: np.ndarray,
        screenshot_shape: tuple[int, ...],
        template_shape: tuple[int, ...],
        mask: np.ndarray | None,
        hsv_match_threshold: float,
        min_distance: int,
    ) -> list[MatchRegion]:
        screenshot_height, screenshot_width = screenshot_shape[:2]
        template_height, template_width = template_shape[:2]
        if template_height > screenshot_height or template_width > screenshot_width:
            return []
        active_mask = (
            np.ones((template_height, template_width), dtype=np.float32)
            if mask is None
            else (mask > 0).astype(np.float32)
        )
        active_count = int(np.count_nonzero(active_mask))
        if active_count <= 0:
            return []
        coverage = cv2.matchTemplate(
            (combined_mask > 0).astype(np.float32), active_mask, cv2.TM_CCORR
        )
        required = _threshold(hsv_match_threshold, 0.9) * active_count
        valid_locations = (coverage >= required - 0.5).astype(np.uint8)
        if not np.any(valid_locations):
            return []
        _, _, stats, _ = cv2.connectedComponentsWithStats(
            valid_locations, connectivity=8
        )
        window = max(3, int(min_distance))
        if window % 2 == 0:
            window += 1
        padding = window // 2
        regions = []
        for component_index in range(1, stats.shape[0]):
            left = int(stats[component_index, cv2.CC_STAT_LEFT])
            top = int(stats[component_index, cv2.CC_STAT_TOP])
            width = int(stats[component_index, cv2.CC_STAT_WIDTH])
            height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
            regions.append(
                (
                    max(0, left - padding),
                    max(0, top - padding),
                    min(
                        int(screenshot_width),
                        left + width - 1 + padding + template_width,
                    ),
                    min(
                        int(screenshot_height),
                        top + height - 1 + padding + template_height,
                    ),
                )
            )
        return self._merge_match_regions(regions)

    def _combined_hsv_mask(
        self, screenshot: np.ndarray, hsv_ranges: Any
    ) -> np.ndarray | None:
        hsv_range_values = _bounded_hsv_range_values(hsv_ranges)
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

    def build_hsv_mask(
        self, screenshot: np.ndarray, hsv_ranges: Any
    ) -> np.ndarray | None:
        """Build one reusable HSV mask for all templates matched on a frame."""
        try:
            screenshot = _as_bgr(screenshot, "screenshot")
        except (ValueError, cv2.error) as exc:
            logger.warning("Invalid HSV mask input: %s", exc)
            return None
        return self._combined_hsv_mask(screenshot, hsv_ranges)

    @staticmethod
    def _validated_hsv_mask(
        hsv_mask: np.ndarray, screenshot_shape: tuple[int, ...]
    ) -> np.ndarray:
        mask = np.asarray(hsv_mask)
        if mask.ndim != 2 or mask.shape != screenshot_shape[:2]:
            raise ValueError("HSV mask does not match screenshot dimensions")
        return mask

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
        max_score: float,
        min_distance: int,
        template_name: str,
    ) -> list[MatchCandidate]:
        result = _match_template(screenshot, template, mask, template_name)
        if result is None:
            return []
        height, width = template.shape[:2]
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
    ) -> list[tuple[Any, ...]]:
        if not candidates:
            return []
        ranked = []
        for index, candidate in enumerate(candidates[:MAX_TEMPLATE_CANDIDATES]):
            box = _candidate_box(candidate)
            try:
                confidence = float(candidate[0])
            except (IndexError, TypeError, ValueError):
                continue
            if box is None or not np.isfinite(confidence):
                continue
            try:
                class_id = 0 if class_ids is None else int(class_ids[index])
            except (IndexError, TypeError, ValueError):
                class_id = 0
            ranked.append((confidence, index, class_id, box, candidate))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        retained: list[tuple[Any, ...]] = []
        threshold = _threshold(iou_threshold, 0.5)
        for item in ranked:
            if all(
                (not class_agnostic and item[2] != kept[2])
                or self._box_iou(item[3], kept[3]) <= threshold
                for kept in retained
            ):
                retained.append(item)
        return [item[4] for item in retained]

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
    def _box_iou(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _normalize_hsv_range(hsv_range: Any) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            lower, upper = hsv_range
            lower = np.asarray(lower, dtype=np.float64)
            upper = np.asarray(upper, dtype=np.float64)
        except (OverflowError, TypeError, ValueError):
            return None
        if lower.shape != (3,) or upper.shape != (3,):
            return None
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
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
            for value in _bounded_hsv_range_values(hsv_ranges)
            if (item := self._normalize_hsv_range(value)) is not None
        ]
        if not ranges:
            return False
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        combined = np.zeros((height, width), dtype=np.uint8)
        for lower, upper in ranges:
            combined = cast(
                np.ndarray,
                cv2.bitwise_or(combined, self._hsv_mask(hsv_region, lower, upper)),
            )
        matched_count = int(np.count_nonzero((combined > 0) & active_mask))
        return matched_count / active_count >= _threshold(hsv_match_threshold, 0.9)
