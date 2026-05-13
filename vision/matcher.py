import logging
from typing import Any

import cv2
import numpy as np

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
Point = tuple[int, int]
HsvRange = tuple[np.ndarray, np.ndarray]


class ImageMatcher:
    def __init__(self, threshold: float = 0.85) -> None:
        self.threshold = self._normalize_threshold(threshold)

    @staticmethod
    def _normalize_threshold(value: Any, default: float = 0.85) -> float:
        try:
            fallback = float(default)
        except (TypeError, ValueError):
            fallback = 0.85
        if not np.isfinite(fallback):
            fallback = 0.85
        fallback = max(0.0, min(1.0, fallback))

        try:
            threshold = float(value)
        except (TypeError, ValueError):
            return fallback
        if not np.isfinite(threshold):
            return fallback
        return max(0.0, min(1.0, threshold))

    @staticmethod
    def _normalize_image(image: np.ndarray, label: str) -> np.ndarray:
        if image is None or not hasattr(image, "shape") or image.size == 0:
            raise ValueError(f"{label} is empty")
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        raise ValueError(f"{label} has unsupported shape {image.shape}")

    @staticmethod
    def _normalize_mask(mask: np.ndarray | None, template_shape: tuple[int, ...], template_name: str) -> np.ndarray | None:
        if mask is None:
            return None
        if not hasattr(mask, "shape") or mask.size == 0:
            logger.warning("[%s] Ignoring empty mask", template_name)
            return None
        if mask.ndim == 3:
            try:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            except cv2.error as exc:
                logger.warning("[%s] Ignoring unsupported mask: %s", template_name, exc)
                return None
        elif mask.ndim != 2:
            logger.warning("[%s] Ignoring mask with unsupported shape %s", template_name, mask.shape)
            return None
        if mask.shape[:2] != template_shape[:2]:
            logger.warning(
                "[%s] Ignoring mask with shape %s for template shape %s",
                template_name,
                mask.shape,
                template_shape,
            )
            return None
        normalized = np.zeros(mask.shape[:2], dtype=np.uint8)
        normalized[mask > 0] = 255
        if not np.any(normalized):
            logger.warning("[%s] Ignoring mask without active pixels", template_name)
            return None
        return normalized

    @staticmethod
    def _safe_match_template(
        screenshot: np.ndarray,
        template: np.ndarray,
        mask: np.ndarray | None,
        template_name: str,
    ) -> np.ndarray | None:
        try:
            result = cv2.matchTemplate(screenshot, template, cv2.TM_SQDIFF_NORMED, mask=mask)
        except cv2.error as exc:
            logger.warning("[%s] Template matching failed: %s", template_name, exc)
            return None
        if result.size == 0:
            return None
        result = np.nan_to_num(result, nan=1.0, posinf=1.0, neginf=1.0)
        return np.clip(result, 0.0, 1.0)

    @staticmethod
    def supervision_available() -> bool:
        return sv is not None

    @staticmethod
    def _candidate_xyxy(candidate: tuple[Any, ...]) -> tuple[float, float, float, float] | None:
        try:
            _, center_x, center_y, width, height = candidate[:5]
            center_x = float(center_x)
            center_y = float(center_y)
            width = float(width)
            height = float(height)
        except (TypeError, ValueError):
            return None

        if not all(np.isfinite(value) for value in (center_x, center_y, width, height)):
            return None
        if width <= 0 or height <= 0:
            return None

        half_width = width / 2.0
        half_height = height / 2.0
        return (
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height,
        )

    @staticmethod
    def _candidate_class_id(class_ids: list[int] | tuple[int, ...] | None, index: int) -> int:
        if class_ids is None:
            return 0
        try:
            return int(class_ids[index])
        except (IndexError, TypeError, ValueError):
            return 0

    @classmethod
    def _normalized_detection_candidate(
        cls: type["ImageMatcher"],
        index: int,
        candidate: tuple[Any, ...],
        class_ids: list[int] | tuple[int, ...] | None,
    ) -> tuple[tuple[float, float, float, float], float, int, int] | None:
        xyxy = cls._candidate_xyxy(candidate)
        if xyxy is None:
            return None
        try:
            confidence = float(candidate[0])
        except (TypeError, ValueError):
            return None
        if not np.isfinite(confidence):
            return None
        return (
            xyxy,
            max(0.0, min(1.0, confidence)),
            cls._candidate_class_id(class_ids, index),
            index,
        )

    @classmethod
    def _candidate_detection_arrays(
        cls: type["ImageMatcher"],
        candidates: list[tuple[Any, ...]],
        class_ids: list[int] | tuple[int, ...] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        normalized_candidates = [
            normalized_candidate
            for index, candidate in enumerate(candidates)
            if (normalized_candidate := cls._normalized_detection_candidate(index, candidate, class_ids)) is not None
        ]

        if not normalized_candidates:
            return None

        boxes, confidences, normalized_class_ids, candidate_indexes = zip(*normalized_candidates)
        return (
            np.asarray(boxes, dtype=np.float32),
            np.asarray(confidences, dtype=np.float32),
            np.asarray(normalized_class_ids, dtype=np.int32),
            np.asarray(candidate_indexes, dtype=np.int32),
        )

    @staticmethod
    def _filtered_detection_candidate_indexes(
        detections: Any,
        boxes: np.ndarray,
        candidate_indexes: np.ndarray,
    ) -> list[int] | None:
        if detections is None or len(detections) == 0:
            return []

        data = getattr(detections, "data", None)
        if isinstance(data, dict) and "candidate_index" in data:
            return [int(index) for index in data["candidate_index"]]

        resolved_indexes = []
        used_source_indexes: set[int] = set()
        for filtered_box in detections.xyxy:
            match_index = None
            for source_index, source_box in enumerate(boxes):
                if source_index in used_source_indexes:
                    continue
                if np.allclose(source_box, filtered_box, atol=0.5):
                    match_index = source_index
                    break
            if match_index is None:
                return None
            used_source_indexes.add(match_index)
            resolved_indexes.append(int(candidate_indexes[match_index]))
        return resolved_indexes

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

        arrays = self._candidate_detection_arrays(candidates, class_ids=class_ids)
        if arrays is None:
            return []
        boxes, confidences, normalized_class_ids, candidate_indexes = arrays

        try:
            detections = sv.Detections(
                xyxy=boxes,
                confidence=confidences,
                class_id=normalized_class_ids,
            )
            detections["candidate_index"] = candidate_indexes
            filtered = detections.with_nms(
                threshold=self._normalize_threshold(iou_threshold, 0.5),
                class_agnostic=bool(class_agnostic),
            )
        except Exception as exc:
            logger.debug("Supervision NMS failed: %s", exc)
            return None

        retained_indexes = self._filtered_detection_candidate_indexes(filtered, boxes, candidate_indexes)
        if retained_indexes is None:
            return None

        retained = [candidates[index] for index in retained_indexes if 0 <= index < len(candidates)]
        return sorted(retained, key=lambda candidate: candidate[0], reverse=True)

    @staticmethod
    def _failed_match(confidence: float = 0.0) -> MatchResult:
        return False, float(confidence), 0, 0

    @staticmethod
    def _template_fits_screenshot(screenshot: np.ndarray, template: np.ndarray, template_name: str) -> bool:
        if template.shape[0] <= screenshot.shape[0] and template.shape[1] <= screenshot.shape[1]:
            return True
        logger.debug(
            "Template is larger than screenshot. Template %s: %s, Screenshot: %s",
            template_name,
            template.shape,
            screenshot.shape,
        )
        return False

    @staticmethod
    def _center_from_location(location: Point, template: np.ndarray) -> Point:
        template_height, template_width = template.shape[:2]
        return location[0] + template_width // 2, location[1] + template_height // 2

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
        template = self._normalize_image(template, str(template_path))
        
        return template, mask

    def _passes_optional_gates(
        self,
        screenshot: np.ndarray,
        template: np.ndarray,
        location: Point,
        mask: np.ndarray | None,
        template_name: str,
        center_x: int,
        center_y: int,
        confidence: float,
        check_color: bool,
        color_threshold: float,
        hsv_ranges: Any,
        hsv_match_threshold: float,
    ) -> bool:
        if check_color and not self._check_color_similarity(
            screenshot,
            template,
            location,
            mask,
            color_threshold=color_threshold,
        ):
            logger.debug(
                "[%s] Color check failed at (%s, %s), confidence: %.2f%%",
                template_name,
                center_x,
                center_y,
                confidence * 100,
            )
            return False

        if not hsv_ranges:
            return True

        hsv_match = self._check_hsv_gate(
            screenshot,
            template,
            location,
            mask,
            hsv_ranges,
            hsv_match_threshold,
        )
        if hsv_match:
            return True
        logger.debug(
            "[%s] HSV gate failed at (%s, %s), confidence: %.2f%%",
            template_name,
            center_x,
            center_y,
            confidence * 100,
        )
        return False

    def find_template(
        self,
        screenshot: np.ndarray,
        template: np.ndarray,
        mask: np.ndarray | None = None,
        threshold: float | None = None,
        template_name: str = "Unknown",
        check_color: bool = False,
        color_threshold: float = 0.7,
        hsv_ranges: Any = None,
        hsv_match_threshold: float = 0.9,
    ) -> MatchResult:
        thresh = self.threshold if threshold is None else self._normalize_threshold(threshold, self.threshold)
        try:
            screenshot = self._normalize_image(screenshot, "screenshot")
            template = self._normalize_image(template, template_name)
        except ValueError as exc:
            logger.warning("[%s] Invalid match input: %s", template_name, exc)
            return self._failed_match()
        mask = self._normalize_mask(mask, template.shape, template_name)

        if not self._template_fits_screenshot(screenshot, template, template_name):
            return self._failed_match()

        result = self._safe_match_template(screenshot, template, mask, template_name)
        if result is None:
            return self._failed_match()

        min_value, _, min_location, _ = cv2.minMaxLoc(result)
        confidence = float(1.0 - min_value)
        if not np.isfinite(confidence):
            return self._failed_match()
        
        if confidence < thresh:
            return self._failed_match(confidence)

        center_x, center_y = self._center_from_location(min_location, template)
        if not self._passes_optional_gates(
            screenshot,
            template,
            min_location,
            mask,
            template_name,
            center_x,
            center_y,
            confidence,
            check_color,
            color_threshold,
            hsv_ranges,
            hsv_match_threshold,
        ):
            return self._failed_match(confidence)
        return True, confidence, center_x, center_y

    def _check_color_similarity(
        self,
        screenshot: np.ndarray,
        template: np.ndarray,
        location: Point,
        mask: np.ndarray | None = None,
        color_threshold: float = 0.7,
    ) -> bool:
        x, y = location
        template_height, template_width = template.shape[:2]
        
        region_of_interest = screenshot[y : y + template_height, x : x + template_width]
        
        if region_of_interest.shape[:2] != template.shape[:2]:
            return False

        if mask is not None and not np.any(mask):
            return False

        try:
            hist_template_b = cv2.calcHist([template], [0], mask, [32], [0, 256])
            hist_template_g = cv2.calcHist([template], [1], mask, [32], [0, 256])
            hist_template_r = cv2.calcHist([template], [2], mask, [32], [0, 256])

            hist_region_b = cv2.calcHist([region_of_interest], [0], mask, [32], [0, 256])
            hist_region_g = cv2.calcHist([region_of_interest], [1], mask, [32], [0, 256])
            hist_region_r = cv2.calcHist([region_of_interest], [2], mask, [32], [0, 256])

            histograms = (
                hist_template_b,
                hist_template_g,
                hist_template_r,
                hist_region_b,
                hist_region_g,
                hist_region_r,
            )
            if any(not np.any(hist) for hist in histograms):
                return False

            for hist in histograms:
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

            correlation_blue = cv2.compareHist(hist_template_b, hist_region_b, cv2.HISTCMP_CORREL)
            correlation_green = cv2.compareHist(hist_template_g, hist_region_g, cv2.HISTCMP_CORREL)
            correlation_red = cv2.compareHist(hist_template_r, hist_region_r, cv2.HISTCMP_CORREL)
        except cv2.error as exc:
            logger.debug("Color similarity check failed: %s", exc)
            return False
        
        correlations = (correlation_blue, correlation_green, correlation_red)
        if not all(np.isfinite(value) for value in correlations):
            return False

        average_correlation = sum(correlations) / 3
        return average_correlation >= self._normalize_threshold(color_threshold, 0.7)

    @staticmethod
    def _normalize_hsv_component(values: np.ndarray) -> np.ndarray:
        return np.array(
            [
                max(0, min(179, int(values[0]))),
                max(0, min(255, int(values[1]))),
                max(0, min(255, int(values[2]))),
            ],
            dtype=np.uint8,
        )

    @classmethod
    def _normalize_hsv_range(cls: type["ImageMatcher"], hsv_range: Any) -> HsvRange | None:
        try:
            lower, upper = hsv_range
            lower = np.array(lower, dtype=np.int16)
            upper = np.array(upper, dtype=np.int16)
        except (TypeError, ValueError):
            return None
        if lower.shape != (3,) or upper.shape != (3,):
            return None
        return cls._normalize_hsv_component(lower), cls._normalize_hsv_component(upper)

    @classmethod
    def _normalize_hsv_ranges(cls: type["ImageMatcher"], hsv_ranges: Any) -> list[HsvRange]:
        return [
            normalized_range
            for hsv_range in hsv_ranges
            if (normalized_range := cls._normalize_hsv_range(hsv_range)) is not None
        ]

    @staticmethod
    def _apply_hsv_range_mask(hsv_region: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
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
        template_height, template_width = template.shape[:2]
        region_of_interest = screenshot[y : y + template_height, x : x + template_width]

        if region_of_interest.shape[:2] != template.shape[:2]:
            return False

        if mask is None:
            active_mask = np.ones((template_height, template_width), dtype=bool)
        else:
            active_mask = mask > 0

        active_count = int(np.count_nonzero(active_mask))
        if active_count <= 0:
            return False

        ranges = self._normalize_hsv_ranges(hsv_ranges)
        if not ranges:
            return False

        try:
            hsv_region = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2HSV)
        except cv2.error as exc:
            logger.debug("HSV gate conversion failed: %s", exc)
            return False

        combined = np.zeros((template_height, template_width), dtype=np.uint8)
        for lower, upper in ranges:
            combined = cv2.bitwise_or(combined, self._apply_hsv_range_mask(hsv_region, lower, upper))

        matched_count = int(np.count_nonzero((combined > 0) & active_mask))
        match_ratio = matched_count / active_count
        return match_ratio >= self._normalize_threshold(hsv_match_threshold, 0.9)

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
        all_matches = self.find_template_candidates(
            screenshot,
            template,
            mask=mask,
            threshold=threshold,
            min_distance=min_distance,
            scales=scales,
            template_name=template_name,
        )
        if all_matches:
            all_matches = self._non_max_suppression(all_matches, min_distance)
            if use_supervision_nms:
                supervision_matches = self.filter_candidates_with_supervision_nms(
                    all_matches,
                    iou_threshold=supervision_iou_threshold,
                    class_agnostic=supervision_class_agnostic,
                )
                if supervision_matches is not None:
                    all_matches = supervision_matches
        return [(conf, x, y) for conf, x, y, _, _ in all_matches]

    @staticmethod
    def _scaled_template_and_mask(
        template: np.ndarray,
        mask: np.ndarray | None,
        scale: float,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if scale == 1.0:
            return template, mask
        scaled_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if mask is None:
            return scaled_template, None
        scaled_mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        scaled_mask[scaled_mask > 0] = 255
        return scaled_template, scaled_mask

    @staticmethod
    def _valid_scale(scale: Any) -> float | None:
        try:
            normalized_scale = float(scale)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(normalized_scale) or normalized_scale <= 0:
            return None
        return normalized_scale

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
        thresh = self.threshold if threshold is None else self._normalize_threshold(threshold, self.threshold)
        all_matches = []
        try:
            screenshot = self._normalize_image(screenshot, "screenshot")
            template = self._normalize_image(template, template_name)
        except ValueError as exc:
            logger.warning("[%s] Invalid multi-match input: %s", template_name, exc)
            return []
        mask = self._normalize_mask(mask, template.shape, template_name)

        if scales is None:
            scales = [1.0]

        for scale_value in scales:
            scale = self._valid_scale(scale_value)
            if scale is None:
                continue
            scaled_template, scaled_mask = self._scaled_template_and_mask(template, mask, scale)

            if scaled_template.shape[0] > screenshot.shape[0] or scaled_template.shape[1] > screenshot.shape[1]:
                continue

            result = self._safe_match_template(screenshot, scaled_template, scaled_mask, template_name)
            if result is None:
                continue

            template_height, template_width = scaled_template.shape[:2]
            candidates = self._local_minima_candidates(result, 1.0 - thresh, min_distance)
            for candidate_x, candidate_y in candidates:
                confidence = float(1.0 - result[candidate_y, candidate_x])
                if not np.isfinite(confidence):
                    continue
                center_x = candidate_x + template_width // 2
                center_y = candidate_y + template_height // 2
                all_matches.append((confidence, center_x, center_y, template_width, template_height))

        return sorted(all_matches, key=lambda match: match[0], reverse=True)

    @staticmethod
    def _local_minima_candidates(result: np.ndarray | None, max_score: float, min_distance: int) -> list[Point]:
        if result is None or result.size == 0:
            return []
        window = max(3, int(min_distance))
        if window % 2 == 0:
            window += 1
        kernel = np.ones((window, window), dtype=np.float32)
        local_min = cv2.erode(result, kernel)
        candidate_mask = (result <= max_score) & (result <= local_min + 1e-6)
        if not np.any(candidate_mask):
            return []
        mask = candidate_mask.astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        candidates = []
        for index in range(1, count):
            component_x, component_y, component_width, component_height, component_area = stats[index]
            if component_area <= 0:
                continue
            region = result[
                component_y : component_y + component_height,
                component_x : component_x + component_width,
            ]
            min_value, _, min_location, _ = cv2.minMaxLoc(region)
            if min_value <= max_score:
                candidates.append((int(component_x + min_location[0]), int(component_y + min_location[1])))
        return candidates

    @staticmethod
    def _box_intersection_over_union(first_match: MatchCandidate, second_match: MatchCandidate) -> float:
        _, first_x, first_y, first_width, first_height = first_match
        _, second_x, second_y, second_width, second_height = second_match
        left = max(first_x - first_width // 2, second_x - second_width // 2)
        top = max(first_y - first_height // 2, second_y - second_height // 2)
        right = min(first_x + first_width // 2, second_x + second_width // 2)
        bottom = min(first_y + first_height // 2, second_y + second_height // 2)
        if right <= left or bottom <= top:
            return 0.0
        intersection = (right - left) * (bottom - top)
        first_area = first_width * first_height
        second_area = second_width * second_height
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    @classmethod
    def _overlaps_existing_match(
        cls: type["ImageMatcher"],
        candidate_match: MatchCandidate,
        filtered_match: MatchCandidate,
        min_distance: int,
    ) -> bool:
        _, candidate_x, candidate_y, _, _ = candidate_match
        _, filtered_x, filtered_y, _, _ = filtered_match
        if abs(candidate_x - filtered_x) >= min_distance or abs(candidate_y - filtered_y) >= min_distance:
            return False
        return cls._box_intersection_over_union(candidate_match, filtered_match) > 0.2

    def _non_max_suppression(self, matches: list[MatchCandidate], min_distance: int) -> list[MatchCandidate]:
        if not matches:
            return []

        matches = sorted(matches, key=lambda match: match[0], reverse=True)
        filtered = []

        for candidate_match in matches:
            if not any(
                self._overlaps_existing_match(candidate_match, filtered_match, min_distance)
                for filtered_match in filtered
            ):
                filtered.append(candidate_match)

        return filtered
