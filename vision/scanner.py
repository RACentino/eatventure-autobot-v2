import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core import config
from bot.types import BoxCandidate, MatchCandidate, RedIcon, TemplatePair

logger = logging.getLogger(__name__)
RED_ICON_ASSET_MIN_RED_PIXELS = 10
RED_ICON_ASSET_MIN_HIGHLIGHT_PIXELS = 8
RED_ICON_ASSET_MIN_RED_SUPPORT_RATIO = 0.60
RED_ICON_ASSET_MIN_HIGHLIGHT_SUPPORT_RATIO = 0.55
RED_ICON_ASSET_HIGHLIGHT_SATURATION_MAX = 95
RED_ICON_ASSET_HIGHLIGHT_VALUE_MIN = 165


class VisionScannerMixin:
    def load_templates(self) -> dict[str, TemplatePair]:
        templates: dict[str, TemplatePair] = {}
        templates_path = Path(config.ASSETS_DIR)
        if not templates_path.exists():
            logger.error("Assets directory not found: %s", templates_path)
            return templates

        for template_file in sorted(templates_path.glob("*.png")):
            try:
                template_name = template_file.stem
                template_img = self.image_matcher.load_template(template_file)
                templates[template_name] = template_img
                logger.info("Loaded template: %s", template_name)
            except Exception as exc:
                logger.error("Failed to load template %s: %s", template_file, exc)

        return templates

    def _available_red_icon_template_count(self) -> int:
        return sum(1 for name in self.templates if name.startswith("RedIcon"))

    def _red_icon_min_matches(self) -> int:
        if bool(config.RED_ICON_FAST_MODE_ENABLED):
            return 1
        available = self._available_red_icon_template_count()
        if available <= 0:
            return 1
        configured = max(1, int(config.RED_ICON_MIN_MATCHES))
        return min(configured, available)

    def _validate_required_templates(self) -> bool:
        missing = [name for name in ("newLevel", "unlock", "upgradeStation") if name not in self.templates]
        red_icon_count = self._available_red_icon_template_count()
        if red_icon_count <= 0:
            missing.append("RedIcon*")
        if missing:
            logger.error("Missing required templates: %s", ", ".join(missing))
            return False
        if red_icon_count < int(config.RED_ICON_MIN_MATCHES):
            logger.warning(
                "Only %s red-icon templates are available; consensus requirement reduced from %s",
                red_icon_count,
                config.RED_ICON_MIN_MATCHES,
            )
        return True

    def _template(self, template_name: str) -> TemplatePair | None:
        return self.templates.get(template_name)

    def _new_level_threshold(self) -> float:
        if self.vision_optimizer.enabled:
            return self.vision_optimizer.new_level_threshold
        return config.NEW_LEVEL_THRESHOLD

    def _red_icon_scan_threshold(self) -> float:
        if not self.vision_optimizer.enabled:
            return config.RED_ICON_THRESHOLD
        return min(
            self.vision_optimizer.red_icon_threshold,
            self.vision_optimizer.new_level_red_icon_threshold,
        )

    def _box_threshold(self) -> float:
        if self.vision_optimizer.enabled:
            return self.vision_optimizer.box_threshold
        return config.BOX_THRESHOLD

    def _stats_upgrade_threshold(self) -> float:
        if self.vision_optimizer.enabled:
            return self.vision_optimizer.stats_upgrade_threshold
        return config.STATS_RED_ICON_THRESHOLD

    @staticmethod
    def _supervision_nms_enabled(detector_name: str) -> bool:
        if not bool(config.SUPERVISION_ENABLED):
            return False
        if detector_name == "box":
            return bool(config.SUPERVISION_BOX_NMS_ENABLED)
        if detector_name == "red_icon":
            return bool(config.SUPERVISION_RED_ICON_NMS_ENABLED)
        if detector_name == "upgrade_station":
            return bool(config.SUPERVISION_UPGRADE_STATION_NMS_ENABLED)
        return False

    def _scan_red_icon_frame(
        self,
        screenshot: Any,
        limited_screenshot: Any,
        scan_threshold: float,
        min_matches: int,
    ) -> tuple[list[RedIcon], list[float], RedIcon | None]:
        all_detections = self._collect_red_icon_detections(
            limited_screenshot,
            scan_threshold,
            min_distance=80,
        )
        red_icons, valid_red_icon_confidences = self._icons_from_detections(
            all_detections,
            min_matches,
        )
        best_new_level_icon = self._find_new_level_red_icon(screenshot, scan_threshold, min_matches)
        return red_icons, valid_red_icon_confidences, best_new_level_icon

    def _find_new_level_button(self, screenshot: Any) -> tuple[bool, float, int, int]:
        template_pair = self._template("newLevel")
        if template_pair is None:
            return False, 0.0, 0, 0
        template, mask = template_pair
        return self.image_matcher.find_template(
            screenshot,
            template,
            mask=mask,
            threshold=self._new_level_threshold(),
            template_name="newLevel",
        )

    def _red_icon_template_names(self) -> list[str]:
        cached = getattr(self, "_red_icon_template_names_cache", None)
        if cached is not None:
            return cached
        template_names = [name for name in self.templates if name.startswith("RedIcon")]
        if not template_names:
            return ["RedIcon"]

        def sort_key(name: str) -> tuple[int, Any]:
            if name == "RedIcon":
                return (0, 0)
            if name == "RedIconNoBG":
                return (2, 0)
            suffix = name.replace("RedIcon", "", 1)
            if suffix.isdigit():
                return (1, int(suffix))
            return (3, suffix)

        return sorted(template_names, key=sort_key)

    @staticmethod
    def _template_size(template_pair: TemplatePair) -> tuple[int, int]:
        template, _ = template_pair
        return int(template.shape[1]), int(template.shape[0])

    def _red_icon_template_span(self) -> tuple[int, int]:
        template_sizes = [
            self._template_size(self.templates[template_name])
            for template_name in self._red_icon_template_names()
            if template_name in self.templates
        ]
        if not template_sizes:
            return 0, 0
        return (
            max(template_width for template_width, _ in template_sizes),
            max(template_height for _, template_height in template_sizes),
        )

    @staticmethod
    def _extract_region(
        screenshot: Any,
        x_min: Any,
        x_max: Any,
        y_min: Any,
        y_max: Any,
        pad_x: Any = 0,
        pad_y: Any = 0,
    ) -> tuple[Any, int, int]:
        height, width = screenshot.shape[:2]
        left = max(0, int(x_min) - int(pad_x))
        right = min(width, int(x_max) + int(pad_x))
        top = max(0, int(y_min) - int(pad_y))
        bottom = min(height, int(y_max) + int(pad_y))
        if left >= right or top >= bottom:
            return screenshot[0:0, 0:0], 0, 0
        return screenshot[top:bottom, left:right], left, top

    @staticmethod
    def _box_iou(first: MatchCandidate, second: MatchCandidate) -> float:
        _, x1, y1, w1, h1 = first[:5]
        _, x2, y2, w2, h2 = second[:5]
        left = max(x1 - w1 / 2, x2 - w2 / 2)
        top = max(y1 - h1 / 2, y2 - h2 / 2)
        right = min(x1 + w1 / 2, x2 + w2 / 2)
        bottom = min(y1 + h1 / 2, y2 + h2 / 2)
        intersection = max(0, right - left) * max(0, bottom - top)
        union = (w1 * h1) + (w2 * h2) - intersection
        return intersection / union if union > 0 else 0

    @classmethod
    def _dedupe_box_candidates(
        cls: type,
        candidates: list[BoxCandidate],
        iou_threshold: float,
    ) -> list[BoxCandidate]:
        ordered_candidates = sorted(candidates, key=lambda item: item[0], reverse=True)
        return cls._dedupe_ordered_box_candidates(ordered_candidates, iou_threshold)

    @classmethod
    def _dedupe_ordered_box_candidates(
        cls: type,
        candidates: list[BoxCandidate],
        iou_threshold: float,
    ) -> list[BoxCandidate]:
        merged = []
        for candidate in candidates:
            if all(cls._box_iou(candidate, existing) <= iou_threshold for existing in merged):
                merged.append(candidate)
        return merged

    @classmethod
    def _merge_box_candidates(cls: type, candidates: list[BoxCandidate]) -> list[BoxCandidate]:
        ordered_candidates = sorted(candidates, key=lambda item: item[0], reverse=True)
        strict = cls._dedupe_ordered_box_candidates(ordered_candidates, 0.20)
        relaxed = cls._dedupe_ordered_box_candidates(ordered_candidates, 0.25)
        if len(relaxed) - len(strict) == 1:
            return relaxed
        return strict

    def _merge_box_candidates_with_supervision(self, candidates: list[BoxCandidate]) -> list[BoxCandidate]:
        legacy_candidates = self._merge_box_candidates(candidates)
        if not candidates or not self._supervision_nms_enabled("box"):
            return legacy_candidates
        supervision_candidates = self.image_matcher.filter_candidates_with_supervision_nms(
            candidates,
            iou_threshold=config.SUPERVISION_BOX_NMS_IOU_THRESHOLD,
            class_agnostic=config.SUPERVISION_CLASS_AGNOSTIC_NMS,
        )
        if supervision_candidates is None or (not supervision_candidates and legacy_candidates):
            return legacy_candidates
        return [candidate for candidate in supervision_candidates]

    @staticmethod
    def _merge_icon_detection(
        detections: dict[tuple[int, int], list[tuple[str, float]]],
        x: int,
        y: int,
        template_name: str,
        confidence: float,
    ) -> None:
        for existing_x, existing_y in detections:
            if abs(x - existing_x) < 10 and abs(y - existing_y) < 10:
                detections[(existing_x, existing_y)].append((template_name, confidence))
                return
        detections[(x, y)] = [(template_name, confidence)]

    def _red_icon_template_match_plan(self, min_distance: int) -> tuple[list[str], int]:
        template_names = self._red_icon_template_names()
        if not bool(config.RED_ICON_FAST_MODE_ENABLED):
            return template_names, min_distance

        fast_template_names = [
            template_name
            for template_name in config.RED_ICON_FAST_TEMPLATE_NAMES
            if template_name in self.templates
        ]
        if not fast_template_names:
            return template_names, min_distance
        return fast_template_names, int(config.RED_ICON_FAST_MIN_DISTANCE)

    def _merge_red_icon_matches(
        self,
        detections: dict[tuple[int, int], list[tuple[str, float]]],
        template_name: str,
        matches: list[tuple[float, int, int]],
        offset_x: int,
        offset_y: int,
    ) -> None:
        for confidence, center_x, center_y in matches:
            self._merge_icon_detection(
                detections,
                center_x + offset_x,
                center_y + offset_y,
                template_name,
                confidence,
            )

    @staticmethod
    def _candidate_region(
        screenshot: Any,
        template: Any,
        center_x: int,
        center_y: int,
    ) -> Any | None:
        template_height, template_width = template.shape[:2]
        left = int(center_x) - template_width // 2
        top = int(center_y) - template_height // 2
        if left < 0 or top < 0:
            return None
        region = screenshot[top : top + template_height, left : left + template_width]
        if region.shape[:2] != template.shape[:2]:
            return None
        return region

    @staticmethod
    def _active_template_mask(template: Any, mask: Any) -> np.ndarray:
        template_height, template_width = template.shape[:2]
        if mask is None:
            return np.ones((template_height, template_width), dtype=bool)
        if mask.shape[:2] != template.shape[:2]:
            return np.ones((template_height, template_width), dtype=bool)
        return mask > 0

    def _red_icon_red_mask(self, image: Any, active_mask: np.ndarray) -> np.ndarray | None:
        color_mask = self.image_matcher.red_hsv_mask(image)
        if color_mask is None or color_mask.shape[:2] != active_mask.shape[:2]:
            return None
        return (color_mask > 0) & active_mask

    @staticmethod
    def _red_icon_highlight_mask(image: Any, active_mask: np.ndarray) -> np.ndarray | None:
        try:
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        except cv2.error:
            return None
        if hsv_image.shape[:2] != active_mask.shape[:2]:
            return None
        return (
            (hsv_image[:, :, 1] <= RED_ICON_ASSET_HIGHLIGHT_SATURATION_MAX)
            & (hsv_image[:, :, 2] >= RED_ICON_ASSET_HIGHLIGHT_VALUE_MIN)
            & active_mask
        )

    @staticmethod
    def _mask_support_ratio(candidate_mask: np.ndarray, required_mask: np.ndarray) -> float:
        required_count = int(np.count_nonzero(required_mask))
        if required_count <= 0:
            return 0.0
        matched_count = int(np.count_nonzero(candidate_mask & required_mask))
        return matched_count / required_count

    def _red_icon_color_support_matches_template(
        self,
        region: Any,
        template: Any,
        active_mask: np.ndarray,
    ) -> bool:
        template_red_mask = self._red_icon_red_mask(template, active_mask)
        region_red_mask = self._red_icon_red_mask(region, active_mask)
        if template_red_mask is None or region_red_mask is None:
            return False
        if int(np.count_nonzero(template_red_mask)) < RED_ICON_ASSET_MIN_RED_PIXELS:
            return False
        return (
            self._mask_support_ratio(region_red_mask, template_red_mask)
            >= RED_ICON_ASSET_MIN_RED_SUPPORT_RATIO
        )

    def _red_icon_highlight_matches_template(
        self,
        region: Any,
        template: Any,
        active_mask: np.ndarray,
    ) -> bool:
        template_highlight_mask = self._red_icon_highlight_mask(template, active_mask)
        region_highlight_mask = self._red_icon_highlight_mask(region, active_mask)
        if template_highlight_mask is None or region_highlight_mask is None:
            return False
        if int(np.count_nonzero(template_highlight_mask)) < RED_ICON_ASSET_MIN_HIGHLIGHT_PIXELS:
            return False
        return (
            self._mask_support_ratio(region_highlight_mask, template_highlight_mask)
            >= RED_ICON_ASSET_MIN_HIGHLIGHT_SUPPORT_RATIO
        )

    def _red_icon_candidate_matches_asset_texture(
        self,
        screenshot: Any,
        template: Any,
        mask: Any,
        center_x: int,
        center_y: int,
    ) -> bool:
        region = self._candidate_region(screenshot, template, center_x, center_y)
        if region is None:
            return False
        active_mask = self._active_template_mask(template, mask)
        return self._red_icon_color_support_matches_template(
            region,
            template,
            active_mask,
        ) and self._red_icon_highlight_matches_template(region, template, active_mask)

    def _verified_red_icon_matches(
        self,
        screenshot: Any,
        template: Any,
        mask: Any,
        template_name: str,
        matches: list[tuple[float, int, int]],
    ) -> list[tuple[float, int, int]]:
        verified_matches = []
        for confidence, center_x, center_y in matches:
            if self._red_icon_candidate_matches_asset_texture(
                screenshot,
                template,
                mask,
                center_x,
                center_y,
            ):
                verified_matches.append((confidence, center_x, center_y))
                continue
            logger.debug(
                "[%s] Rejected red-icon texture candidate at (%s, %s) [%.3f]",
                template_name,
                center_x,
                center_y,
                confidence,
            )
        return verified_matches

    def _collect_template_red_icon_detections(
        self,
        detections: dict[tuple[int, int], list[tuple[str, float]]],
        screenshot: Any,
        template_names: list[str],
        threshold: float,
        min_distance: int,
        offset_x: int,
        offset_y: int,
    ) -> None:
        for template_name in template_names:
            template_pair = self._template(template_name)
            if template_pair is None:
                continue
            template, mask = template_pair
            matches = self.image_matcher.find_all_templates(
                screenshot,
                template,
                mask=mask,
                threshold=threshold,
                min_distance=min_distance,
                template_name=template_name,
                use_supervision_nms=self._supervision_nms_enabled("red_icon"),
                supervision_iou_threshold=config.SUPERVISION_RED_ICON_NMS_IOU_THRESHOLD,
                supervision_class_agnostic=config.SUPERVISION_CLASS_AGNOSTIC_NMS,
            )
            verified_matches = self._verified_red_icon_matches(
                screenshot,
                template,
                mask,
                template_name,
                matches,
            )
            self._merge_red_icon_matches(detections, template_name, verified_matches, offset_x, offset_y)

    def _collect_hsv_red_icon_detections(
        self,
        detections: dict[tuple[int, int], list[tuple[str, float]]],
        screenshot: Any,
        template_names: list[str],
        threshold: float,
        min_distance: int,
        offset_x: int,
        offset_y: int,
    ) -> None:
        screenshot_color_mask = self.image_matcher.red_hsv_mask(screenshot)
        if screenshot_color_mask is None:
            return

        for template_name in template_names:
            template_pair = self._template(template_name)
            if template_pair is None:
                continue
            template, mask = template_pair
            matches = self.image_matcher.find_all_hsv_mask_templates(
                screenshot,
                template,
                mask=mask,
                threshold=threshold,
                min_distance=min_distance,
                template_name=template_name,
                screenshot_color_mask=screenshot_color_mask,
                use_supervision_nms=self._supervision_nms_enabled("red_icon"),
                supervision_iou_threshold=config.SUPERVISION_RED_ICON_NMS_IOU_THRESHOLD,
                supervision_class_agnostic=config.SUPERVISION_CLASS_AGNOSTIC_NMS,
            )
            verified_matches = self._verified_red_icon_matches(
                screenshot,
                template,
                mask,
                template_name,
                matches,
            )
            self._merge_red_icon_matches(detections, template_name, verified_matches, offset_x, offset_y)

    def _collect_red_icon_detections(
        self,
        screenshot: Any,
        threshold: float,
        min_distance: int = 80,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> dict[tuple[int, int], list[tuple[str, float]]]:
        detections: dict[tuple[int, int], list[tuple[str, float]]] = {}
        if screenshot.size == 0:
            return detections

        template_match_names, template_match_min_distance = self._red_icon_template_match_plan(min_distance)
        hsv_match_names = self._red_icon_template_names()
        self._collect_template_red_icon_detections(
            detections,
            screenshot,
            template_match_names,
            threshold,
            template_match_min_distance,
            offset_x,
            offset_y,
        )
        self._collect_hsv_red_icon_detections(
            detections,
            screenshot,
            hsv_match_names,
            threshold,
            template_match_min_distance,
            offset_x,
            offset_y,
        )
        return detections

    @staticmethod
    def _best_confidence_by_template(matches: list[tuple[str, float]]) -> dict[str, float]:
        by_template: dict[str, float] = {}
        for template_name, confidence in matches:
            existing = by_template.get(template_name)
            if existing is None or confidence > existing:
                by_template[template_name] = confidence
        return by_template

    @classmethod
    def _icons_from_detections(
        cls: type,
        detections: dict[tuple[int, int], list[tuple[str, float]]],
        min_matches: int,
    ) -> tuple[list[RedIcon], list[float]]:
        icons = [
            (max(by_template.values()), x, y)
            for (x, y), matches in detections.items()
            if len(by_template := cls._best_confidence_by_template(matches)) >= min_matches
        ]
        confidences = [confidence for confidence, _, _ in icons]
        return icons, confidences

    @staticmethod
    def _red_icon_in_bounds(icon: RedIcon, x_min: int, x_max: int, y_min: int, y_max: int) -> bool:
        _, x, y = icon
        return x_min <= x <= x_max and y_min <= y <= y_max

    @classmethod
    def _best_red_icon_in_bounds(
        cls: type,
        icons: list[RedIcon],
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        minimum_confidence: float = 0.0,
    ) -> RedIcon | None:
        return max(
            (
                icon
                for icon in icons
                if icon[0] >= minimum_confidence and cls._red_icon_in_bounds(icon, x_min, x_max, y_min, y_max)
            ),
            key=lambda icon: icon[0],
            default=None,
        )

    def _find_best_zone_red_icon(
        self,
        screenshot: Any,
        threshold: float,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        min_distance: int = 80,
    ) -> RedIcon | None:
        region, offset_x, offset_y = self._extract_region(
            screenshot,
            x_min,
            x_max,
            y_min,
            y_max,
            pad_x=self._red_icon_max_width,
            pad_y=self._red_icon_max_height,
        )
        if region.size == 0:
            return None

        detections = self._collect_red_icon_detections(
            region,
            threshold,
            min_distance=min_distance,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        min_matches = self._red_icon_min_matches()
        icons, _ = self._icons_from_detections(detections, min_matches)
        return self._best_red_icon_in_bounds(icons, x_min, x_max, y_min, y_max)

    def _find_new_level_red_icon(
        self,
        screenshot: Any = None,
        scan_threshold: float | None = None,
        min_matches: int | None = None,
    ) -> RedIcon | None:
        if screenshot is None:
            screenshot = self.window_capture.capture(max_y=config.EXTENDED_SEARCH_Y)
        if scan_threshold is None:
            scan_threshold = self._red_icon_scan_threshold()
        if min_matches is None:
            min_matches = self._red_icon_min_matches()

        footer_region, offset_x, offset_y = self._extract_region(
            screenshot,
            config.NEW_LEVEL_RED_ICON_X_MIN,
            config.NEW_LEVEL_RED_ICON_X_MAX,
            config.NEW_LEVEL_RED_ICON_Y_MIN,
            config.NEW_LEVEL_RED_ICON_Y_MAX,
            pad_x=self._red_icon_max_width,
            pad_y=self._red_icon_max_height,
        )
        footer_detections = self._collect_red_icon_detections(
            footer_region,
            scan_threshold,
            min_distance=80,
            offset_x=offset_x,
            offset_y=offset_y,
        )
        all_red_icons_extended, _ = self._icons_from_detections(footer_detections, min_matches)

        new_level_icon_threshold = (
            self.vision_optimizer.new_level_red_icon_threshold
            if self.vision_optimizer.enabled
            else config.NEW_LEVEL_RED_ICON_THRESHOLD
        )
        return self._best_red_icon_in_bounds(
            all_red_icons_extended,
            config.NEW_LEVEL_RED_ICON_X_MIN,
            config.NEW_LEVEL_RED_ICON_X_MAX,
            config.NEW_LEVEL_RED_ICON_Y_MIN,
            config.NEW_LEVEL_RED_ICON_Y_MAX,
            minimum_confidence=new_level_icon_threshold,
        )

    def _upgrade_station_threshold(self) -> float:
        if self.vision_optimizer.enabled:
            return self.vision_optimizer.upgrade_station_threshold
        return config.UPGRADE_STATION_THRESHOLD

    def _candidate_passes_template_gates(
        self,
        screenshot: Any,
        template: Any,
        mask: Any,
        location: tuple[int, int],
        color_check_enabled: bool,
        color_threshold: float,
        hsv_gate_enabled: bool,
        hsv_ranges: Any,
        hsv_min_match_ratio: float,
    ) -> bool:
        if color_check_enabled and not self.image_matcher._check_color_similarity(
            screenshot,
            template,
            location,
            mask,
            color_threshold=color_threshold,
        ):
            return False
        if not hsv_gate_enabled:
            return True
        return self.image_matcher._check_hsv_gate(
            screenshot,
            template,
            location,
            mask,
            hsv_ranges,
            hsv_min_match_ratio,
        )

    @staticmethod
    def _upgrade_station_distance_squared(match: RedIcon, expected_position: tuple[int, int]) -> int:
        _, x, y = match
        expected_x, expected_y = expected_position
        return (int(x) - int(expected_x)) ** 2 + (int(y) - int(expected_y)) ** 2

    @staticmethod
    def _normalized_candidate_center(candidate: tuple[Any, ...]) -> RedIcon | None:
        try:
            confidence, center_x, center_y = candidate[:3]
            confidence = float(confidence)
            center_x = int(center_x)
            center_y = int(center_y)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(confidence):
            return None
        return confidence, center_x, center_y

    def _upgrade_station_candidate_match(
        self,
        limited_screenshot: Any,
        template: Any,
        mask: Any,
        candidate: tuple[Any, ...],
    ) -> RedIcon | None:
        candidate_center = self._normalized_candidate_center(candidate)
        if candidate_center is None:
            return None
        confidence, x, y = candidate_center
        template_height, template_width = template.shape[:2]
        location = (x - template_width // 2, y - template_height // 2)
        if not self._candidate_passes_template_gates(
            limited_screenshot,
            template,
            mask,
            location,
            config.UPGRADE_STATION_COLOR_CHECK,
            0.7,
            config.UPGRADE_STATION_HSV_COLOR_GATE_ENABLED,
            config.UPGRADE_STATION_HSV_RANGES,
            config.UPGRADE_STATION_HSV_MIN_MATCH_RATIO,
        ):
            return None
        if self.mouse_controller.is_in_forbidden_zone(x, y, relative=True):
            return None
        return float(confidence), x, y

    @staticmethod
    def _maximum_distance_squared(maximum_distance: float | None) -> float | None:
        if maximum_distance is None:
            return None
        return max(0.0, float(maximum_distance)) ** 2

    def _best_upgrade_station_match(
        self,
        candidates: list[MatchCandidate],
        limited_screenshot: Any,
        template: Any,
        mask: Any,
        expected_position: tuple[int, int] | None,
        maximum_distance: float | None,
    ) -> RedIcon | None:
        best_target_match = None
        best_target_distance = None
        maximum_distance_squared = self._maximum_distance_squared(maximum_distance)
        for candidate in candidates:
            match = self._upgrade_station_candidate_match(limited_screenshot, template, mask, candidate)
            if match is None:
                continue
            if expected_position is None:
                return match
            distance_squared = self._upgrade_station_distance_squared(match, expected_position)
            if maximum_distance_squared is not None and distance_squared > maximum_distance_squared:
                continue
            if best_target_distance is None or distance_squared < best_target_distance:
                best_target_match = match
                best_target_distance = distance_squared
        return best_target_match

    def _find_upgrade_station_match(
        self,
        threshold: float,
        expected_position: tuple[int, int] | None = None,
        maximum_distance: float | None = None,
    ) -> RedIcon | None:
        if "upgradeStation" not in self.templates:
            return None

        limited_screenshot = self.window_capture.capture(max_y=config.UPGRADE_STATION_SEARCH_Y)
        template, mask = self.templates["upgradeStation"]
        candidates = self.image_matcher.find_all_templates(
            limited_screenshot,
            template,
            mask=mask,
            threshold=threshold,
            min_distance=15,
            template_name="upgradeStation",
            use_supervision_nms=self._supervision_nms_enabled("upgrade_station"),
            supervision_iou_threshold=config.SUPERVISION_UPGRADE_STATION_NMS_IOU_THRESHOLD,
            supervision_class_agnostic=config.SUPERVISION_CLASS_AGNOSTIC_NMS,
        )
        if not candidates:
            return None

        return self._best_upgrade_station_match(
            candidates,
            limited_screenshot,
            template,
            mask,
            expected_position,
            maximum_distance,
        )

    def _find_verified_upgrade_station_match(
        self,
        base_threshold: float,
        relaxed_threshold: float,
        expected_position: tuple[int, int],
    ) -> tuple[RedIcon | None, bool]:
        verify_attempts = max(1, int(config.UPGRADE_STATION_VERIFY_SEARCH_ATTEMPTS))
        verify_radius = float(config.UPGRADE_STATION_VERIFY_RADIUS)
        for attempt in range(verify_attempts):
            current_threshold = base_threshold if attempt == 0 else relaxed_threshold
            verified_match = self._find_upgrade_station_match(
                current_threshold,
                expected_position=expected_position,
                maximum_distance=verify_radius,
            )
            if verified_match is not None:
                return verified_match, True
            if attempt < verify_attempts - 1 and not self._sleep(config.UPGRADE_STATION_VERIFY_SEARCH_INTERVAL):
                return None, False
        return None, True

    def _box_template_names(self) -> list[str]:
        cached = getattr(self, "_box_template_names_cache", None)
        if cached is not None:
            return cached
        return [box_name for box_name in ("box1", "box2", "box3", "box4") if box_name in self.templates]

    def _box_candidate_from_match(
        self,
        limited_screenshot: Any,
        template: Any,
        mask: Any,
        box_name: str,
        match: MatchCandidate,
    ) -> BoxCandidate | None:
        confidence, x, y, candidate_width, candidate_height = match
        candidate_width = int(candidate_width)
        candidate_height = int(candidate_height)
        location = (int(x) - candidate_width // 2, int(y) - candidate_height // 2)
        if not self._candidate_passes_template_gates(
            limited_screenshot,
            template,
            mask,
            location,
            config.BOX_COLOR_CHECK,
            config.BOX_COLOR_THRESHOLD,
            config.BOX_HSV_COLOR_GATE_ENABLED,
            config.BOX_HSV_RANGES,
            config.BOX_HSV_MIN_MATCH_RATIO,
        ):
            return None
        return confidence, int(x), int(y), candidate_width, candidate_height, box_name

    def _collect_box_candidates(self, limited_screenshot: Any, box_threshold: float) -> list[BoxCandidate]:
        box_candidates: list[BoxCandidate] = []
        for box_name in self._box_template_names():
            template, mask = self.templates[box_name]
            candidates = self.image_matcher.find_template_candidates(
                limited_screenshot,
                template,
                mask=mask,
                threshold=box_threshold,
                min_distance=12,
                template_name=box_name,
            )
            box_candidates.extend(
                box_candidate
                for candidate in candidates
                if (
                    box_candidate := self._box_candidate_from_match(
                        limited_screenshot,
                        template,
                        mask,
                        box_name,
                        candidate,
                    )
                )
                is not None
            )
        return box_candidates
