"""Optimized image matching and color verification logic."""

import cv2
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path

from core import config
from core.logger import setup_logger

logger = setup_logger("vision.matcher")

class ImageMatcher:
    """
    Optimized OpenCV-based image matcher.
    Supports template matching with masks and color gate checks.
    """
    def __init__(self, global_threshold: float = 0.85):
        self.global_threshold = global_threshold
        # Performance Tuning: Use 1 thread for OpenCV to prevent GDI/Monitor thread contention.
        cv2.setUseOptimized(True)
        cv2.setNumThreads(1)

    def load_template(self, template_path: Union[str, Path]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Loads a template and generates an alpha mask if present."""
        template_file = str(template_path)
        img = cv2.imread(template_file, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Template not found at: {template_file}")
        
        # Check for 4-channel image (BGRA) for transparency mask support
        if len(img.shape) == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3]
            mask = np.zeros_like(alpha)
            mask[alpha > 0] = 255
            bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return bgr, mask
        
        return img, None

    def find_template(
        self, 
        screenshot: np.ndarray, 
        template: np.ndarray, 
        mask: Optional[np.ndarray] = None, 
        threshold: Optional[float] = None, 
        template_name: str = "Unknown"
    ) -> Tuple[bool, float, int, int]:
        """
        Locates the best single match for a template.
        Returns (found, confidence, center_x, center_y).
        """
        matches = self.find_all_templates(
            screenshot, template, mask=mask, threshold=threshold, min_distance=1000, template_name=template_name
        )
        
        if not matches:
            return False, 0.0, 0, 0
            
        # Return the best match (already sorted by find_all_templates)
        conf, x, y = matches[0]
        return True, conf, x, y

    def find_all_templates(
        self, 
        screenshot: np.ndarray, 
        template: np.ndarray, 
        mask: Optional[np.ndarray] = None, 
        threshold: Optional[float] = None, 
        min_distance: int = 15,
        template_name: str = "Unknown"
    ) -> List[Tuple[float, int, int]]:
        """
        Detects all occurrences of a template within a screenshot.
        
        Args:
            screenshot: The BGR image to search within.
            template: The BGR template to search for.
            mask: Optional transparency mask.
            threshold: Confidence threshold (0.0 - 1.0).
            min_distance: Minimum pixel distance between detections.
            
        Returns:
            A list of (confidence, center_x, center_y).
        """
        active_thresh = threshold if threshold is not None else self.global_threshold
        
        # Guard: Check if template fits in screenshot
        if template.shape[0] > screenshot.shape[0] or template.shape[1] > screenshot.shape[1]:
            logger.debug(f"Template '{template_name}' is larger than ROI. Skipping.")
            return []

        # Perform SQDIFF_NORMED matching
        result = cv2.matchTemplate(screenshot, template, cv2.TM_SQDIFF_NORMED, mask=mask)
        
        matches = self._find_sqdiff_matches(result, threshold=active_thresh, min_distance=min_distance)
        
        # Convert top-left coordinates to center coordinates
        h, w = template.shape[:2]
        centered_matches = []
        for confidence, tl_x, tl_y in matches:
            cx = int(round(tl_x + (w / 2.0)))
            cy = int(round(tl_y + (h / 2.0)))
            centered_matches.append((confidence, cx, cy))
            
        return centered_matches

    def _find_sqdiff_matches(self, result: np.ndarray, threshold: float, min_distance: int) -> List[Tuple[float, int, int]]:
        """Internal SQDIFF match extraction with suppression."""
        max_error = 1.0 - threshold
        if max_error < 0:
            return []

        working = result.copy()
        height, width = working.shape[:2]
        matches: List[Tuple[float, int, int]] = []

        while True:
            # Locate the global minimum (best match for SQDIFF)
            min_val, _, min_loc, _ = cv2.minMaxLoc(working)
            if min_val > max_error:
                break

            x, y = min_loc
            matches.append((1.0 - min_val, x, y))

            # Suppress the surrounding area to find subsequent matches
            x1 = max(0, x - min_distance)
            x2 = min(width, x + min_distance + 1)
            y1 = max(0, y - min_distance)
            y2 = min(height, y + min_distance + 1)
            working[y1:y2, x1:x2] = 1.0 # Set error to maximum in suppression zone

        return matches

    def count_red_pixels(
        self, 
        image: np.ndarray, 
        x: int, 
        y: int, 
        size: Optional[int] = None, 
        lower1: Optional[Tuple[int, int, int]] = None, 
        upper1: Optional[Tuple[int, int, int]] = None,
        lower2: Optional[Tuple[int, int, int]] = None,
        upper2: Optional[Tuple[int, int, int]] = None
    ) -> int:
        """
        Counts red pixels in a region of interest around (x, y).
        Used for the Color Gate safety check.
        """
        sample_size = size if size is not None else config.RED_ICON_COLOR_SAMPLE_SIZE
        l1 = lower1 if lower1 is not None else config.RED_HSV_LOWER1
        u1 = upper1 if upper1 is not None else config.RED_HSV_UPPER1
        l2 = lower2 if lower2 is not None else config.RED_HSV_LOWER2
        u2 = upper2 if upper2 is not None else config.RED_HSV_UPPER2

        half = max(1, sample_size // 2)
        x1, y1 = max(0, x - half), max(0, y - half)
        x2, y2 = min(image.shape[1], x + half), min(image.shape[0], y + half)

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return 0

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Handle Red HSV wraparound (0-15 and 165-180)
        mask1 = cv2.inRange(hsv, np.array(l1), np.array(u1))
        mask2 = cv2.inRange(hsv, np.array(l2), np.array(u2))
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphological dilation to inflate/connect small red clusters
        kernel_size = config.RED_ICON_DILATE_KERNEL
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        
        return int(cv2.countNonZero(dilated))
