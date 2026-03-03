"""Configuration module with runtime validation and type enforcement."""

import os
from typing import List, Tuple, Dict, Any, Optional

###############################
###    WINDOW & UI SETTINGS   ###
###############################

# WINDOW_TITLE: The exact title of the scrcpy window (visible at the top of the window)
WINDOW_TITLE: str = "EatventureAuto"

# Window dimensions used for capturing and relative positioning
WINDOW_WIDTH: int = int(300 * 1.2)
WINDOW_HEIGHT: int = int(650 * 1.2)

# Debug and Visualization Settings
DEBUG: bool = True
DEBUG_VISION: bool = False  # Enables masked view for tuning pixel density
ShowForbiddenArea: bool = False  # Enables a visual overlay showing forbidden zones in red


###############################
###  DIRECTORY & FILE PATHS ###
###############################

TEMPLATES_DIR: str = "templates"
ASSETS_DIR: str = "Assets"
LOGS_DIR: str = "logs"


###############################
###   DETECTION THRESHOLDS  ###
###############################

# General template matching confidence (0.0 - 1.0)
MATCH_THRESHOLD: float = 0.98

# Specific thresholds for different game assets
RED_ICON_THRESHOLD: float = 0.94
NEW_LEVEL_RED_ICON_THRESHOLD: float = 0.95
STATS_RED_ICON_THRESHOLD: float = 0.97
UPGRADE_STATION_THRESHOLD: float = 0.94
BOX_THRESHOLD: float = 0.97
UNLOCK_THRESHOLD: float = 0.95
NEW_LEVEL_THRESHOLD: float = 0.98

# Detection gate settings
RED_ICON_MIN_MATCHES: int = 1
NEW_LEVEL_RED_ICON_MIN_MATCHES: int = 1
RED_ICON_PIXEL_THRESHOLD: int = 50  # Min red pixels in ROI to trigger
RED_ICON_DILATE_KERNEL: int = 3     # Size of dilation kernel to 'inflate' red pixels

# Red Color HSV bounds (wider range for better detection)
RED_HSV_LOWER1: Tuple[int, int, int] = (0, 100, 100)
RED_HSV_UPPER1: Tuple[int, int, int] = (15, 255, 255)
RED_HSV_LOWER2: Tuple[int, int, int] = (165, 100, 100)
RED_HSV_UPPER2: Tuple[int, int, int] = (180, 255, 255)

# Color verification for Red Icons
RED_ICON_COLOR_CHECK: bool = True
RED_ICON_COLOR_MIN_RATIO: float = 1.15
RED_ICON_COLOR_MIN_MEAN: int = 35
RED_ICON_COLOR_SAMPLE_SIZE: int = 24

# Position refinement and verification
RED_ICON_VERIFY_PADDING: int = 24
RED_ICON_VERIFY_TOLERANCE: int = 12
RED_ICON_REFINE_RADIUS: int = 18
RED_ICON_REFINE_THRESHOLD_DROP: float = 0.02

# Upgrade station specific detection
UPGRADE_STATION_COLOR_CHECK: bool = False
UPGRADE_STATION_REFINE_RADIUS: int = 28
UPGRADE_STATION_CLICK_REFINE_RADIUS: int = 18


###############################
###  MOUSE & INTERACTION    ###
###############################

# Base interaction timings
CLICK_DELAY: float = 0.045        # Padded handoff for UI consistency
MOUSE_MOVE_DELAY: float = 0.006   
CLICK_DURATION: float = 0.026     # Padded dwell ensures registration on all systems
MOUSE_DOWN_UP_DELAY: float = 0.026
DOUBLE_CLICK_DELAY: float = 0.042

# Mouse movement retry and correction logic
MOUSE_MOVE_RETRIES: int = 2
MOUSE_MOVE_RETRY_DELAY: float = 0.002
MOUSE_TARGET_SETTLE_DELAY: float = 0.002
MOUSE_TARGET_TIMEOUT: float = 0.045
MOUSE_TARGET_CHECK_INTERVAL: float = 0.003
MOUSE_TARGET_HOVER_DELAY: float = 0.002
MOUSE_STABILIZE_DURATION: float = 0.008  # Longer stabilization at target before click
MOUSE_TARGET_RETRIES: int = 3
MOUSE_TARGET_CORRECTION_DELAY: float = 0.002

# Stability delays before clicking
MOUSE_PRE_CLICK_STABILIZE_BASE: float = 0.004
MOUSE_PRE_CLICK_STABILIZE_MAX: float = 0.015
MOUSE_PRE_CLICK_STABILIZE_DISTANCE_FACTOR: float = 0.00004

# Click retry logic for robustness
MOUSE_CLICK_RETRY_COUNT: int = 2
MOUSE_CLICK_RETRY_SETTLE_DELAY: float = 0.004


###############################
###    SCROLLING BEHAVIOR   ###
###############################

# Start position for search scrolls (relative to window)
SCROLL_START_POS: Tuple[int, int] = (180, 390)

# Distance in pixels for a single "standard" scroll step
SCROLL_PIXEL_STEP: int = 100     # Tightened: Finer search resolution for better locking
SCROLL_DISTANCE_RATIO: float = 1.0

# Arithmetic Search Strategy (Numerous but Short)
MAX_SCROLL_CYCLES: int = 12     # Increased cycles to compensate for shorter steps
SCROLL_INCREMENT_STEP: int = 3   
SCROLL_INTERVAL_PAUSE: float = 0.08 # Rhythmic gap between swipes
POST_SCROLL_SETTLE: float = 0.24    # CRITICAL: Pure scan window. Guarantees a static frame.
CYCLE_PAUSE_DURATION: float = 0.20  # Stabilized direction flip

# Visual smoothness and stability
SCROLL_DURATION: float = 0.28     # Slower, more deliberate glide reduces motion blur
SCROLL_STEP_COUNT: int = 55     # High density linear motion
SCROLL_MIN_INTERVAL: float = 0.004
SCROLL_SETTLE_DELAY: float = 0.12  # Mechanical stabilization (Stop Inertia)
DRAG_STEPS: int = 20


###############################
###    BOT LOGIC & TIMING   ###
###############################

# Main loop execution speed
FSM_TICK_DELAY: float = 0.015     # Aligned with 60FPS frame timing (16ms)
MAIN_LOOP_DELAY: float = FSM_TICK_DELAY

# Minimum time to wait between state handler executions
STATE_DELAY: float = 0.025
STATE_MIN_INTERVAL_DEFAULT: float = 0.02
STATE_MIN_INTERVALS: Dict[str, float] = {
    "FIND_RED_ICONS": 0.05,  # Forces a "stare" before giving up and scrolling
    "OPEN_BOXES": 0.015,
    "SCROLL": 0.025,
}

# Red Icon and detection offsets
RED_ICON_OFFSET_X: int = 10
RED_ICON_OFFSET_Y: int = 10

# Fixed click positions for specific UI elements
NEW_LEVEL_POS: Tuple[int, int] = (171, 434)
LEVEL_TRANSITION_POS: Tuple[int, int] = (174, 520)
IDLE_CLICK_POS: Tuple[int, int] = (14, 360)
STATS_UPGRADE_POS: Tuple[int, int] = (270, 304)
STATS_UPGRADE_BUTTON_POS: Tuple[int, int] = (310, 698)
NEW_LEVEL_BUTTON_POS: Tuple[int, int] = (30, 692)

# Timing for interaction sequences
UPGRADE_HOLD_DURATION: float = 5.0  # How long to hold the upgrade button
UPGRADE_CLICK_INTERVAL: float = 0.012  # Slower hold-loop tap cadence improves upgrade registration consistency.
UPGRADE_SEARCH_INTERVAL: float = 0.08  # More time between upgrade scans avoids CV while UI counters are animating.
UPGRADE_CHECK_INTERVAL: float = 0.07  # Slower polling reduces overlap between click effects and verification reads.
STATS_UPGRADE_CLICK_DURATION: float = 2.0
STATS_UPGRADE_CLICK_DELAY: float = 0.02  # Added spacing between stat taps to prevent dropped clicks on low FPS moments.
STATS_ICON_PADDING: int = 20

# UI render and settle delays
IDLE_CLICK_SETTLE_DELAY: float = 0.05  # Longer idle settle prevents immediate post-idle scans from reading transition blur.
IDLE_CLICK_COOLDOWN: float = 0.15

# Red Icon and detection logic constants
RED_ICON_MIN_DISTANCE: int = 80
RED_ICON_MERGE_PROXIMITY: int = 10
RED_ICON_MERGE_BUCKET_SIZE: int = 10

# Forbidden-zone red icon arbitration (debounced 4-state matrix)
FORBIDDEN_ZONE_DETECTION_PRE_DELAY: float = 0.02
FORBIDDEN_ZONE_DETECTION_POST_DELAY: float = 0.03
FORBIDDEN_ZONE_DEBOUNCE_TICKS: int = 3
FORBIDDEN_ZONE_DEBOUNCE_REQUIRED_CONSENSUS: int = 2
FORBIDDEN_ZONE_SCROLL_REENTRY_COOLDOWN: float = 0.18
FORBIDDEN_BLACKOUT_DURATION: float = 3.5 # World-space coordinate ignore time

# Strict pre-click boundary validator timing (Slow is Smooth, Smooth is Fast)
FORBIDDEN_ZONE_PRECLICK_VALIDATION_DELAY: float = 0.012
FORBIDDEN_ZONE_DOUBLE_CHECK_DELAY: float = 0.008
ASSET_BOUNDARY_PRECHECK_DELAY: float = 0.02
ASSET_BOUNDARY_CONFIRM_DELAY: float = 0.01
ASSET_SEGREGATION_DELAY: float = 0.04  # Deliberate pause for coordinate categorization

# Upgrade station interaction settings
UPGRADE_STATION_SEARCH_MAX_ATTEMPTS: int = 5
UPGRADE_STATION_RELAXED_THRESHOLD_DROP: float = 0.05
UPGRADE_STATION_RELAXED_ATTEMPT_TRIGGER: int = 2

# Level transition and completion settings
LEVEL_TRANSITION_MAX_ATTEMPTS: int = 5
LEVEL_COMPLETION_RECENCY_WINDOW: float = 5.0
NEW_LEVEL_FAIL_COOLDOWN: float = 15.0

NEW_LEVEL_BUTTON_DELAY: float = 0.5
NEW_LEVEL_FOLLOWUP_DELAY: float = 0.3
UI_TRANSITION_PADDING: float = 1.1  # Unified transition padding so post-click travel/menu animations fully complete before CV.
TRANSITION_POST_CLICK_DELAY: float = 1.1  # Reuses the padded transition constant for all transition waits.
TRANSITION_RETRY_DELAY: float = 0.1
UNLOCK_POST_CLICK_DELAY: float = 0.8
WAIT_UNLOCK_RETRY_DELAY: float = 0.08  # Added unlock retry spacing avoids rapid-clicking while unlock modal is still opening.
WAIT_FOR_UNLOCK_TIMEOUT: float = 5.0
WAIT_UNLOCK_HOT_LOOP: float = 0.05
PRE_UNLOCK_DELAY: float = 0.0
UNLOCK_BACKOFF_THRESHOLD: int = 5
UNLOCK_MAX_RETRY_DELAY: float = 0.5

# Performance caching
CAPTURE_CACHE_TTL: float = 0.015  # Aligned with FSM_TICK_DELAY for efficient frame sharing.
MAX_SEARCH_Y: int = 665 # Excludes the bottom navigation bar from general scans
NEW_LEVEL_RED_ICON_CACHE_TTL: float = 0.01
RED_ICON_STABILITY_CACHE_TTL: float = 4.0 # Extended history for deliberate locking
RED_ICON_STABILITY_RADIUS: int = 16    
RED_ICON_STABILITY_MIN_HITS: int = 3    # INCREASED: Requires 3 frames of consistency for lock
RED_ICON_STABILITY_MAX_HISTORY: int = 15 # Deeper pool for hit verification

# Scan regions for Red Icons
NEW_LEVEL_RED_ICON_X_MIN: int = 40
NEW_LEVEL_RED_ICON_X_MAX: int = 60
NEW_LEVEL_RED_ICON_Y_MIN: int = 665
NEW_LEVEL_RED_ICON_Y_MAX: int = 680

UPGRADE_RED_ICON_X_MIN: int = 280
UPGRADE_RED_ICON_X_MAX: int = 310
UPGRADE_RED_ICON_Y_MIN: int = 665
UPGRADE_RED_ICON_Y_MAX: int = 680

# Background monitoring frequency
NEW_LEVEL_INTERRUPT_INTERVAL: float = 0.035 # 2x faster exit from sleep states
NEW_LEVEL_MONITOR_INTERVAL: float = 0.055   # Consistent background scan rate
NEW_LEVEL_OVERRIDE_COOLDOWN: float = 0.25


###############################
### ADAPTIVE TUNER SETTINGS ###
###############################

ADAPTIVE_TUNER_ENABLED: bool = True
ADAPTIVE_TUNER_ALPHA: float = 0.2  # EMA smoothing factor

# Success rate thresholds for triggering delay adjustments
ADAPTIVE_TUNER_CLICK_LOW_THRESHOLD: float = 0.85
ADAPTIVE_TUNER_CLICK_HIGH_THRESHOLD: float = 0.97
ADAPTIVE_TUNER_SEARCH_LOW_THRESHOLD: float = 0.70
ADAPTIVE_TUNER_SEARCH_HIGH_THRESHOLD: float = 0.90

# Step values for delay adjustments
ADAPTIVE_TUNER_CLICK_DELAY_STEP: float = 0.01
ADAPTIVE_TUNER_MOVE_DELAY_STEP: float = 0.001
ADAPTIVE_TUNER_CLICK_DECREMENT: float = 0.005
ADAPTIVE_TUNER_MOVE_DECREMENT: float = 0.001
ADAPTIVE_TUNER_SEARCH_INTERVAL_STEP: float = 0.01
ADAPTIVE_TUNER_UPGRADE_INTERVAL_STEP: float = 0.001
ADAPTIVE_TUNER_SEARCH_DECREMENT: float = 0.005
ADAPTIVE_TUNER_UPGRADE_DECREMENT: float = 0.001

# Range limits for adaptive delays
ADAPTIVE_TUNER_MIN_CLICK_DELAY: float = 0.035
ADAPTIVE_TUNER_MAX_CLICK_DELAY: float = 0.11
ADAPTIVE_TUNER_MIN_MOVE_DELAY: float = 0.003
ADAPTIVE_TUNER_MAX_MOVE_DELAY: float = 0.012
ADAPTIVE_TUNER_MIN_UPGRADE_INTERVAL: float = 0.006
ADAPTIVE_TUNER_MAX_UPGRADE_INTERVAL: float = 0.012
ADAPTIVE_TUNER_MIN_SEARCH_INTERVAL: float = 0.015
ADAPTIVE_TUNER_MAX_SEARCH_INTERVAL: float = 0.09  # Must stay above UPGRADE_SEARCH_INTERVAL so low-success tuning can only slow scans, never snap faster.


###############################
###  AI VISION & LEARNING   ###
###############################

AI_VISION_ENABLED: bool = True
AI_VISION_ALPHA: float = 0.2
AI_VISION_ALPHA_MAX: float = 0.45
AI_VISION_CONFIDENCE_BOOST: float = 0.3
AI_VISION_CONFIDENCE_THRESHOLD: float = 0.9  # Higher confidence gate avoids over-boosting thresholds from transient/blurred matches.

# Box detection specific AI settings
AI_BOX_THRESHOLD_MIN: float = 0.85
AI_BOX_THRESHOLD_MAX: float = 0.995
AI_BOX_MISS_WINDOW: int = 3
AI_BOX_MISS_STEP: float = 0.005

# Threshold limits for AI-driven detection
AI_RED_ICON_THRESHOLD_MIN: float = 0.92
AI_RED_ICON_THRESHOLD_MAX: float = 0.985
AI_RED_ICON_MARGIN: float = 0.01
AI_RED_ICON_MISS_WINDOW: int = 2
AI_RED_ICON_MISS_STEP: float = 0.006

AI_NEW_LEVEL_THRESHOLD_MIN: float = 0.965
AI_NEW_LEVEL_THRESHOLD_MAX: float = 0.995
AI_NEW_LEVEL_MISS_WINDOW: int = 2
AI_NEW_LEVEL_MISS_STEP: float = 0.004

AI_NEW_LEVEL_RED_ICON_THRESHOLD_MIN: float = 0.92
AI_NEW_LEVEL_RED_ICON_THRESHOLD_MAX: float = 0.99
AI_NEW_LEVEL_RED_ICON_MISS_WINDOW: int = 2
AI_NEW_LEVEL_RED_ICON_MISS_STEP: float = 0.005

AI_UPGRADE_STATION_THRESHOLD_MIN: float = 0.9
AI_UPGRADE_STATION_THRESHOLD_MAX: float = 0.99
AI_UPGRADE_STATION_MISS_WINDOW: int = 2
AI_UPGRADE_STATION_MISS_STEP: float = 0.005

AI_STATS_UPGRADE_THRESHOLD_MIN: float = 0.9
AI_STATS_UPGRADE_THRESHOLD_MAX: float = 0.99
AI_STATS_UPGRADE_MISS_WINDOW: int = 2
AI_STATS_UPGRADE_MISS_STEP: float = 0.005

# Persistence files
AI_VISION_STATE_FILE: str = f"{LOGS_DIR}/vision_state.json"
AI_VISION_SAVE_INTERVAL: float = 1.0

# Historical Learning
AI_LEARNING_ENABLED: bool = True
AI_LEARNING_STATE_FILE: str = f"{LOGS_DIR}/learning_state.json"
AI_LEARNING_SAVE_INTERVAL: float = 1.5
AI_LEARNING_RECORDS_LIMIT: int = 120
AI_LEARNING_THREAD_JOIN_TIMEOUT: float = 1.0

# Learning range limits
AI_LEARNING_MIN_CLICK_DELAY: float = 0.035
AI_LEARNING_MAX_CLICK_DELAY: float = 0.12
AI_LEARNING_MIN_MOVE_DELAY: float = 0.002
AI_LEARNING_MAX_MOVE_DELAY: float = 0.012
AI_LEARNING_MIN_UPGRADE_INTERVAL: float = 0.006
AI_LEARNING_MAX_UPGRADE_INTERVAL: float = 0.013
AI_LEARNING_MIN_SEARCH_INTERVAL: float = 0.012
AI_LEARNING_MAX_SEARCH_INTERVAL: float = 0.09  # Keep learner clamp aligned with tuner max to preserve monotonic reliability-focused search pacing.


###############################
###  TELEGRAM NOTIFICATIONS ###
###############################

TELEGRAM_ENABLED: bool = False
TELEGRAM_BOT_TOKEN: str = ""
TELEGRAM_CHAT_ID: str = ""


###############################
###     FORBIDDEN ZONES     ###
###############################

# Zones prevent the bot from clicking on critical UI elements
# Each zone is defined by name and bounding box (min/max X and Y)
# Optional field: "coordinate_space"
# - "image" (default): x/y are relative to emulator client area (same space as template matching output)
# - "monitor": x/y are absolute desktop coordinates
FORBIDDEN_ZONES: List[Dict[str, Any]] = [
    {
        "name": "General bottom bar",
        "coordinate_space": "image",
        "x_min": 60, "x_max": 280, "y_min": 668, "y_max": 1000
    },
    {
        "name": "Zone 1: Right side menu area",
        "coordinate_space": "image",
        "x_min": 290, "x_max": 350, "y_min": 93, "y_max": 320
    },
    {
        "name": "Zone 2: Left side top menu area",
        "coordinate_space": "image",
        "x_min": 0, "x_max": 60, "y_min": 50, "y_max": 280
    },
    {
        "name": "Zone 3: Left side bottom menu area",
        "coordinate_space": "image",
        "x_min": 0, "x_max": 60, "y_min": 590, "y_max": 667
    },
    {
        "name": "Zone 4: Top center notification area",
        "coordinate_space": "image",
        "x_min": 145, "x_max": 200, "y_min": 65, "y_max": 110
    },
    {
        "name": "Zone 5: Bottom navigation bar",
        "coordinate_space": "image",
        "x_min": 55, "x_max": 285, "y_min": 660, "y_max": 725
    },
    {
        "name": "Zone 6: Top bar area",
        "coordinate_space": "image",
        "x_min": 0, "x_max": 360, "y_min": 0, "y_max": 70
    }
]

# Coordinate limits for searching Red Icons
# Asset Names
RED_ICON_TEMPLATES: List[str] = [
    "RedIcon", "RedIcon2", "RedIcon3", "RedIcon4", "RedIcon5", "RedIcon6",
    "RedIcon7", "RedIcon8", "RedIcon9", "RedIcon10", "RedIcon11", "RedIcon12",
    "RedIcon13", "RedIcon14", "RedIcon15", "RedIconNoBG"
]

REQUIRED_TEMPLATES: List[str] = RED_ICON_TEMPLATES + [
    "newLevel", "unlock", "upgradeStation",
    "box1", "box2", "box3", "box4", "box5"
]

def validate_config() -> None:
    """Validates configuration parameters at runtime to prevent invalid states."""
    if not isinstance(WINDOW_WIDTH, int) or WINDOW_WIDTH <= 0:
        raise ValueError(f"WINDOW_WIDTH must be a positive integer, got: {WINDOW_WIDTH}")
    if not isinstance(WINDOW_HEIGHT, int) or WINDOW_HEIGHT <= 0:
        raise ValueError(f"WINDOW_HEIGHT must be a positive integer, got: {WINDOW_HEIGHT}")
    if not isinstance(CLICK_DELAY, float) or CLICK_DELAY < 0:
        raise ValueError("CLICK_DELAY must be a non-negative float")
    
    # Ensure directories exist
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)

validate_config()
