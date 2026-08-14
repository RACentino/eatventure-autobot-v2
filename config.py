import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Path to the image template assets directory.
ASSETS_DIR = str(PROJECT_ROOT / "assets")

# Path to the runtime log output directory.
LOGS_DIR = str(PROJECT_ROOT / "logs")


# Window and Logging

# Exact desktop window title that the automation targets.
WINDOW_TITLE = "EatventureAuto"

# Width applied when resizing the target window.
WINDOW_WIDTH = 360

# Height applied when resizing the target window.
WINDOW_HEIGHT = 780

# Enables verbose debug logging when true.
DEBUG = False

# Duration of a single frame at 60 FPS. Used as the base unit for input-event timing so physical dispatch matches display refresh rate.
SIXTY_FPS_FRAME_DURATION_SECONDS = 0.016666666666666666

# Global delay between state-machine steps.
STATE_DELAY = 0.0


# Detection overlap suppression

# IoU thresholds used by the built-in class-agnostic NMS.
SUPERVISION_BOX_NMS_IOU_THRESHOLD = 0.14
SUPERVISION_RED_ICON_NMS_IOU_THRESHOLD = 0.20
SUPERVISION_UPGRADE_STATION_NMS_IOU_THRESHOLD = 0.20


# SCRCPY Recovery

# Enables a short retry delay after SCRCPY capture misses.
SCRCPY_MISS_RECOVERY_ENABLED = True

# Retry delay after a red icon scan miss.
SCRCPY_RED_ICON_MISS_RECOVERY_DELAY = 0.03333333333333333

# Retry delay after a box scan miss.
SCRCPY_BOX_MISS_RECOVERY_DELAY = 0.03333333333333333


# Vision Matching

# Default template matching threshold used by ImageMatcher.
MATCH_THRESHOLD = 0.98

# Template confidence threshold for red icon scans.
RED_ICON_THRESHOLD = 0.950

# Template confidence threshold for new-level red icon validation.
NEW_LEVEL_RED_ICON_THRESHOLD = 0.958

# Template confidence threshold for stats upgrade red icon validation.
STATS_RED_ICON_THRESHOLD = 0.950

# Template confidence threshold for upgrade station scans.
UPGRADE_STATION_THRESHOLD = 0.910

# Accepted HSV ranges for upgrade station candidate gating.
UPGRADE_STATION_HSV_RANGES = (
    ((12, 88, 185), (29, 199, 252)),
    ((100, 135, 204), (103, 191, 255)),
)

# Minimum HSV mask ratio for accepting an upgrade station candidate.
UPGRADE_STATION_HSV_MIN_MATCH_RATIO = 0.4

# Template confidence threshold for box scans.
BOX_THRESHOLD = 0.950

# Accepted HSV ranges for box candidate gating.
BOX_HSV_RANGES = (
    ((10, 65, 180), (13, 105, 255)),
    ((13, 90, 120), (15, 190, 245)),
    ((18, 90, 120), (18, 129, 245)),
    ((18, 130, 120), (18, 130, 229)),
    ((18, 130, 235), (18, 130, 245)),
    ((18, 131, 120), (18, 190, 245)),
    ((20, 115, 220), (20, 125, 245)),
    ((23, 65, 140), (30, 115, 255)),
)

# Minimum HSV mask ratio for accepting a box candidate.
BOX_HSV_MIN_MATCH_RATIO = 0.45

# Template confidence threshold for unlock button scans.
UNLOCK_THRESHOLD = 0.905

# Template confidence threshold for new-level button scans.
NEW_LEVEL_THRESHOLD = 0.965

# Uses only the configured fast templates when true; uses all red-icon templates with consensus when false.
RED_ICON_FAST_MODE_ENABLED = True

# Red icon template filename stems matched exclusively in fast mode.
RED_ICON_FAST_TEMPLATE_NAMES = ("RedIcon5",)

# Minimum distinct templates that must agree in full mode.
RED_ICON_MIN_MATCHES = 4

# Minimum pixel distance between red icon matches.
RED_ICON_FAST_MIN_DISTANCE = 32

# Accepted HSV ranges for red icon candidate gating.
RED_ICON_HSV_RANGES = (
    ((0, 100, 180), (15, 255, 255)),
    ((170, 100, 180), (179, 255, 255)),
)

# Minimum HSV mask ratio for accepting a red icon candidate.
RED_ICON_HSV_MIN_MATCH_RATIO = 0.6


# Mouse and Action Timing

# Delay after normal click actions.
CLICK_DELAY = 0.03333333333333333

# Delay after cursor movement.
MOUSE_MOVE_DELAY = 0.016666666666666666

# Duration to hold the mouse button down during click actions.
MOUSE_DOWN_DURATION = 0.11666666666666667

# Delay after releasing the mouse button.
MOUSE_UP_DURATION = 0.11666666666666667

# Enables hover movement before click actions.
HOVER_ENABLED = False

# Duration for hover movement before click actions.
HOVER_DURATION = 0.0

# Delay between upgrade station search attempts.
UPGRADE_SEARCH_INTERVAL = 0.05

# Settling delay before verifying an upgrade station hold target.
UPGRADE_STATION_VERIFY_SETTLE_DELAY = 0.13333333333333333

# Maximum attempts for upgrade station verification.
UPGRADE_STATION_VERIFY_SEARCH_ATTEMPTS = 2

# Delay between upgrade station verification attempts.
UPGRADE_STATION_VERIFY_SEARCH_INTERVAL = 0.1

# Maximum duration for holding an upgrade station click.
CLICK_HOLD_MAX_DURATION = 9.0

# Duration for stats upgrade click bursts.
STATS_UPGRADE_CLICK_DURATION = 1.75

# Delay between stats upgrade click actions.
STATS_UPGRADE_CLICK_DELAY = 0.016


# Telegram Notifications

# Enables Telegram notifications.
TELEGRAM_ENABLED = False

# Telegram bot token used for notification delivery.
TELEGRAM_BOT_TOKEN = os.environ.get("EATVENTURE_TELEGRAM_BOT_TOKEN", "").strip()

# Telegram chat identifier that receives notifications.
TELEGRAM_CHAT_ID = os.environ.get("EATVENTURE_TELEGRAM_CHAT_ID", "").strip()

# Maximum Telegram request and shutdown waits in seconds.
TELEGRAM_REQUEST_TIMEOUT = 5.0
TELEGRAM_SHUTDOWN_TIMEOUT = 5.0


# Capture Regions

# Maximum vertical capture area for normal scans.
MAX_SEARCH_Y = 660

# Extended vertical capture area for red icon and upgrade scans.
EXTENDED_SEARCH_Y = 710

# Vertical capture area for upgrade station searches.
UPGRADE_STATION_SEARCH_Y = 760

# Vertical capture area for box searches.
BOX_SEARCH_Y = 780


# Click Coordinates

# Relative idle click coordinate used before scan actions.
IDLE_CLICK_POS = (2, 390)

# Relative coordinate for opening the stats upgrade panel.
STATS_UPGRADE_BUTTON_POS = (310, 698)

# Relative coordinate for the stats upgrade button inside the panel.
STATS_UPGRADE_POS = (270, 304)

# Relative start coordinate for scroll drags.
SCROLL_START_POS = (170, 380)

# Relative coordinate for the new-level button.
NEW_LEVEL_BUTTON_POS = (30, 692)

# Relative coordinate for the level transition confirmation.
LEVEL_TRANSITION_POS = (174, 520)


# Icon Regions and Offsets

# Horizontal offset added to red icon click coordinates.
RED_ICON_OFFSET_X = 10

# Vertical offset added to red icon click coordinates.
RED_ICON_OFFSET_Y = 10

# Minimum x-coordinate for a new-level red icon.
NEW_LEVEL_RED_ICON_X_MIN = 40

# Maximum x-coordinate for a new-level red icon.
NEW_LEVEL_RED_ICON_X_MAX = 60

# Minimum y-coordinate for a new-level red icon.
NEW_LEVEL_RED_ICON_Y_MIN = 665

# Maximum y-coordinate for a new-level red icon.
NEW_LEVEL_RED_ICON_Y_MAX = 680

# Minimum x-coordinate for an upgrade red icon.
UPGRADE_RED_ICON_X_MIN = 280

# Maximum x-coordinate for an upgrade red icon.
UPGRADE_RED_ICON_X_MAX = 310

# Minimum y-coordinate for an upgrade red icon.
UPGRADE_RED_ICON_Y_MIN = 665

# Maximum y-coordinate for an upgrade red icon.
UPGRADE_RED_ICON_Y_MAX = 680


# Scrolling

# Pixel distance for each scroll drag.
SCROLL_PIXEL_STEP = 180

# Multiplier applied to scroll pixel distance.
SCROLL_DISTANCE_RATIO = 1.0

# Maximum oscillating scroll cycles before reset.
MAX_SCROLL_CYCLES = 1

# Step increment applied to each oscillating scroll cycle.
SCROLL_INCREMENT_STEP = 5

# Delay between scroll intervals.
SCROLL_INTERVAL_PAUSE = 0.200

# Settling delay after a scroll.
POST_SCROLL_SETTLE = 0.200

# Duration of the scroll drag gesture.
SCROLL_DURATION = 0.300


# Forbidden Zones

# Minimum x-coordinate for the general forbidden click band.
FORBIDDEN_CLICK_X_MIN = 60

# Maximum x-coordinate for the general forbidden click band.
FORBIDDEN_CLICK_X_MAX = 260

# Minimum y-coordinate for the general forbidden click band.
FORBIDDEN_CLICK_Y_MIN = 668

NUMBERED_FORBIDDEN_ZONE_BOUNDS = (
    (290, 350, 93, 270),
    (0, 60, 50, 280),
    (0, 60, 600, 667),
    (145, 200, 65, 110),
    (55, 260, 660, 725),
)
