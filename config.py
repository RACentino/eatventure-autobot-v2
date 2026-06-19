# Paths

# Path to the image template assets directory (relative to project root).
ASSETS_DIR = "assets"

# Path to the runtime log output directory (relative to project root).
LOGS_DIR = "logs"


# Window and Logging

# Exact desktop window title that the automation targets.
WINDOW_TITLE = "EatventureAuto"

# Width applied when resizing the target window.
WINDOW_WIDTH = 360

# Height applied when resizing the target window.
WINDOW_HEIGHT = 780

# Enables verbose debug logging when true.
DEBUG = False

# Duration of a single frame at 60 FPS. Used as the base unit for
# input-event timing so physical dispatch matches display refresh rate.
SIXTY_FPS_FRAME_DURATION_SECONDS = 0.017


# Supervision NMS

# Enables supervision-based non-max suppression globally.
SUPERVISION_ENABLED = True

# Enables supervision NMS for box detections.
SUPERVISION_BOX_NMS_ENABLED = True

# Enables supervision NMS for red icon detections.
SUPERVISION_RED_ICON_NMS_ENABLED = True

# Enables supervision NMS for upgrade station detections.
SUPERVISION_UPGRADE_STATION_NMS_ENABLED = True

# Allows supervision NMS to suppress candidates across classes.
SUPERVISION_CLASS_AGNOSTIC_NMS = True

# IoU threshold for supervision NMS on box detections.
SUPERVISION_BOX_NMS_IOU_THRESHOLD = 0.15

# IoU threshold for supervision NMS on red icon detections.
SUPERVISION_RED_ICON_NMS_IOU_THRESHOLD = 0.20

# IoU threshold for supervision NMS on upgrade station detections.
SUPERVISION_UPGRADE_STATION_NMS_IOU_THRESHOLD = 0.20


# ByteTrack Asset Tracking

# Enables continuous background asset tracking.
ASSET_TRACKING_ENABLED = True

# Maximum vertical capture area used by the background tracker.
ASSET_TRACKING_CAPTURE_Y = 780

# Delay between tracker capture passes.
ASSET_TRACKING_INTERVAL = 0.7

# Frame rate passed to supervision ByteTrack.
ASSET_TRACKING_FRAME_RATE = 1.43

# Maximum age in seconds for tracker snapshots used by bot actions.
ASSET_TRACKING_MAX_SNAPSHOT_AGE = 0.75

# Maximum detections passed through each tracking frame.
ASSET_TRACKING_MAX_DETECTIONS = 256

# Detection confidence threshold for ByteTrack track activation.
ASSET_TRACKING_TRACK_ACTIVATION_THRESHOLD = 0.25

# Number of lost frames ByteTrack keeps before removing a track.
ASSET_TRACKING_LOST_TRACK_BUFFER = 2

# ByteTrack association threshold between existing tracks and detections.
ASSET_TRACKING_MINIMUM_MATCHING_THRESHOLD = 0.8

# Consecutive tracked frames required before a track is considered valid.
ASSET_TRACKING_MINIMUM_CONSECUTIVE_FRAMES = 2

# Enables background red icon detections.
ASSET_TRACKING_RED_ICON_ENABLED = True

# Enables background upgrade station detections.
ASSET_TRACKING_UPGRADE_STATION_ENABLED = True

# Enables background box detections.
ASSET_TRACKING_BOX_ENABLED = True

# Maximum wait for the background tracker thread to stop.
ASSET_TRACKING_THREAD_JOIN_TIMEOUT = 1.5

# SCRCPY Recovery

# Enables a short retry delay after SCRCPY capture misses.
SCRCPY_MISS_RECOVERY_ENABLED = True

# Retry delay after a red icon scan miss.
SCRCPY_RED_ICON_MISS_RECOVERY_DELAY = 0.083

# Retry delay after a box scan miss.
SCRCPY_BOX_MISS_RECOVERY_DELAY = 0.083

# Vision Matching

# Default template matching threshold used by ImageMatcher.
MATCH_THRESHOLD = 0.98

# Template confidence threshold for red icon scans.
RED_ICON_THRESHOLD = 0.960

# Template confidence threshold for new-level red icon validation.
NEW_LEVEL_RED_ICON_THRESHOLD = 0.950

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
BOX_THRESHOLD = 0.930

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
BOX_HSV_MIN_MATCH_RATIO = 0.5

# Template confidence threshold for unlock button scans.
UNLOCK_THRESHOLD = 0.905

# Template confidence threshold for new-level button scans.
NEW_LEVEL_THRESHOLD = 0.965

# Minimum red icon templates that must agree outside fast mode.
RED_ICON_MIN_MATCHES = 3

# Enables single-template red icon_scan mode for faster passes.
RED_ICON_FAST_MODE_ENABLED = False

# Red icon template names used when fast mode is enabled.
RED_ICON_FAST_TEMPLATE_NAMES = (
    "RedIcon3",
    "RedIcon6",
    "RedIcon9",
    "RedIcon13",
)

# Minimum pixel distance between fast-mode red icon matches.
RED_ICON_FAST_MIN_DISTANCE = 64

RED_ICON_HSV_RANGES = (
    ((0, 110, 190), (12, 255, 255)),
    ((174, 110, 190), (179, 255, 255)),
)

RED_ICON_HSV_MIN_MATCH_RATIO = 0.5

# Mouse and Action Timing

# Delay after normal click actions.
CLICK_DELAY = 0.083

# Delay after cursor movement.
MOUSE_MOVE_DELAY = 0.033

# Duration to hold the mouse button down during click actions.
MOUSE_DOWN_DURATION = 0.05

# Delay after releasing the mouse button.
MOUSE_UP_DURATION = 0.033

# Enables hover movement before click actions.
HOVER_ENABLED = True

# Duration for hover movement before click actions.
HOVER_DURATION = 0.1

# Delay between upgrade station search attempts.
UPGRADE_SEARCH_INTERVAL = 0.083

# Delay between state-machine actions.
STATE_DELAY = 0.15

# Settling delay before verifying an upgrade station hold target.
UPGRADE_STATION_VERIFY_SETTLE_DELAY = 0.15

# Maximum attempts for upgrade station verification.
UPGRADE_STATION_VERIFY_SEARCH_ATTEMPTS = 3

# Delay between upgrade station verification attempts.
UPGRADE_STATION_VERIFY_SEARCH_INTERVAL = 0.067

# Maximum duration for holding an upgrade station click.
CLICK_HOLD_MAX_DURATION = 10.0

# Duration for stats upgrade click bursts.
STATS_UPGRADE_CLICK_DURATION = 3.0

# Delay between stats upgrade click actions.
STATS_UPGRADE_CLICK_DELAY = 0.067


# Telegram Notifications

# Enables Telegram notifications.
TELEGRAM_ENABLED = False

# Telegram bot token used for notification delivery.
TELEGRAM_BOT_TOKEN = ""

# Telegram chat identifier that receives notifications.
TELEGRAM_CHAT_ID = ""

# Maximum Telegram request timeout in seconds.
TELEGRAM_CLOSE_TIMEOUT = 1.5


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
MAX_SCROLL_CYCLES = 6

# Step increment applied to each oscillating scroll cycle.
SCROLL_INCREMENT_STEP = 1

# Delay between scroll intervals.
SCROLL_INTERVAL_PAUSE = 0.1

# Settling delay after a scroll.
POST_SCROLL_SETTLE = 0.1

# Duration of the scroll drag gesture.
SCROLL_DURATION = 0.3


# Adaptive Runtime Tuning

# Enables adaptive runtime timing adjustments.
ADAPTIVE_TUNER_ENABLED = False

# EMA alpha used by the adaptive tuner.
ADAPTIVE_TUNER_ALPHA = 0.3

# Click success rate below which delays are increased.
ADAPTIVE_TUNER_CLICK_LOW_THRESHOLD = 1.0

# Click success rate above which delays are reduced.
ADAPTIVE_TUNER_CLICK_HIGH_THRESHOLD = 0.0

# Search success rate below which search interval is increased.
ADAPTIVE_TUNER_SEARCH_LOW_THRESHOLD = 1.0

# Search success rate above which search interval is reduced.
ADAPTIVE_TUNER_SEARCH_HIGH_THRESHOLD = 0.0

# Amount added to click delay after low click success.
ADAPTIVE_TUNER_CLICK_DELAY_STEP = 0.017

# Amount added to move delay after low click success.
ADAPTIVE_TUNER_MOVE_DELAY_STEP = 0.008

# Amount subtracted from click delay after high click success.
ADAPTIVE_TUNER_CLICK_DECREMENT = 0.008

# Amount subtracted from move delay after high click success.
ADAPTIVE_TUNER_MOVE_DECREMENT = 0.004

# Amount added to search interval after low search success.
ADAPTIVE_TUNER_SEARCH_INTERVAL_STEP = 0.017

# Amount subtracted from search interval after high search success.
ADAPTIVE_TUNER_SEARCH_DECREMENT = 0.008

# Minimum click delay allowed by adaptive tuning.
ADAPTIVE_TUNER_MIN_CLICK_DELAY = 0.067

# Maximum click delay allowed by adaptive tuning.
ADAPTIVE_TUNER_MAX_CLICK_DELAY = 0.133

# Minimum move delay allowed by adaptive tuning.
ADAPTIVE_TUNER_MIN_MOVE_DELAY = 0.017

# Maximum move delay allowed by adaptive tuning.
ADAPTIVE_TUNER_MAX_MOVE_DELAY = 0.067

# Minimum search interval allowed by adaptive tuning.
ADAPTIVE_TUNER_MIN_SEARCH_INTERVAL = 0.067

# Maximum search interval allowed by adaptive tuning.
ADAPTIVE_TUNER_MAX_SEARCH_INTERVAL = 0.167


# Historical Learning

# Enables historical learning from completed levels.
AI_LEARNING_ENABLED = False

# Path for persisted historical learning state (relative to project root).
AI_LEARNING_STATE_FILE = "memory/learning_state_stable.json"

# Minimum interval between historical learning state saves.
AI_LEARNING_SAVE_INTERVAL = 30.0

# Maximum historical learning records kept in memory and persisted state.
AI_LEARNING_RECORDS_LIMIT = 500

# Maximum wait for the historical learning worker to stop.
AI_LEARNING_THREAD_JOIN_TIMEOUT = 1.0

# Delay between historical learning worker passes.
AI_LEARNING_THREAD_INTERVAL = 5.0

# Number of level records considered for a learning batch.
AI_LEARNING_BATCH_WINDOW = 2

# EMA alpha used for historical behavior profiles.
AI_LEARNING_EMA_ALPHA = 0.8

# Number of best profiles blended into a learned behavior.
AI_LEARNING_PROFILE_BLEND_TOP_K = 1

# Minimum improvement ratio needed before applying learned behavior.
AI_LEARNING_MIN_IMPROVEMENT_RATIO = 0.0

# Cooldown between learned behavior applications.
AI_LEARNING_APPLY_COOLDOWN = 30.0

# Minimum click delay allowed by historical learning.
AI_LEARNING_MIN_CLICK_DELAY = 0.067

# Maximum click delay allowed by historical learning.
AI_LEARNING_MAX_CLICK_DELAY = 0.133

# Minimum move delay allowed by historical learning.
AI_LEARNING_MIN_MOVE_DELAY = 0.017

# Maximum move delay allowed by historical learning.
AI_LEARNING_MAX_MOVE_DELAY = 0.067

# Minimum search interval allowed by historical learning.
AI_LEARNING_MIN_SEARCH_INTERVAL = 0.067

# Maximum search interval allowed by historical learning.
AI_LEARNING_MAX_SEARCH_INTERVAL = 0.167

# Minimum sleep between historical learning worker loops.
LEARNING_LOOP_MIN_SLEEP = 1.0


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
