# Paths

# Project-relative path to the image template assets directory.
ASSETS_DIR = "assets"

# Project-relative path to the runtime log output directory.
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

# Shows the forbidden click area overlay while the bot is running.
SHOW_FORBIDDEN_AREA = False


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
SUPERVISION_BOX_NMS_IOU_THRESHOLD = 0.25

# IoU threshold for supervision NMS on red icon detections.
SUPERVISION_RED_ICON_NMS_IOU_THRESHOLD = 0.20

# IoU threshold for supervision NMS on upgrade station detections.
SUPERVISION_UPGRADE_STATION_NMS_IOU_THRESHOLD = 0.20


# SCRCPY Recovery

# Enables a short retry delay after SCRCPY capture misses.
SCRCPY_MISS_RECOVERY_ENABLED = True

# Retry delay after a red icon scan miss.
SCRCPY_RED_ICON_MISS_RECOVERY_DELAY = 0.240

# Retry delay after a box scan miss.
SCRCPY_BOX_MISS_RECOVERY_DELAY = 0.240

# Retry delay after an upgrade station scan miss.
SCRCPY_UPGRADE_MISS_RECOVERY_DELAY = 0.280


# Vision Matching

# Default template matching threshold used by ImageMatcher.
MATCH_THRESHOLD = 0.98

# Template confidence threshold for red icon scans.
RED_ICON_THRESHOLD = 0.920

# Template confidence threshold for new-level red icon validation.
NEW_LEVEL_RED_ICON_THRESHOLD = 0.942

# Template confidence threshold for stats upgrade red icon validation.
STATS_RED_ICON_THRESHOLD = 0.943

# Template confidence threshold for upgrade station scans.
UPGRADE_STATION_THRESHOLD = 0.910

# Template confidence threshold for box scans.
BOX_THRESHOLD = 0.930

# Enables HSV range gating for box candidates.
BOX_HSV_COLOR_GATE_ENABLED = True

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
BOX_HSV_MIN_MATCH_RATIO = 0.50

# Template confidence threshold for unlock button scans.
UNLOCK_THRESHOLD = 0.905

# Template confidence threshold for new-level button scans.
NEW_LEVEL_THRESHOLD = 0.965

# Minimum red icon templates that must agree outside fast mode.
RED_ICON_MIN_MATCHES = 3

# Enables single-template red icon_scan mode for faster passes.
RED_ICON_FAST_MODE_ENABLED = False

# Red icon template names used when fast mode is enabled.
RED_ICON_FAST_TEMPLATE_NAMES = ("RedIcon6","RedIcon15", "RedIcon12",)

# Minimum pixel distance between fast-mode red icon matches.
RED_ICON_FAST_MIN_DISTANCE = 64

# Red icon template scales used for multi-resolution screenshots.
RED_ICON_TEMPLATE_SCALES = (1.0, 2.0, 2.5)

RED_ICON_HSV_COLOR_GATE_ENABLED = True

RED_ICON_HSV_RANGES = (
    ((0, 110, 190), (8, 255, 255)),
    ((176, 110, 190), (179, 255, 255)),
)

RED_ICON_HSV_MIN_MATCH_RATIO = 0.40

# Maximum red component span accepted during red icon texture validation.
RED_ICON_ASSET_MAX_RED_COMPONENT_SPAN = 112


# Mouse and Action Timing

# Delay after normal click actions.
CLICK_DELAY = 0.140

# Delay after cursor movement.
MOUSE_MOVE_DELAY = 0.055

# Duration to hold the mouse button down during click actions.
MOUSE_DOWN_DURATION = 0.105

# Delay after releasing the mouse button.
MOUSE_UP_DURATION = 0.065

# Enables hover movement before click actions.
HOVER_ENABLED = True

# Duration for hover movement before click actions.
HOVER_DURATION = 0.085

# Delay between upgrade station search attempts.
UPGRADE_SEARCH_INTERVAL = 0.560

# Delay between state-machine actions.
STATE_DELAY = 0.360

# Settling delay before verifying an upgrade station hold target.
UPGRADE_STATION_VERIFY_SETTLE_DELAY = 0.300

# Maximum attempts for upgrade station verification.
UPGRADE_STATION_VERIFY_SEARCH_ATTEMPTS = 1

# Delay between upgrade station verification attempts.
UPGRADE_STATION_VERIFY_SEARCH_INTERVAL = 0.140

# Maximum pixel radius for accepting a verified upgrade station match.
UPGRADE_STATION_VERIFY_RADIUS = 36

# Maximum duration for holding an upgrade station click.
CLICK_HOLD_MAX_DURATION = 8.0

# Duration for stats upgrade click bursts.
STATS_UPGRADE_CLICK_DURATION = 1.4

# Delay between stats upgrade click actions.
STATS_UPGRADE_CLICK_DELAY = 0.050

# Settling delay before scan actions in transition handlers.
IDLE_SETTLE_DELAY = 0.180

# Delay after clicking the new-level button.
NEW_LEVEL_CLICK_SETTLE_DELAY = 0.400

# Delay after clicking the level transition confirmation.
LEVEL_TRANSITION_CONFIRM_DELAY = 0.700

# Delay after the level transition animation completes.
LEVEL_TRANSITION_COMPLETE_DELAY = 0.850

# Retry interval during level transition attempts.
LEVEL_TRANSITION_RETRY_INTERVAL = 0.380

# Retry interval when waiting for the unlock button to appear.
WAIT_FOR_UNLOCK_RETRY_INTERVAL = 0.400

# Settling delay after clicking the unlock button.
WAIT_FOR_UNLOCK_SETTLE_DELAY = 0.440


# Telegram Notifications

# Enables Telegram notifications.
TELEGRAM_ENABLED = False

# Telegram bot token used for notification delivery.
TELEGRAM_BOT_TOKEN = ""

# Telegram chat identifier that receives notifications.
TELEGRAM_CHAT_ID = ""

# Maximum queued Telegram messages before new messages are dropped.
TELEGRAM_QUEUE_MAXSIZE = 100

# Maximum Telegram worker shutdown wait in seconds.
TELEGRAM_CLOSE_TIMEOUT = 5.6


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


# Upgrade Station HSV Gate

# Enables HSV range gating for upgrade station candidates.
UPGRADE_STATION_HSV_COLOR_GATE_ENABLED = True

# Accepted HSV ranges for upgrade station candidate gating.
UPGRADE_STATION_HSV_RANGES = (
    ((12, 88, 185), (29, 199, 252)),
    ((100, 135, 204), (103, 191, 255)),
)

# Minimum HSV mask ratio for accepting an upgrade station candidate.
UPGRADE_STATION_HSV_MIN_MATCH_RATIO = 0.50


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
SCROLL_PIXEL_STEP = 90

# Multiplier applied to scroll pixel distance.
SCROLL_DISTANCE_RATIO = 1.0

# Maximum oscillating scroll cycles before reset.
MAX_SCROLL_CYCLES = 8

# Step increment applied to each oscillating scroll cycle.
SCROLL_INCREMENT_STEP = 1

# Delay between scroll intervals.
SCROLL_INTERVAL_PAUSE = 0.260

# Settling delay after a scroll gesture.
POST_SCROLL_SETTLE = 0.360

# Duration of each scroll drag gesture.
SCROLL_DURATION = 0.320


# Handler Limits

# Maximum attempts when searching for an upgrade station.
UPGRADE_STATION_SEARCH_MAX_ATTEMPTS = 2

# Minimum upgrade counter before opening the stats panel.
UPGRADE_STATION_STATS_THRESHOLD = 1

# Maximum retry attempts during level transition.
TRANSITION_LEVEL_MAX_ATTEMPTS = 11

# Consecutive failed scan cycles before triggering idle recovery.
CONSECUTIVE_FAILED_CYCLES_THRESHOLD = 3

# Idle passes without progress before triggering a scroll search.
IDLE_PASS_SCROLL_THRESHOLD = 2


# Adaptive Runtime Tuning

# Enables adaptive runtime timing adjustments.
ADAPTIVE_TUNER_ENABLED = False

# EMA alpha used by the adaptive tuner.
ADAPTIVE_TUNER_ALPHA = 0.18

# Click success rate below which delays are increased.
ADAPTIVE_TUNER_CLICK_LOW_THRESHOLD = 0.96

# Click success rate above which delays are reduced.
ADAPTIVE_TUNER_CLICK_HIGH_THRESHOLD = 0.995

# Search success rate below which search interval is increased.
ADAPTIVE_TUNER_SEARCH_LOW_THRESHOLD = 0.90

# Search success rate above which search interval is reduced.
ADAPTIVE_TUNER_SEARCH_HIGH_THRESHOLD = 0.985

# Amount added to click delay after low click success.
ADAPTIVE_TUNER_CLICK_DELAY_STEP = 0.014

# Amount added to move delay after low click success.
ADAPTIVE_TUNER_MOVE_DELAY_STEP = 0.006

# Amount subtracted from click delay after high click success.
ADAPTIVE_TUNER_CLICK_DECREMENT = 0.007

# Amount subtracted from move delay after high click success.
ADAPTIVE_TUNER_MOVE_DECREMENT = 0.003

# Amount added to search interval after low search success.
ADAPTIVE_TUNER_SEARCH_INTERVAL_STEP = 0.060

# Amount subtracted from search interval after high search success.
ADAPTIVE_TUNER_SEARCH_DECREMENT = 0.030

# Minimum click delay allowed by adaptive tuning.
ADAPTIVE_TUNER_MIN_CLICK_DELAY = 0.105

# Maximum click delay allowed by adaptive tuning.
ADAPTIVE_TUNER_MAX_CLICK_DELAY = 0.200

# Minimum move delay allowed by adaptive tuning.
ADAPTIVE_TUNER_MIN_MOVE_DELAY = 0.040

# Maximum move delay allowed by adaptive tuning.
ADAPTIVE_TUNER_MAX_MOVE_DELAY = 0.085

# Minimum search interval allowed by adaptive tuning.
ADAPTIVE_TUNER_MIN_SEARCH_INTERVAL = 0.420

# Maximum search interval allowed by adaptive tuning.
ADAPTIVE_TUNER_MAX_SEARCH_INTERVAL = 0.760


# Adaptive Vision

# Enables adaptive vision threshold updates.
AI_VISION_ENABLED = False

# Base EMA alpha used for vision confidence updates.
AI_VISION_ALPHA = 0.18

# Maximum EMA alpha used when confidence exceeds the threshold strongly.
AI_VISION_ALPHA_MAX = 0.35

# Confidence boost applied to persisted vision thresholds.
AI_VISION_CONFIDENCE_BOOST = 0.10

# Confidence level that starts stronger adaptive alpha scaling.
AI_VISION_CONFIDENCE_THRESHOLD = 0.96

# Minimum adaptive threshold for box scans.
AI_BOX_THRESHOLD_MIN = 0.903

# Maximum adaptive threshold for box scans.
AI_BOX_THRESHOLD_MAX = 0.903

# Consecutive box misses before adaptive lowering is applied.
AI_BOX_MISS_WINDOW = 3

# Amount to lower the box threshold after miss window exhaustion.
AI_BOX_MISS_STEP = 0.002

# Minimum adaptive threshold for red icon scans.
AI_RED_ICON_THRESHOLD_MIN = 0.942

# Maximum adaptive threshold for red icon scans.
AI_RED_ICON_THRESHOLD_MAX = 0.942

# Safety margin subtracted from average red icon confidence.
AI_RED_ICON_MARGIN = 0.012

# Consecutive red icon misses before adaptive lowering is applied.
AI_RED_ICON_MISS_WINDOW = 5

# Amount to lower the red icon threshold after miss window exhaustion.
AI_RED_ICON_MISS_STEP = 0.001

# Minimum adaptive threshold for new-level button scans.
AI_NEW_LEVEL_THRESHOLD_MIN = 0.945

# Maximum adaptive threshold for new-level button scans.
AI_NEW_LEVEL_THRESHOLD_MAX = 0.988

# Consecutive new-level misses before adaptive lowering is applied.
AI_NEW_LEVEL_MISS_WINDOW = 3

# Amount to lower the new-level threshold after miss window exhaustion.
AI_NEW_LEVEL_MISS_STEP = 0.002

# Minimum adaptive threshold for new-level red icon scans.
AI_NEW_LEVEL_RED_ICON_THRESHOLD_MIN = 0.942

# Maximum adaptive threshold for new-level red icon scans.
AI_NEW_LEVEL_RED_ICON_THRESHOLD_MAX = 0.942

# Consecutive new-level red icon misses before adaptive lowering is applied.
AI_NEW_LEVEL_RED_ICON_MISS_WINDOW = 5

# Amount to lower the new-level red icon threshold after miss window exhaustion.
AI_NEW_LEVEL_RED_ICON_MISS_STEP = 0.001

# Minimum adaptive threshold for upgrade station scans.
AI_UPGRADE_STATION_THRESHOLD_MIN = 0.918

# Maximum adaptive threshold for upgrade station scans.
AI_UPGRADE_STATION_THRESHOLD_MAX = 0.918

# Consecutive upgrade station misses before adaptive lowering is applied.
AI_UPGRADE_STATION_MISS_WINDOW = 3

# Amount to lower the upgrade station threshold after miss window exhaustion.
AI_UPGRADE_STATION_MISS_STEP = 0.002

# Minimum adaptive threshold for stats upgrade scans.
AI_STATS_UPGRADE_THRESHOLD_MIN = 0.942

# Maximum adaptive threshold for stats upgrade scans.
AI_STATS_UPGRADE_THRESHOLD_MAX = 0.942

# Consecutive stats upgrade misses before adaptive lowering is applied.
AI_STATS_UPGRADE_MISS_WINDOW = 3

# Amount to lower the stats upgrade threshold after miss window exhaustion.
AI_STATS_UPGRADE_MISS_STEP = 0.001

# Project-relative file path for persisted adaptive vision state.
AI_VISION_STATE_FILE = "memory/vision_state.json"

# Minimum interval between adaptive vision state saves.
AI_VISION_SAVE_INTERVAL = 4.0


# Historical Learning

# Enables historical learning from completed levels.
AI_LEARNING_ENABLED = False

# Project-relative file path for persisted historical learning state.
AI_LEARNING_STATE_FILE = "memory/learning_state_stable.json"

# Minimum interval between historical learning state saves.
AI_LEARNING_SAVE_INTERVAL = 12.0

# Maximum historical learning records kept in memory and persisted state.
AI_LEARNING_RECORDS_LIMIT = 256

# Maximum wait for the historical learning worker to stop.
AI_LEARNING_THREAD_JOIN_TIMEOUT = 1.8

# Delay between historical learning worker passes.
AI_LEARNING_THREAD_INTERVAL = 0.9

# Number of adjacent level records considered as pairs.
AI_LEARNING_PAIR_WINDOW = 5

# Number of level records considered for a learning batch.
AI_LEARNING_BATCH_WINDOW = 12

# EMA alpha used for historical behavior profiles.
AI_LEARNING_EMA_ALPHA = 0.14

# Number of best profiles blended into a learned behavior.
AI_LEARNING_PROFILE_BLEND_TOP_K = 3

# Minimum improvement ratio needed before applying learned behavior.
AI_LEARNING_MIN_IMPROVEMENT_RATIO = 0.05

# Cooldown between learned behavior applications.
AI_LEARNING_APPLY_COOLDOWN = 50.0

# Minimum click delay allowed by historical learning.
AI_LEARNING_MIN_CLICK_DELAY = 0.105

# Maximum click delay allowed by historical learning.
AI_LEARNING_MAX_CLICK_DELAY = 0.200

# Minimum move delay allowed by historical learning.
AI_LEARNING_MIN_MOVE_DELAY = 0.040

# Maximum move delay allowed by historical learning.
AI_LEARNING_MAX_MOVE_DELAY = 0.085

# Minimum search interval allowed by historical learning.
AI_LEARNING_MIN_SEARCH_INTERVAL = 0.420

# Maximum search interval allowed by historical learning.
AI_LEARNING_MAX_SEARCH_INTERVAL = 0.760

# Minimum sleep between historical learning worker loops.
LEARNING_LOOP_MIN_SLEEP = 0.450


# Forbidden Zones

# Minimum x-coordinate for the general forbidden click band.
FORBIDDEN_CLICK_X_MIN = 60

# Maximum x-coordinate for the general forbidden click band.
FORBIDDEN_CLICK_X_MAX = 260

# Minimum y-coordinate for the general forbidden click band.
FORBIDDEN_CLICK_Y_MIN = 668

# Minimum x-coordinate for forbidden zone 1.
FORBIDDEN_ZONE_1_X_MIN = 290

# Maximum x-coordinate for forbidden zone 1.
FORBIDDEN_ZONE_1_X_MAX = 350

# Minimum y-coordinate for forbidden zone 1.
FORBIDDEN_ZONE_1_Y_MIN = 93

# Maximum y-coordinate for forbidden zone 1.
FORBIDDEN_ZONE_1_Y_MAX = 320

# Minimum x-coordinate for forbidden zone 2.
FORBIDDEN_ZONE_2_X_MIN = 0

# Maximum x-coordinate for forbidden zone 2.
FORBIDDEN_ZONE_2_X_MAX = 60

# Minimum y-coordinate for forbidden zone 2.
FORBIDDEN_ZONE_2_Y_MIN = 50

# Maximum y-coordinate for forbidden zone 2.
FORBIDDEN_ZONE_2_Y_MAX = 280

# Minimum x-coordinate for forbidden zone 3.
FORBIDDEN_ZONE_3_X_MIN = 0

# Maximum x-coordinate for forbidden zone 3.
FORBIDDEN_ZONE_3_X_MAX = 60

# Minimum y-coordinate for forbidden zone 3.
FORBIDDEN_ZONE_3_Y_MIN = 600

# Maximum y-coordinate for forbidden zone 3.
FORBIDDEN_ZONE_3_Y_MAX = 667

# Minimum x-coordinate for forbidden zone 4.
FORBIDDEN_ZONE_4_X_MIN = 145

# Maximum x-coordinate for forbidden zone 4.
FORBIDDEN_ZONE_4_X_MAX = 200

# Minimum y-coordinate for forbidden zone 4.
FORBIDDEN_ZONE_4_Y_MIN = 65

# Maximum y-coordinate for forbidden zone 4.
FORBIDDEN_ZONE_4_Y_MAX = 110

# Minimum x-coordinate for forbidden zone 5.
FORBIDDEN_ZONE_5_X_MIN = 55

# Maximum x-coordinate for forbidden zone 5.
FORBIDDEN_ZONE_5_X_MAX = 260

# Minimum y-coordinate for forbidden zone 5.
FORBIDDEN_ZONE_5_Y_MIN = 660

# Maximum y-coordinate for forbidden zone 5.
FORBIDDEN_ZONE_5_Y_MAX = 725

NUMBERED_FORBIDDEN_ZONE_BOUNDS = (
    (290, 350, 93, 320),
    (0, 60, 50, 280),
    (0, 60, 600, 667),
    (145, 200, 65, 110),
    (55, 260, 660, 725),
)
