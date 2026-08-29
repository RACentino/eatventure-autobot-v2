import math
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = str(PROJECT_ROOT / "assets")
LOGS_DIR = str(PROJECT_ROOT / "logs")

WINDOW_TITLE = "EatventureAuto"
WINDOW_WIDTH = 360
WINDOW_HEIGHT = 780
DEBUG = False

SCRCPY_MISS_RECOVERY_ENABLED = True
SCRCPY_RED_ICON_MISS_RECOVERY_DELAY = 0.144
SCRCPY_BOX_MISS_RECOVERY_DELAY = 0.144
SCRCPY_ACTION_SETTLE_DELAY = 0.016

MATCH_THRESHOLD = 0.98
RED_ICON_THRESHOLD = 0.930
NEW_LEVEL_RED_ICON_THRESHOLD = 0.942
STATS_RED_ICON_THRESHOLD = 0.930
UPGRADE_STATION_THRESHOLD = 0.910
BOX_THRESHOLD = 0.86581389
UNLOCK_THRESHOLD = 0.905
NEW_LEVEL_THRESHOLD = 0.965

UPGRADE_STATION_HSV_RANGES = (
    ((12, 88, 185), (29, 199, 252)),
    ((100, 135, 204), (103, 191, 255)),
)
UPGRADE_STATION_HSV_MIN_MATCH_RATIO = 0.50

BOX_HSV_RANGES = (
    ((18, 90, 120), (18, 129, 245)),
    ((18, 130, 120), (18, 130, 229)),
    ((18, 130, 235), (18, 130, 245)),
    ((20, 115, 220), (20, 125, 245)),
    ((23, 65, 154), (30, 115, 255)),
    ((24, 48, 208), (25, 63, 223)),
    ((26, 48, 192), (27, 63, 207)),
    ((22, 120, 248), (22, 127, 255)),
    ((13, 136, 136), (13, 143, 143)),
    ((14, 136, 136), (14, 143, 151)),
    ((14, 144, 144), (14, 151, 151)),
    ((14, 144, 169), (14, 151, 175)),
    ((14, 152, 144), (14, 159, 175)),
    ((14, 160, 136), (14, 167, 143)),
    ((15, 128, 168), (15, 135, 175)),
    ((15, 144, 152), (15, 151, 183)),
    ((15, 152, 144), (15, 159, 175)),
    ((15, 160, 136), (15, 167, 143)),
)
BOX_HSV_MIN_MATCH_RATIO = 0.26424162
BOX_NMS_IOU_THRESHOLD = 0.16770229
BOXES_MIN_MATCHES = 1

RED_ICON_HSV_RANGES = (
    ((0, 45, 120), (12, 255, 255)),
    ((166, 45, 120), (179, 255, 255)),
)
RED_ICON_HSV_MIN_MATCH_RATIO = 0.65
RED_ICON_FAST_MODE_ENABLED = True
RED_ICON_FAST_TEMPLATE_NAMES = ("RedIcon5",)
RED_ICON_MIN_MATCHES = 2
RED_ICON_FAST_MIN_DISTANCE = 30
RED_ICON_OFFSET_X = 10
RED_ICON_OFFSET_Y = 10

CLICK_DELAY = 0.175
MOUSE_MOVE_DELAY = 0.016
MOUSE_DOWN_DURATION = 0.125
MOUSE_UP_DURATION = 0.125
INPUT_RETRY_COUNT = 3
INPUT_RETRY_DELAY = 0.016

UPGRADE_SEARCH_INTERVAL = 0.080
UPGRADE_SEARCH_ATTEMPTS = 5
FAILED_UPGRADE_SEARCHES_BEFORE_SCROLL = 3
UPGRADES_BEFORE_STATS = 2
STATE_STALL_TIMEOUT_SECONDS = 9.0
EVENT_LOOP_INTERVAL = 0.016
FOCUS_SETTLE_DELAY = 0.016
UPGRADE_STATION_VERIFY_SETTLE_DELAY = 0.144
UPGRADE_STATION_VERIFY_SEARCH_ATTEMPTS = 4
UPGRADE_STATION_VERIFY_SEARCH_INTERVAL = 0.080
UPGRADE_STATION_DISAPPEAR_CONFIRMATION_COUNT = 2
CLICK_HOLD_MAX_DURATION = 9.0
UPGRADE_HOLD_CHECK_INTERVAL_MIN = 0.080
UPGRADE_HOLD_CHECK_INTERVAL_MAX = 0.144

MAX_SEARCH_Y = 660
EXTENDED_SEARCH_Y = WINDOW_HEIGHT
UPGRADE_STATION_SEARCH_Y = 760
BOX_SEARCH_Y = 780

IDLE_CLICK_POS = (8, 390)
STATS_UPGRADE_BUTTON_POS = (320, 730)
STATS_UPGRADE_POS = (285, 330)
SCROLL_START_POS = (170, 380)
NEW_LEVEL_BUTTON_POS = (40, 730)
LEVEL_TRANSITION_POS = (190, 560)

NEW_LEVEL_RED_ICON_X_MIN = 40
NEW_LEVEL_RED_ICON_X_MAX = 60
NEW_LEVEL_RED_ICON_Y_MIN = 700
NEW_LEVEL_RED_ICON_Y_MAX = 722
UPGRADE_RED_ICON_X_MIN = 280
UPGRADE_RED_ICON_X_MAX = 310
UPGRADE_RED_ICON_Y_MIN = 700
UPGRADE_RED_ICON_Y_MAX = 722

NEW_LEVEL_SEARCH_ATTEMPTS = 5
NEW_LEVEL_SEARCH_INTERVAL = 0.080
LEVEL_TRANSITION_SETTLE_DELAY = 0.300
NEW_LEVEL_CONFIRMATION_DELAY = 0.300
LEVEL_TRANSITION_SECONDARY_SETTLE_DELAY = 0.300
UNLOCK_SEARCH_ATTEMPTS = 4
UNLOCK_SEARCH_INTERVAL = 0.300
UNLOCK_SETTLE_DELAY = 0.016

SCROLL_PIXEL_STEP = 180
SCROLL_DISTANCE_RATIO = 1.0
MAX_SCROLL_CYCLES = 6
SCROLL_INCREMENT_STEP = 1
MAX_IDLE_PASS_ATTEMPTS = 1
SCROLL_INTERVAL_PAUSE = 0.300
POST_SCROLL_SETTLE = 0.300
SCROLL_DURATION = 0.300

STATS_UPGRADE_CLICK_DURATION = 1.5
STATS_UPGRADE_CLICK_DELAY = 0.016

FORBIDDEN_CLICK_X_MIN = 60
FORBIDDEN_CLICK_X_MAX = 260
FORBIDDEN_CLICK_Y_MIN = 668
EVENT_FORBIDDEN_ZONE_OPTIONS = {
    1: (290, 350, 93, 270),
    2: (290, 350, 93, 330),
    3: (290, 350, 93, 380),
}
NUMBERED_FORBIDDEN_ZONE_BOUNDS = (
    (0, 60, 50, 270),
    (0, 60, 640, 700),
    (150, 205, 70, 115),
    (65, 295, 700, 760),
)

TELEGRAM_ENABLED = os.getenv("EATVENTURE_TELEGRAM_ENABLED", "").lower() in {
    "1", "true", "yes", "on"
}
TELEGRAM_BOT_TOKEN = os.getenv("EATVENTURE_TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("EATVENTURE_TELEGRAM_CHAT_ID", "").strip()

RECOVERY_INITIAL_DELAY = 0.1
RECOVERY_MAX_DELAY = 1.0
HEARTBEAT_INTERVAL_SECONDS = 300.0
LOG_QUEUE_MAX_RECORDS = 2048


def validate_configuration() -> None:
    errors: list[str] = []

    def number(name: str, value: Any, minimum: float, maximum: float | None = None) -> None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            errors.append(f"{name} must be numeric")
            return
        if not math.isfinite(parsed) or parsed < minimum or (
            maximum is not None and parsed > maximum
        ):
            limit = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
            errors.append(f"{name} must be finite and within {limit}")

    if not str(WINDOW_TITLE).strip():
        errors.append("WINDOW_TITLE must not be empty")
    number("WINDOW_WIDTH", WINDOW_WIDTH, 1)
    number("WINDOW_HEIGHT", WINDOW_HEIGHT, 1)

    for name in (
        "MATCH_THRESHOLD",
        "RED_ICON_THRESHOLD",
        "NEW_LEVEL_RED_ICON_THRESHOLD",
        "STATS_RED_ICON_THRESHOLD",
        "UPGRADE_STATION_THRESHOLD",
        "BOX_THRESHOLD",
        "UNLOCK_THRESHOLD",
        "NEW_LEVEL_THRESHOLD",
        "UPGRADE_STATION_HSV_MIN_MATCH_RATIO",
        "BOX_HSV_MIN_MATCH_RATIO",
        "BOX_NMS_IOU_THRESHOLD",
        "RED_ICON_HSV_MIN_MATCH_RATIO",
    ):
        number(name, globals()[name], 0.0, 1.0)

    for name in (
        "SCRCPY_RED_ICON_MISS_RECOVERY_DELAY",
        "SCRCPY_BOX_MISS_RECOVERY_DELAY",
        "SCRCPY_ACTION_SETTLE_DELAY",
        "CLICK_DELAY",
        "MOUSE_MOVE_DELAY",
        "MOUSE_DOWN_DURATION",
        "MOUSE_UP_DURATION",
        "INPUT_RETRY_DELAY",
        "UPGRADE_SEARCH_INTERVAL",
        "STATE_STALL_TIMEOUT_SECONDS",
        "EVENT_LOOP_INTERVAL",
        "FOCUS_SETTLE_DELAY",
        "UPGRADE_STATION_VERIFY_SETTLE_DELAY",
        "UPGRADE_STATION_VERIFY_SEARCH_INTERVAL",
        "CLICK_HOLD_MAX_DURATION",
        "UPGRADE_HOLD_CHECK_INTERVAL_MIN",
        "UPGRADE_HOLD_CHECK_INTERVAL_MAX",
        "NEW_LEVEL_SEARCH_INTERVAL",
        "LEVEL_TRANSITION_SETTLE_DELAY",
        "NEW_LEVEL_CONFIRMATION_DELAY",
        "LEVEL_TRANSITION_SECONDARY_SETTLE_DELAY",
        "UNLOCK_SEARCH_INTERVAL",
        "UNLOCK_SETTLE_DELAY",
        "SCROLL_INTERVAL_PAUSE",
        "POST_SCROLL_SETTLE",
        "SCROLL_DURATION",
        "STATS_UPGRADE_CLICK_DURATION",
        "STATS_UPGRADE_CLICK_DELAY",
        "RECOVERY_INITIAL_DELAY",
        "RECOVERY_MAX_DELAY",
        "HEARTBEAT_INTERVAL_SECONDS",
    ):
        number(name, globals()[name], 0.0)

    for name in (
        "INPUT_RETRY_COUNT",
        "UPGRADE_SEARCH_ATTEMPTS",
        "FAILED_UPGRADE_SEARCHES_BEFORE_SCROLL",
        "UPGRADES_BEFORE_STATS",
        "UPGRADE_STATION_VERIFY_SEARCH_ATTEMPTS",
        "UPGRADE_STATION_DISAPPEAR_CONFIRMATION_COUNT",
        "BOXES_MIN_MATCHES",
        "RED_ICON_MIN_MATCHES",
        "RED_ICON_FAST_MIN_DISTANCE",
        "NEW_LEVEL_SEARCH_ATTEMPTS",
        "UNLOCK_SEARCH_ATTEMPTS",
        "SCROLL_PIXEL_STEP",
        "MAX_SCROLL_CYCLES",
        "SCROLL_INCREMENT_STEP",
        "MAX_IDLE_PASS_ATTEMPTS",
        "LOG_QUEUE_MAX_RECORDS",
    ):
        value = globals()[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(f"{name} must be a positive integer")

    number("SCROLL_DISTANCE_RATIO", SCROLL_DISTANCE_RATIO, 0.0)
    if UPGRADE_HOLD_CHECK_INTERVAL_MIN > UPGRADE_HOLD_CHECK_INTERVAL_MAX:
        errors.append("UPGRADE_HOLD_CHECK_INTERVAL_MIN must not exceed MAX")
    if RECOVERY_INITIAL_DELAY > RECOVERY_MAX_DELAY:
        errors.append("RECOVERY_INITIAL_DELAY must not exceed RECOVERY_MAX_DELAY")
    for name in ("STATE_STALL_TIMEOUT_SECONDS", "RECOVERY_INITIAL_DELAY", "RECOVERY_MAX_DELAY", "HEARTBEAT_INTERVAL_SECONDS", "SCROLL_DISTANCE_RATIO"):
        if globals()[name] <= 0:
            errors.append(f"{name} must be greater than zero")

    for name in ("MAX_SEARCH_Y", "EXTENDED_SEARCH_Y", "UPGRADE_STATION_SEARCH_Y", "BOX_SEARCH_Y"):
        value = globals()[name]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= WINDOW_HEIGHT:
            errors.append(f"{name} must be an integer within the client height")

    for name in (
        "IDLE_CLICK_POS",
        "STATS_UPGRADE_BUTTON_POS",
        "STATS_UPGRADE_POS",
        "SCROLL_START_POS",
        "NEW_LEVEL_BUTTON_POS",
        "LEVEL_TRANSITION_POS",
    ):
        point = globals()[name]
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            errors.append(f"{name} must contain two coordinates")
            continue
        try:
            x, y = map(int, point)
        except (TypeError, ValueError):
            errors.append(f"{name} coordinates must be integers")
            continue
        if not (0 <= x < WINDOW_WIDTH and 0 <= y < WINDOW_HEIGHT):
            errors.append(f"{name} must be inside the target client")

    if any(not isinstance(count, int) or count <= 0 for count in EVENT_FORBIDDEN_ZONE_OPTIONS):
        errors.append("EVENT_FORBIDDEN_ZONE_OPTIONS keys must be positive integers")
    zones = (
        tuple(NUMBERED_FORBIDDEN_ZONE_BOUNDS)
        + tuple(EVENT_FORBIDDEN_ZONE_OPTIONS.values())
        + ((FORBIDDEN_CLICK_X_MIN, FORBIDDEN_CLICK_X_MAX, FORBIDDEN_CLICK_Y_MIN, WINDOW_HEIGHT - 1),)
    )
    for index, zone in enumerate(zones, 1):
        if not isinstance(zone, (tuple, list)) or len(zone) != 4:
            errors.append(f"forbidden zone {index} must contain four coordinates")
            continue
        try:
            x_min, x_max, y_min, y_max = map(int, zone)
        except (TypeError, ValueError):
            errors.append(f"forbidden zone {index} coordinates must be integers")
            continue
        if not (0 <= x_min <= x_max < WINDOW_WIDTH and 0 <= y_min <= y_max < WINDOW_HEIGHT):
            errors.append(f"forbidden zone {index} is outside or has reversed bounds")

    for name, minimum, maximum, limit in (
        ("NEW_LEVEL_RED_ICON_X", NEW_LEVEL_RED_ICON_X_MIN, NEW_LEVEL_RED_ICON_X_MAX, WINDOW_WIDTH),
        ("NEW_LEVEL_RED_ICON_Y", NEW_LEVEL_RED_ICON_Y_MIN, NEW_LEVEL_RED_ICON_Y_MAX, WINDOW_HEIGHT),
        ("UPGRADE_RED_ICON_X", UPGRADE_RED_ICON_X_MIN, UPGRADE_RED_ICON_X_MAX, WINDOW_WIDTH),
        ("UPGRADE_RED_ICON_Y", UPGRADE_RED_ICON_Y_MIN, UPGRADE_RED_ICON_Y_MAX, WINDOW_HEIGHT),
    ):
        if not (0 <= minimum <= maximum < limit):
            errors.append(f"{name}_MIN/MAX are outside or reversed")

    scroll_distance = round(SCROLL_PIXEL_STEP * SCROLL_DISTANCE_RATIO)
    if not (
        0 <= SCROLL_START_POS[1] - scroll_distance < WINDOW_HEIGHT
        and 0 <= SCROLL_START_POS[1] + scroll_distance < WINDOW_HEIGHT
    ):
        errors.append("configured scroll endpoints must stay inside the target client")

    for name in ("UPGRADE_STATION_HSV_RANGES", "BOX_HSV_RANGES", "RED_ICON_HSV_RANGES"):
        ranges = globals()[name]
        if not ranges:
            errors.append(f"{name} must not be empty")
            continue
        for index, hsv_range in enumerate(ranges, 1):
            try:
                lower, upper = hsv_range
                if len(lower) != 3 or len(upper) != 3:
                    raise ValueError
                values = tuple(map(int, lower + upper))
            except (TypeError, ValueError):
                errors.append(f"{name}[{index}] must contain two HSV triples")
                continue
            if not (
                0 <= values[0] <= values[3] <= 179
                and 0 <= values[1] <= values[4] <= 255
                and 0 <= values[2] <= values[5] <= 255
            ):
                errors.append(f"{name}[{index}] has invalid HSV bounds")

    if errors:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(errors))
