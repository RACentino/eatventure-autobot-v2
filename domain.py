from enum import StrEnum
from typing import Any


class AssetType(StrEnum):
    RED_ICON = "red_icon"
    UPGRADE_STATION = "upgrade_station"
    BOX = "box"


class BehaviorKey(StrEnum):
    CLICK_DELAY = "click_delay"
    MOVE_DELAY = "move_delay"
    SEARCH_INTERVAL = "search_interval"


class BoxTemplateName(StrEnum):
    BOX_ONE = "box1"
    BOX_TWO = "box2"
    BOX_THREE = "box3"
    BOX_FOUR = "box4"
    BOX_FIVE = "box5"


class SupervisionFlag(StrEnum):
    BOX_NMS = "SUPERVISION_BOX_NMS_ENABLED"
    RED_ICON_NMS = "SUPERVISION_RED_ICON_NMS_ENABLED"
    UPGRADE_STATION_NMS = "SUPERVISION_UPGRADE_STATION_NMS_ENABLED"


class TemplateName(StrEnum):
    NEW_LEVEL = "newLevel"
    UNLOCK = "unlock"
    UPGRADE_STATION = "upgradeStation"


ASSET_CLASS_IDS = {
    AssetType.RED_ICON: 0,
    AssetType.UPGRADE_STATION: 1,
    AssetType.BOX: 2,
}
ASSET_CLASS_NAMES = {
    class_id: asset_type.value for asset_type, class_id in ASSET_CLASS_IDS.items()
}

BEHAVIOR_KEYS = tuple(behavior_key.value for behavior_key in BehaviorKey)
BOX_TEMPLATE_NAMES = tuple(template_name.value for template_name in BoxTemplateName)
REQUIRED_TEMPLATE_NAMES = tuple(template_name.value for template_name in TemplateName)

RED_ICON_TEMPLATE_PREFIX = "RedIcon"
RED_ICON_MISSING_PATTERN = "RedIcon*"
RED_ICON_PRIMARY_TEMPLATE = "RedIcon"
RED_ICON_NO_BACKGROUND_TEMPLATE = "RedIconNoBG"

BOT_RUN_ITERATION_LIMIT = 2_147_483_647
LEARNING_LOOP_ITERATION_LIMIT = 2_147_483_647
MAX_TEMPLATE_FILES = 128
MAX_TEMPLATE_NAMES = 32
MAX_UPGRADE_SEARCH_ATTEMPTS = 5
MAX_LEVEL_TRANSITION_ATTEMPTS = 5
MAX_WAIT_FOR_UNLOCK_ATTEMPTS = 4
MAX_RUNTIME_STATE_FILE_BYTES = 4_000_000

SUCCESSFUL_RED_ICON_ROWS_LIMIT = 24
SEARCH_CYCLES_BEFORE_SCROLL = 2
FAILED_CYCLES_BEFORE_SCROLL = 3
FAILED_CLICK_BUCKET_LIMIT = 3
RED_ICON_FALLBACK_MIN_DISTANCE = 80
ICON_MERGE_DISTANCE_PIXELS = 10
ROW_PRIORITY_DISTANCE_PIXELS = 50
CLICK_BUCKET_SIZE_PIXELS = 30
SUCCESSFUL_ROW_DEDUP_DISTANCE_PIXELS = 12
MIN_TEMPLATE_DIMENSION = 1

UPGRADE_STATION_THRESHOLD_RELAXATION = 0.05
UPGRADE_STATION_HOLD_MIN_VERIFY_INTERVAL = 0.05
UPGRADE_STATION_HOLD_MAX_VERIFY_INTERVAL = 0.20
UPGRADE_STATS_CYCLE_INTERVAL = 2

BOT_STATE_LOOP_SLEEP_SECONDS = 0.10
CHECK_NEW_LEVEL_PRE_CLICK_DELAY = 0.05
TRANSITION_LEVEL_BUTTON_WAIT_SECONDS = 1.00
TRANSITION_LEVEL_RETRY_DELAY_SECONDS = 0.20
WAIT_FOR_UNLOCK_PRE_SCAN_DELAY = 0.05
WAIT_FOR_UNLOCK_RETRY_DELAY = 0.30
WAIT_FOR_UNLOCK_POST_CLICK_DELAY = 0.50


def asset_type_value(asset_type: AssetType | str) -> str:
    if isinstance(asset_type, AssetType):
        return asset_type.value
    return str(asset_type)


def normalize_asset_type(
    value: Any, default: AssetType = AssetType.RED_ICON
) -> AssetType:
    try:
        return AssetType(str(value))
    except (TypeError, ValueError):
        return default


def asset_class_id_for(asset_type: Any) -> int:
    return ASSET_CLASS_IDS[normalize_asset_type(asset_type)]


def asset_class_name_for(class_id: Any) -> str:
    try:
        normalized_class_id = int(class_id)
    except (TypeError, ValueError):
        return AssetType.RED_ICON.value
    return ASSET_CLASS_NAMES.get(normalized_class_id, AssetType.RED_ICON.value)
