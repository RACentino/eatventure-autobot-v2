from enum import Enum, auto


class State(Enum):
    FIND_RED_ICONS = auto()
    CLICK_RED_ICON = auto()
    CHECK_UNLOCK = auto()
    SEARCH_UPGRADE_STATION = auto()
    HOLD_UPGRADE_STATION = auto()
    UPGRADE_STATS = auto()
    OPEN_BOXES = auto()
    SCROLL = auto()
    CHECK_NEW_LEVEL = auto()
    TRANSITION_LEVEL = auto()
    WAIT_FOR_UNLOCK = auto()
