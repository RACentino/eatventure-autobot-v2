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


# The New Level dialog has a red X of its own, and it looks exactly like the boot popup's. While one
# of these states owns the screen the popup-closing rule is hard-locked: no capture, no click.
X_LOCKED_STATES = frozenset(
    {State.CHECK_NEW_LEVEL, State.TRANSITION_LEVEL, State.WAIT_FOR_UNLOCK}
)
