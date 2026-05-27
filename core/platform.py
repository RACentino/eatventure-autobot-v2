import importlib
import os
import sys
from types import ModuleType

SYSTEM_NAME_BY_PLATFORM = {
    "linux": "Linux",
    "win32": "Windows",
    "cygwin": "Windows",
}
SYSTEM_NAME = SYSTEM_NAME_BY_PLATFORM.get(sys.platform, sys.platform)
IS_WINDOWS = SYSTEM_NAME == "Windows"
IS_LINUX = SYSTEM_NAME == "Linux"
SUPPORTED_SYSTEMS = {"Windows", "Linux"}


class AutomationBackendError(RuntimeError):
    pass


class BackendDependencyError(AutomationBackendError):
    pass


def _optional_import(module_name: str) -> tuple[ModuleType | None, Exception | None]:
    try:
        return importlib.import_module(module_name), None
    except Exception as exc:
        return None, exc


pywinctl, _pywinctl_error = _optional_import("pywinctl")
mss, _mss_error = _optional_import("mss")
pynput_mouse, _pynput_mouse_error = _optional_import("pynput.mouse")
pynput_keyboard, _pynput_keyboard_error = _optional_import("pynput.keyboard")


def _dependency_messages(dependency_errors: tuple[tuple[str, Exception | None], ...]) -> list[str]:
    return [f"{name} ({error})" for name, error in dependency_errors if error is not None]


def _missing_automation_dependency_messages() -> list[str]:
    return _dependency_messages(
        (
            ("PyWinCtl", _pywinctl_error),
            ("mss", _mss_error),
            ("pynput.mouse", _pynput_mouse_error),
        )
    )


def _missing_keyboard_dependency_messages() -> list[str]:
    return _dependency_messages((("pynput.keyboard", _pynput_keyboard_error),))


def _unsupported_platform_error() -> str | None:
    if SYSTEM_NAME in SUPPORTED_SYSTEMS:
        return None
    return f"unsupported desktop platform: {SYSTEM_NAME}"


def _display_session_error() -> str | None:
    if not IS_LINUX:
        return None
    if os.environ.get("DISPLAY"):
        return None
    session_type = os.environ.get("XDG_SESSION_TYPE") or "unknown"
    return f"Linux runtime automation requires an X11/XWayland DISPLAY; session={session_type}"


def _runtime_environment_errors() -> list[str]:
    return [
        error
        for error in (_unsupported_platform_error(), _display_session_error())
        if error is not None
    ]


def _raise_backend_error(feature_name: str, dependency_messages: list[str]) -> None:
    details = []
    if dependency_messages:
        details.append("missing dependencies: " + ", ".join(dependency_messages))
    details.extend(_runtime_environment_errors())
    if details:
        raise BackendDependencyError(f"{feature_name} requires the automation backend; {'; '.join(details)}")


def automation_backend_available() -> bool:
    return not _missing_automation_dependency_messages() and not _runtime_environment_errors()


def keyboard_backend_available() -> bool:
    return not _missing_keyboard_dependency_messages() and not _runtime_environment_errors()


AUTOMATION_BACKEND_AVAILABLE = automation_backend_available()
KEYBOARD_BACKEND_AVAILABLE = keyboard_backend_available()


def require_automation_backend(feature_name: str) -> None:
    _raise_backend_error(feature_name, _missing_automation_dependency_messages())


def require_keyboard_backend(feature_name: str) -> None:
    _raise_backend_error(feature_name, _missing_keyboard_dependency_messages())
