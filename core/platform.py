import importlib
import platform
from types import ModuleType

SYSTEM_NAME = platform.system()
IS_WINDOWS = SYSTEM_NAME == "Windows"
IS_LINUX = SYSTEM_NAME == "Linux"


class _OptionalModuleFallback:
    class error(Exception):
        pass


def _optional_import(module_name: str) -> tuple[ModuleType | None, Exception | None]:
    try:
        return importlib.import_module(module_name), None
    except Exception as exc:
        return None, exc


# Windows-specific imports
pywintypes, _pywintypes_error = _optional_import("pywintypes")
if pywintypes is None:
    pywintypes = _OptionalModuleFallback()

win32api, _win32api_error = _optional_import("win32api")
win32con, _win32con_error = _optional_import("win32con")
win32gui, _win32gui_error = _optional_import("win32gui")
win32ui, _win32ui_error = _optional_import("win32ui")

WIN32_AVAILABLE = bool(
    IS_WINDOWS
    and all(m is not None for m in (win32api, win32con, win32gui, win32ui))
)

# Linux-specific / Cross-platform imports
pywinctl, _pywinctl_error = _optional_import("pywinctl")
mss, _mss_error = _optional_import("mss")

LINUX_AVAILABLE = bool(
    IS_LINUX
    and all(m is not None for m in (pywinctl, mss))
)


def require_windows_backend(feature_name: str) -> None:
    if WIN32_AVAILABLE:
        return
    if not IS_WINDOWS:
        raise RuntimeError(f"{feature_name} requires Windows automation APIs; current platform is {SYSTEM_NAME}")
    raise RuntimeError(f"{feature_name} requires pywin32 modules")


def require_linux_backend(feature_name: str) -> None:
    if LINUX_AVAILABLE:
        return
    if not IS_LINUX:
        raise RuntimeError(f"{feature_name} requires Linux automation APIs; current platform is {SYSTEM_NAME}")
    raise RuntimeError(f"{feature_name} requires pywinctl and mss modules")
