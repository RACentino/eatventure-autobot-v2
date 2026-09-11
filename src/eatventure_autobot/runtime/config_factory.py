"""Builds a validated BotConfig for the running app. Replaces the old PathsConfig.under /
BotConfig.default classmethods, which moved out of domain/config.py so that module holds only
literal field declarations."""

from pathlib import Path

from eatventure_autobot.domain.config import BotConfig, PathsConfig
from eatventure_autobot.domain.config_validation import validate_bot_config
from eatventure_autobot.notifier.telegram import load_telegram_config_from_env


def build_paths_config(project_root: Path) -> PathsConfig:
    return PathsConfig(
        project_root=project_root,
        assets_dir=project_root / "assets",
        logs_dir=project_root / "logs",
    )


def build_default_config(project_root: Path) -> BotConfig:
    config = BotConfig(
        paths=build_paths_config(project_root),
        telegram=load_telegram_config_from_env(),
    )
    validate_bot_config(config)
    return config
