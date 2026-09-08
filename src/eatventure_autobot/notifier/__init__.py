from eatventure_autobot.domain.config import TelegramConfig
from eatventure_autobot.domain.protocols import Notifier
from eatventure_autobot.notifier.telegram import TelegramNotifier


def create_notifier(config: TelegramConfig) -> Notifier:
    return TelegramNotifier(config)
