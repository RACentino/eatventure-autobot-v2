import logging
from typing import Any

import requests

import config

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, bot_token: Any, chat_id: Any, enabled: bool = True) -> None:
        self.bot_token = str(bot_token or "").strip()
        self.chat_id = str(chat_id or "").strip()
        self.timeout = 5.0
        self.enabled = bool(enabled and self.bot_token and self.chat_id)
        self._session = requests.Session() if self.enabled else None
        logger.info("Telegram notifier %s", "enabled" if self.enabled else "disabled")

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
        self.enabled = False

    def send_message(self, message: Any) -> bool:
        if not self.enabled or self._session is None:
            return False
        text = str(message).strip()
        if not text:
            return False
        payload = {"chat_id": self.chat_id, "text": text[:4096]}
        try:
            response = self._session.post(
                f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                json=payload,
                timeout=max(0.0, float(config.TELEGRAM_CLOSE_TIMEOUT), self.timeout),
            )
        except requests.RequestException as exc:
            logger.error("Error sending Telegram message: %s", exc.__class__.__name__)
            return False
        if response.ok:
            return True
        logger.error("Failed to send Telegram message: status=%s", response.status_code)
        return False

    def _notify(self, message: str) -> None:
        if not self.send_message(message):
            logger.debug("Telegram notification was not delivered")

    def notify_bot_started(self) -> None:
        self._notify("Bot Started")

    def notify_bot_stopped(self) -> None:
        self._notify("Bot Stopped")

    def notify_new_level(self, level_number: int, time_spent: float) -> None:
        minutes = int(time_spent // 60)
        seconds = int(time_spent % 60)
        self._notify(
            f"{level_number}. restaurant completed! Time spent: {minutes:02d}:{seconds:02d}"
        )

    def notify_level_milestone(self, total_levels: int) -> None:
        self._notify(f"Milestone Reached\nTotal cities completed: {total_levels}")
