import logging
import math
from typing import Any
from urllib.parse import quote

import requests

import config

logger = logging.getLogger(__name__)
TELEGRAM_API_BASE_URL = "https://api.telegram.org"
DEFAULT_TELEGRAM_REQUEST_TIMEOUT = 5.0
MIN_TELEGRAM_REQUEST_TIMEOUT = 0.001


def _positive_timeout(value: Any, default: float) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        timeout = float(default)
    if not math.isfinite(timeout) or timeout <= 0:
        timeout = float(default)
    return max(MIN_TELEGRAM_REQUEST_TIMEOUT, timeout)


class TelegramNotifier:
    def __init__(self, bot_token: Any, chat_id: Any, enabled: bool = True) -> None:
        self._bot_token = str(bot_token or "").strip()
        self._chat_id = str(chat_id or "").strip()
        if enabled and (not self._bot_token or not self._chat_id):
            raise ValueError("Telegram is enabled but its credentials are missing")
        self.timeout = _positive_timeout(
            config.TELEGRAM_CLOSE_TIMEOUT, DEFAULT_TELEGRAM_REQUEST_TIMEOUT
        )
        self.enabled = bool(enabled)
        self._session = requests.Session() if self.enabled else None
        if self._session is not None:
            self._session.trust_env = False
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
        payload = {"chat_id": self._chat_id, "text": text[:4096]}
        try:
            response = self._session.post(
                self._send_message_url(),
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            logger.error("Error sending Telegram message: %s", exc.__class__.__name__)
            return False
        if response.ok:
            return True
        logger.error("Failed to send Telegram message: status=%s", response.status_code)
        return False

    def _send_message_url(self) -> str:
        encoded_token = quote(self._bot_token, safe=":")
        return f"{TELEGRAM_API_BASE_URL}/bot{encoded_token}/sendMessage"

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
