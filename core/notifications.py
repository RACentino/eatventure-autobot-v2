import html
import logging
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


@dataclass
class TelegramConfig:
    bot_token: str
    chat_id: str
    enabled: bool = True
    timeout_seconds: float = 5.0


class TelegramNotifier:
    def __init__(self, bot_token, chat_id, enabled=True, timeout_seconds=5.0):
        # FIX: Centralize and validate constructor inputs to avoid malformed URL state.
        clean_token = (bot_token or "").strip()
        clean_chat_id = str(chat_id).strip() if chat_id is not None else ""
        self.config = TelegramConfig(
            bot_token=clean_token,
            chat_id=clean_chat_id,
            enabled=bool(enabled),
            timeout_seconds=max(1.0, float(timeout_seconds)),
        )

        # FIX: Reuse a Session for better connection pooling/performance under many notifications.
        self._session = requests.Session()
        self.base_url = f"https://api.telegram.org/bot{self.config.bot_token}"
        self.enabled = self.config.enabled and bool(self.config.bot_token and self.config.chat_id)

        if self.enabled:
            logger.info("Telegram notifier enabled")
        else:
            logger.warning("Telegram notifier disabled")

    def send_message(self, message):
        if not self.enabled:
            return False

        # FIX: Escape outgoing message text while still allowing HTML parse mode safely.
        safe_message = html.escape(str(message or "")).strip()
        if not safe_message:
            logger.warning("Skipping Telegram send because message is empty")
            return False

        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.config.chat_id,
            "text": safe_message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        try:
            response = self._session.post(url, json=payload, timeout=self.config.timeout_seconds)
            response.raise_for_status()
            body = response.json()
            # FIX: Handle Telegram API-level failures even on HTTP 200.
            if not body.get("ok", False):
                logger.error("Telegram API rejected message: %s", body)
                return False
            logger.debug("Telegram message sent")
            return True
        except requests.RequestException as exc:
            logger.error("Error sending Telegram message: %s", exc)
            return False
        except ValueError as exc:
            logger.error("Invalid Telegram API JSON response: %s", exc)
            return False

    def notify_bot_started(self):
        # FIX: Keep formatting plain text so escaping does not break tags.
        self.send_message("🤖 Bot Started")

    def notify_bot_stopped(self):
        self.send_message("⏹️ Bot Stopped")

    def notify_new_level(self, level_number, time_spent):
        # FIX: Clamp and normalize user-provided values to avoid bad formatting.
        safe_level = max(1, int(level_number))
        safe_time_spent = max(0.0, float(time_spent))
        minutes = int(safe_time_spent // 60)
        seconds = int(safe_time_spent % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"
        self.send_message(f"{safe_level}. restaurant completed! Time spent: {time_str}")

    def notify_level_milestone(self, total_levels):
        safe_levels = max(0, int(total_levels))
        self.send_message(f"📊 Milestone reached! Total cities completed: {safe_levels}")
