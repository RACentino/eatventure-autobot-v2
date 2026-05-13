import logging
import queue
import threading
from typing import Any

import requests

from core import config

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, bot_token: Any, chat_id: Any, enabled: bool = True) -> None:
        self.bot_token = str(bot_token or "").strip()
        self.chat_id = str(chat_id or "").strip()
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.timeout = 5
        self.enabled = bool(enabled and self.bot_token and self.chat_id)
        try:
            queue_size = int(getattr(config, "TELEGRAM_QUEUE_MAXSIZE", 100))
        except (TypeError, ValueError):
            queue_size = 100
        self._queue = queue.Queue(maxsize=max(1, queue_size))
        self._stop = threading.Event()
        self._thread = None
        self._session = requests.Session() if self.enabled else None

        if self.enabled:
            self._thread = threading.Thread(target=self._worker_loop, name="telegram_notifier", daemon=True)
            self._thread.start()
            logger.info("Telegram notifier enabled")
        else:
            logger.info("Telegram notifier disabled")

    def _worker_loop(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                message = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._send_message_now(message)
            except Exception:
                logger.exception("Unexpected Telegram notification failure")
            finally:
                self._queue.task_done()

    def _send_message_now(self, message: str) -> bool:
        if not self.enabled or self._session is None:
            return False

        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
            }

            response = self._session.post(url, json=data, timeout=self.timeout)
            try:
                response_data = response.json()
            except ValueError:
                response_data = {}

            if response.ok and response_data.get("ok"):
                logger.debug("Telegram message sent successfully")
                return True

            description = response_data.get("description") if isinstance(response_data, dict) else None
            if not description:
                description = response.text[:200] if response.text else "unavailable"
            logger.error(
                "Failed to send Telegram message: status=%s description=%s",
                response.status_code,
                description,
            )
            return False
        except requests.RequestException as exc:
            logger.error("Error sending Telegram message: %s", exc.__class__.__name__)
            return False

    def send_message(self, message: Any) -> bool:
        if not self.enabled or self._stop.is_set():
            return False

        message = str(message).strip()
        if not message:
            return False
        if len(message) > 4096:
            message = message[:4093] + "..."
        try:
            self._queue.put_nowait(message)
            return True
        except queue.Full:
            logger.warning("Telegram queue is full; dropping notification")
            return False

    def _discard_pending_messages(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
            self._queue.task_done()

    def close(self) -> None:
        if not self.enabled and self._session is None:
            return
        self._stop.set()
        self.enabled = False
        self._discard_pending_messages()
        if self._thread is not None and self._thread.is_alive():
            try:
                close_timeout = float(getattr(config, "TELEGRAM_CLOSE_TIMEOUT", 2.0))
            except (TypeError, ValueError):
                close_timeout = 2.0
            close_timeout = max(0.0, close_timeout, float(self.timeout) + 0.5)
            self._thread.join(timeout=close_timeout)
            if self._thread.is_alive():
                logger.warning("Telegram notifier did not stop before timeout; session left open")
                return
        if self._session is not None:
            self._session.close()
            self._session = None

    def notify_bot_started(self) -> None:
        message = "Bot Started"
        self.send_message(message)

    def notify_bot_stopped(self) -> None:
        message = "Bot Stopped"
        self.send_message(message)

    def notify_new_level(self, level_number: int, time_spent: float) -> None:
        minutes = int(time_spent // 60)
        seconds = int(time_spent % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"

        message = f"{level_number}. restaurant completed! Time spent: {time_str}"
        self.send_message(message)

    def notify_level_milestone(self, total_levels: int) -> None:
        message = f"Milestone Reached\nTotal cities completed: {total_levels}"
        self.send_message(message)
