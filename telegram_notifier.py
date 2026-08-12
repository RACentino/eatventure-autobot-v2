import logging
import math
import queue
import threading
from typing import Any
from urllib.parse import quote

import requests

import config
from domain import MAX_RUNTIME_LOOP_ITERATIONS

logger = logging.getLogger(__name__)
TELEGRAM_API_BASE_URL = "https://api.telegram.org"
DEFAULT_TELEGRAM_REQUEST_TIMEOUT = 5.0
MIN_TELEGRAM_REQUEST_TIMEOUT = 0.001
TELEGRAM_QUEUE_SIZE = 16
TELEGRAM_WORKER_POLL_SECONDS = 0.1


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
        self.request_timeout = _positive_timeout(
            config.TELEGRAM_REQUEST_TIMEOUT, DEFAULT_TELEGRAM_REQUEST_TIMEOUT
        )
        self.shutdown_timeout = _positive_timeout(
            config.TELEGRAM_SHUTDOWN_TIMEOUT, DEFAULT_TELEGRAM_REQUEST_TIMEOUT
        )
        self.enabled = bool(enabled)
        self._session = requests.Session() if self.enabled else None
        self._messages: queue.Queue[str] | None = (
            queue.Queue(maxsize=TELEGRAM_QUEUE_SIZE) if self.enabled else None
        )
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        if self._session is not None:
            self._session.trust_env = False
            self._worker = threading.Thread(
                target=self._run, name="telegram_notifier", daemon=True
            )
            try:
                self._worker.start()
            except RuntimeError:
                self._worker = None
                self._session.close()
                self._session = None
                raise
        logger.info("Telegram notifier %s", "enabled" if self.enabled else "disabled")

    def close(self) -> None:
        self.enabled = False
        self._stop.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=self.shutdown_timeout)
        if self._worker is not None and self._worker.is_alive():
            logger.error("Telegram worker did not stop before timeout")
            return
        self._worker = None
        if self._session is not None:
            self._session.close()
            self._session = None

    def send_message(self, message: Any) -> bool:
        if not self.enabled or self._messages is None:
            return False
        text = str(message).strip()
        if not text:
            return False
        try:
            self._messages.put_nowait(text[:4096])
            return True
        except queue.Full:
            logger.warning("Telegram queue is full; dropping newest notification")
            return False

    def _run(self) -> None:
        for _ in range(MAX_RUNTIME_LOOP_ITERATIONS):
            if self._stop.is_set():
                return
            if self._messages is None:
                return
            try:
                message = self._messages.get(timeout=TELEGRAM_WORKER_POLL_SECONDS)
            except queue.Empty:
                continue
            try:
                self._deliver(message)
            finally:
                self._messages.task_done()
        logger.critical("Telegram worker exhausted its iteration limit")

    def _deliver(self, text: str) -> bool:
        if self._session is None:
            return False
        payload: dict[str, Any] = {
            "chat_id": self._chat_id,
            "text": text[:4096],
        }
        try:
            response = self._session.post(
                self._send_message_url(),
                json=payload,
                timeout=self.request_timeout,
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

    def notify_failure(self, message: str) -> None:
        self._notify(f"Bot recovering: {message}")

    def notify_recovered(self, message: str) -> None:
        self._notify(f"Bot resumed: {message}")
