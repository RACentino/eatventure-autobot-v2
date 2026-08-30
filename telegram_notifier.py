import logging
import queue
import threading
from typing import Any

import requests

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, bot_token: Any, chat_id: Any, enabled: bool = True) -> None:
        token = str(bot_token or "").strip()
        self.chat_id = str(chat_id or "").strip()
        self.enabled = bool(enabled and token and self.chat_id)
        if enabled and not self.enabled:
            logger.warning(
                "Telegram requested but credentials are incomplete; notifications disabled"
            )
        self.url = f"https://api.telegram.org/bot{token}/sendMessage"
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=100)
        self._session = requests.Session() if self.enabled else None
        self._thread = None
        if self.enabled:
            self._thread = threading.Thread(
                target=self._worker, name="telegram_notifier", daemon=True
            )
            self._thread.start()
        logger.info("Telegram notifier %s", "enabled" if self.enabled else "disabled")

    def _worker(self) -> None:
        while (message := self._queue.get()) is not None:
            try:
                self._send(message)
            except Exception:
                logger.exception("Unexpected Telegram worker error")
            finally:
                self._queue.task_done()
        self._queue.task_done()

    def _send(self, message: str) -> None:
        if self._session is None:
            return
        try:
            response = self._session.post(
                self.url,
                json={"chat_id": self.chat_id, "text": message},
                timeout=5,
            )
            if not response.ok:
                logger.error("Telegram request failed: HTTP %s", response.status_code)
        except requests.RequestException as exc:
            logger.error("Telegram request failed: %s", type(exc).__name__)

    def send_message(self, message: Any) -> bool:
        if not self.enabled:
            return False
        text = str(message).strip()
        if not text:
            return False
        try:
            self._queue.put_nowait(text[:4096])
            return True
        except queue.Full:
            logger.warning("Telegram queue is full; dropping notification")
            return False

    def close(self) -> None:
        if not self.enabled:
            return
        self.enabled = False
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        self._queue.put_nowait(None)
        if self._thread is not None:
            self._thread.join(timeout=5.5)
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Telegram notifier did not stop before timeout")
            return
        if self._session is not None:
            self._session.close()
            self._session = None

    def notify_bot_started(self) -> None:
        self.send_message("Bot Started")

    def notify_bot_stopped(self) -> None:
        self.send_message("Bot Stopped")

    def notify_new_level(self, level_number: int, time_spent: float) -> None:
        minutes, seconds = divmod(int(time_spent), 60)
        self.send_message(
            f"{level_number}. restaurant completed! "
            f"Time spent: {minutes:02d}:{seconds:02d}"
        )
