import logging
import os
import queue
import threading
from typing import Protocol, cast

from eatventure_autobot.domain.config import TelegramConfig

logger = logging.getLogger(__name__)

_MAX_MESSAGE_LENGTH = 4096


def load_telegram_config_from_env() -> TelegramConfig:
    """Builds TelegramConfig from EATVENTURE_TELEGRAM_* env vars. Credentials are deliberately
    env-var gated and never live in domain/config.py, so they can't end up in a config dump or
    committed alongside it."""
    enabled = os.environ.get("EATVENTURE_TELEGRAM_ENABLED", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    bot_token = os.environ.get("EATVENTURE_TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("EATVENTURE_TELEGRAM_CHAT_ID", "").strip()
    if enabled and not (bot_token and chat_id):
        return TelegramConfig(enabled=False, bot_token=bot_token, chat_id=chat_id)
    return TelegramConfig(enabled=enabled, bot_token=bot_token, chat_id=chat_id)


class _HttpSession(Protocol):
    """The slice of requests.Session this module depends on, injectable for tests."""

    def post(self, url: str, json: dict[str, object], timeout: float) -> object: ...

    def close(self) -> None: ...


class TelegramNotifier:
    def __init__(self, config: TelegramConfig, session: _HttpSession | None = None) -> None:
        self.enabled = config.enabled
        self._chat_id = config.chat_id
        self._close_timeout = config.close_timeout
        self._url = f"https://api.telegram.org/bot{config.bot_token}/sendMessage"
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=config.queue_maxsize)
        self._session: _HttpSession | None = None
        self._thread: threading.Thread | None = None
        if self.enabled:
            if session is None:
                import requests

                session = cast("_HttpSession", requests.Session())
            self._session = session
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
                self._url, json={"chat_id": self._chat_id, "text": message}, timeout=5
            )
            if not getattr(response, "ok", True):
                status = getattr(response, "status_code", "?")
                logger.error("Telegram request failed: HTTP %s", status)
        except Exception as exc:
            logger.error("Telegram request failed: %s", type(exc).__name__)

    def send_message(self, message: str) -> bool:
        if not self.enabled:
            return False
        text = str(message).strip()
        if not text:
            return False
        try:
            self._queue.put_nowait(text[:_MAX_MESSAGE_LENGTH])
            return True
        except queue.Full:
            logger.warning("Telegram queue is full; dropping notification")
            return False

    def notify_bot_started(self) -> None:
        self.send_message("Bot Started")

    def notify_bot_stopped(self) -> None:
        self.send_message("Bot Stopped")

    def notify_new_level(self, level_number: int, time_spent: float) -> None:
        minutes, seconds = divmod(int(time_spent), 60)
        self.send_message(
            f"{level_number}. restaurant completed! Time spent: {minutes:02d}:{seconds:02d}"
        )

    def close(self) -> None:
        """Stops accepting new messages and lets the worker drain what's already queued
        (including a message sent moments before close(), e.g. notify_bot_stopped()) before
        exiting, bounded by close_timeout — it does not discard pending messages outright."""
        if not self.enabled:
            return
        self.enabled = False
        try:
            self._queue.put(None, timeout=self._close_timeout)
        except queue.Full:
            logger.warning("Telegram queue never drained enough to accept the stop sentinel")
            return
        if self._thread is not None:
            self._thread.join(timeout=self._close_timeout)
            if self._thread.is_alive():
                logger.warning("Telegram notifier did not stop before timeout")
                return
        if self._session is not None:
            self._session.close()
            self._session = None
