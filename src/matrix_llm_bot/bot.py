from __future__ import annotations

import logging
from collections import deque

from nio import AsyncClient, MatrixRoom, RoomMessageText, SyncResponse

from .config import Config
from .ollama import OllamaClient

logger = logging.getLogger(__name__)


class MatrixLLMBot:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.client = AsyncClient(config.matrix.server, config.matrix.username)
        self.ollama = OllamaClient(config.ollama.url, config.ollama.model)
        # history keyed by (room_id, sender) -> deque of {"role": ..., "content": ...}
        self._history: dict[tuple[str, str], deque[dict]] = {}
        self._started = False

    async def run(self) -> None:
        logger.info("Logging in as %s", self.config.matrix.username)
        resp = await self.client.login(self.config.matrix.password)
        logger.info("Logged in: %s", resp)

        await self.client.set_displayname(self.config.bot_name)
        logger.info("Display name set to %s", self.config.bot_name)

        for room_id in self.config.rooms:
            await self.client.join(room_id)
            logger.info("Joined room %s", room_id)

        self.client.add_event_callback(self._on_message, RoomMessageText)

        # First sync to consume backlog without processing it
        await self.client.sync(timeout=0)
        self._started = True
        logger.info("Bot ready, listening for messages")

        try:
            await self.client.sync_forever(timeout=30_000)
        finally:
            await self.ollama.aclose()
            await self.client.close()

    async def _on_message(self, room: MatrixRoom, event: RoomMessageText) -> None:
        if not self._started:
            return

        sender = event.sender
        if sender == self.client.user_id:
            return

        body = event.body.strip()
        prompt = _extract_prompt(body, self.config.bot_name)
        if prompt is None:
            return

        logger.info("[%s] %s: %s", room.room_id, sender, prompt)

        history = self._get_history(room.room_id, sender)
        history.append({"role": "user", "content": prompt})

        try:
            messages = list(history)
            if self.config.system_prompt:
                messages.insert(0, {"role": "system", "content": self.config.system_prompt})
            reply = await self.ollama.chat(messages)
        except Exception as exc:
            logger.error("Ollama error: %s", exc)
            await self.client.room_send(
                room.room_id,
                message_type="m.room.message",
                content={"msgtype": "m.text", "body": f"Error: {exc}"},
            )
            history.pop()
            return

        history.append({"role": "assistant", "content": reply})
        logger.info("[%s] -> %s", room.room_id, reply[:120])

        await self.client.room_send(
            room.room_id,
            message_type="m.room.message",
            content={"msgtype": "m.text", "body": reply},
        )

    def _get_history(self, room_id: str, sender: str) -> deque[dict]:
        key = (room_id, sender)
        if key not in self._history:
            self._history[key] = deque(maxlen=self.config.history_size)
        return self._history[key]


def _extract_prompt(body: str, bot_name: str) -> str | None:
    prefix_ai = ".ai "
    prefix_name = f"{bot_name}:"
    if body.lower().startswith(prefix_ai):
        return body[len(prefix_ai):].strip()
    if body.lower().startswith(prefix_name.lower()):
        return body[len(prefix_name):].strip()
    return None


