from __future__ import annotations

import base64
import logging
import re
import time
from collections import deque

from nio import AsyncClient, MatrixRoom, RoomMessageImage, RoomMessageText

from .config import Config
from .ollama import OllamaClient
from .search import SearXNGClient

logger = logging.getLogger(__name__)

IMAGE_TTL = 120  # seconds before a pending image is discarded

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information, news, or facts you don't know.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
}


class MatrixLLMBot:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.client = AsyncClient(config.matrix.server, config.matrix.username)
        self.ollama = OllamaClient(config.ollama.url, config.ollama.model)
        self.search = SearXNGClient(config.searxng_url) if config.searxng_url else None
        self._history: dict[tuple[str, str], deque[dict]] = {}
        # room_id -> (image_bytes, timestamp)
        self._pending_images: dict[str, tuple[bytes, float]] = {}
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
        if self.config.ollama.vision_model:
            self.client.add_event_callback(self._on_image, RoomMessageImage)

        await self.client.sync(timeout=0)
        self._started = True
        logger.info("Bot ready, listening for messages")

        try:
            await self.client.sync_forever(timeout=30_000)
        finally:
            if self.search:
                await self.search.aclose()
            await self.ollama.aclose()
            await self.client.close()

    async def _on_image(self, room: MatrixRoom, event: RoomMessageImage) -> None:
        if not self._started:
            return
        if event.sender == self.client.user_id:
            return

        logger.info("[%s] %s sent an image, storing as pending", room.room_id, event.sender)
        response = await self.client.download(event.url)
        if not hasattr(response, "body"):
            logger.error("Failed to download image: %s", response)
            return

        self._pending_images[room.room_id] = (response.body, time.monotonic())

    async def _on_message(self, room: MatrixRoom, event: RoomMessageText) -> None:
        if not self._started:
            return
        if event.sender == self.client.user_id:
            return

        body = event.body.strip()
        prompt = _extract_prompt(body, self.config.bot_name, self.client.user_id, event.source)
        if prompt is None:
            return

        logger.info("[%s] %s: %s", room.room_id, event.sender, prompt)

        # Check for a pending image in this room
        image_b64: str | None = None
        pending = self._pending_images.pop(room.room_id, None)
        if pending is not None:
            image_bytes, ts = pending
            if time.monotonic() - ts <= IMAGE_TTL:
                image_b64 = base64.b64encode(image_bytes).decode()
                logger.info("[%s] Pairing message with pending image", room.room_id)
            else:
                logger.info("[%s] Pending image expired, discarding", room.room_id)

        history = self._get_history(room.room_id, event.sender)
        # Store [image] placeholder in history so the model has context
        if image_b64:
            history.append({"role": "user", "content": f"[image] {prompt}"})
        else:
            history.append({"role": "user", "content": prompt})

        try:
            reply = await self._llm_reply(list(history), image_b64=image_b64)
        except Exception as exc:
            logger.error("LLM error: %s", exc)
            await self._send(room.room_id, f"Error: {exc}")
            history.pop()
            return

        history.append({"role": "assistant", "content": reply})
        logger.info("[%s] -> %s", room.room_id, reply[:120])
        await self._send(room.room_id, reply)

    async def _llm_reply(self, history: list[dict], image_b64: str | None = None) -> str:
        messages = list(history)
        if self.config.system_prompt:
            messages.insert(0, {"role": "system", "content": self.config.system_prompt})

        if image_b64:
            return await self.ollama.chat_with_image(
                messages, image_b64, model=self.config.ollama.vision_model
            )

        if not self.search:
            return await self.ollama.chat(messages)

        # Tool calling: let the model decide if it needs to search
        msg = await self.ollama.chat_with_tools(messages, [WEB_SEARCH_TOOL])
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            return msg.get("content", "")

        messages.append(msg)
        for call in tool_calls:
            fn = call.get("function", {})
            if fn.get("name") == "web_search":
                query = fn.get("arguments", {}).get("query", "")
                logger.info("Web search: %s", query)
                results = await self.search.search(query)
                tool_result = "\n\n".join(
                    f"{r['title']}\n{r['url']}\n{r['content']}" for r in results
                )
                messages.append({"role": "tool", "content": tool_result})

        return await self.ollama.chat(messages)

    async def _send(self, room_id: str, text: str) -> None:
        await self.client.room_send(
            room_id,
            message_type="m.room.message",
            content={"msgtype": "m.text", "body": text},
        )

    def _get_history(self, room_id: str, sender: str) -> deque[dict]:
        key = (room_id, sender)
        if key not in self._history:
            self._history[key] = deque(maxlen=self.config.history_size)
        return self._history[key]


def _is_mentioned(body: str, bot_name: str, user_id: str, source: dict) -> bool:
    mentioned_ids = source.get("content", {}).get("m.mentions", {}).get("user_ids", [])
    if user_id in mentioned_ids:
        return True
    if re.search(r"@?" + re.escape(bot_name), body, re.IGNORECASE):
        return True
    return False


def _extract_prompt(body: str, bot_name: str, user_id: str, source: dict) -> str | None:
    if not _is_mentioned(body, bot_name, user_id, source):
        return None
    return _strip_mention(body, bot_name, user_id)


def _strip_mention(body: str, bot_name: str, user_id: str) -> str:
    pattern = re.compile(
        r"@?" + re.escape(user_id) + r"|@?" + re.escape(bot_name),
        re.IGNORECASE,
    )
    cleaned = pattern.sub("", body).strip(" ,:\t\n")
    return cleaned or body.strip()
