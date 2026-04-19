from __future__ import annotations

import base64
import html as html_module
import logging
import re
import time

from nio import AsyncClient, MatrixRoom, RoomMessageImage, RoomMessageText

from .config import Config
from .handlers.commands import CommandHandler
from .handlers.reply import ConversationHistory, ReplyHandler
from .integrations.k8s import K8sClient
from .integrations.ollama import OllamaClient
from .integrations.search import SearXNGClient

logger = logging.getLogger(__name__)

IMAGE_TTL = 120  # seconds before a pending image is discarded


class MatrixLLMBot:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.client = AsyncClient(config.matrix.server, config.matrix.username)
        self.ollama = OllamaClient(
            config.ollama.url,
            config.ollama.model,
            config.ollama.routing_model,
            api_kind=config.ollama.api_kind,
            api_key=config.ollama.api_key,
        )
        self.search = SearXNGClient(config.searxng_url) if config.searxng_url else None
        self.k8s = K8sClient() if config.k8s_enabled else None

        self._reply_handler = ReplyHandler(config, self.ollama, self.k8s, self.search)
        self._command_handler = CommandHandler(config, self.client, self.k8s, bool(self.search), self._send)
        self._history = ConversationHistory(config.history_size)
        self._pending_images: dict[tuple[str, str], tuple[bytes, float]] = {}
        self._seen_events: set[str] = set()
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
            if self.k8s:
                await self.k8s.aclose()
            await self.ollama.aclose()
            await self.client.close()

    async def _on_image(self, room: MatrixRoom, event: RoomMessageImage) -> None:
        if not self._started or event.sender == self.client.user_id:
            return
        logger.info("[%s] %s sent an image, storing as pending", room.room_id, event.sender)
        response = await self.client.download(event.url)
        if not hasattr(response, "body"):
            logger.error("Failed to download image: %s", response)
            return
        self._pending_images[(room.room_id, event.sender)] = (response.body, time.monotonic())

    async def _on_message(self, room: MatrixRoom, event: RoomMessageText) -> None:
        if not self._started or event.sender == self.client.user_id:
            return

        if event.event_id in self._seen_events:
            return
        self._seen_events.add(event.event_id)
        if len(self._seen_events) > 2000:
            self._seen_events.clear()

        body = event.body.strip()
        explicit_mention = _is_explicit_mention(event.source, self.client.user_id)
        prompt = _extract_prompt(body, self.config.bot_name, self.client.user_id, event.source)
        if prompt is None:
            return
        logger.debug("[%s] Mention detected from %s: %r", room.room_id, event.sender, body[:120])

        if prompt.strip().lower() == "reset":
            self._history.clear(room.room_id, event.sender)
            await self._send(room.room_id, "Conversation history cleared.")
            return

        if await self._command_handler.handle(room.room_id, prompt, event.sender):
            return

        # Skip the LLM gate when the bot was explicitly mentioned via m.mentions —
        # the user unambiguously addressed us. Only run it for name-based detection
        # where a peer bot might be the real target.
        if not explicit_mention:
            try:
                if not await self.ollama.should_respond(self.config.bot_name, body, self.config.peer_bots):
                    logger.info("[%s] Gate rejected message from %s: %r", room.room_id, event.sender, body[:120])
                    return
                logger.debug("[%s] Gate accepted message from %s", room.room_id, event.sender)
            except Exception as exc:
                logger.warning("Gate call failed, proceeding anyway: %s", exc)

        logger.info("[%s] %s: %s", room.room_id, event.sender, prompt)

        image_b64: str | None = None
        pending = self._pending_images.pop((room.room_id, event.sender), None)
        if pending is not None:
            image_bytes, ts = pending
            if time.monotonic() - ts <= IMAGE_TTL:
                image_b64 = base64.b64encode(image_bytes).decode()
                logger.info("[%s] Pairing message with pending image from %s", room.room_id, event.sender)
            else:
                logger.info("[%s] Pending image from %s expired, discarding", room.room_id, event.sender)

        history = self._history.get(room.room_id, event.sender)
        history.append({"role": "user", "content": f"[image] {prompt}" if image_b64 else prompt})

        await self.client.room_typing(room.room_id, typing_state=True, timeout=120_000)
        reply: str | None = None
        try:
            reply = await self._reply_handler.reply(
                list(history),
                prompt=prompt,
                image_b64=image_b64,
                room=room,
                sender=event.sender,
            )
        except Exception as exc:
            logger.error("LLM error: %s", exc)
            await self._send(room.room_id, f"Error: {exc}")
        finally:
            await self.client.room_typing(room.room_id, typing_state=False)
            if reply is None:
                history.pop()

        if reply is not None:
            history.append({"role": "assistant", "content": reply})
            logger.info("[%s] -> %s", room.room_id, reply[:120])
            await self._send_reply(room, event.sender, reply)

    async def _send_reply(self, room: MatrixRoom, sender: str, text: str) -> None:
        display_name = (
            room.users[sender].display_name if sender in room.users and room.users[sender].display_name else sender
        )
        plain = f"{display_name}: {text}"
        mention_html = (
            f'<a href="https://matrix.to/#/{html_module.escape(sender)}">{html_module.escape(display_name)}</a>'
        )
        formatted = f"{mention_html}: {html_module.escape(text)}"
        await self.client.room_send(
            room.room_id,
            message_type="m.room.message",
            content={
                "msgtype": "m.text",
                "body": plain,
                "format": "org.matrix.custom.html",
                "formatted_body": formatted,
                "m.mentions": {"user_ids": [sender]},
            },
        )

    async def _send(self, room_id: str, text: str) -> None:
        await self.client.room_send(
            room_id,
            message_type="m.room.message",
            content={"msgtype": "m.text", "body": text},
        )


def _is_explicit_mention(source: dict, user_id: str) -> bool:
    """True when the Matrix event explicitly named this bot via m.mentions."""
    return user_id in source.get("content", {}).get("m.mentions", {}).get("user_ids", [])


def _is_mentioned(body: str, bot_name: str, user_id: str, source: dict) -> bool:
    if _is_explicit_mention(source, user_id):
        return True
    if re.search(r"@?" + re.escape(bot_name), body, re.IGNORECASE):
        return True
    return False


def _extract_prompt(body: str, bot_name: str, user_id: str, source: dict) -> str | None:
    if not _is_mentioned(body, bot_name, user_id, source):
        return None
    return _strip_mention(body, bot_name, user_id)


def _strip_mention(body: str, bot_name: str, user_id: str) -> str:
    mention = r"(?:" + re.escape(user_id) + r"|@?" + re.escape(bot_name) + r")"
    cleaned = re.sub(r"(?i)^\s*" + mention + r"[,:\s]*", "", body)
    cleaned = re.sub(r"(?i)[,:\s]*" + mention + r"\s*$", "", cleaned)
    return cleaned.strip() or body.strip()
