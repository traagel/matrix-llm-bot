from __future__ import annotations

import base64
import io
import logging
import mimetypes
import re
import time
from collections import deque
from datetime import datetime, timezone

import httpx
from nio import AsyncClient, MatrixRoom, RoomMessageImage, RoomMessageText, UploadResponse

from .config import Config
from .k8s import K8sClient
from .ollama import OllamaClient
from .search import SearXNGClient

logger = logging.getLogger(__name__)

IMAGE_TTL = 120  # seconds before a pending image is discarded

K8S_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "k8s_check_health",
            "description": "Check if a specific service/deployment is healthy and running in the k3s cluster. Use when someone asks if a service is up, running, or healthy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string", "description": "Name of the service to check, e.g. 'jellyfin', 'ollama', 'caddy', 'pihole'"},
                },
                "required": ["service_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "k8s_get_pods",
            "description": "List pods and their status in the k3s cluster. Optionally filter by namespace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "description": "Kubernetes namespace to filter by. Leave empty for all namespaces."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "k8s_get_logs",
            "description": "Get recent logs from a pod in the k3s cluster. Use this to debug issues or check what a service is doing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "description": "Kubernetes namespace the pod is in"},
                    "pod_name": {"type": "string", "description": "Full or partial pod name"},
                    "tail_lines": {"type": "integer", "description": "Number of log lines to retrieve (default 30)"},
                },
                "required": ["namespace", "pod_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "k8s_list_namespaces",
            "description": "List all namespaces in the k3s cluster.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "k8s_get_deployments",
            "description": "List deployments and their replica status. Optionally filter by namespace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "description": "Kubernetes namespace to filter by. Leave empty for all."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "k8s_get_services",
            "description": "List Kubernetes services and their ports. Optionally filter by namespace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "description": "Kubernetes namespace to filter by. Leave empty for all."},
                },
                "required": [],
            },
        },
    },
]

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current, real-time, or factual information you do not already know. "
            "Only use this for explicit questions about news, current events, prices, weather, or facts. "
            "Do NOT use for greetings, casual conversation, opinions, or anything you can answer yourself."
        ),
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
        self.ollama = OllamaClient(config.ollama.url, config.ollama.model, config.ollama.routing_model)
        self.search = SearXNGClient(config.searxng_url) if config.searxng_url else None
        self.k8s = K8sClient() if config.k8s_enabled else None
        self._history: dict[tuple[str, str], deque[dict]] = {}
        # (room_id, sender) -> (image_bytes, timestamp)  — keyed per sender, not per room
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

        # Key by (room_id, sender) so each user's image is isolated
        self._pending_images[(room.room_id, event.sender)] = (response.body, time.monotonic())

    async def _on_message(self, room: MatrixRoom, event: RoomMessageText) -> None:
        if not self._started:
            return
        if event.sender == self.client.user_id:
            return

        # Deduplicate — nio can deliver the same event more than once
        if event.event_id in self._seen_events:
            return
        self._seen_events.add(event.event_id)
        if len(self._seen_events) > 2000:
            self._seen_events.clear()

        body = event.body.strip()
        prompt = _extract_prompt(body, self.config.bot_name, self.client.user_id, event.source)
        if prompt is None:
            return

        # LLM gate: confirm the message is actually directed at this bot
        try:
            if not await self.ollama.should_respond(self.config.bot_name, body):
                logger.debug("[%s] Gate rejected message from %s: %s", room.room_id, event.sender, body)
                return
        except Exception as exc:
            logger.warning("Gate call failed, proceeding anyway: %s", exc)

        logger.info("[%s] %s: %s", room.room_id, event.sender, prompt)

        # User commands (available to everyone)
        if prompt.strip().lower() == "reset":
            self._history.pop((room.room_id, event.sender), None)
            await self._send(room.room_id, "Conversation history cleared.")
            return

        # Admin commands
        if event.sender in self.config.admins:
            handled = await self._handle_admin_command(room.room_id, prompt)
            if handled:
                return

        # Check for a pending image from this specific sender
        image_b64: str | None = None
        pending = self._pending_images.pop((room.room_id, event.sender), None)
        if pending is not None:
            image_bytes, ts = pending
            if time.monotonic() - ts <= IMAGE_TTL:
                image_b64 = base64.b64encode(image_bytes).decode()
                logger.info("[%s] Pairing message with pending image from %s", room.room_id, event.sender)
            else:
                logger.info("[%s] Pending image from %s expired, discarding", room.room_id, event.sender)

        history = self._get_history(room.room_id, event.sender)
        if image_b64:
            history.append({"role": "user", "content": f"[image] {prompt}"})
        else:
            history.append({"role": "user", "content": prompt})

        await self.client.room_typing(room.room_id, typing_state=True, timeout=30_000)
        try:
            reply = await self._llm_reply(
                list(history), prompt=prompt, image_b64=image_b64, room=room, sender=event.sender
            )
        except Exception as exc:
            logger.error("LLM error: %s", exc)
            await self.client.room_typing(room.room_id, typing_state=False)
            await self._send(room.room_id, f"Error: {exc}")
            history.pop()
            return
        finally:
            await self.client.room_typing(room.room_id, typing_state=False)

        history.append({"role": "assistant", "content": reply})
        logger.info("[%s] -> %s", room.room_id, reply[:120])
        await self._send(room.room_id, reply)

    async def _llm_reply(
        self,
        history: list[dict],
        prompt: str,
        image_b64: str | None = None,
        room: MatrixRoom | None = None,
        sender: str | None = None,
    ) -> str:
        room_id = room.room_id if room else None
        messages = list(history)
        system = _build_system_prompt(self.config.system_prompt, room=room, sender=sender)
        messages.insert(0, {"role": "system", "content": system})

        if image_b64:
            return await self.ollama.chat_with_image(
                messages, image_b64, model=self.config.ollama.vision_model
            )

        if not self.search:
            return await self.ollama.chat(messages)

        # Bypass gate if user explicitly requests a search
        if not _explicit_search_request(prompt):
            needs_search = await self.ollama.should_search(prompt)
            if not needs_search and not self.k8s:
                return await self.ollama.chat(messages)

        # Build tool list
        tools = []
        if self.search:
            tools.append(WEB_SEARCH_TOOL)
        if self.k8s:
            tools.extend(K8S_TOOLS)

        if not tools:
            return await self.ollama.chat(messages)

        msg = await self.ollama.chat_with_tools(messages, tools)
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            return msg.get("content", "")

        # Execute tool calls
        tool_blocks: list[str] = []
        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})

            if name == "web_search" and self.search:
                query = args.get("query", "")
                logger.info("Web search: %s", query)
                if room_id:
                    try:
                        ack = await self.ollama.acknowledgment(system, query)
                        await self._send(room_id, ack)
                    except Exception:
                        pass
                results = await self.search.search(query)
                tool_blocks.append("\n\n".join(
                    f"{r['title']}\n{r['url']}\n{r['content']}" for r in results
                ))

            elif name.startswith("k8s_") and self.k8s:
                if sender not in self.config.admins:
                    tool_blocks.append("Access denied: only admins can use cluster tools.")
                    continue
                logger.info("K8s tool: %s %s", name, args)
                try:
                    if name == "k8s_check_health":
                        result = await self.k8s.check_service_health(args["service_name"])
                    elif name == "k8s_get_pods":
                        result = await self.k8s.get_pods(args.get("namespace", ""))
                    elif name == "k8s_get_logs":
                        result = await self.k8s.get_logs(
                            args["namespace"], args["pod_name"], args.get("tail_lines", 30)
                        )
                    elif name == "k8s_list_namespaces":
                        result = await self.k8s.list_namespaces()
                    elif name == "k8s_get_deployments":
                        result = await self.k8s.get_deployments(args.get("namespace", ""))
                    elif name == "k8s_get_services":
                        result = await self.k8s.get_services(args.get("namespace", ""))
                    else:
                        result = f"Unknown k8s tool: {name}"
                    tool_blocks.append(result)
                except Exception as exc:
                    tool_blocks.append(f"K8s error: {exc}")

        synthesis_instruction = (
            "You have been given tool results below. "
            "Synthesize the information into a clear, concise answer. "
            "Do not dump raw data — respond naturally and in character."
        )
        synthesis_messages = [
            {"role": "system", "content": system + "\n\n" + synthesis_instruction},
            {"role": "user", "content": prompt},
            {"role": "tool", "content": "\n\n---\n\n".join(tool_blocks)},
        ]
        return await self.ollama.chat(synthesis_messages)

    async def _handle_admin_command(self, room_id: str, prompt: str) -> bool:
        """Returns True if the prompt was an admin command and has been handled."""
        lower = prompt.strip().lower()

        if lower.startswith("avatar "):
            url = prompt.strip()[7:].strip()
            await self._set_avatar(room_id, url)
            return True

        return False

    async def _set_avatar(self, room_id: str, url: str) -> None:
        logger.info("Setting avatar from URL: %s", url)
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
                image_data = response.content
                content_type = response.headers.get("content-type", "").split(";")[0].strip()

            if not content_type.startswith("image/"):
                guessed, _ = mimetypes.guess_type(url)
                content_type = guessed if guessed and guessed.startswith("image/") else "image/jpeg"

            upload_resp, _ = await self.client.upload(
                io.BytesIO(image_data),
                content_type=content_type,
                filesize=len(image_data),
            )
            if not isinstance(upload_resp, UploadResponse):
                raise RuntimeError(f"Upload failed: {upload_resp}")

            await self.client.set_avatar(upload_resp.content_uri)
            logger.info("Avatar set to %s", upload_resp.content_uri)
            await self._send(room_id, "Avatar updated.")
        except Exception as exc:
            logger.error("Failed to set avatar: %s", exc)
            await self._send(room_id, f"Failed to set avatar: {exc}")

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


_EXPLICIT_SEARCH_PATTERNS = re.compile(
    r"\b(search|look\s+it\s+up|look\s+up|use\s+the\s+internet|google|find\s+out|check\s+online|browse)\b",
    re.IGNORECASE,
)

def _explicit_search_request(prompt: str) -> bool:
    return bool(_EXPLICIT_SEARCH_PATTERNS.search(prompt))


def _build_system_prompt(
    base: str,
    room: MatrixRoom | None = None,
    sender: str | None = None,
) -> str:
    now = datetime.now(timezone.utc)
    lines = [f"Current date and time: {now.strftime('%A, %Y-%m-%d %H:%M UTC')}"]

    if room:
        room_name = room.display_name or room.room_id
        lines.append(f"Room: {room_name}")

    if sender and room and sender in room.users:
        display_name = room.users[sender].display_name or sender
        lines.append(f"You are speaking with: {display_name} ({sender})")
    elif sender:
        lines.append(f"You are speaking with: {sender}")

    context = "\n".join(lines)
    if base:
        return f"{base}\n\n[Context]\n{context}"
    return f"[Context]\n{context}"


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
    mention = r"@?(?:" + re.escape(user_id) + r"|" + re.escape(bot_name) + r")"
    cleaned = re.sub(r"(?i)^\s*" + mention + r"[,:\s]*", "", body)
    cleaned = re.sub(r"(?i)[,:\s]*" + mention + r"\s*$", "", cleaned)
    return cleaned.strip() or body.strip()
