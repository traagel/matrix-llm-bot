from __future__ import annotations

import io
import logging
import mimetypes
from collections.abc import Awaitable, Callable

import httpx
from nio import AsyncClient, UploadResponse

from ..config import Config
from ..integrations.k8s import K8sClient

logger = logging.getLogger(__name__)

SendFn = Callable[[str, str], Awaitable[None]]


class CommandHandler:
    def __init__(
        self,
        config: Config,
        matrix_client: AsyncClient,
        k8s: K8sClient | None,
        search_enabled: bool,
        send_fn: SendFn,
    ) -> None:
        self.config = config
        self.matrix_client = matrix_client
        self.k8s = k8s
        self.search_enabled = search_enabled
        self._send = send_fn

    async def handle(self, room_id: str, prompt: str, sender: str) -> bool:
        """Returns True if the prompt was handled as a builtin command."""
        lower = prompt.strip().lower()

        if lower == "help":
            await self._handle_help(room_id)
            return True

        if lower == "tools":
            await self._handle_tools(room_id)
            return True

        if lower == "status":
            await self._handle_status(room_id)
            return True

        if lower.startswith("k8s ") and self.k8s:
            await self._handle_k8s(room_id, prompt)
            return True

        if sender in self.config.admins and lower.startswith("avatar "):
            url = prompt.strip()[7:].strip()
            await self._handle_avatar(room_id, url)
            return True

        return False

    async def _handle_help(self, room_id: str) -> None:
        n = self.config.bot_name
        lines = [f"{n} commands:"]
        lines.append(f"  {n} help              — this message")
        lines.append(f"  {n} tools             — show model and feature config")
        lines.append(f"  {n} status            — on/off summary of all features")
        lines.append(f"  {n} reset             — clear your conversation history")
        if self.k8s:
            lines.append(f"  {n} k8s health              — health of all pods")
            lines.append(f"  {n} k8s health <service>    — health of matching pods")
            lines.append(f"  {n} k8s status <service>    — deployment + pod detail")
            lines.append(f"  {n} k8s version <service>   — image tags")
        if self.search_enabled:
            lines.append(f"  {n} <question>     — answer with web search if needed")
        if self.config.ollama.vision_model:
            lines.append(f"  [upload image] then {n} <question>  — vision")
        await self._send(room_id, "\n".join(lines))

    async def _handle_tools(self, room_id: str) -> None:
        lines = [f"{self.config.bot_name} config:"]
        lines.append(f"  model: {self.config.ollama.model}")
        if self.config.ollama.routing_model and self.config.ollama.routing_model != self.config.ollama.model:
            lines.append(f"  routing model: {self.config.ollama.routing_model}")
        if self.config.ollama.vision_model:
            lines.append(f"  vision: {self.config.ollama.vision_model}")
        if self.search_enabled:
            lines.append(f"  web search: {self.config.searxng_url}")
        if self.k8s:
            lines.append(f"  k8s services: {', '.join(self.config.k8s_services) or 'none configured'}")
        if self.config.peer_bots:
            lines.append(f"  peer bots: {', '.join(self.config.peer_bots)}")
        lines.append(f"  history: {self.config.history_size} messages per user")
        await self._send(room_id, "\n".join(lines))

    async def _handle_status(self, room_id: str) -> None:
        lines = [f"{self.config.bot_name} status:"]
        lines.append(f"  model: {self.config.ollama.model}")
        lines.append(f"  search: {'on' if self.search_enabled else 'off'}")
        lines.append(f"  k8s: {'on' if self.k8s else 'off'}")
        lines.append(f"  vision: {'on' if self.config.ollama.vision_model else 'off'}")
        lines.append(f"  history: {self.config.history_size}")
        await self._send(room_id, "\n".join(lines))

    async def _handle_k8s(self, room_id: str, prompt: str) -> None:
        rest = prompt.strip()[4:].strip()
        if not rest:
            return
        rest_lower = rest.lower()
        if rest_lower.startswith("version "):
            service = rest[8:].strip()
            cmd = "version"
        elif rest_lower == "health" or rest_lower.startswith("health "):
            service = rest[7:].strip() if len(rest) > 6 else ""
            cmd = "health"
        elif rest_lower.startswith("status "):
            service = rest[7:].strip()
            cmd = "status"
        else:
            service = rest
            cmd = "status"
        resolved = self.config.k8s_aliases.get(service.lower(), service)
        logger.info("[%s] Builtin k8s %s for %r", room_id, cmd, resolved)
        try:
            if cmd == "version":
                result = await self.k8s.service_version(resolved)
            elif cmd == "health":
                result = await self.k8s.service_health(resolved)
            else:
                result = await self.k8s.service_status(resolved)
        except Exception as exc:
            result = f"K8s error: {exc}"
        await self._send(room_id, result)

    async def _handle_avatar(self, room_id: str, url: str) -> None:
        logger.info("Setting avatar from URL: %s", url)
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as http:
                response = await http.get(url)
                response.raise_for_status()
                image_data = response.content
                content_type = response.headers.get("content-type", "").split(";")[0].strip()

            if not content_type.startswith("image/"):
                guessed, _ = mimetypes.guess_type(url)
                content_type = guessed if guessed and guessed.startswith("image/") else "image/jpeg"

            upload_resp, _ = await self.matrix_client.upload(
                io.BytesIO(image_data),
                content_type=content_type,
                filesize=len(image_data),
            )
            if not isinstance(upload_resp, UploadResponse):
                raise RuntimeError(f"Upload failed: {upload_resp}")

            await self.matrix_client.set_avatar(upload_resp.content_uri)
            logger.info("Avatar set to %s", upload_resp.content_uri)
            await self._send(room_id, "Avatar updated.")
        except Exception as exc:
            logger.error("Failed to set avatar: %s", exc)
            await self._send(room_id, f"Failed to set avatar: {exc}")
