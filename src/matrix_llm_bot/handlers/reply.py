from __future__ import annotations

import logging
import re
import time
from collections import deque
from datetime import UTC, datetime

from nio import MatrixRoom

from ..config import Config
from ..integrations.k8s import K8S_TOOLS, K8sClient
from ..integrations.ollama import OllamaClient
from ..integrations.search import WEB_SEARCH_TOOL, SearXNGClient

logger = logging.getLogger(__name__)

_EXPLICIT_SEARCH_PATTERNS = re.compile(
    r"\b(search|look\s+it\s+up|look\s+up|use\s+the\s+internet|google|find\s+out|check\s+online|browse)\b",
    re.IGNORECASE,
)


def build_system_prompt(
    base: str,
    room: MatrixRoom | None = None,
    sender: str | None = None,
) -> str:
    now = datetime.now(UTC)
    parts: list[str] = []

    if base:
        parts.append(base)

    meta: list[str] = [f"Today is {now.strftime('%A, %Y-%m-%d')}. Current time: {now.strftime('%H:%M UTC')}."]
    if room:
        room_name = room.display_name or room.room_id
        meta.append(f"This conversation is in the room: {room_name}.")
    if sender and room and sender in room.users:
        display_name = room.users[sender].display_name or sender
        meta.append(f"You are speaking with {display_name}.")
    elif sender:
        meta.append(f"You are speaking with {sender}.")

    parts.append(" ".join(meta))
    parts.append(
        "Do not fabricate conversation turns, invent messages from other participants, "
        "or reproduce structural labels like '[Context]' or '[System]' in your replies. "
        "Respond only to what was actually said."
    )
    return "\n\n".join(parts)


class ConversationHistory:
    def __init__(self, maxlen: int) -> None:
        # maxlen=0 means unlimited
        self._maxlen = maxlen if maxlen > 0 else None
        self._store: dict[tuple[str, str], deque[dict]] = {}

    def get(self, room_id: str, sender: str) -> deque[dict]:
        key = (room_id, sender)
        if key not in self._store:
            self._store[key] = deque(maxlen=self._maxlen)
        return self._store[key]

    def clear(self, room_id: str, sender: str) -> None:
        self._store.pop((room_id, sender), None)


class ClusterMapCache:
    def __init__(self, k8s: K8sClient, ttl: float = 60.0) -> None:
        self._k8s = k8s
        self._ttl = ttl
        self._value: str = ""
        self._ts: float = 0.0

    async def get(self) -> str:
        now = time.monotonic()
        if self._value and (now - self._ts) < self._ttl:
            return self._value
        self._value = await self._k8s.cluster_map()
        self._ts = now
        return self._value


class ReplyHandler:
    def __init__(
        self,
        config: Config,
        ollama: OllamaClient,
        k8s: K8sClient | None,
        search: SearXNGClient | None,
    ) -> None:
        self.config = config
        self.ollama = ollama
        self.k8s = k8s
        self.search = search
        self._cache = ClusterMapCache(k8s) if k8s else None

    async def reply(
        self,
        history: list[dict],
        prompt: str,
        image_b64: str | None = None,
        room: MatrixRoom | None = None,
        sender: str | None = None,
    ) -> str:
        room_id = room.room_id if room else None
        messages = list(history)
        system = build_system_prompt(self.config.system_prompt, room=room, sender=sender)
        messages.insert(0, {"role": "system", "content": system})

        if image_b64:
            logger.info("[%s] Vision path: model=%s", room_id, self.config.ollama.vision_model)
            return await self.ollama.chat_with_image(messages, image_b64, model=self.config.ollama.vision_model)

        k8s_resolved = False
        is_k8s = self.k8s and _is_k8s_query(
            prompt, self.config.k8s_keywords, self.config.k8s_services, self.config.k8s_aliases
        )
        logger.debug("[%s] K8s query detection: %s", room_id, bool(is_k8s))

        if is_k8s:
            k8s_context = await self._resolve_k8s_query(prompt, sender=sender)
            if k8s_context:
                logger.info("[%s] K8s context injected: %s", room_id, k8s_context[:120])
                messages.insert(1, {"role": "system", "content": f"Current cluster data: {k8s_context}"})
                k8s_resolved = True
            else:
                logger.warning("[%s] K8s query matched but returned no context", room_id)

        if k8s_resolved:
            logger.debug("[%s] Skipping search: k8s already resolved", room_id)
            return await self.ollama.chat(messages)

        if not self.search:
            logger.debug("[%s] Plain chat (search disabled)", room_id)
            return await self.ollama.chat(messages)

        if _explicit_search_request(prompt):
            logger.info("[%s] Explicit search request detected", room_id)
            needs_search = True
        else:
            needs_search = await self.ollama.should_search(prompt)

        if not needs_search:
            logger.debug("[%s] Search gate: not needed, plain chat", room_id)
            return await self.ollama.chat(messages)

        logger.info("[%s] Search gate: search needed", room_id)
        msg = await self.ollama.chat_with_tools(messages, [WEB_SEARCH_TOOL])
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            logger.debug("[%s] Web search: model returned no tool calls, using direct response", room_id)
            return msg.get("content", "")

        messages.append(msg)
        for call in tool_calls:
            fn = call.get("function", {})
            if fn.get("name") == "web_search":
                query = fn.get("arguments", {}).get("query", "")
                logger.info("Web search: %s", query)
                results = await self.search.search(query)
                result_text = "\n\n".join(f"{r['title']}\n{r['url']}\n{r['content']}" for r in results)
                messages.append({"role": "tool", "content": result_text, "tool_name": "web_search"})

        return await self.ollama.chat(messages)

    async def _resolve_k8s_query(self, prompt: str, sender: str | None = None) -> str:
        cluster_map = await self._cache.get()

        translated = prompt
        for alias, real_name in self.config.k8s_aliases.items():
            before = translated
            translated = re.sub(re.escape(alias), real_name, translated, flags=re.IGNORECASE)
            if translated != before:
                logger.debug("K8s alias translated: %r -> %r", alias, real_name)

        alias_context = ""
        if self.config.k8s_aliases:
            mappings = ", ".join(f'"{k}" = {v}' for k, v in self.config.k8s_aliases.items())
            alias_context = f"\nService aliases: {mappings}. Translate these names when looking up services.\n"

        live_checks: list[str] = []
        checked: set[str] = set()
        for service in self.config.k8s_services:
            if service.lower() in translated.lower() and service not in checked:
                checked.add(service)
                logger.info("K8s pre-fetch health: %s", service)
                try:
                    result = await self.k8s.check_service_health(service)
                    logger.debug("K8s health result for %s: %s", service, result[:120])
                    live_checks.append(result)
                except Exception as exc:
                    logger.warning("K8s health check failed for %s: %s", service, exc)
                    live_checks.append(f"Could not check {service}: {exc}")
        for alias, real_name in self.config.k8s_aliases.items():
            if alias.lower() in prompt.lower() and real_name not in checked:
                checked.add(real_name)
                logger.info("K8s pre-fetch health (via alias %r): %s", alias, real_name)
                try:
                    result = await self.k8s.check_service_health(real_name)
                    logger.debug("K8s health result for %s: %s", real_name, result[:120])
                    live_checks.append(result)
                except Exception as exc:
                    logger.warning("K8s health check failed for %s: %s", real_name, exc)
                    live_checks.append(f"Could not check {real_name}: {exc}")

        if not live_checks:
            logger.debug("K8s: no specific services matched in prompt, relying on cluster_map only")

        live_section = "\nLive service checks:\n" + "\n\n".join(live_checks) + "\n" if live_checks else ""

        k8s_system = (
            "You are a Kubernetes cluster assistant. "
            "Below is the current cluster state.\n\n"
            f"{cluster_map}\n"
            f"{live_section}"
            f"{alias_context}\n"
            "Answer the user's question using ONLY this data. "
            "Include exact image tags as versions. "
            "Reply in plain prose — one or two sentences. "
            "No JSON, no markdown, no bullet points, no structured data."
        )
        messages: list[dict] = [
            {"role": "system", "content": k8s_system},
            {"role": "user", "content": translated},
        ]

        is_log_request = any(kw in translated.lower() for kw in ("log", "logs"))
        if is_log_request:
            logger.info("K8s log request detected, sender=%s is_admin=%s", sender, sender in self.config.admins)
            if sender not in self.config.admins:
                logger.info("K8s log request denied: %s is not an admin", sender)
                return "Access denied: only admins can view logs."
            log_tools = [t for t in K8S_TOOLS if t["function"]["name"] == "k8s_get_logs"]
            msg = await self.ollama.chat_with_tools(messages, log_tools)
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                messages.append(msg)
                for call in tool_calls:
                    fn = call.get("function", {})
                    name = fn.get("name", "")
                    args = fn.get("arguments", {})
                    logger.info("K8s tool call: %s %s", name, args)
                    try:
                        result = await self.k8s.dispatch_tool(name, args)
                        logger.debug("K8s tool result: %s", result[:120])
                    except Exception as exc:
                        logger.error("K8s tool error (%s): %s", name, exc)
                        result = f"K8s error: {exc}"
                    messages.append({"role": "tool", "content": result})
                return await self.ollama.chat(messages)
            logger.debug("K8s log request: model returned no tool calls")
            return msg.get("content", "")

        logger.debug("K8s: synthesizing answer from cluster_map + live checks")
        return await self.ollama.chat(messages)


def _is_k8s_query(
    prompt: str,
    keywords: list[str],
    services: list[str],
    aliases: dict[str, str] | None = None,
) -> bool:
    words = set(re.sub(r"[^\w\s-]", " ", prompt.lower()).split())
    all_terms = {k.lower() for k in keywords} | {s.lower() for s in services} | {a.lower() for a in (aliases or {})}
    return bool(words & all_terms)


def _explicit_search_request(prompt: str) -> bool:
    return bool(_EXPLICIT_SEARCH_PATTERNS.search(prompt))
