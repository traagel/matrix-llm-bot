from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


def _parse_yes_no(answer: str, default: bool) -> bool:
    answer = answer.strip().upper()
    yes_pos = answer.find("YES")
    no_pos = answer.find("NO")
    if yes_pos == -1 and no_pos == -1:
        return default
    if yes_pos == -1:
        return False
    if no_pos == -1:
        return True
    return yes_pos < no_pos


class OllamaClient:
    def __init__(self, url: str, model: str, routing_model: str = "") -> None:
        self.url = url.rstrip("/")
        self.model = model
        self._routing_model = routing_model or model
        self._client = httpx.AsyncClient(timeout=120.0)

    async def chat(self, messages: list[dict], model: str | None = None) -> str:
        response = await self._client.post(
            f"{self.url}/api/chat",
            json={"model": model or self.model, "messages": messages, "stream": False},
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    async def chat_with_image(self, messages: list[dict], image_b64: str, model: str | None = None) -> str:
        msgs = list(messages)
        if msgs and msgs[-1]["role"] == "user":
            msgs[-1] = {**msgs[-1], "images": [image_b64]}
        response = await self._client.post(
            f"{self.url}/api/chat",
            json={"model": model or self.model, "messages": msgs, "stream": False},
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    async def should_search(self, message: str) -> bool:
        """Decide if a message requires a web search to answer."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You decide if a message requires a web search to answer. "
                    "Answer YES only if the message explicitly asks for: current news, live prices, "
                    "today's weather, recent events, or time-sensitive facts you cannot know. "
                    "Answer NO for: greetings, small talk, opinions, jokes, questions about what was "
                    "previously said in the conversation, memory questions ('what did I tell you', "
                    "'do you remember'), personal questions, or anything answerable from general knowledge. "
                    "If the message asks about current events, prices, or news — answer YES. "
                    "Reply with a single word: YES or NO. Nothing else."
                ),
            },
            {"role": "user", "content": message},
        ]
        response = await self._client.post(
            f"{self.url}/api/chat",
            json={"model": self._routing_model, "messages": messages, "stream": False},
        )
        response.raise_for_status()
        raw = response.json()["message"]["content"]
        result = _parse_yes_no(raw, default=False)
        logger.debug("Search gate: %r -> %s (raw: %r)", message[:80], "YES" if result else "NO", raw[:40])
        return result

    async def should_respond(self, bot_name: str, message: str, peer_bots: list[str] | None = None) -> bool:
        """Ask the model if this message is directed at bot_name and not at a peer bot."""
        others = peer_bots or []
        others_line = (
            f" Other bots in this room: {', '.join(others)}."
            f" Answer NO if the message is clearly directed at one of them instead."
            if others
            else ""
        )
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a routing assistant deciding if a chat message is directly addressed "
                    f"to '{bot_name}' (match case-insensitively, including nicknames or partial matches).{others_line} "
                    f"Answer YES if '{bot_name}' is mentioned, named, or appears to be the one being spoken to. "
                    f"When in doubt, answer YES. "
                    f"Answer NO only if the message is clearly addressed to a specific other person or bot and '{bot_name}' is not involved. "
                    f"Reply with a single word: YES or NO. Nothing else."
                ),
            },
            {"role": "user", "content": message},
        ]
        response = await self._client.post(
            f"{self.url}/api/chat",
            json={"model": self._routing_model, "messages": messages, "stream": False},
        )
        response.raise_for_status()
        raw = response.json()["message"]["content"]
        result = _parse_yes_no(raw, default=True)
        logger.debug("Respond gate: %r -> %s (raw: %r)", message[:80], "YES" if result else "NO", raw[:40])
        return result

    async def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        """Returns the raw assistant message dict (may contain tool_calls)."""
        tool_names = [t["function"]["name"] for t in tools]
        logger.debug("chat_with_tools: offering %s", tool_names)
        response = await self._client.post(
            f"{self.url}/api/chat",
            json={"model": self.model, "messages": messages, "tools": tools, "stream": False},
        )
        response.raise_for_status()
        msg = response.json()["message"]
        calls = [c.get("function", {}).get("name") for c in (msg.get("tool_calls") or [])]
        logger.debug("chat_with_tools: model called %s", calls if calls else "no tools")
        return msg

    async def aclose(self) -> None:
        await self._client.aclose()
